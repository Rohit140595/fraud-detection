"""
FastAPI serving layer for real-time fraud detection.

Design decisions:
  - Model loaded once at startup via lifespan — avoids per-request disk I/O.
  - Behavioral features (velocity, hist_mean_amt, card aggregates) are caller-supplied:
    the API has no database, so the caller is responsible for fetching history.
    Defaults to zero/None so the endpoint works without a history store.
  - Static features (time, email, amount, D1) are computed server-side from raw inputs —
    they require no historical context and are cheap to recompute.
  - _coerce_dtypes runs before feature engineering: a single None in an optional field
    causes pandas to infer object dtype, which LightGBM rejects.
  - cat_features saved alongside the model ensures the serving layer casts exactly
    the same columns to 'category' dtype that were used during training.
"""

from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import load_model
from src.features import (
    add_user_proxy, compute_time_features,
    compute_email_features, compute_amount_features,
    compute_d_features,
)

# Columns that must be numeric dtype in the inference DataFrame.
# Optional fields default to None, which causes pandas to infer object dtype
# for those columns in a single-row DataFrame — LightGBM rejects non-numeric dtypes.
_NUMERIC_COLS = [
    "TransactionAmt", "card1", "addr1", "TransactionDT",
    "velocity_1h", "velocity_24h", "velocity_7d",
    "hist_mean_amt", "amt_deviation",
    "card_unique_addr", "card_unique_amt", "D1",
]


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast numeric columns to float64, converting None → NaN.

    Pydantic validates request types; this step fixes a downstream pandas artifact.
    When a single-row DataFrame is built from a dict containing None, pandas cannot
    infer the dtype and falls back to object. pd.to_numeric(..., errors='coerce')
    converts None/non-numeric values to NaN and returns float64, which LightGBM accepts.

    Args:
        df: Single-row DataFrame built from the incoming request dict.

    Returns:
        DataFrame with all _NUMERIC_COLS cast to float64.
    """
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Startup / shutdown ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once at startup and attach it to app.state.

    Loading inside lifespan (not at module import time) means the model is only
    read from disk once per server process, and the app fails fast on startup if
    the model file is missing — rather than failing silently on the first request.
    """
    app.state.model, app.state.cat_features = load_model()
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


# ── Request / Response schemas ──────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """
    Incoming transaction payload.

    Split into three groups:
      - Core fields: always required — minimum needed to construct the user proxy
        and run feature engineering.
      - Optional raw fields: present in the original dataset but not always
        available at inference time; missing values are handled by the model.
      - Behavioral features: pre-computed by the caller from transaction history
        (e.g., a Redis lookup). Defaulted to 0/None so the endpoint works without
        a history store, at the cost of slightly less accurate scores.
    """
    # Core fields — required
    TransactionAmt: float
    card1:          int
    addr1:          float
    TransactionDT:  int
    ProductCD:      str

    # Optional raw fields the model uses directly
    card4:          Optional[str]   = None
    card6:          Optional[str]   = None
    P_emaildomain:  Optional[str]   = None
    R_emaildomain:  Optional[str]   = None
    D1:             Optional[float] = None

    # Behavioral features — caller fetches from history store (e.g., Redis)
    # Default to 0/None so the endpoint is usable without a history backend.
    velocity_1h:      Optional[float] = 0.0
    velocity_24h:     Optional[float] = 0.0
    velocity_7d:      Optional[float] = 0.0
    hist_mean_amt:    Optional[float] = None   # None → deviation computed vs. current amount
    card_unique_addr: Optional[float] = 0.0
    card_unique_amt:  Optional[float] = 0.0


class PredictResponse(BaseModel):
    """
    Fraud score returned for a single transaction.

    fraud_probability: raw model output from predict_proba — useful for ranking
                       or applying a custom threshold downstream.
    is_fraud:          binary decision at the default 0.5 threshold.
    threshold:         threshold used to derive is_fraud, returned explicitly so
                       the caller knows what operating point was applied.
    """
    fraud_probability: float
    is_fraud:          bool
    threshold:         float


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check — returns 200 if the server is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: TransactionRequest):
    """
    Score a single transaction for fraud.

    Pipeline:
      1. Build a flat feature dict from the request fields.
      2. Compute amt_deviation from hist_mean_amt (falls back to current amount
         if no history is available, giving a deviation of 0).
      3. Create a single-row DataFrame and coerce numeric dtypes (None → NaN).
      4. Run static feature engineering: user proxy, time, email, amount, D1.
      5. Align columns to the model's expected feature set (reindex fills gaps with NaN).
      6. Cast categorical columns to 'category' dtype — must match training dtype
         exactly or LightGBM raises a categorical_feature mismatch error.
      7. Score and return fraud_probability + is_fraud decision.

    Args:
        request: Validated TransactionRequest payload.

    Returns:
        PredictResponse with fraud_probability, is_fraud, and threshold.
    """
    # 1. Flatten request into a dict — single source of truth before DataFrame creation
    features_dict = {
        "TransactionAmt":   request.TransactionAmt,
        "card1":            request.card1,
        "addr1":            request.addr1,
        "TransactionDT":    request.TransactionDT,
        "ProductCD":        request.ProductCD,
        "card4":            request.card4,
        "card6":            request.card6,
        "P_emaildomain":    request.P_emaildomain,
        "R_emaildomain":    request.R_emaildomain,
        "D1":               request.D1,
        "velocity_1h":      request.velocity_1h,
        "velocity_24h":     request.velocity_24h,
        "velocity_7d":      request.velocity_7d,
        "hist_mean_amt":    request.hist_mean_amt,
        "card_unique_addr": request.card_unique_addr,
        "card_unique_amt":  request.card_unique_amt,
    }

    # 2. Amount deviation — how far this transaction is from the user's historical mean.
    # If no history is available, fall back to the current amount so deviation = 0.
    hist = features_dict["hist_mean_amt"] or features_dict["TransactionAmt"]
    features_dict["amt_deviation"] = features_dict["TransactionAmt"] - hist

    # 3. Build DataFrame and fix dtype inference before any feature engineering
    df = pd.DataFrame([features_dict])
    df = _coerce_dtypes(df)   # None → NaN; prevents object-typed numeric columns

    # 4. Static feature engineering — mirrors the training pipeline in features.py
    df = add_user_proxy(df)          # card_addr = card1 + addr1 user identity proxy
    df = compute_time_features(df)   # hour_of_day, day_of_week from TransactionDT
    df = compute_email_features(df)  # email_domain_match, is_free_email
    df = compute_amount_features(df) # amt_cents, is_round_amt
    df = compute_d_features(df)      # log_d1, d1_null_flag

    # 5. Align to the exact feature set the model was trained on.
    # reindex fills any column the model expects but the request didn't supply with NaN —
    # LightGBM handles NaN natively, so no imputation step is needed here.
    df = df.reindex(columns=app.state.model.feature_names_in_)

    # 6. Cast categorical columns to 'category' dtype.
    # app.state.cat_features is the list saved alongside the model at training time,
    # ensuring we cast exactly the same columns and avoid a LightGBM dtype mismatch.
    for col in app.state.cat_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 7. Score
    fraud_probability = float(app.state.model.predict_proba(df)[0, 1])
    is_fraud = bool(app.state.model.predict(df)[0])

    return PredictResponse(
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        threshold=0.5,
    )
