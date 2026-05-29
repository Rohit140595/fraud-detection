"""
FastAPI serving layer for real-time fraud detection.

Design decisions:
  - Ensemble loaded once at startup via lifespan — avoids per-request disk I/O.
  - Behavioral features (velocity, hist_mean_amt, card aggregates) are caller-supplied:
    the API has no database, so the caller is responsible for fetching history.
    Defaults to zero/None so the endpoint works without a history store.
  - Static features (time, email, amount, D columns) are computed server-side from
    raw inputs — they require no historical context and are cheap to recompute.
  - TransactionRequest uses extra='allow' so payment-processor features (V/C/M columns)
    can be forwarded without declaring every field explicitly. model_dump() captures
    them all and passed through to the ensemble via reindex.
  - _coerce_dtypes coerces all non-string columns to float64 — a None in any optional
    field causes pandas to infer object dtype, which models reject at inference time.
  - cat_cols saved alongside the ensemble ensures the serving layer ordinal-encodes
    exactly the same columns with the same codes used during training.
"""

from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.ensemble import load_ensemble
from src.features import (
    add_user_proxy, compute_time_features,
    compute_email_features, compute_amount_features,
    compute_d_features, compute_identity_features,
)

# String columns that must stay as object — everything else is numeric
_STRING_COLS = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
               "DeviceType", "DeviceInfo"}


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all non-string columns to float64, converting None → NaN.

    A single None in a column makes pandas infer object dtype for that column.
    Coercing everything except known string columns ensures all models receive
    correct numeric dtypes regardless of which extra fields are passed.
    """
    for col in df.columns:
        if col not in _STRING_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Startup / shutdown ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ensemble once at startup and attach it to app.state.

    Loading inside lifespan (not at module import time) means the ensemble is only
    read from disk once per server process, and the app fails fast on startup if
    the model file is missing — rather than failing silently on the first request.
    """
    (app.state.models, app.state.cat_cols, app.state.cat_encoders,
     app.state.calibrator, app.state.threshold) = load_ensemble()
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
        available at inference time; missing values are handled by the models.
      - Behavioral features: pre-computed by the caller from transaction history
        (e.g., a Redis lookup). Defaulted to 0/None so the endpoint works without
        a history store, at the cost of slightly less accurate scores.

    extra='allow' lets payment-processor features (V/C/M columns) pass through
    without declaring each one explicitly. They are captured by model_dump() and
    forwarded to the ensemble via reindex in the predict endpoint.
    """
    model_config = ConfigDict(extra='allow')

    # Core fields — required
    TransactionAmt: float
    card1:          int
    TransactionDT:  int
    ProductCD:      str

    # addr1 is the billing zip code — present for most transactions but not all
    addr1: Optional[float] = None

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

    fraud_probability: soft-vote average of predict_proba across all ensemble models —
                       useful for ranking or applying a custom threshold downstream.
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
    Score a single transaction for fraud using the soft-voting ensemble.

    Pipeline:
      1. Flatten request fields into a dict (includes V/C/M passthrough columns).
      2. Compute amt_deviation from hist_mean_amt (falls back to current amount
         if no history is available, giving a deviation of 0).
      3. Create a single-row DataFrame and coerce numeric dtypes (None → NaN).
      4. Run static feature engineering: user proxy, time, email, amount, D1.
      5. Ordinal-encode categorical columns using the codes saved at training time.
      6. Align columns to the ensemble's expected feature set (reindex fills gaps
         with NaN — all three models handle NaN natively).
      7. Average predict_proba across LightGBM, XGBoost, and CatBoost (soft vote).

    Args:
        request: Validated TransactionRequest payload.

    Returns:
        PredictResponse with fraud_probability, is_fraud, and threshold.
    """
    # 1. Flatten request — captures declared fields AND extra V/C/M columns.
    features_dict = request.model_dump()

    # 2. Amount deviation — how far this transaction is from the user's historical mean.
    hist = features_dict["hist_mean_amt"] or features_dict["TransactionAmt"]
    features_dict["amt_deviation"] = features_dict["TransactionAmt"] - hist

    # 3. Build DataFrame and fix dtype inference before feature engineering.
    df = pd.DataFrame([features_dict])
    df = _coerce_dtypes(df)

    # 4. Static feature engineering — mirrors the training pipeline in features.py.
    df = add_user_proxy(df)
    df = compute_time_features(df)
    df = compute_email_features(df)
    df = compute_amount_features(df)
    df = compute_d_features(df)
    df = compute_identity_features(df)

    # 5. Ordinal-encode categoricals using the exact same label→code mapping that
    #    was fit on X_train during training. Unknown/missing values → -1 (same
    #    sentinel used by encode_for_ensemble for unseen categories).
    for col in app.state.cat_cols:
        if col in df.columns:
            encoder = app.state.cat_encoders.get(col, {})
            df[col] = df[col].map(encoder).fillna(-1).astype(int)

    # 6 & 7. Score: average predict_proba across all ensemble models (soft vote).
    lgbm_m = app.state.models["lgbm"]
    xgb_m  = app.state.models["xgb"]
    cat_m  = app.state.models["catboost"]

    lgbm_df = df.reindex(columns=lgbm_m.feature_names_in_)
    xgb_df  = df.reindex(columns=xgb_m.feature_names_in_)
    cat_df  = df.reindex(columns=cat_m.feature_names_)

    # CatBoost rejects float NaN for declared cat_features — reindex fills missing
    # columns with NaN, so explicitly replace with -1 (unseen-category sentinel).
    for col in app.state.cat_cols:
        if col in cat_df.columns:
            cat_df[col] = cat_df[col].fillna(-1).astype(int)

    raw_prob = (
        lgbm_m.predict_proba(lgbm_df)[0, 1]
        + xgb_m.predict_proba(xgb_df)[0, 1]
        + cat_m.predict_proba(cat_df)[0, 1]
    ) / 3.0

    # 8. Apply calibration if a calibrator was saved with the model.
    if app.state.calibrator is not None:
        fraud_probability = float(app.state.calibrator.predict([raw_prob])[0])
    else:
        fraud_probability = float(raw_prob)

    is_fraud = bool(fraud_probability >= app.state.threshold)

    return PredictResponse(
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        threshold=app.state.threshold,
    )
