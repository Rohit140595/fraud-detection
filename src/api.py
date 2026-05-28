from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.model import load_model
from src.features import ( 
    add_user_proxy, compute_time_features,
    compute_email_features, compute_amount_features,
    compute_d_features, FREE_EMAIL_DOMAINS
)

# ── Startup / shutdown ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.cat_features = load_model()
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


# ── Request / Response schemas ───────────────────────────────────
class TransactionRequest(BaseModel):
    # Core fields (required)
    TransactionAmt: float
    card1:          int
    addr1:          float
    TransactionDT:  int
    ProductCD:      str

    # Optional raw fields the model uses
    card4:          Optional[str] = None
    card6:          Optional[str] = None
    P_emaildomain:  Optional[str] = None
    R_emaildomain:  Optional[str] = None
    D1:             Optional[float] = None

    # Pre-computed behavioral features (caller fetches from history)
    velocity_1h:      Optional[float] = 0.0
    velocity_24h:     Optional[float] = 0.0
    velocity_7d:      Optional[float] = 0.0
    hist_mean_amt:    Optional[float] = None
    card_unique_addr: Optional[float] = 0.0
    card_unique_amt:  Optional[float] = 0.0


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    threshold:         float


# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    # TODO: return status ok
    return {'status': 'ok'}

@app.post("/predict", response_model=PredictResponse)
def predict(request: TransactionRequest):
    # Initialize empty features dict
    features_dict = {}
    # TODO: 1. build feature dict from request
    features_dict['TransactionAmt']     = request.TransactionAmt
    features_dict['card1']              = request.card1    
    features_dict['addr1']              = request.addr1
    features_dict['TransactionDT']      = request.TransactionDT   
    features_dict['ProductCD']          = request.ProductCD
    features_dict['velocity_1h']        = request.velocity_1h
    features_dict['velocity_24h']       = request.velocity_24h
    features_dict['velocity_7d']        = request.velocity_7d
    features_dict['hist_mean_amt']      = request.hist_mean_amt
    features_dict['card_unique_addr']   = request.card_unique_addr
    features_dict['card_unique_amt']    = request.card_unique_amt
    features_dict['card4']              = request.card4
    features_dict['card6']              = request.card6
    features_dict['P_emaildomain']      = request.P_emaildomain
    features_dict['R_emaildomain']      = request.R_emaildomain
    features_dict['D1']                 = request.D1
    
    hist = features_dict['hist_mean_amt'] or features_dict['TransactionAmt']
    features_dict['amt_deviation'] = features_dict['TransactionAmt'] - hist
    
    # 2. compute static features (hour_of_day, amt_cents, etc.)
    # 3. create a single-row DataFrame
    df = pd.DataFrame([features_dict])
    df = add_user_proxy(df)
    df = compute_time_features(df)
    df = compute_email_features(df)
    df = compute_amount_features(df)
    df = compute_d_features(df)

    # 4. score with app.state.model
    # Align to model's expected features — fills missing columns with NaN
    df = df.reindex(columns=app.state.model.feature_names_in_)
    # Cast exactly the columns the model was trained with as 'category' — must match training dtype
    for col in app.state.cat_features:
        if col in df.columns:
            df[col] = df[col].astype("category")
    fraud_probability = app.state.model.predict_proba(df)[0, 1]
    is_fraud = bool(app.state.model.predict(df)[0])
    
    # 5. return PredictResponse
    response = PredictResponse(fraud_probability = fraud_probability
                               , is_fraud = is_fraud
                               , threshold = 0.5)
    
    return response