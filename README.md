# Fraud Detection — IEEE-CIS Dataset

End-to-end ML pipeline for real-time transaction fraud detection, built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

## Results

| Metric | Score |
|---|---|
| **PR-AUC** | **0.5603** |
| ROC-AUC | 0.9100 |
| Fraud rate (test set) | 3.44% |

Primary metric is PR-AUC — more informative than ROC-AUC on heavily imbalanced data since it focuses on the minority class and is not inflated by the large number of true negatives.

## Project Structure

```
fraud-detection/
├── config.yaml                      # Central config: paths, tuning, ensemble settings
├── data/                            # Raw CSVs (not tracked — download from Kaggle)
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── test_features.parquet        # Pre-engineered test set for streaming simulation
├── models/                          # Saved artifacts (not tracked — regenerable)
│   ├── ensemble_fraud.pkl           # Trained ensemble (LightGBM + XGBoost + CatBoost)
│   └── shap_features.json           # SHAP feature selection cache
├── notebooks/
│   ├── data_exploration.ipynb       # EDA: class imbalance, missing data, distributions
│   ├── model_training.ipynb         # End-to-end training pipeline (load → save)
│   └── streaming_simulation.ipynb   # Replay test set against live API, plot metrics
├── src/
│   ├── features.py                  # Leak-free feature engineering
│   ├── model.py                     # Data prep and SHAP feature selection
│   ├── ensemble.py                  # Ensemble training, tuning, persistence
│   ├── api.py                       # FastAPI serving layer
│   └── simulator.py                 # Streaming simulation utilities
└── requirements.txt
```

## Feature Engineering

All features in `src/features.py` are computed **leak-free** — only prior transactions are used at each step.

| Feature | Description |
|---|---|
| `velocity_1h / 24h / 7d` | Transactions per cardholder in the last 1h / 24h / 7d (bisect-based, O(n log n)) |
| `hist_mean_amt` | Expanding mean of prior transaction amounts per cardholder |
| `amt_deviation` | Current amount minus historical mean |
| `card_unique_addr` | Distinct billing addresses seen per card before this transaction |
| `card_unique_amt` | Distinct transaction amounts seen per card before this transaction |
| `hour_of_day` | Hour extracted from `TransactionDT` (modulo arithmetic) |
| `day_of_week` | Day of week extracted from `TransactionDT` |
| `is_night` | 1 if transaction occurred between midnight and 5am |
| `is_weekend` | 1 if transaction occurred on Saturday or Sunday |
| `month_of_year` | Approximate month index (0–11) for seasonal fraud patterns |
| `email_domain_match` | 1 if purchaser and recipient share the same email domain |
| `is_free_email` | 1 if purchaser uses a free provider (gmail, yahoo, hotmail, etc.) |
| `amt_cents` | Fractional cents portion of `TransactionAmt` |
| `is_round_amt` | 1 if transaction amount has no cents |
| `log_D{n}` | Log-transformed days-since columns D1–D9 (compresses right skew) |
| `D{n}_null_flag` | 1 when D column is missing — distinguishes absence from zero |
| `is_unknown_os` | 1 when DeviceInfo is absent (no device fingerprint available) |
| `is_mobile` | 1 if DeviceType is mobile |
| `device_os` | Coarse OS category from DeviceInfo (windows / ios / macos / android / linux / other) |

## Training Pipeline

`notebooks/model_training.ipynb` orchestrates the full pipeline:

1. **Load** raw transaction and identity CSVs
2. **Feature engineering** — all features above, computed leak-free via `build_features`
3. **Prepare** — drop >99% missing columns, drop constant columns, encode categoricals
4. **Time-based split** — 80/20 chronological split (no random shuffling)
5. **SHAP feature selection** — top 40 features by mean |SHAP| value; cached to disk
6. **Tune** — Optuna (50 trials, TimeSeriesSplit 5-fold) per model: LightGBM, XGBoost, CatBoost
7. **Train** soft-voting ensemble with best params + early stopping
8. **Evaluate** — PR-AUC and ROC-AUC on held-out test set
9. **Save** ensemble to `models/ensemble_fraud.pkl`

## Architecture

```
                        ┌─────────────┐
Transaction ──────────► │  FastAPI    │
(JSON payload)          │  /predict   │
                        └──────┬──────┘
                               │ feature engineering
                               ▼
                        ┌─────────────┐
                        │  Ensemble   │
                        │  LightGBM   │──► avg(predict_proba)
                        │  XGBoost    │──► fraud_probability
                        │  CatBoost   │
                        └─────────────┘
```

The API accepts raw transaction fields plus optional behavioral features (velocity, historical mean amount) that the caller pre-fetches from a history store (e.g., Redis). Static features are computed server-side.

## Setup

```bash
git clone https://github.com/Rohit140595/fraud-detection.git
cd fraud-detection
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name fraud-detection --display-name "Python (fraud-detection)"
```

Download the data from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place the four CSV files in `data/`.

## Running

**Train the ensemble:**
```bash
jupyter notebook notebooks/model_training.ipynb
# Run all cells — takes ~75 min with n_trials=50
```

**Serve the API:**
```bash
uvicorn src.api:app --reload
# POST to http://localhost:8000/predict
```

**Run streaming simulation:**
```bash
# Start the API first, then:
jupyter notebook notebooks/streaming_simulation.ipynb
```

## Key Design Decisions

- **Time-based split over random split** — behavioral features (velocity, historical mean) would leak future information into training if rows were shuffled.
- **SHAP feature selection** — ranks features by actual prediction contribution (mean |SHAP|) rather than split counts, which is biased toward high-cardinality features.
- **Soft-voting ensemble** — LightGBM, XGBoost, and CatBoost each make different errors; averaging probabilities reduces variance without requiring a meta-learner.
- **`scale_pos_weight = neg / pos`** — handles 3.5% class imbalance without SMOTE, which is unreliable on mixed/missing data.
- **`TimeSeriesSplit` in Optuna** — each CV fold's validation is strictly later than its training data, consistent with production temporal ordering.
- **SHAP cache** — `models/shap_features.json` stores the selected features so SHAP only runs once; subsequent notebook runs load from cache instantly.
- **`_expanding_nunique` with a running set** — O(n) unique-count aggregation instead of the naive O(n²) `expanding().apply(nunique)`.
