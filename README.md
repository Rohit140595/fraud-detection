# Fraud Detection — IEEE-CIS Dataset

End-to-end ML pipeline for real-time transaction fraud detection, built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

## Results

| Metric | Score |
|---|---|
| **PR-AUC** | **0.5712** |
| ROC-AUC | 0.9100 |
| Precision / Recall (F2 threshold) | 0.436 / 0.614 |
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
│   ├── ensemble_fraud.pkl           # Trained ensemble (LightGBM + XGBoost, meta: XGBoost)
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
| `dt_card1_last` | Seconds since the previous transaction on the same card1 |
| `dt_uid_last` | Seconds since the previous transaction for the same UID (card + addr + email) |
| `uid_D{n}_mean` | Expanding mean of D1–D9 per UID up to but not including the current transaction |
| `uid_D{n}_std` | Expanding std of D1–D9 per UID up to but not including the current transaction |
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
4. **Time-based split** — 80/20 train/test chronological split (no random shuffling)
5. **SHAP feature selection** — top 100 features by mean |SHAP| value; cached to disk
6. **Tune** — Optuna (50 trials, TimeSeriesSplit 5-fold) per model: LightGBM, XGBoost
7. **Train** stacking ensemble — base models with early stopping, then XGBoost meta-learner on OOF predictions
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
                        │  Base models│
                        │  LightGBM   │──► lgbm_prob ─┐
                        │  XGBoost    │──► xgb_prob  ─┤
                        └─────────────┘               │
                                                       ▼
                                               ┌──────────────┐
                                               │ Meta-learner │
                                               │  XGBoost     │──► fraud_probability
                                               └──────────────┘
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

**Run streaming simulation (Layer 4):**
```bash
# Start the API first, then:
jupyter notebook notebooks/streaming_simulation.ipynb
```

**Run Kafka pipeline (Layer 5):**
```bash
# 1. Start Kafka
docker-compose up -d

# 2. Start the consumer/scorer (terminal 1)
python -m src.kafka_consumer --offset earliest

# 3. Stream transactions from the test set (terminal 2)
python -m src.kafka_producer --n 500 --delay 100   # 500 rows, 100ms apart

# 4. Tail the fraud-scores topic to see results (terminal 3)
docker exec fraud-kafka kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic fraud-scores
```

## Kafka Architecture

```
test_features.parquet
        │
        ▼
┌─────────────────┐        raw-transactions topic
│  kafka_producer │ ──────────────────────────────►┐
└─────────────────┘                                 │
                                                    ▼
                                         ┌──────────────────┐
                                         │  kafka_consumer  │
                                         │  (loads ensemble │
                                         │   at startup)    │
                                         └────────┬─────────┘
                                                  │
                                                  ▼
                                         fraud-scores topic
                                  { fraud_probability, is_fraud,
                                    threshold, latency_ms,
                                    true_label (for evaluation) }
```

The consumer scores each transaction directly — no HTTP hop — making it
suitable for high-throughput production deployments. The FastAPI layer
remains available for synchronous/external use cases.

## Key Design Decisions

- **Time-based split over random split** — behavioral features (velocity, historical mean) would leak future information into training if rows were shuffled.
- **SHAP feature selection** — ranks features by actual prediction contribution (mean |SHAP|) rather than split counts, which is biased toward high-cardinality features.
- **Stacking ensemble** — LightGBM and XGBoost base models generate out-of-fold predictions via TimeSeriesSplit; a shallow XGBoost meta-learner (max_depth=3) learns to blend them. No `scale_pos_weight` on the meta-learner — OOF inputs are already probability-like scores from base models that handled class imbalance themselves.
- **`scale_pos_weight = neg / pos`** — handles 3.5% class imbalance without SMOTE, which is unreliable on mixed/missing data.
- **`TimeSeriesSplit` in Optuna** — each CV fold's validation is strictly later than its training data, consistent with production temporal ordering.
- **SHAP cache** — `models/shap_features.json` stores the selected features so SHAP only runs once; subsequent notebook runs load from cache instantly.
- **`_expanding_nunique` with a running set** — O(n) unique-count aggregation instead of the naive O(n²) `expanding().apply(nunique)`.
