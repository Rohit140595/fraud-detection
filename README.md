# Fraud Detection — IEEE-CIS Dataset

End-to-end ML pipeline for real-time transaction fraud detection, built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

## Overview

The project trains a LightGBM classifier to identify fraudulent transactions. It emphasises **temporal correctness** — every feature and every train/test split is designed to prevent data leakage from future transactions into past predictions, matching real-world production conditions.

## Project Structure

```
fraud-detection/
├── data/                        # Raw CSVs (not tracked — download from Kaggle)
│   ├── train_transaction.csv
│   └── train_identity.csv
├── models/
│   └── lgbm_fraud.pkl           # Trained model artifact
├── notebooks/
│   ├── data_exploration.ipynb   # EDA: class imbalance, missing data, feature distributions
│   └── model_training.ipynb     # End-to-end training pipeline (load → save)
├── src/
│   ├── features.py              # Feature engineering functions
│   └── model.py                 # Training, tuning, evaluation, persistence
└── requirements.txt
```

## Feature Engineering

All features in `src/features.py` are computed **leak-free** — only prior transactions are used at each step.

| Feature | Description |
|---|---|
| `velocity_1h / 24h / 7d` | Transactions per cardholder in the last 1h / 24h / 7d (bisect-based, O(n log n)) |
| `hist_mean_amt` | Expanding mean of prior transaction amounts per cardholder |
| `amt_deviation` | Current amount minus historical mean |
| `hour_of_day` | Hour extracted from `TransactionDT` via modulo arithmetic |
| `day_of_week` | Day of week extracted from `TransactionDT` via modulo arithmetic |
| `card_unique_addr` | Distinct billing addresses seen per card before this transaction |
| `card_unique_amt` | Distinct transaction amounts seen per card before this transaction |
| `email_domain_match` | 1 if purchaser and recipient share the same email domain |
| `is_free_email` | 1 if purchaser uses a free provider (gmail, yahoo, hotmail, etc.) |
| `amt_cents` | Fractional cents portion of `TransactionAmt` |
| `is_round_amt` | 1 if transaction amount has no cents |

## Training Pipeline

The notebook `notebooks/model_training.ipynb` orchestrates the full pipeline:

1. Load raw transaction and identity data
2. Feature engineering (`build_features`)
3. Drop columns with >99% missing values; encode categoricals
4. Time-based 80/20 train/test split (chronological — no random shuffling)
5. Train baseline LightGBM
6. Feature selection — drop features with zero importance in the baseline
7. Hyperparameter tuning — Optuna over 50 trials, optimising PR-AUC with `TimeSeriesSplit`
8. Train final model with best params + early stopping
9. Evaluate — PR-AUC and ROC-AUC
10. Feature importance plot
11. Save model to `models/lgbm_fraud.pkl`

**Primary metric: PR-AUC.** The dataset is heavily imbalanced (~3.5% fraud rate). PR-AUC focuses on the minority class and is not inflated by the large number of true negatives, making it more informative than ROC-AUC here.

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/Rohit140595/fraud-detection.git
cd fraud-detection
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register the kernel for Jupyter
python -m ipykernel install --user --name fraud-detection --display-name "Python (fraud-detection)"
```

Download the data from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place the CSV files in the `data/` directory.

## Running

Open `notebooks/model_training.ipynb` and select the **Python (fraud-detection)** kernel, then run all cells top to bottom.

## Key Design Decisions

- **Time-based split over random split** — behavioural features (velocity, historical mean) would leak future information into training if rows were shuffled.
- **`scale_pos_weight = neg / pos`** — handles class imbalance without SMOTE, which is unreliable on mixed/missing data.
- **`TimeSeriesSplit` in Optuna** — each CV fold's validation is strictly later than its training data, consistent with production.
- **Zero-importance feature selection** — the baseline model identifies which features LightGBM actually uses before the expensive hyperparameter search.
- **`_expanding_nunique` with a running set** — O(n) unique-count aggregation instead of the naive O(n²) `expanding().apply(nunique)`.
