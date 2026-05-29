"""
Data preparation and feature selection for real-time fraud detection.

Design decisions:
  - Three-way chronological split (train / cal / test) to keep calibration
    data fully independent of model training and early stopping.
  - SHAP-based feature selection — uses actual prediction contribution
    (not split counts) to rank features. Results cached to disk so SHAP
    only runs once regardless of how many times the notebook is re-executed.
"""

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
import pandas as pd
import shap


LABEL = "isFraud"
DROP_COLS = ["TransactionID", "card_addr"]
MISSING_THRESHOLD = 0.99
RANDOM_STATE = 42
SHAP_CACHE_PATH = Path(__file__).parent.parent / "models" / "shap_features.json"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the feature matrix for modeling.

    Steps:
      1. Drop columns with >99% missing values — heavily sparse columns add
         noise and inflate the feature space without contributing stable signal.
      2. Drop constant columns (only 1 unique value) — zero variance means
         no discriminative power; keeping them wastes splits in the trees.
      3. Drop helper columns not needed for modeling.
      4. Cast object columns to 'category' so LightGBM handles them natively
         (no manual label encoding needed).

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Cleaned DataFrame ready for train/test split.
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{MISSING_THRESHOLD:.0%} missing")
    df = df.drop(columns=cols_to_drop)

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    print(f"Dropping {len(constant_cols)} constant columns")
    df = df.drop(columns=constant_cols)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Encode categoricals — LightGBM handles 'category' dtype natively
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    return df


def three_way_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    cal_frac:   float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train / calibration / test sets.

    All three cuts are strictly ordered by TransactionDT so no future
    information leaks into earlier splits.

    Split roles:
      train — model training and Optuna hyperparameter tuning
              (TimeSeriesSplit CV stays entirely within this slice)
      cal   — calibration only; never seen during training or early stopping,
              ensuring the calibrator is fit on genuinely unseen scores
      test  — early stopping for the final ensemble fit + held-out evaluation

    Args:
        df:          Prepared feature DataFrame (must contain 'TransactionDT').
        train_frac:  Fraction of rows for training (default 0.70).
        cal_frac:    Fraction of rows for calibration (default 0.10).
                     Remainder (1 - train_frac - cal_frac) goes to test.

    Returns:
        (train, cal, test) DataFrames.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    cal_end   = int(n * (train_frac + cal_frac))
    return df.iloc[:train_end], df.iloc[train_end:cal_end], df.iloc[cal_end:]


def select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    X_cal: pd.DataFrame | None = None,
    top_n: int = 40,
    sample_size: int = 50_000,
    cache_path: Path = SHAP_CACHE_PATH,
    force_recompute: bool = False,
) -> tuple:
    """
    Select the top N features by mean absolute SHAP value.

    SHAP measures each feature's actual contribution to predictions — more
    accurate than split-count importance (used by SelectFromModel), which is
    biased toward high-cardinality features and misrepresents low-frequency
    but highly predictive features.

    Results are cached to disk so SHAP only runs once. On subsequent calls
    the cached feature list is loaded directly unless force_recompute=True
    or the cache is stale (contains features no longer in X_train).

    A 50K subsample is used for SHAP computation — sufficient for stable
    feature ranking while keeping runtime to ~1–2 min.

    Args:
        X_train:         Training feature matrix.
        y_train:         Training labels.
        X_test:          Test feature matrix.
        X_cal:           Optional calibration feature matrix — filtered alongside
                         train/test when provided.
        top_n:           Number of top features to keep (default 40).
        sample_size:     Rows used for SHAP computation (default 50,000).
        cache_path:      Path to the JSON cache file.
        force_recompute: Ignore existing cache and rerun SHAP (default False).

    Returns:
        (X_train_f, X_test_f, selected_features) when X_cal is None.
        (X_train_f, X_test_f, X_cal_f, selected_features) when X_cal is provided.
    """
    # ── Load from cache if available ──────────────────────────────────────────
    if not force_recompute and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        selected = cached.get("features", [])
        missing = [feat for feat in selected if feat not in X_train.columns]
        if not missing:
            print(f"SHAP cache loaded : {len(selected)} features  ({cache_path.name})")
            if X_cal is not None:
                return X_train[selected], X_test[selected], X_cal[selected], selected
            return X_train[selected], X_test[selected], selected
        print(f"Cache stale — {len(missing)} features missing, recomputing ...")

    # ── Compute SHAP ──────────────────────────────────────────────────────────
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    # Subsample for speed — 50K rows is sufficient for stable feature ranking
    if len(X_train) > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[idx]
        y_sample = y_train.iloc[idx]
    else:
        X_sample, y_sample = X_train, y_train

    print(f"Computing SHAP on {len(X_sample):,} rows × {X_train.shape[1]} features ...")

    # Baseline LightGBM — fast defaults, just enough to rank features reliably
    baseline = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=64,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    baseline.fit(X_sample, y_sample)

    # TreeExplainer is fast for tree models — uses the tree structure directly
    explainer = shap.TreeExplainer(baseline)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary classification returns a list [neg_class, pos_class];
    # index [1] is the positive (fraud) class contribution
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Rank by mean |SHAP| — average absolute contribution per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    selected = X_train.columns[top_idx].tolist()

    print(f"SHAP selection : top {len(selected)} / {X_train.shape[1]} features")
    print("Top 10 by mean |SHAP|:")
    for name, score in zip(selected[:10], mean_abs_shap[top_idx[:10]]):
        print(f"  {name:<35} {score:.4f}")

    # ── Save to cache ─────────────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"features": selected, "top_n": top_n}, f, indent=2)
    print(f"SHAP features cached to {cache_path.name}")

    if X_cal is not None:
        return X_train[selected], X_test[selected], X_cal[selected], selected
    return X_train[selected], X_test[selected], selected
