"""
Model training and evaluation for real-time fraud detection.

Design decisions:
  - Time-based train/test split to avoid temporal data leakage.
  - LightGBM with scale_pos_weight to handle class imbalance.
  - PR-AUC as primary metric — more informative than ROC-AUC on imbalanced data.
  - Model persisted to disk so it can be loaded by the FastAPI serving layer.
"""

import joblib
from pathlib import Path

import numpy as np
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


LABEL = "isFraud"
DROP_COLS = ["TransactionID", "card_addr"]
MISSING_THRESHOLD = 0.99
MODEL_PATH = Path(__file__).parent.parent / "models" / "lgbm_fraud.pkl"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the feature matrix for modeling.

    Steps:
      1. Drop columns with >99% missing values — pure noise at that rate.
      2. Drop helper columns not needed for modeling.
      3. Cast object columns to 'category' so LightGBM handles them natively
         (no manual label encoding needed).

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Cleaned DataFrame ready for train/test split.
    """
    # Drop high-missingness columns
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{MISSING_THRESHOLD:.0%} missing")
    df = df.drop(columns=cols_to_drop)

    # Drop non-feature columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Encode categoricals — LightGBM handles 'category' dtype natively
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically — train on earlier, test on later transactions.

    A random split would leak future transaction history into training features
    (velocity, historical mean) that wouldn't be available at prediction time
    in production. Time-based split prevents this.

    Args:
        df:          Prepared feature DataFrame (must contain 'TransactionDT').
        train_frac:  Fraction of data to use for training (default 0.8).

    Returns:
        (train, test) DataFrames.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: str = "mean",
) -> tuple:
    """
    Train a fast baseline LightGBM internally, then select features
    whose importance meets the threshold using SelectFromModel.

    Using 'mean' or 'median' is more principled than keeping any feature
    with non-zero importance — it retains only features that contribute
    meaningfully relative to the average, reducing overfitting noise.

    Args:
        X_train:   Training feature matrix.
        y_train:   Training labels.
        X_test:    Test feature matrix.
        threshold: Importance cutoff — 'mean', 'median', or a float.
                   'mean' keeps features above average importance (recommended).
                   'median' is more aggressive, keeping the top 50%.

    Returns:
        (X_train_filtered, X_test_filtered, selected_feature_names)
    """
    from sklearn.feature_selection import SelectFromModel

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    # Fast baseline — just enough trees to estimate stable feature importances
    baseline = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=64,
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # SelectFromModel fits the baseline and applies the importance threshold
    selector = SelectFromModel(baseline, threshold=threshold)
    selector.fit(X_train, y_train)

    # Boolean mask → column names from original DataFrame
    support = selector.get_support()
    selected_features = X_train.columns[support].tolist()

    print(f"Features kept   : {len(selected_features)} / {X_train.shape[1]}")
    print(f"Features dropped: {X_train.shape[1] - len(selected_features)}")

    return X_train[selected_features], X_test[selected_features], selected_features


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> dict:
    """
    Search for the best LightGBM hyperparameters using Optuna.

    Uses TimeSeriesSplit(n_splits=3) so each fold respects temporal order —
    validation data is always later than training data, matching production.
    PR-AUC is the optimisation target because it is more informative than
    ROC-AUC on the heavily imbalanced fraud dataset.

    Args:
        X_train:  Training features.
        y_train:  Training labels.
        n_trials: Number of Optuna trials (default 50).

    Returns:
        Dictionary of best hyperparameters, ready to pass to LGBMClassifier.
    """
    # Compute once here — reused in every trial without recalculating
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    def objective(trial):
        # Optuna samples one combination of params per trial.
        # log=True for learning_rate so the search is uniform on a log scale
        # (treats 0.01→0.1 the same as 0.1→1.0, which is correct for rates).
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "bagging_freq":      1,     # required for bagging_fraction to take effect
            "n_estimators":      300,   # fixed — early stopping handled in final train()
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,    # suppress per-fold LightGBM output
        }

        # TimeSeriesSplit ensures each fold's validation is strictly later than
        # its training data — prevents leakage from future transactions into
        # behavioural features (velocity, historical mean).
        tscv = TimeSeriesSplit(n_splits=3)
        pr_aucs = []

        for train_idx, val_idx in tscv.split(X_train):
            X_fold_tr, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_tr, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_fold_tr, y_fold_tr)

            # [:, 1] gives the probability of the positive (fraud) class
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_fold_val, y_prob)
            pr_aucs.append(auc(recall, precision))

        # Optuna maximises this return value across all trials
        return np.mean(pr_aucs)

    # Suppress Optuna's per-trial INFO logs — progress bar is enough
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # TPESampler with fixed seed ensures the same trials are sampled every run
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best PR-AUC : {study.best_value:.4f}")
    print(f"Best params : {study.best_params}")

    return study.best_params


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier with early stopping.

    scale_pos_weight = (# negatives) / (# positives) tells LightGBM to penalise
    missed fraud proportionally to its rarity — a simple, effective imbalance fix
    that avoids the complexity of SMOTE on mixed/missing data.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val:     Validation features and labels for early stopping.

    Returns:
        Trained LGBMClassifier.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight: {scale_pos_weight:.1f}  (neg={neg:,}, pos={pos:,})")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(50),  # matches early-stopping patience — always fires at least once
        ],
    )
    print(f"Best iteration : {model.best_iteration_}")

    return model


def evaluate(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot: bool = True,
) -> dict:
    """
    Evaluate model using PR-AUC and ROC-AUC.

    PR-AUC is the primary metric for imbalanced fraud detection — it focuses
    on the minority class and is not inflated by the large number of true negatives.

    Args:
        model:  Trained LGBMClassifier.
        X_test: Test features.
        y_test: True labels.
        plot:   Whether to display the precision-recall curve.

    Returns:
        Dictionary with pr_auc and roc_auc scores.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")

    if plot:
        baseline = y_test.mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(recall, precision, color="steelblue", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
        ax.axhline(baseline, color="tomato", linestyle="--", label=f"Random baseline ({baseline:.2%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """
    Plot the top N most important features.

    Useful for checking whether engineered features (velocity_1h, amt_deviation)
    added meaningful signal on top of the raw columns.

    Args:
        model:         Trained LGBMClassifier.
        feature_names: List of feature names used during training.
        top_n:         Number of top features to display.
    """
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    print(importance.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(x="feature", y="importance", kind="barh", ax=ax, color="steelblue", legend=False)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


def save_model(model: lgb.LGBMClassifier, path: Path = MODEL_PATH) -> None:
    """Persist the trained model to disk for use by the serving layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH) -> lgb.LGBMClassifier:
    """Load a persisted model from disk."""
    return joblib.load(path)
