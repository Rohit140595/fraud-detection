"""
Soft-voting ensemble for fraud detection (LightGBM + XGBoost + CatBoost).

Design decisions:
  - Category columns are ordinal-encoded to integers before any model sees them.
    Encoding is fit on X_train and applied to test/val to keep codes consistent.
  - LightGBM: receives integer-encoded cats + categorical_feature list → categorical splits.
  - XGBoost:  receives integers, treated as continuous — standard and effective for trees.
  - CatBoost: receives integers + cat_features constructor arg → ordered target encoding.
  - Soft voting: averages predict_proba[:, 1] across all three models.
  - Each model tuned independently with the same 5-param Optuna grid mapped to
    framework-specific names. Same TimeSeriesSplit(n_splits=5) + PR-AUC objective
    as the single-model tuner in model.py.
  - scale_pos_weight (LightGBM / XGBoost) and class_weights (CatBoost) are always
    derived from the training class ratio — not tuned.
  - Early stopping only in train_ensemble (not during Optuna trials), consistent
    with model.py's tune_hyperparameters.
  - Single-model pipeline in model.py is untouched. To revert, call train() from
    model.py and ignore this module entirely.
"""

import joblib
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


ENSEMBLE_PATH = Path(__file__).parent.parent / "models" / "ensemble_fraud.pkl"
RANDOM_STATE = 42


# ── Categorical encoding ───────────────────────────────────────────────────────

def encode_for_ensemble(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """
    Ordinal-encode category columns to integers for cross-framework compatibility.

    Encoding is fit on X_train's category labels, then applied to X_test using
    the same mapping — prevents code mismatches if X_test contains categories
    not seen in X_train (those map to -1, which all three models treat as missing).

    Args:
        X_train: Training features with 'category' dtype columns.
        X_test:  Test features with 'category' dtype columns.

    Returns:
        (X_train_enc, X_test_enc, cat_cols, cat_encoders)
        cat_encoders: {col: {label: int_code}} — saved alongside the model so the
        serving layer can apply the exact same mapping at inference time.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()
    cat_cols = X_train.select_dtypes(include="category").columns.tolist()
    cat_encoders: dict[str, dict] = {}

    for col in cat_cols:
        # Build label→code mapping from X_train's category set
        cat_to_code = {
            cat: code
            for code, cat in enumerate(X_train[col].cat.categories)
        }
        cat_encoders[col] = cat_to_code
        X_train[col] = X_train[col].cat.codes                              # NaN → -1
        X_test[col]  = X_test[col].astype(object).map(cat_to_code).fillna(-1).astype(int)

    return X_train, X_test, cat_cols, cat_encoders


# ── Shared CV helper ───────────────────────────────────────────────────────────

def _pr_auc_cv(model_cls, params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """3-fold time-series cross-validation, returns mean PR-AUC."""
    tscv = TimeSeriesSplit(n_splits=5)
    pr_aucs = []
    for train_idx, val_idx in tscv.split(X_train):
        model = model_cls(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_prob = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        precision, recall, _ = precision_recall_curve(y_train.iloc[val_idx], y_prob)
        pr_aucs.append(auc(recall, precision))
    return float(np.mean(pr_aucs))


# ── Per-model tuners ───────────────────────────────────────────────────────────

def _tune_lgbm(X_train, y_train, n_trials: int, scale_pos_weight: float) -> dict:
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_freq":     1,
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
        }
        return _pr_auc_cv(lgb.LGBMClassifier, params, X_train, y_train)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  LightGBM best PR-AUC : {study.best_value:.4f}  {study.best_params}")
    return study.best_params


def _tune_xgb(X_train, y_train, n_trials: int, scale_pos_weight: float) -> dict:
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0,
            # No early_stopping_rounds — fit() has no eval_set during CV
        }
        return _pr_auc_cv(xgb.XGBClassifier, params, X_train, y_train)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  XGBoost  best PR-AUC : {study.best_value:.4f}  {study.best_params}")
    return study.best_params


def _tune_catboost(
    X_train, y_train, n_trials: int, scale_pos_weight: float, cat_cols: list[str],
) -> dict:
    def objective(trial):
        params = {
            "iterations":        trial.suggest_int("iterations", 100, 500),
            "depth":             trial.suggest_int("depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "bootstrap_type":    "Bernoulli",   # required for subsample to take effect
            "class_weights":     [1.0, scale_pos_weight],
            "cat_features":      cat_cols,
            "random_seed": RANDOM_STATE, "verbose": 0,
            # No early_stopping_rounds — fit() has no eval_set during CV
        }
        return _pr_auc_cv(CatBoostClassifier, params, X_train, y_train)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  CatBoost best PR-AUC : {study.best_value:.4f}  {study.best_params}")
    return study.best_params


# ── Public API ─────────────────────────────────────────────────────────────────

def tune_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: list[str],
    n_trials: int = 50,
) -> dict[str, dict]:
    """
    Tune LightGBM, XGBoost, and CatBoost independently using Optuna.

    All three use the same 5-parameter grid philosophy and
    TimeSeriesSplit(n_splits=5) + PR-AUC objective — consistent with
    tune_hyperparameters() in model.py.

    scale_pos_weight / class_weights are fixed (derived from class ratio),
    not tuned — they are data properties, not model complexity knobs.

    Args:
        X_train:  Training features (integer-encoded, from encode_for_ensemble).
        y_train:  Training labels.
        cat_cols: Categorical column names (from encode_for_ensemble).
        n_trials: Optuna trials per model (default 50, ~3 × 50 = 150 total trials).

    Returns:
        {"lgbm": best_params, "xgb": best_params, "catboost": best_params}
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    print(f"Tuning LightGBM  ({n_trials} trials) ...")
    lgbm_params = _tune_lgbm(X_train, y_train, n_trials, spw)

    print(f"\nTuning XGBoost   ({n_trials} trials) ...")
    xgb_params = _tune_xgb(X_train, y_train, n_trials, spw)

    print(f"\nTuning CatBoost  ({n_trials} trials) ...")
    catboost_params = _tune_catboost(X_train, y_train, n_trials, spw, cat_cols)

    return {"lgbm": lgbm_params, "xgb": xgb_params, "catboost": catboost_params}


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict[str, dict],
    cat_cols: list[str],
) -> dict[str, object]:
    """
    Train LightGBM, XGBoost, and CatBoost with early stopping on the validation set.

    Merges tuned hyperparameters (from tune_all) with sensible defaults, then
    trains with early_stopping_rounds=50 to prevent overfitting.

    Args:
        X_train, y_train: Training features and labels (integer-encoded).
        X_val, y_val:     Validation set for early stopping.
        best_params:      Output of tune_all().
        cat_cols:         Categorical column names (from encode_for_ensemble).

    Returns:
        {"lgbm": fitted_model, "xgb": fitted_model, "catboost": fitted_model}
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos
    print(f"scale_pos_weight: {spw:.1f}  (neg={neg:,}, pos={pos:,})")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\nTraining LightGBM ...")
    lgbm_p = {
        "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64,
        "scale_pos_weight": spw, "bagging_freq": 1,
        "random_state": RANDOM_STATE, "n_jobs": -1,
    }
    lgbm_p.update(best_params.get("lgbm", {}))
    lgbm_p["scale_pos_weight"] = spw   # always from data, not tuned
    lgbm_p["bagging_freq"]     = 1     # required for bagging_fraction

    lgbm_model = lgb.LGBMClassifier(**lgbm_p)
    fit_kw = {
        "eval_set": [(X_val, y_val)],
        "callbacks": [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    }
    if cat_cols:
        fit_kw["categorical_feature"] = cat_cols
    lgbm_model.fit(X_train, y_train, **fit_kw)
    print(f"  Best iteration: {lgbm_model.best_iteration_}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost ...")
    xgb_p = {
        "n_estimators": 1000, "learning_rate": 0.05,
        "scale_pos_weight": spw, "random_state": RANDOM_STATE, "n_jobs": -1,
        "verbosity": 0, "eval_metric": "aucpr", "early_stopping_rounds": 50,
    }
    xgb_p.update(best_params.get("xgb", {}))
    xgb_p["scale_pos_weight"]    = spw
    xgb_p["early_stopping_rounds"] = 50   # re-enforce after update

    xgb_model = xgb.XGBClassifier(**xgb_p)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ── CatBoost ──────────────────────────────────────────────────────────────
    print("\nTraining CatBoost ...")
    cat_p = {
        "iterations": 1000, "learning_rate": 0.05,
        "class_weights": [1.0, spw], "bootstrap_type": "Bernoulli",
        "cat_features": cat_cols, "random_seed": RANDOM_STATE, "verbose": 0,
        "early_stopping_rounds": 50,
    }
    cat_p.update(best_params.get("catboost", {}))
    cat_p["class_weights"]         = [1.0, spw]   # always from data
    cat_p["cat_features"]          = cat_cols       # always from encoding
    cat_p["early_stopping_rounds"] = 50

    catboost_model = CatBoostClassifier(**cat_p)
    catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    return {"lgbm": lgbm_model, "xgb": xgb_model, "catboost": catboost_model}


def predict_proba_ensemble(models: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Average predict_proba[:, 1] across all models (soft voting).

    Args:
        models: Dict from train_ensemble.
        X:      Integer-encoded features (same schema as training).

    Returns:
        1-D array of fraud probabilities, one per row.
    """
    probs = [model.predict_proba(X)[:, 1] for model in models.values()]
    return np.mean(probs, axis=0)


def evaluate_ensemble(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    plot: bool = True,
) -> dict:
    """
    Evaluate the ensemble using PR-AUC and ROC-AUC.

    Args:
        models:    Dict from train_ensemble.
        X_test:    Test features (integer-encoded).
        y_test:    True labels.
        threshold: Decision threshold for is_fraud (default 0.5).
        plot:      Whether to display the precision-recall curve.

    Returns:
        Dict with pr_auc and roc_auc.
    """
    y_prob = predict_proba_ensemble(models, X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc  = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Ensemble PR-AUC  : {pr_auc:.4f}")
    print(f"Ensemble ROC-AUC : {roc_auc:.4f}")

    if plot:
        baseline_rate = y_test.mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(recall, precision, color="steelblue", lw=2,
                label=f"Ensemble PR-AUC = {pr_auc:.4f}")
        ax.axhline(baseline_rate, color="tomato", linestyle="--",
                   label=f"Random baseline ({baseline_rate:.2%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Ensemble Precision-Recall Curve")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def calibrate_ensemble(
    models: dict,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "isotonic",
) -> IsotonicRegression | LogisticRegression:
    """
    Fit a post-hoc calibrator on raw ensemble scores from the calibration set.

    Boosted tree ensembles tend to produce overconfident probabilities (pushed
    toward 0 and 1). Isotonic regression corrects this by learning a monotone
    mapping from raw scores → calibrated probabilities.

    The calibration set must be independent of model training and early stopping
    (i.e., the dedicated cal split, not the train or test sets).

    Args:
        models:  Dict from train_ensemble.
        X_cal:   Calibration features (integer-encoded, same schema as training).
        y_cal:   Calibration labels.
        method:  'isotonic' (default) or 'sigmoid' (Platt scaling).
                 Isotonic is non-parametric and more flexible; sigmoid assumes
                 a logistic relationship between raw score and true probability.

    Returns:
        Fitted calibrator (IsotonicRegression or LogisticRegression).
    """
    raw_probs = predict_proba_ensemble(models, X_cal)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)
    else:  # sigmoid / Platt
        calibrator = LogisticRegression()
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    cal_probs = (
        calibrator.predict(raw_probs) if method == "isotonic"
        else calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    )
    brier_before = brier_score_loss(y_cal, raw_probs)
    brier_after  = brier_score_loss(y_cal, cal_probs)
    print(f"Brier score  before calibration : {brier_before:.4f}")
    print(f"Brier score  after  calibration : {brier_after:.4f}  ({method})")

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(7, 5))
    for probs, label, style in [
        (raw_probs, "Before calibration", "--"),
        (cal_probs, f"After calibration ({method})", "-"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_cal, probs, n_bins=15, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linestyle=style, label=label)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return calibrator


def tune_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    beta: float = 2.0,
) -> dict:
    """
    Find the decision threshold that maximises F-beta on calibrated probabilities.

    F-beta weights recall beta² times more than precision. beta=2 means missing
    fraud (false negative) is penalised 4× more than a false alarm (false positive),
    which reflects the typical cost asymmetry in fraud detection.

    Also plots the precision-recall curve and F-beta curve so the operating point
    can be inspected visually before committing to a threshold.

    Args:
        y_true: True binary labels.
        y_prob: Calibrated fraud probabilities (output of calibrator.predict).
        beta:   F-beta weight (default 2.0).

    Returns:
        Dict with threshold, precision, recall, f_beta at the optimal point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # F-beta is undefined when precision + recall = 0; add epsilon to avoid /0
    denom  = beta**2 * precision[:-1] + recall[:-1]
    f_beta = np.where(
        denom > 0,
        (1 + beta**2) * precision[:-1] * recall[:-1] / denom,
        0.0,
    )

    best_idx   = int(np.argmax(f_beta))
    best       = {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall":    float(recall[best_idx]),
        "f_beta":    float(f_beta[best_idx]),
    }

    print(f"Optimal threshold (F{beta:.0f}) : {best['threshold']:.4f}")
    print(f"  Precision : {best['precision']:.4f}")
    print(f"  Recall    : {best['recall']:.4f}")
    print(f"  F{beta:.0f}       : {best['f_beta']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # PR curve
    pr_auc = auc(recall, precision)
    axes[0].plot(recall, precision, color="steelblue", lw=2,
                 label=f"PR-AUC = {pr_auc:.4f}")
    axes[0].scatter(best["recall"], best["precision"],
                    color="tomato", zorder=5, s=100,
                    label=f"Optimal threshold = {best['threshold']:.4f}")
    axes[0].axhline(y_true.mean(), color="grey", linestyle="--",
                    label=f"Baseline ({y_true.mean():.2%})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()

    # F-beta curve
    axes[1].plot(thresholds, f_beta, color="seagreen", lw=2)
    axes[1].axvline(best["threshold"], color="tomato", linestyle="--",
                    label=f"Optimal = {best['threshold']:.4f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel(f"F{beta:.0f} score")
    axes[1].set_title(f"F{beta:.0f} vs Threshold")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return best


def save_ensemble(
    models: dict,
    cat_cols: list[str],
    cat_encoders: dict,
    calibrator: Optional[object] = None,
    threshold: float = 0.5,
    path: Path = ENSEMBLE_PATH,
) -> None:
    """Persist the ensemble, cat_cols, cat_encoders, calibrator, and threshold to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models":      models,
        "cat_cols":    cat_cols,
        "cat_encoders": cat_encoders,
        "calibrator":  calibrator,
        "threshold":   threshold,
    }, path)
    print(f"Ensemble saved to {path}  ({len(models)} models, threshold={threshold:.4f})")


def load_ensemble(path: Path = ENSEMBLE_PATH) -> tuple:
    """
    Load a persisted ensemble from disk.

    Returns:
        (models, cat_cols, cat_encoders, calibrator, threshold)
        calibrator: fitted IsotonicRegression / LogisticRegression, or None.
        threshold:  saved decision threshold (defaults to 0.5 if not present).
    """
    data = joblib.load(path)
    return (
        data["models"],
        data["cat_cols"],
        data.get("cat_encoders", {}),
        data.get("calibrator", None),
        data.get("threshold", 0.5),
    )
