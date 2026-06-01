"""
Kafka consumer / scorer for real-time fraud detection.

Consumes raw transaction messages from the raw-transactions topic, scores each
one through the stacking ensemble, and publishes results to the fraud-scores topic.

Design decisions:
  - Ensemble loaded once at startup — avoids per-message disk I/O.
  - Feature engineering mirrors api.py exactly: add_user_proxy → time → email →
    amount → D-columns → identity → ordinal encode → reindex → score.
  - Results include latency_ms so a downstream consumer can monitor throughput.
  - _true_label is passed through (if present) so offline PR-AUC can be computed
    by tailing fraud-scores without re-joining the source data.
  - Consumer group 'fraud-scorer' allows horizontal scaling: spin up multiple
    instances sharing the same group and Kafka distributes partitions across them.
    Use a different group id to replay all messages from the beginning.

Usage:
    python -m src.kafka_consumer                               # default topics
    python -m src.kafka_consumer --input my-topic             # custom input
    python -m src.kafka_consumer --group replay --offset earliest  # replay all
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from src.ensemble import ENSEMBLE_PATH, apply_calibrator, load_ensemble, predict_stack
from src.features import (
    add_user_proxy,
    compute_amount_features,
    compute_d_features,
    compute_email_features,
    compute_identity_features,
    compute_time_features,
)

DEFAULT_BOOTSTRAP    = "localhost:9092"
DEFAULT_INPUT_TOPIC  = "raw-transactions"
DEFAULT_OUTPUT_TOPIC = "fraud-scores"
DEFAULT_GROUP_ID     = "fraud-scorer"

# Columns that must stay as strings — everything else is coerced to float64
_STRING_COLS = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
                "DeviceType", "DeviceInfo"}


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all non-string columns to float64, converting None → NaN."""
    for col in df.columns:
        if col not in _STRING_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _score_transaction(
    payload: dict,
    models: dict,
    cat_cols: list[str],
    cat_encoders: dict,
    calibrator,
    threshold: float,
    meta_learner,
) -> dict:
    """
    Score a single transaction payload through the stacking ensemble.

    Pipeline mirrors api.py/predict() exactly:
      1. Compute amt_deviation from hist_mean_amt.
      2. Build single-row DataFrame and coerce dtypes.
      3. Static feature engineering.
      4. Ordinal-encode categoricals using training-time codes.
      5. Reindex to each model's expected feature set and score.
      6. Blend via meta-learner (or soft-vote if no meta-learner).
      7. Apply calibrator if present.

    Args:
        payload:      Transaction fields (same shape as /predict request body).
        models:       Dict of fitted base models from load_ensemble().
        cat_cols:     Categorical column names.
        cat_encoders: Label→code mappings fit at training time.
        calibrator:   Fitted calibrator or None.
        threshold:    Decision threshold for is_fraud.
        meta_learner: Fitted XGBoost meta-learner or None.

    Returns:
        Dict with fraud_probability, is_fraud, threshold, latency_ms.
    """
    t0 = time.perf_counter()

    # Amount deviation — how far this transaction is from historical mean
    hist = payload.get("hist_mean_amt") or payload.get("TransactionAmt", 0)
    payload["amt_deviation"] = payload.get("TransactionAmt", 0) - hist

    df = pd.DataFrame([payload])
    df = _coerce_dtypes(df)

    df = add_user_proxy(df)
    df = compute_time_features(df)
    df = compute_email_features(df)
    df = compute_amount_features(df)
    df = compute_d_features(df)
    df = compute_identity_features(df)

    for col in cat_cols:
        if col in df.columns:
            encoder = cat_encoders.get(col, {})
            df[col] = df[col].map(encoder).fillna(-1).astype(int)

    lgbm_m    = models["lgbm"]
    xgb_m     = models["xgb"]
    lgbm_prob = lgbm_m.predict_proba(df.reindex(columns=lgbm_m.feature_names_in_))[0, 1]
    xgb_prob  = xgb_m.predict_proba(df.reindex(columns=xgb_m.feature_names_in_))[0, 1]

    if meta_learner is not None:
        meta_in  = pd.DataFrame({"lgbm_oof": [lgbm_prob], "xgb_oof": [xgb_prob]})
        raw_prob = float(meta_learner.predict_proba(meta_in)[0, 1])
    else:
        raw_prob = (lgbm_prob + xgb_prob) / 2.0

    fraud_probability = (
        float(apply_calibrator(calibrator, np.array([raw_prob]))[0])
        if calibrator is not None else float(raw_prob)
    )

    return {
        "fraud_probability": round(fraud_probability, 6),
        "is_fraud":          bool(fraud_probability >= threshold),
        "threshold":         threshold,
        "latency_ms":        round((time.perf_counter() - t0) * 1_000, 2),
    }


def _safe_deserialize(raw: bytes):
    """Return parsed JSON dict, or None if the message is not valid JSON."""
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def run_consumer(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    input_topic:  str = DEFAULT_INPUT_TOPIC,
    output_topic: str = DEFAULT_OUTPUT_TOPIC,
    group_id: str = DEFAULT_GROUP_ID,
    auto_offset_reset: str = "latest",
    model_path: Path | None = None,
) -> None:
    """
    Consume transactions, score them, publish results — runs until interrupted.

    Args:
        bootstrap_servers: Kafka bootstrap address.
        input_topic:        Topic to consume raw transactions from.
        output_topic:       Topic to publish fraud scores to.
        group_id:           Consumer group ID.
        auto_offset_reset:  'latest' (new messages only) or 'earliest' (replay all).
        model_path:         Path to ensemble pickle; defaults to config path.
    """
    path = Path(model_path) if model_path else ENSEMBLE_PATH
    print(f"Loading ensemble from {path} ...")
    models, cat_cols, cat_encoders, calibrator, threshold, meta_learner = load_ensemble(path)
    print(f"Ensemble loaded — threshold={threshold:.4f}, meta_learner={meta_learner is not None}")

    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        value_deserializer=lambda v: _safe_deserialize(v),
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Listening on '{input_topic}' → scoring → '{output_topic}'  (Ctrl+C to stop)")
    scored = flagged = 0

    try:
        for msg in consumer:
            # Skip messages that failed deserialization (e.g. plain-text or empty)
            if msg.value is None:
                continue
            envelope   = msg.value
            # Skip messages that don't match our envelope schema
            if "transaction" not in envelope:
                continue
            payload    = dict(envelope.get("transaction", {}))
            true_label = payload.pop("_true_label", None)

            try:
                result = _score_transaction(
                    payload, models, cat_cols, cat_encoders,
                    calibrator, threshold, meta_learner,
                )
            except Exception as exc:
                print(f"  [ERROR] scoring failed for message {envelope.get('message_id')}: {exc}")
                continue

            out = {
                "message_id":  envelope.get("message_id"),
                "produced_at": envelope.get("produced_at"),
                "scored_at":   datetime.now(timezone.utc).isoformat(),
                **result,
            }
            if true_label is not None:
                out["true_label"] = true_label

            producer.send(output_topic, value=out)
            scored  += 1
            flagged += int(result["is_fraud"])

            if scored % 100 == 0:
                print(
                    f"  Scored {scored:,} | flagged {flagged} ({flagged/scored:.1%})"
                    f" | last latency {result['latency_ms']:.1f}ms"
                )

    except KeyboardInterrupt:
        print(f"\nShutting down — scored {scored:,} transactions, flagged {flagged}.")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud detection Kafka consumer/scorer")
    parser.add_argument("--bootstrap",  default=DEFAULT_BOOTSTRAP,    help="Kafka bootstrap servers")
    parser.add_argument("--input",      default=DEFAULT_INPUT_TOPIC,   help="Input topic")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT_TOPIC,  help="Output topic")
    parser.add_argument("--group",      default=DEFAULT_GROUP_ID,      help="Consumer group ID")
    parser.add_argument("--offset",     default="latest",
                        choices=["latest", "earliest"],                help="Auto offset reset")
    parser.add_argument("--model",      default=None,                  help="Path to ensemble pkl")
    args = parser.parse_args()

    run_consumer(
        bootstrap_servers=args.bootstrap,
        input_topic=args.input,
        output_topic=args.output,
        group_id=args.group,
        auto_offset_reset=args.offset,
        model_path=args.model,
    )
