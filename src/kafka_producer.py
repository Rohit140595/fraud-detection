"""
Kafka producer for real-time fraud detection.

Reads the pre-engineered test set (data/test_features.parquet) and publishes
each transaction to the raw-transactions topic as a JSON message.

Design decisions:
  - row_to_payload() from simulator.py is reused — strips the label and all
    server-side computed features, keeping only fields the scoring consumer needs.
  - _true_label is included in the message payload for offline evaluation; in a
    real production feed this field would not exist.
  - Configurable delay lets you simulate realistic arrival rates (e.g. 100ms) or
    run at full speed for load testing.
  - Messages are keyed by TransactionID so partitioning is deterministic.

Usage:
    python -m src.kafka_producer                        # full test set, no delay
    python -m src.kafka_producer --n 500 --delay 100   # 500 rows, 100ms apart
    python -m src.kafka_producer --topic custom-topic   # custom topic
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaProducer

from src.simulator import row_to_payload

DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_TOPIC     = "raw-transactions"
DEFAULT_DATA_PATH = "data/test_features.parquet"


def make_producer(bootstrap_servers: str = DEFAULT_BOOTSTRAP) -> KafkaProducer:
    """Create a KafkaProducer with JSON serialisation."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def publish_transactions(
    data_path: str = DEFAULT_DATA_PATH,
    topic: str = DEFAULT_TOPIC,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP,
    n: int | None = None,
    delay_ms: float = 0.0,
    random_state: int = 42,
) -> None:
    """
    Stream transactions from a parquet file to a Kafka topic.

    Each message is a JSON envelope:
      {
        "message_id":  "<uuid>",
        "produced_at": "<iso8601 utc>",
        "transaction": { ...payload fields... , "_true_label": 0/1 }
      }

    Args:
        data_path:         Path to test_features.parquet.
        topic:             Kafka topic to publish to.
        bootstrap_servers: Kafka bootstrap address.
        n:                 Rows to publish (None = all).
        delay_ms:          Sleep between messages in ms (0 = full speed).
        random_state:      Seed for reproducible sampling when n is set.
    """
    df = pd.read_parquet(data_path)
    if n is not None:
        df = df.sample(min(n, len(df)), random_state=random_state).reset_index(drop=True)

    producer = make_producer(bootstrap_servers)
    print(f"Publishing {len(df):,} transactions → topic '{topic}' ...")

    sent = 0
    for _, row in df.iterrows():
        payload = row_to_payload(row)

        # Pass true label through for offline evaluation — absent in production
        if "isFraud" in row.index:
            payload["_true_label"] = int(row["isFraud"])

        message = {
            "message_id":  str(uuid.uuid4()),
            "produced_at": datetime.now(timezone.utc).isoformat(),
            "transaction": payload,
        }

        key = str(int(row["TransactionID"])) if "TransactionID" in row.index else None
        producer.send(topic, key=key, value=message)
        sent += 1

        if sent % 1_000 == 0:
            print(f"  Sent {sent:,} / {len(df):,}")

        if delay_ms > 0:
            time.sleep(delay_ms / 1_000)

    producer.flush()
    print(f"Done — {sent:,} messages published to '{topic}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud detection Kafka producer")
    parser.add_argument("--data",      default=DEFAULT_DATA_PATH, help="Path to parquet file")
    parser.add_argument("--topic",     default=DEFAULT_TOPIC,     help="Kafka topic name")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP, help="Kafka bootstrap servers")
    parser.add_argument("--n",         type=int,   default=None,  help="Rows to send (default: all)")
    parser.add_argument("--delay",     type=float, default=0.0,   help="Delay between messages in ms")
    args = parser.parse_args()

    publish_transactions(
        data_path=args.data,
        topic=args.topic,
        bootstrap_servers=args.bootstrap,
        n=args.n,
        delay_ms=args.delay,
    )
