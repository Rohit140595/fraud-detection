
"""
Streaming simulation for real-time fraud detection.

Replays a sample of the test set against the live /predict endpoint,
recording fraud scores and latency for each transaction.
Consumed by notebooks/streaming_simulation.ipynb for rolling metric visualisation.
"""

import pandas as pd
import time
import requests

# Columns to exclude from the payload — isFraud is the label, the rest are
# computed server-side by the API's feature engineering pipeline
_EXCLUDE_COLS = {
    "isFraud", "card_addr",
    "hour_of_day", "day_of_week",
    "email_domain_match", "is_free_email",
    "amt_cents", "is_round_amt",
    "log_d1", "d1_null_flag",
    "amt_deviation",
}

def load_test_sample(df: pd.DataFrame, n: int, random_state: int = 414) -> pd.DataFrame:
    """
    Draw a reproducible random sample from a transaction DataFrame.

    Args:
        df:           Source DataFrame (e.g. the test split from model_training.ipynb).
        n:            Number of rows to sample.
        random_state: Seed for reproducibility (default 414).

    Returns:
        DataFrame of n sampled rows.
    """
    
    sample = df.sample(n, random_state=random_state).reset_index(drop=True)
    
    return sample
    
    
def row_to_payload(row: pd.Series) -> dict:
    """
    Convert a DataFrame row into a JSON-serialisable dict for the /predict endpoint.

    Sends all columns except those in _EXCLUDE_COLS (label, server-computed features).
    This includes raw fields, behavioral features, and V/C/M columns so the model
    receives the same feature set it was trained on.

    NaN values are mapped to None (JSON null) so optional fields deserialise
    correctly as None in TransactionRequest rather than failing JSON validation.

    Args:
        row: A single row from the test sample DataFrame (pd.Series).

    Returns:
        Dict of all non-excluded fields with NaN replaced by None, ready to POST as JSON.
    """
    feat_dict = {}
    for field in row.index:
        if field in _EXCLUDE_COLS:
            continue
        val = row[field]
        feat_dict[field] = None if pd.isna(val) else val
    return feat_dict
    
    
def send_request(url: str, payload: dict) -> dict:
    """
    POST a transaction payload to the /predict endpoint and return the result with latency.

    Latency is measured as wall-clock round-trip time including network + server processing.
    On non-200 responses, returns a dict with None scores so run_simulation can still
    record the row (with true_label) without crashing — failed rows are visible in the
    results DataFrame and excluded from metric computations via NaN handling.

    Args:
        url:     Full URL of the /predict endpoint (e.g. 'http://localhost:8000/predict').
        payload: JSON-serialisable dict produced by row_to_payload.

    Returns:
        Dict with fraud_probability, is_fraud, threshold, and latency_ms.
        On failure, fraud_probability/is_fraud/threshold are None.
    """
    start = time.perf_counter()
    response = requests.post(url, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        return {"fraud_probability": None, "is_fraud": None,
                "threshold": None, "latency_ms": elapsed_ms}

    result = response.json()
    result["latency_ms"] = elapsed_ms
    return result
        
        
def run_simulation(
    sample: pd.DataFrame,
    url: str,
    delay_ms: float = 0.0,
) -> pd.DataFrame:
    """
    Send each row in sample to the /predict endpoint and collect results.

    Iterates the sample sequentially, sleeping delay_ms between requests to
    simulate a realistic transaction arrival rate.

    Args:
        sample:   DataFrame produced by load_test_sample.
        url:      Full URL of the /predict endpoint.
        delay_ms: Sleep between requests in milliseconds (default 0 — no delay).

    Returns:
        DataFrame with one row per transaction containing true_label,
        fraud_probability, is_fraud, threshold, and latency_ms.
    """
    results = []

    for _, row in sample.iterrows():
        payload = row_to_payload(row)
        result = send_request(url, payload)
        result["true_label"] = int(row["isFraud"])
        results.append(result)

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    return pd.DataFrame(results)