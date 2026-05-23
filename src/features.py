"""
Feature engineering for real-time fraud detection.

Two core feature families:
  - Velocity   : how many transactions has this user made in the last N seconds?
  - Deviation  : how far is this transaction amount from the user's historical mean?

Both are computed in a leak-free way — only prior transactions are used.
"""

from bisect import bisect_left
import pandas as pd


def add_user_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user identity proxy from card number + billing region.

    There is no explicit user ID in this dataset. card1 (high-cardinality
    card identifier) combined with addr1 (billing region) is a reliable proxy
    for a unique cardholder.
    """
    df = df.copy()
    df["card_addr"] = df["card1"].astype(str) + "-" + df["addr1"].astype(str)
    return df


def compute_velocity(
    df: pd.DataFrame,
    window_seconds: int = 3600,
    group_col: str = "card_addr",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Count prior transactions per user within a rolling time window.

    Uses binary search (O(log k) per transaction) instead of filtering,
    giving overall O(n log n) complexity vs the naive O(n^2).

    Args:
        df:             DataFrame containing group_col and time_col.
        window_seconds: Size of the lookback window in seconds (default 1 hour).
        group_col:      Column identifying the user (default 'card_addr').
        time_col:       Column with transaction timestamps in seconds (default 'TransactionDT').

    Returns:
        DataFrame with a new column 'velocity_1h'.
    """
    df = df.sort_values([group_col, time_col]).copy()
    df["velocity_1h"] = 0

    for _, group in df.groupby(group_col):
        times = group[time_col].tolist()
        counts = [
            bisect_left(times, t) - bisect_left(times, t - window_seconds)
            for t in times
        ]
        df.loc[group.index, "velocity_1h"] = counts

    return df


def compute_amount_deviation(
    df: pd.DataFrame,
    group_col: str = "card_addr",
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """
    Compute how far each transaction deviates from the user's historical mean.

    Uses an expanding mean shifted by 1 row so the current transaction is
    never included — leak-free by design.

    For a user's first transaction, hist_mean_amt is filled with the current
    transaction amount (deviation = 0), since no prior history exists.

    Args:
        df:         DataFrame with group_col, time_col, and amount_col.
        group_col:  Column identifying the user.
        time_col:   Column with transaction timestamps.
        amount_col: Column with transaction amounts.

    Returns:
        DataFrame with two new columns: 'hist_mean_amt' and 'amt_deviation'.
    """
    df = df.sort_values([group_col, time_col]).copy()

    df["hist_mean_amt"] = (
        df.groupby(group_col)[amount_col]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # First transaction per user has no history — fill with current amount
    df["hist_mean_amt"] = df["hist_mean_amt"].fillna(df[amount_col])
    df["amt_deviation"] = df[amount_col] - df["hist_mean_amt"]

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. Add user proxy (card_addr)
      2. Compute transaction velocity (last 1 hour)
      3. Compute amount deviation from historical mean

    Args:
        df: Raw transaction DataFrame.

    Returns:
        DataFrame with all engineered features added.
    """
    df = add_user_proxy(df)
    df = compute_velocity(df)
    df = compute_amount_deviation(df)
    return df
