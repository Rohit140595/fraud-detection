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
    # Concatenate as strings so NaN addr1 values don't silently collide across cards
    df["card_addr"] = df["card1"].astype(str) + "-" + df["addr1"].astype(str)
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
    # Sort so expanding window always looks backward in time
    df = df.sort_values([group_col, time_col]).copy()

    # shift(1) excludes the current transaction — mean is over all prior rows only
    df["hist_mean_amt"] = (
        df.groupby(group_col)[amount_col]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # First transaction per user has no history — fill with current amount so deviation = 0
    df["hist_mean_amt"] = df["hist_mean_amt"].fillna(df[amount_col])
    df["amt_deviation"] = df[amount_col] - df["hist_mean_amt"]

    return df


def compute_time_features(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Extract cyclic time features from the transaction timestamp.

    TransactionDT is seconds elapsed from an arbitrary reference point —
    no real calendar conversion needed. Modulo arithmetic gives hour and
    day-of-week directly.

    Adds:
        hour_of_day  : 0–23
        day_of_week  : 0–6  (0 = Monday relative to reference)
    """
    df = df.copy()
    # Integer division to seconds → hours → modulo for 24-hour cycle
    df["hour_of_day"] = (df[time_col] // 3600) % 24
    # 86400 seconds per day; modulo 7 gives day index within the week
    df["day_of_week"] = (df[time_col] // 86400) % 7
    return df


def compute_velocity_multi_window(
    df: pd.DataFrame,
    windows: dict = None,
    group_col: str = "card_addr",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Count prior transactions per user across multiple time windows.

    Uses the bisect approach — O(n log n) per window.
    Adds one column per window key, e.g. velocity_1h, velocity_24h, velocity_7d.

    Args:
        df:        DataFrame with group_col and time_col.
        windows:   Dict of {column_suffix: window_seconds}.
        group_col: Column identifying the user.
        time_col:  Column with transaction timestamps in seconds.
    """
    if windows is None:
        windows = {"1h": 3600, "24h": 86400, "7d": 604800}

    # Sort once — bisect requires a sorted list within each group
    df = df.sort_values([group_col, time_col]).copy()

    for label, secs in windows.items():
        col = f"velocity_{label}"
        df[col] = 0
        for _, group in df.groupby(group_col):
            times = group[time_col].tolist()
            # bisect_left(times, t) gives index of t; subtracting the index of
            # (t - secs) gives the count of transactions in the window [t-secs, t)
            counts = [
                bisect_left(times, t) - bisect_left(times, t - secs)
                for t in times
            ]
            df.loc[group.index, col] = counts

    return df


def _expanding_nunique(series: pd.Series) -> pd.Series:
    """Count distinct values seen before each row — O(n), leak-free by construction."""
    seen = set()
    result = []
    for val in series:
        # Append count BEFORE adding current value → leak-free without needing shift
        result.append(len(seen))
        seen.add(val)
    return pd.Series(result, index=series.index)


def compute_card_aggregates(
    df: pd.DataFrame,
    group_col: str = "card1",
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Expanding unique-count aggregates per card — leak-free, O(n log n).

    A card appearing at many billing addresses or with many distinct amounts
    is a strong fraud signal. _expanding_nunique records the count before
    adding the current value, so no shift is needed.

    Adds:
        card_unique_addr : distinct addr1 values seen before this transaction
        card_unique_amt  : distinct TransactionAmt values seen before this transaction
    """
    # Sort so _expanding_nunique processes each card's transactions in time order
    df = df.sort_values([group_col, time_col]).copy()

    # How many distinct billing addresses has this card used before this transaction?
    df["card_unique_addr"] = (
        df.groupby(group_col)["addr1"]
        .transform(_expanding_nunique)
        .astype(int)
    )

    # How many distinct transaction amounts has this card produced before this transaction?
    df["card_unique_amt"] = (
        df.groupby(group_col)["TransactionAmt"]
        .transform(_expanding_nunique)
        .astype(int)
    )

    return df


def merge_identity(trn: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """
    Left join identity features onto transactions.

    Not all transactions have a matching identity record — unmatched rows
    receive NaN for all identity columns, which LightGBM handles natively
    without requiring imputation.

    Args:
        trn: Raw transaction DataFrame (must contain 'TransactionID').
        idn: Raw identity DataFrame (must contain 'TransactionID').

    Returns:
        Merged DataFrame with identity columns appended.
    """
    # Left join preserves every transaction row; unmatched identity rows are dropped
    df = pd.merge(trn, idn, on="TransactionID", how="left")

    print(f"Merged            : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Identity coverage : {idn['TransactionID'].nunique() / len(trn):.1%}")

    return df


def build_features(trn: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. Merge identity features onto transactions (left join)
      2. Add user proxy (card_addr)
      3. Velocity across 1h / 24h / 7d windows
      4. Amount deviation from historical mean
      5. Time features (hour of day, day of week)
      6. Card-level unique address and amount counts

    Args:
        trn: Raw transaction DataFrame.
        idn: Raw identity DataFrame.

    Returns:
        DataFrame with all engineered features added.
    """
    df = merge_identity(trn, idn)
    df = add_user_proxy(df)
    df = compute_velocity_multi_window(df)
    df = compute_amount_deviation(df)
    df = compute_time_features(df)
    df = compute_card_aggregates(df)
    return df
