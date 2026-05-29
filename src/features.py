"""
Feature engineering for real-time fraud detection.

Two core feature families:
  - Velocity   : how many transactions has this user made in the last N seconds?
  - Deviation  : how far is this transaction amount from the user's historical mean?

Both are computed in a leak-free way — only prior transactions are used.
"""

from bisect import bisect_left
import pandas as pd
import numpy as np

FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com"}

# Identity columns id_01–id_38 from the Vesta identity dataset
_ID_COLS = [f"id_{i:02d}" for i in range(1, 39)]

# id columns with zero importance in the trained model — confirmed noise.
# Dropped early to reduce feature space before SelectFromModel runs.
_WEAK_ID_COLS = {
    "id_01", "id_03", "id_04", "id_09", "id_10",
    "id_14", "id_15", "id_18", "id_19", "id_30", "id_38",
}

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
        hour_of_day   : 0–23
        day_of_week   : 0–6  (0 = Monday relative to reference)
        is_night      : 1 if hour_of_day is 0–5 (midnight to 5am)
        is_weekend    : 1 if day_of_week is 5 or 6 (Saturday/Sunday)
        month_of_year : 0–11, approximate month index using 30-day months.
                        Captures seasonal fraud patterns without leaking the
                        absolute timestamp into the model.
    """
    df = df.copy()
    # Integer division to seconds → hours → modulo for 24-hour cycle
    df["hour_of_day"] = (df[time_col] // 3600) % 24
    # 86400 seconds per day; modulo 7 gives day index within the week
    df["day_of_week"] = (df[time_col] // 86400) % 7
    # Fraud is more common in off-hours when monitoring is lighter
    df["is_night"] = (df["hour_of_day"].between(0, 5)).astype(int)
    # 1 on Saturday/Sunday — weekend fraud patterns differ from weekday
    df["is_weekend"] = (df["day_of_week"].isin([5, 6])).astype(int)
    # Approximate month index (0–11) using 30-day months relative to reference.
    # Captures seasonal fraud patterns without leaking the absolute timestamp.
    df["month_of_year"] = (df[time_col] // (86400 * 30)) % 12
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


def compute_email_features(
    df: pd.DataFrame,
    p_col: str = "P_emaildomain",
    r_col: str = "R_emaildomain",
) -> pd.DataFrame:
    """
    Extract risk signals from purchaser and recipient email domains.

    Two features:
      email_domain_match : 1 if purchaser and recipient share the same domain.
                           Mismatched domains can indicate account takeover or
                           third-party purchases — higher fraud risk.
      is_free_email      : 1 if the purchaser email is from a free provider
                           (gmail, yahoo, hotmail, etc.). Free accounts are
                           cheaper to create and more common in fraud.

    NaN domains (no email on file) are treated as non-matching and non-free.

    Args:
        df:    DataFrame containing p_col and r_col.
        p_col: Purchaser email domain column.
        r_col: Recipient email domain column.

    Returns:
        DataFrame with 'email_domain_match' and 'is_free_email' columns added.
    """
    df = df.copy()
    # Both sides must be non-null for a valid match; NaN == NaN would be a false positive
    df["email_domain_match"] = (
        df[p_col].notna() & df[r_col].notna() & (df[p_col] == df[r_col])
    ).astype(int)

    # Map purchaser domain to free-provider flag; NaN → 0
    df["is_free_email"] = df[p_col].isin(FREE_EMAIL_DOMAINS).astype(int)

    return df


def compute_amount_features(
    df: pd.DataFrame,
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """
    Extract fraud signals from transaction amount structure.

    Fraudsters often use amounts ending in .00 (round numbers) or
    specific cent patterns (.99, .95) that differ from normal spending.

    Two features:
      amt_cents     : fractional part of the amount (e.g. 99.99 → 0.99).
                      Recurring cent values can be a fraud fingerprint.
      is_round_amt  : 1 if the amount has no cents (e.g. 100.00).
                      Round amounts are overrepresented in card-testing fraud.

    Args:
        df:         DataFrame containing amount_col.
        amount_col: Column with transaction amounts.

    Returns:
        DataFrame with 'amt_cents' and 'is_round_amt' columns added.
    """
    df = df.copy()
    # Round to 2 decimal places first to avoid floating-point noise (e.g. 99.9999999)
    rounded = df[amount_col].round(2)
    df["amt_cents"] = (rounded % 1).round(2)
    # Amount is round when cents portion is exactly zero
    df["is_round_amt"] = (df["amt_cents"] == 0).astype(int)

    return df


def compute_d_features(
    df: pd.DataFrame,
    d_col_list: list = ['D' + str(x) for x in range(1, 10)],
) -> pd.DataFrame:
    """
    Engineer log-transform and null-flag features for D1–D9 (days-since columns).

    D columns are right-skewed — most cardholders transact recently but outliers
    can be very large. Log-transforming compresses the scale so LightGBM splits
    are more evenly distributed. Columns absent from the DataFrame are silently
    skipped — safe for API inference where only D1 is supplied.

    For each D column present, adds two features:
      log_{d}      : log1p(D) — compressed scale, preserves zero.
      {d}_null_flag: 1 when D is missing (no prior transaction on record).
                     NaN and a genuinely dormant card are different signals —
                     this flag lets the model distinguish them.

    Args:
        df:         DataFrame that may contain any subset of d_col_list.
        d_col_list: D columns to process (default D1–D9).

    Returns:
        DataFrame with log_{d} and {d}_null_flag columns added for each
        D column present in the input.
    """
    df = df.copy()
    for d_col in d_col_list:
        if d_col not in df.columns:
            continue
        # Coerce to float first — None (from API) becomes NaN, which log1p handles cleanly
        df[d_col] = pd.to_numeric(df[d_col], errors="coerce")
        # Flag missing D1 before the transform — log1p(-1) = -inf, not NaN,
        # so checking the transformed column would silently miss invalid negatives
        null_col = d_col + '_null_flag'
        df[null_col] = df[d_col].isnull().astype(int)
        # log1p compresses the right tail while keeping log1p(0) = 0
        log_col = 'log_' + d_col
        df[log_col] = np.log1p(df[d_col])

    return df


def _extract_os(device_info) -> str:
    """
    Map a raw DeviceInfo string to a coarse OS category.

    Returns None for missing DeviceInfo so the resulting device_os column
    contains NaN — LightGBM handles NaN natively and treats it differently
    from a real category value, avoiding the dominant 'unknown' sentinel
    that previously distorted splits.
    """
    if pd.isna(device_info):
        return None   # NaN → LightGBM handles natively; is_unknown_os=1 flags this
    d = str(device_info).lower()
    if "windows" in d:
        return "windows"
    if "ios" in d or "iphone" in d or "ipad" in d:
        return "ios"
    if "mac" in d:
        return "macos"
    if "android" in d or "samsung" in d or "sm-" in d:
        return "android"
    if "linux" in d:
        return "linux"
    return "other"


def compute_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer risk signals from identity and device columns.

    Identity records are only available for ~24% of transactions — column
    guards ensure this function is safe to call when any field is absent
    (e.g. at API inference time when identity columns are not supplied).

    Adds:
        is_unknown_os : 1 when DeviceInfo is absent — explicit flag for
                        transactions with no device fingerprint. Replaces
                        the old 'has_identity' and 'id_null_count' features
                        which were both near-constant (76% rows had no identity
                        record) and redundant with each other.
        is_mobile     : 1 if DeviceType is 'mobile'.
        device_os     : coarse OS category from DeviceInfo
                        (windows / ios / macos / android / linux / other).
                        NaN when DeviceInfo is absent — LightGBM handles NaN
                        natively, avoiding the dominant 'unknown' sentinel that
                        previously distorted tree splits.

    Args:
        df: DataFrame that may contain DeviceType and DeviceInfo.

    Returns:
        DataFrame with is_unknown_os, is_mobile, and device_os added.
    """
    df = df.copy()

    # 1. is_unknown_os — explicit flag for missing device fingerprint.
    # Replaces has_identity (correlated inverse) and id_null_count (near-constant).
    if "DeviceInfo" in df.columns:
        df["is_unknown_os"] = df["DeviceInfo"].isna().astype(int)
    else:
        df["is_unknown_os"] = 1  # no device info column at all

    # 2. is_mobile — mobile and desktop fraud patterns differ meaningfully
    if "DeviceType" in df.columns:
        df["is_mobile"] = (df["DeviceType"].str.lower() == "mobile").astype(int)
    else:
        df["is_mobile"] = 0

    # 3. device_os — NaN for missing DeviceInfo (not "unknown") so LightGBM
    # treats absence as missing rather than a dominant categorical value.
    if "DeviceInfo" in df.columns:
        df["device_os"] = pd.Categorical(df["DeviceInfo"].apply(_extract_os))
    else:
        df["device_os"] = pd.Categorical([None] * len(df))

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


def build_features(trn: pd.DataFrame, idn: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
      1. Merge identity features onto transactions (left join, optional)
      2. Drop confirmed zero-importance id columns (_WEAK_ID_COLS)
      3. Add user proxy (card_addr)
      4. Velocity across 1h / 24h / 7d windows
      5. Amount deviation from historical mean
      6. Time features (hour of day, day of week, month of year)
      7. Card-level unique address and amount counts
      8. Email domain features (match flag, free-provider flag)
      9. Amount structure features (cents portion, round-number flag)
     10. D1–D9 features (log-transform, null flag)
     11. Identity features (is_unknown_os, is_mobile, device_os)

    Args:
        trn: Raw transaction DataFrame.
        idn: Optional identity DataFrame. If provided, left-joined onto transactions.

    Returns:
        DataFrame with all engineered features added.
    """
    df = merge_identity(trn, idn) if idn is not None else trn.copy()

    # Drop id columns with confirmed zero importance — reduces noise before
    # SelectFromModel runs without relying on the model to filter them out.
    weak_cols = [c for c in _WEAK_ID_COLS if c in df.columns]
    if weak_cols:
        df = df.drop(columns=weak_cols)
        print(f"Dropped {len(weak_cols)} weak id columns")

    df = add_user_proxy(df)
    df = compute_velocity_multi_window(df)
    df = compute_amount_deviation(df)
    df = compute_time_features(df)
    df = compute_card_aggregates(df)
    df = compute_email_features(df)
    df = compute_amount_features(df)
    df = compute_d_features(df)
    df = compute_identity_features(df)
    return df
