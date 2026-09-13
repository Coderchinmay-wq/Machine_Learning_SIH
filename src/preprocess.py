"""
preprocess.py
=============
This file contains ALL the data-cleaning and feature-engineering logic
in ONE place, so that:

    - train.py uses it to prepare the training data
    - api.py / predict.py use the EXACT SAME logic to prepare a new,
      single incoming sensor reading before asking the model to predict

This is very important: if training and prediction preprocess data
differently, the model will silently give wrong answers. Keeping the
logic in one shared file prevents that class of bug.

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
1. Loads the raw CSV and fixes obvious data problems (missing values,
   duplicate rows, bad timestamps).
2. Adds new "engineered" features that describe HOW readings are
   CHANGING over time for each node (deltas and rolling averages),
   because a mine roof that is moving fast is more dangerous than one
   that is stable, even at the same absolute distance.
3. Defines the final, fixed list and ORDER of feature columns the
   model will be trained on - this list is saved alongside the model
   so that predictions later use identical columns in identical order.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# The exact raw sensor columns we expect to receive from the hardware.
RAW_SENSOR_COLUMNS = [
    "distance", "tilt_x", "tilt_y", "vibration", "temperature", "humidity"
]

# How many previous readings to average over for the rolling-mean
# features. 3 means "the last 3 readings including the current one".
ROLLING_WINDOW = 3

# The final, ordered list of feature columns used to train/predict.
# This is intentionally built as a function so both train.py and
# predict.py/api.py agree on it without copy-pasting a list twice.
def get_feature_columns() -> list[str]:
    delta_cols = [f"delta_{c}" for c in ["distance", "tilt_x", "tilt_y", "vibration"]]
    rolling_cols = [f"rollmean_{c}" for c in ["distance", "tilt_x", "tilt_y", "vibration"]]
    return RAW_SENSOR_COLUMNS + delta_cols + rolling_cols


# ---------------------------------------------------------------------
# STAGE 1: Loading + cleaning
# ---------------------------------------------------------------------
def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the raw sensor CSV and clean it.

    Beginner explanation of every step:
      - We parse 'timestamp' as a real datetime object (not just text)
        so that we can sort readings in the correct time order later.
      - We drop rows where the target label 'status' is missing,
        because a row with no label is useless for supervised learning.
      - We drop exact duplicate rows (same node, same timestamp, same
        readings) which can happen due to sensor/network retransmits.
      - For missing values in numeric sensor columns, we do NOT throw
        the row away automatically (that wastes real data). Instead we
        fill the missing value with that node's median for that column.
        The median is used (rather than the mean) because it is less
        affected by extreme/outlier sensor spikes.
    """
    df = pd.read_csv(csv_path)

    required_cols = ["timestamp", "node_id", *RAW_SENSOR_COLUMNS, "status"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"CSV is missing required columns: {missing_required}")

    # --- Convert timestamp to real datetime ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_timestamps = df["timestamp"].isna().sum()
    if n_bad_timestamps > 0:
        print(f"[preprocess] Dropping {n_bad_timestamps} rows with unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])

    # --- Drop rows with missing labels (can't train on unlabeled rows) ---
    n_missing_status = df["status"].isna().sum()
    if n_missing_status > 0:
        print(f"[preprocess] Dropping {n_missing_status} rows with missing 'status' label.")
        df = df.dropna(subset=["status"])

    # --- Normalize label text (lowercase, strip spaces) ---
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    valid_labels = {"normal", "warning", "critical"}
    invalid_mask = ~df["status"].isin(valid_labels)
    if invalid_mask.any():
        print(f"[preprocess] Dropping {invalid_mask.sum()} rows with invalid status values: "
              f"{df.loc[invalid_mask, 'status'].unique().tolist()}")
        df = df[~invalid_mask]

    # --- Remove exact duplicate rows ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"[preprocess] Removed {n_dupes} exact duplicate rows.")

    # --- Handle missing numeric sensor values (impute per-node median) ---
    for col in RAW_SENSOR_COLUMNS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df.groupby("node_id")[col].transform(
                lambda s: s.fillna(s.median())
            )
            # Fallback: if a node has ALL values missing for a column,
            # fall back to the global median.
            df[col] = df[col].fillna(df[col].median())
            print(f"[preprocess] Filled {n_missing} missing values in '{col}' "
                  f"using per-node median.")

    # --- Sort chronologically within each node (required for feature engineering) ---
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# STAGE 2: Feature engineering (used on the full training dataframe)
# ---------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "change over time" features, computed SEPARATELY per node_id so
    that node A's readings never leak into node B's delta/rolling
    calculations.

    Features added:
      - delta_<col>:    value minus the PREVIOUS reading from the same
                         node ("how much did this change since last time").
                         This directly captures the RATE at which the
                         ground is moving/tilting/vibrating, which is a
                         strong physical indicator of developing
                         instability - a fast-changing small tilt can be
                         more dangerous than a large but stable tilt.
      - rollmean_<col>:  short-term rolling average of the last
                         ROLLING_WINDOW readings (including current).
                         This smooths out single-reading sensor noise
                         and reflects the recent short-term trend rather
                         than one possibly-noisy instant.

    WHY THIS AVOIDS DATA LEAKAGE:
      Both `.diff()` and `.rolling().mean()` are CAUSAL - for row i they
      only look at row i and EARLIER rows (i-1, i-2, ...), never at
      future rows. As long as the dataframe is sorted chronologically
      per node before calling this function (which load_and_clean_csv
      already guarantees), no information from the future leaks into
      a given row's features.
    """
    df = df.copy()

    for col in ["distance", "tilt_x", "tilt_y", "vibration"]:
        # Change since previous reading, per node
        df[f"delta_{col}"] = df.groupby("node_id")[col].diff()
        # Rolling short-term average, per node (min_periods=1 so the
        # very first reading(s) of a node still get a value instead of NaN)
        df[f"rollmean_{col}"] = (
            df.groupby("node_id")[col]
              .transform(lambda s: s.rolling(window=ROLLING_WINDOW, min_periods=1).mean())
        )

    # The very first reading of each node has no "previous" reading, so
    # delta_* will be NaN there. We fill these with 0 (interpreted as
    # "no known change yet"), which is a reasonable, explicit default
    # rather than silently dropping the node's first reading.
    delta_cols = [f"delta_{c}" for c in ["distance", "tilt_x", "tilt_y", "vibration"]]
    df[delta_cols] = df[delta_cols].fillna(0.0)

    return df


# ---------------------------------------------------------------------
# STAGE 3: Prepare a SINGLE new reading (used at prediction time)
# ---------------------------------------------------------------------
def build_features_for_new_reading(new_reading: dict, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the same feature set for ONE new incoming sensor reading, for
    use at prediction time (predict.py / api.py).

    Parameters
    ----------
    new_reading : dict
        Raw sensor values for the new reading, e.g.
        {"node_id": "NODE_01", "distance": 80.1, "tilt_x": 4.0, ...}
    history_df : pd.DataFrame or None
        Optional recent history (previous readings) for the SAME node,
        with at least the raw sensor columns, sorted oldest -> newest.
        If provided, it is used to compute real delta_/rollmean_
        features (this is what a production backend would supply from
        its database of recent readings for that node).
        If None, this is the FIRST reading we have ever seen for this
        node (or no history is available), so delta_* features are set
        to 0 and rollmean_* features simply equal the raw values -
        exactly matching how add_engineered_features() treats a node's
        very first row.

    Returns
    -------
    A single-row DataFrame with columns in the exact order returned by
    get_feature_columns(), ready to be passed into model.predict().
    """
    row = {c: float(new_reading[c]) for c in RAW_SENSOR_COLUMNS}

    if history_df is not None and len(history_df) > 0:
        recent = history_df.tail(ROLLING_WINDOW - 1)  # previous readings only
        for col in ["distance", "tilt_x", "tilt_y", "vibration"]:
            prev_value = history_df[col].iloc[-1]
            row[f"delta_{col}"] = row[col] - float(prev_value)
            window_values = pd.concat([recent[col], pd.Series([row[col]])])
            row[f"rollmean_{col}"] = float(window_values.mean())
    else:
        for col in ["distance", "tilt_x", "tilt_y", "vibration"]:
            row[f"delta_{col}"] = 0.0
            row[f"rollmean_{col}"] = row[col]

    feature_cols = get_feature_columns()
    return pd.DataFrame([row], columns=feature_cols)
