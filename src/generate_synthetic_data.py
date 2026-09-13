"""
generate_synthetic_data.py
===========================
THIS FILE CREATES **SYNTHETIC / DEMONSTRATION DATA ONLY**.

It does NOT represent real underground coal mine behaviour. It exists
purely so that, as a student, you can:

    1. Run the entire ML pipeline (train -> evaluate -> save -> predict -> API)
       end to end BEFORE you have real sensor data.
    2. Confirm that your code works correctly.
    3. Later swap this fake CSV for a real one (with the same column
       names) and re-run the exact same pipeline with no code changes.

WHY SYNTHETIC DATA CANNOT REPLACE REAL DATA
--------------------------------------------
Real subsidence behaviour depends on geology, mining method, overburden
depth, water ingress, seismic activity, and many other factors that we
have not modelled here. The patterns below are simple, hand-picked
approximations designed only to be "learnable" by a Random Forest so
that you can test your code. A model trained only on this data must
NEVER be used to make real safety decisions.

HOW THE FAKE DATA IS GENERATED
-------------------------------
For each of a few sensor "nodes" (simulating physical ESP32 nodes
placed at different points in a mine), we simulate a sequence of
readings over time. We slowly drift some underlying "ground stress"
value up and down. When that hidden stress value is:
    - low                -> status = normal
    - medium             -> status = warning
    - high                -> status = critical

The observable sensor readings (distance, tilt_x, tilt_y, vibration,
temperature, humidity) are generated FROM that hidden stress value,
plus random noise, so that the RandomForest actually has to learn a
relationship between sensor readings and status - we are not just
writing the label directly into the columns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def _simulate_one_node(node_id: str, n_readings: int, start_time: datetime,
                        interval_minutes: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate a single sensor node's time series.

    The "hidden_stress" variable is a slow random walk between 0 and 1.
    It is NOT saved in the final CSV - it only exists to generate
    believable, correlated sensor readings.
    """
    # Hidden ground-stress process (0 = calm, 1 = severe).
    # This is a MEAN-REVERTING random walk (it drifts back toward a
    # resting baseline over time) with occasional upward "event" spikes
    # that then decay away. This keeps ALL three condition classes
    # (normal/warning/critical) appearing throughout the WHOLE timeline
    # for each node, rather than the node simply drifting into one
    # extreme and staying there - which is both more realistic (real
    # ground stress rises and falls with mining activity) and better
    # for demonstrating a chronological train/test split, since both
    # the earlier (train) and later (test) portions contain a mix of
    # classes.
    baseline = rng.uniform(0.1, 0.2)     # this node's normal resting stress
    reversion_rate = 0.035               # how strongly it pulls back to baseline
    hidden_stress = np.zeros(n_readings)
    hidden_stress[0] = baseline

    for i in range(1, n_readings):
        pull_to_baseline = reversion_rate * (baseline - hidden_stress[i - 1])
        noise = rng.normal(0, 0.015)
        spike = 0.0
        if rng.random() < 0.012:  # rare event pushing stress up sharply
            spike = rng.uniform(0.3, 0.7)
        hidden_stress[i] = np.clip(
            hidden_stress[i - 1] + pull_to_baseline + noise + spike, 0.0, 1.0
        )

    rows = []
    # A baseline "resting" distance for this node (distance from sensor
    # to rock face/roof, in cm, for example) - varies slightly per node
    base_distance = rng.uniform(75, 95)

    for i in range(n_readings):
        stress = hidden_stress[i]
        timestamp = start_time + timedelta(minutes=interval_minutes * i)

        # As stress increases, distance tends to shrink (roof/face moving
        # closer), tilt increases, vibration increases.
        distance = base_distance - stress * rng.uniform(8, 15) + rng.normal(0, 0.3)
        tilt_x = stress * rng.uniform(6, 10) + rng.normal(0, 0.2)
        tilt_y = stress * rng.uniform(4, 8) + rng.normal(0, 0.2)
        vibration = stress * rng.uniform(0.8, 1.4) + abs(rng.normal(0, 0.05))
        temperature = 26 + rng.normal(0, 1.2) + stress * rng.uniform(0, 1.5)
        humidity = 70 + rng.normal(0, 3) + stress * rng.uniform(0, 5)

        # Label derived from the (hidden) stress level with soft thresholds
        if stress < 0.33:
            status = "normal"
        elif stress < 0.66:
            status = "warning"
        else:
            status = "critical"

        rows.append({
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "node_id": node_id,
            "distance": round(distance, 2),
            "tilt_x": round(tilt_x, 2),
            "tilt_y": round(tilt_y, 2),
            "vibration": round(max(vibration, 0), 3),
            "temperature": round(temperature, 2),
            "humidity": round(min(max(humidity, 0), 100), 2),
            "status": status,
        })

    return pd.DataFrame(rows)


def generate_synthetic_dataset(output_path: str = "data/sensor_data.csv",
                                n_nodes: int = 4,
                                n_readings_per_node: int = 500,
                                interval_minutes: int = 10,
                                seed: int = 42) -> pd.DataFrame:
    """Generate and save the full synthetic/demo dataset."""
    rng = np.random.default_rng(seed)
    start_time = datetime(2026, 1, 1, 0, 0, 0)

    all_dfs = []
    for n in range(1, n_nodes + 1):
        node_id = f"NODE_{n:02d}"
        # Give each node a slightly different start time so timestamps
        # are not perfectly identical across nodes
        node_start = start_time + timedelta(minutes=int(rng.integers(0, 30)))
        df_node = _simulate_one_node(node_id, n_readings_per_node, node_start,
                                      interval_minutes, rng)
        all_dfs.append(df_node)

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Introduce a handful of realistic messiness: a few missing values
    # and a duplicate row, so preprocess.py has something real to clean.
    missing_idx = rng.choice(full_df.index, size=5, replace=False)
    full_df.loc[missing_idx, "humidity"] = np.nan

    duplicate_row = full_df.iloc[[10]]
    full_df = pd.concat([full_df, duplicate_row], ignore_index=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print(f"[SYNTHETIC DATA] Generated {len(full_df)} rows across {n_nodes} nodes.")
    print(f"[SYNTHETIC DATA] Saved to: {output_path}")
    print("[SYNTHETIC DATA] Class distribution:")
    print(full_df["status"].value_counts())
    print("\nREMINDER: This is DEMONSTRATION/SYNTHETIC data only. It does NOT")
    print("represent real Indian underground coal mine behaviour.")

    return full_df


if __name__ == "__main__":
    generate_synthetic_dataset()
