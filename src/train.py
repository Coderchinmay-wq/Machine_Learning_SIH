"""
train.py
========
This is the main script you run to TRAIN the Random Forest model.

WHAT IT DOES, STEP BY STEP
---------------------------
1. Loads and cleans data/sensor_data.csv (using preprocess.py).
2. Adds engineered "change over time" features (using preprocess.py).
3. Splits the data into TRAIN and TEST sets CHRONOLOGICALLY (explained
   below), not randomly.
4. Trains a RandomForestClassifier on the training set.
5. Prints a quick evaluation summary (full evaluation lives in
   evaluate.py, which is more detailed).
6. Saves the trained model + feature column list + label list to
   models/mine_subsidence_model.pkl using joblib, so the API/predict
   script can load it later without retraining.

WHY A CHRONOLOGICAL SPLIT INSTEAD OF A RANDOM SPLIT?
-----------------------------------------------------
scikit-learn's default `train_test_split(..., shuffle=True)` picks rows
RANDOMLY for train vs test. For sensor/time-series data like this, that
causes a subtle but serious problem:

  - Readings that are close together in time are usually very SIMILAR
    (ground conditions don't change instantly). If a 10:00am reading
    ends up in the training set and the 10:10am reading (almost
    identical) ends up in the test set, the model can score very well
    on the test set simply by "memorizing" a near-duplicate neighbour,
    NOT because it actually learned to recognise developing instability
    patterns.
  - This makes test accuracy look artificially high - a form of DATA
    LEAKAGE from the future into training.
  - In real deployment, the model will only ever have PAST data to
    learn from and will be asked to predict the FUTURE. So the fairest
    test is: "train on the earlier readings, test on the later
    readings", per sensor node. That's what we do below.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocess import load_and_clean_csv, add_engineered_features, get_feature_columns

DATA_PATH = "data/sensor_data.csv"
MODEL_PATH = "models/mine_subsidence_model.pkl"
TRAIN_SPLIT_PATH = "data/train_split.csv"
TEST_SPLIT_PATH = "data/test_split.csv"

TEST_FRACTION = 0.2  # last 20% of each node's timeline is held out for testing
LABEL_ORDER = ["normal", "warning", "critical"]  # fixed, meaningful order


def chronological_train_test_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """
    Split each node's readings by TIME: the earliest (1 - test_fraction)
    portion goes to training, the latest test_fraction portion goes to
    testing. Done per-node so every node contributes to both sets.

    Because `df` is already sorted chronologically per node (guaranteed
    by preprocess.load_and_clean_csv), we can simply slice by position.
    """
    train_parts, test_parts = [], []
    for node_id, group in df.groupby("node_id"):
        n = len(group)
        split_point = int(np.floor(n * (1 - test_fraction)))
        train_parts.append(group.iloc[:split_point])
        test_parts.append(group.iloc[split_point:])

    train_df = pd.concat(train_parts).sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(["node_id", "timestamp"]).reset_index(drop=True)
    return train_df, test_df


def main():
    print("=" * 70)
    print("STAGE 1: Load and clean data")
    print("=" * 70)
    df = load_and_clean_csv(DATA_PATH)
    print(f"Loaded {len(df)} clean rows across {df['node_id'].nunique()} node(s).")
    print("\nClass distribution (full dataset):")
    print(df["status"].value_counts())

    print("\n" + "=" * 70)
    print("STAGE 2: Feature engineering")
    print("=" * 70)
    df = add_engineered_features(df)
    feature_cols = get_feature_columns()
    print(f"Feature columns used for training ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    print("\n" + "=" * 70)
    print("STAGE 3: Chronological train/test split")
    print("=" * 70)
    train_df, test_df = chronological_train_test_split(df)
    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print("\nTrain class distribution:")
    print(train_df["status"].value_counts())
    print("\nTest class distribution:")
    print(test_df["status"].value_counts())

    # Save the splits so evaluate.py can reuse the EXACT same test set
    Path(TRAIN_SPLIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    X_train = train_df[feature_cols]
    y_train = train_df["status"]
    X_test = test_df[feature_cols]
    y_test = test_df["status"]

    print("\n" + "=" * 70)
    print("STAGE 4: Train RandomForestClassifier")
    print("=" * 70)
    # Parameter explanations (beginner-friendly):
    #   n_estimators=300     -> number of decision trees in the forest.
    #                           More trees = generally more stable
    #                           predictions, at the cost of more compute.
    #   max_depth=10         -> maximum depth of each tree. Limits how
    #                           complex/specific each tree can get, which
    #                           helps prevent overfitting on a small
    #                           student-project dataset.
    #   min_samples_leaf=5   -> each leaf (final decision) must be
    #                           supported by at least 5 training examples.
    #                           Again, this discourages the model from
    #                           memorising single noisy readings.
    #   class_weight="balanced" -> automatically up-weights rarer classes
    #                           (e.g. "critical" is likely rarer than
    #                           "normal"). This matters a lot for safety:
    #                           without it, the model could get high
    #                           accuracy just by mostly predicting
    #                           "normal" and rarely flagging "critical".
    #   random_state=42      -> fixes the randomness so results are
    #                           reproducible when you re-run this script.
    #   n_jobs=-1            -> use all available CPU cores to train faster.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n" + "=" * 70)
    print("STAGE 5: Quick evaluation on the held-out test set")
    print("=" * 70)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"Test accuracy   : {acc:.3f}")
    print(f"Test macro F1   : {f1_macro:.3f}")
    print("\n(Run evaluate.py for a full breakdown: precision/recall per")
    print(" class, confusion matrix, and time-series cross-validation.)")

    print("\n" + "=" * 70)
    print("STAGE 6: Save the trained model with joblib")
    print("=" * 70)
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": feature_cols,   # exact column order used in training
        "label_order": LABEL_ORDER,        # fixed, meaningful class ordering
        "rolling_window": 3,
        "raw_sensor_columns": ["distance", "tilt_x", "tilt_y", "vibration",
                                "temperature", "humidity"],
        "sklearn_target_classes": list(model.classes_),  # what the model itself learned
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved trained model bundle to: {MODEL_PATH}")
    print("\nBundle contents: trained model, feature column order, label order,")
    print("rolling window size, and raw sensor column names - everything")
    print("needed to make consistent predictions later.")


if __name__ == "__main__":
    main()
