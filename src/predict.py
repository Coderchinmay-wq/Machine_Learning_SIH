"""
predict.py
==========
A standalone script that loads the SAVED, TRAINED model and makes a
prediction for ONE new sensor reading you type in below (or pass in).

This is the simplest possible way to test "does my saved model work?"
before wiring it into the FastAPI backend (api.py) or the real-time
analyzer (realtime_analyzer.py).

Run it directly:
    python predict.py

It will print:
    - the predicted class (NORMAL / WARNING / CRITICAL)
    - the model's probability for EACH class (not just the winner)
    - the highest probability value
    - a plain-English explanation of what that means
"""

import joblib
import pandas as pd

from preprocess import build_features_for_new_reading

MODEL_PATH = "models/mine_subsidence_model.pkl"


def load_model_bundle(model_path: str = MODEL_PATH) -> dict:
    """Load the joblib bundle saved by train.py."""
    return joblib.load(model_path)


def predict_single_reading(new_reading: dict, history_df: pd.DataFrame | None = None,
                            bundle: dict | None = None) -> dict:
    """
    Predict the class for one new sensor reading.

    Parameters
    ----------
    new_reading : dict
        Raw sensor values, e.g.
        {"node_id": "NODE_01", "distance": 80.1, "tilt_x": 4.0,
         "tilt_y": 2.8, "vibration": 0.48, "temperature": 27.4,
         "humidity": 73}
    history_df : optional DataFrame of recent PAST readings for the same
        node (used to compute real delta_/rollmean_ features). If not
        given, this reading is treated as the node's first known reading.
    bundle : optional pre-loaded model bundle (to avoid reloading from
        disk on every call, e.g. inside the FastAPI server).

    Returns
    -------
    dict with keys: prediction, probabilities (per class), confidence
    """
    if bundle is None:
        bundle = load_model_bundle()

    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    X_new = build_features_for_new_reading(new_reading, history_df=history_df)
    X_new = X_new[feature_cols]  # enforce exact training column order

    # model.classes_ gives the class order the model itself uses internally
    proba = model.predict_proba(X_new)[0]
    class_names = model.classes_

    probabilities = {cls: float(p) for cls, p in zip(class_names, proba)}
    prediction = model.predict(X_new)[0]
    confidence = float(max(proba))  # NOT called "certainty" - see explanation below

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "highest_probability": confidence,
    }


def explain_result(result: dict) -> str:
    pred = result["prediction"].upper()
    conf_pct = result["highest_probability"] * 100
    probs_str = ", ".join(f"{k}={v:.1%}" for k, v in result["probabilities"].items())

    explanation = (
        f"\nPredicted condition: {pred}\n"
        f"Model probabilities: {probs_str}\n"
        f"Highest probability: {conf_pct:.1f}%\n\n"
        "WHAT THIS MEANS: The 'probability' is the trained Random Forest's "
        "internal estimate of how likely each class is, based on the "
        "patterns it learned from historical data. It is NOT the same as "
        "'certainty' or 'proof' - it reflects how closely this reading "
        "resembles patterns seen during training. A high probability on a "
        "model trained on limited or synthetic data still does not "
        "guarantee real-world correctness."
    )
    return explanation


if __name__ == "__main__":
    # Example new reading - EDIT these values to test different scenarios.
    example_reading = {
        "node_id": "NODE_01",
        "distance": 80.1,
        "tilt_x": 4.0,
        "tilt_y": 2.8,
        "vibration": 0.48,
        "temperature": 27.4,
        "humidity": 73,
    }

    print("Input reading:")
    for k, v in example_reading.items():
        print(f"  {k}: {v}")

    result = predict_single_reading(example_reading)
    print(explain_result(result))
