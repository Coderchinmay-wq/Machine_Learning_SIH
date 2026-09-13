"""
realtime_analyzer.py
=====================
This module simulates the "brain" that would sit on your backend,
continuously receiving JSON sensor readings (eventually arriving via:
ESP32 sensor node -> LoRa -> Gateway ESP32 -> Internet -> Backend) and
producing a live prediction for each one.

WHY THIS IS SEPARATE FROM predict.py
--------------------------------------
predict.py handles ONE isolated reading with no memory of the past.
But your engineered features (delta_*, rollmean_*) are much more
meaningful when we actually know the last few real readings from that
same node. RealtimeAnalyzer keeps a small in-memory history buffer PER
NODE so that each new reading gets proper delta/rolling features
computed from genuine recent history - not a "first reading" fallback.

In this student prototype the history buffer lives in memory (a Python
dictionary) and is lost if the process restarts. In a real deployment,
recent history would instead be read from the backend's database.
"""

from collections import deque
import pandas as pd

from preprocess import build_features_for_new_reading, RAW_SENSOR_COLUMNS, ROLLING_WINDOW
from predict import load_model_bundle, predict_single_reading


class RealtimeAnalyzer:
    """
    Keeps a short rolling history of recent readings PER node_id and
    turns each new incoming JSON reading into a live prediction.
    """

    def __init__(self, model_path: str = "models/mine_subsidence_model.pkl",
                 history_length: int = 10):
        self.bundle = load_model_bundle(model_path)
        # history_length: how many recent raw readings to remember per
        # node. Only the last (ROLLING_WINDOW - 1) are actually needed
        # for feature engineering, but we keep a bit more for context /
        # future extensions (e.g. plotting trends on a dashboard).
        self.history_length = max(history_length, ROLLING_WINDOW)
        self._node_history: dict[str, deque] = {}

    def _get_history_df(self, node_id: str) -> pd.DataFrame | None:
        if node_id not in self._node_history or len(self._node_history[node_id]) == 0:
            return None
        return pd.DataFrame(list(self._node_history[node_id]))

    def _update_history(self, node_id: str, reading: dict) -> None:
        if node_id not in self._node_history:
            self._node_history[node_id] = deque(maxlen=self.history_length)
        raw_only = {c: reading[c] for c in RAW_SENSOR_COLUMNS}
        self._node_history[node_id].append(raw_only)

    def process_reading(self, reading: dict) -> dict:
        """
        Process ONE incoming sensor reading JSON (already parsed into a
        Python dict) and return a prediction result.

        Expected keys in `reading`:
            node_id, timestamp (optional here, used only for logging),
            distance, tilt_x, tilt_y, vibration, temperature, humidity
        """
        node_id = reading.get("node_id", "UNKNOWN_NODE")

        required = RAW_SENSOR_COLUMNS
        missing = [c for c in required if c not in reading]
        if missing:
            raise ValueError(f"Reading is missing required sensor fields: {missing}")

        history_df = self._get_history_df(node_id)
        result = predict_single_reading(reading, history_df=history_df, bundle=self.bundle)

        # Only AFTER computing the prediction do we add this reading to
        # history, so it becomes "previous" context for the NEXT reading
        # from this node (never leaking into its own prediction).
        self._update_history(node_id, reading)

        result["node_id"] = node_id
        result["timestamp"] = reading.get("timestamp")
        return result


if __name__ == "__main__":
    # Small demo: simulate a stream of readings for one node and watch
    # the analyzer's predictions update as history builds up.
    analyzer = RealtimeAnalyzer()

    demo_stream = [
        {"node_id": "NODE_01", "timestamp": "2026-09-08T10:00:00",
         "distance": 88.0, "tilt_x": 1.0, "tilt_y": 0.8, "vibration": 0.10,
         "temperature": 26.0, "humidity": 70},
        {"node_id": "NODE_01", "timestamp": "2026-09-08T10:10:00",
         "distance": 86.5, "tilt_x": 1.6, "tilt_y": 1.2, "vibration": 0.18,
         "temperature": 26.2, "humidity": 71},
        {"node_id": "NODE_01", "timestamp": "2026-09-08T10:20:00",
         "distance": 80.1, "tilt_x": 4.0, "tilt_y": 2.8, "vibration": 0.48,
         "temperature": 27.4, "humidity": 73},
    ]

    for reading in demo_stream:
        result = analyzer.process_reading(reading)
        print(f"[{result['timestamp']}] node={result['node_id']} "
              f"-> {result['prediction'].upper()}  "
              f"(probabilities: {result['probabilities']})")
