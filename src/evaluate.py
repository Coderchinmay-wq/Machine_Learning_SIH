"""
evaluate.py
===========
Run this AFTER train.py. It loads the saved model and the held-out
test set (saved by train.py) and produces a full evaluation report:

  - accuracy
  - precision, recall, F1-score (per class AND averaged)
  - confusion matrix
  - a specific check on how many CRITICAL cases were missed
  - a time-series-aware cross-validation demonstration

WHY WE LOOK AT MORE THAN JUST ACCURACY
----------------------------------------
Imagine a dataset where 90% of readings are "normal", 7% are "warning",
and only 3% are "critical" (this is realistic - dangerous conditions
should be rare). A lazy model that ALWAYS predicts "normal" would score
90% accuracy while being completely useless (and dangerous) as an early
warning system, because it would miss every single critical event.

That is why, for a safety system, RECALL on the "critical" class
(i.e. "of all the times it was truly critical, how many did we catch?")
matters far more than overall accuracy. A missed critical event
(false negative) is much more dangerous than a false alarm (false
positive) that sends someone to check a mine that turns out to be fine.
"""

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/mine_subsidence_model.pkl"
TEST_SPLIT_PATH = "data/test_split.csv"
TRAIN_SPLIT_PATH = "data/train_split.csv"


def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "            " + "".join(f"{l:>10}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:>10}  " + "".join(f"{v:>10}" for v in cm[i])
        print(row_str)
    return cm


def main():
    print("=" * 70)
    print("Loading saved model and held-out test set")
    print("=" * 70)
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    label_order = bundle["label_order"]

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    X_test = test_df[feature_cols]
    y_test = test_df["status"]

    y_pred = model.predict(X_test)

    print("\n" + "=" * 70)
    print("STAGE 5: Full evaluation on the test set")
    print("=" * 70)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy: {acc:.3f}")
    print("  (Fraction of test readings whose class was predicted correctly.")
    print("   On its own this can be MISLEADING for imbalanced safety data -")
    print("   see the per-class breakdown below.)")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=label_order, zero_division=0
    )
    print("\nPer-class metrics:")
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    for lbl, p, r, f, s in zip(label_order, precision, recall, f1, support):
        print(f"{lbl:>10} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>10}")

    print("\nMetric definitions (plain English):")
    print("  precision = of the readings we PREDICTED as this class, what")
    print("              fraction actually were that class? (low precision")
    print("              on 'critical' = too many false alarms)")
    print("  recall    = of the readings that WERE actually this class, what")
    print("              fraction did we correctly catch? (low recall on")
    print("              'critical' = missed dangerous events - the most")
    print("              serious failure mode for this system)")
    print("  f1-score  = a single number balancing precision and recall")
    print("  support   = how many true examples of this class were in the")
    print("              test set (small support = less reliable estimate)")

    print("\n" + "=" * 70)
    print("Confusion matrix (rows = true label, columns = predicted label)")
    print("=" * 70)
    print_confusion_matrix(y_test, y_pred, label_order)

    # Explicit safety-focused check
    print("\n" + "=" * 70)
    print("SAFETY-FOCUSED CHECK: missed CRITICAL cases")
    print("=" * 70)
    critical_mask = (y_test == "critical")
    n_critical_total = critical_mask.sum()
    n_critical_missed = ((y_test == "critical") & (y_pred != "critical")).sum()
    if n_critical_total > 0:
        miss_rate = n_critical_missed / n_critical_total
        print(f"True CRITICAL readings in test set : {n_critical_total}")
        print(f"Of those, MISSED (predicted as something else) : {n_critical_missed} "
              f"({miss_rate:.1%})")
        if miss_rate > 0:
            print("WARNING: The model is currently missing some CRITICAL events.")
            print("This must be reduced before any real-world use is even")
            print("considered - see the README for real-deployment requirements.")
    else:
        print("No CRITICAL examples were present in this test set, so recall")
        print("on CRITICAL could not be evaluated here. This itself is a")
        print("limitation - a real evaluation needs enough true CRITICAL")
        print("examples to trust the recall estimate.")

    print("\n" + "=" * 70)
    print("STAGE 6: Time-series-aware cross-validation")
    print("=" * 70)
    print("""
WHY NOT ORDINARY (RANDOM) K-FOLD CROSS-VALIDATION?
Ordinary KFold cross-validation shuffles rows randomly into K groups.
For time-series sensor data, this again mixes readings from many
different points in time into both the 'training' and 'validation'
folds within each split, so near-identical neighbouring readings can
end up on both sides. That inflates the cross-validation score in the
same way a random train/test split does, for the same underlying
reason (temporal leakage).

sklearn's TimeSeriesSplit instead creates folds where the validation
set always comes AFTER the training set in time, e.g.:
  fold 1: train on readings [0:100],  validate on [100:150]
  fold 2: train on readings [0:150],  validate on [150:200]
  fold 3: train on readings [0:200],  validate on [200:250]
This mimics how the model will actually be used - always predicting
readings that occur after the data it was trained on.

NOTE: This demonstration runs TimeSeriesSplit on the combined training
data sorted by time (across all nodes together) purely to illustrate
the technique on a small student dataset. For a larger, production
dataset, you would typically run this validation separately PER NODE,
since different nodes are physically independent time series.
""")

    train_df = pd.read_csv(TRAIN_SPLIT_PATH).sort_values("timestamp").reset_index(drop=True)
    X_full_train = train_df[feature_cols]
    y_full_train = train_df["status"]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_accuracies = []
    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_full_train), start=1):
        X_tr, X_val = X_full_train.iloc[train_idx], X_full_train.iloc[val_idx]
        y_tr, y_val = y_full_train.iloc[train_idx], y_full_train.iloc[val_idx]

        # Fresh model per fold, same hyperparameters as train.py
        fold_model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        fold_model.fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_val)
        fold_acc = accuracy_score(y_val, fold_pred)
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold_i}: train_size={len(train_idx):4d}  val_size={len(val_idx):4d}  "
              f"accuracy={fold_acc:.3f}")

    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\nAverage TimeSeriesSplit validation accuracy: {avg_acc:.3f}")
    print("A large spread between folds suggests the model's performance is")
    print("unstable / sensitive to which time period it sees - worth")
    print("investigating further before trusting it.")

    print("\n" + "=" * 70)
    print("IMPORTANT SAFETY / SCIENTIFIC DISCLAIMER")
    print("=" * 70)
    print("""
High accuracy or good recall on this dataset (real OR synthetic) does
NOT mean this model can safely predict a real mine collapse. It only
means the model has learned to classify patterns similar to what it
was trained on. Real-world deployment would additionally require:
  - substantially more real, labeled mine sensor data
  - labels defined and checked by mining/geotechnical experts
  - field validation against real, verified subsidence events
  - sensor calibration and reliability/failure testing
  - independent third-party validation
  - integration with proper mine safety engineering procedures
This project is a prototype for detecting/classifying sensor patterns,
not a certified life-safety system.
""")


if __name__ == "__main__":
    main()
