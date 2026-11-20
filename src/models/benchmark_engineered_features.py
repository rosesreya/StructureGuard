import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    cohen_kappa_score
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CV_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vibration"
    / "cv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vibration"
    / "stage2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_COUNT = 4
RANDOM_SEED = 42


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_window_features(X):
    """
    X shape: (samples, time_steps, sensors)

    Returns:
        (samples, sensors * 8)
    """

    features = []

    for window in X:

        window_features = []

        for sensor in range(window.shape[1]):

            signal = window[:, sensor]

            mean = np.mean(signal)
            std = np.std(signal)
            minimum = np.min(signal)
            maximum = np.max(signal)
            signal_range = maximum - minimum
            rms = np.sqrt(np.mean(signal ** 2))
            slope = np.polyfit(
                np.arange(len(signal)),
                signal,
                1
            )[0]
            last_value = signal[-1]

            window_features.extend([
                mean,
                std,
                minimum,
                maximum,
                signal_range,
                rms,
                slope,
                last_value
            ])

        features.append(window_features)

    return np.asarray(features, dtype=np.float32)


# --------------------------------------------------
# Model
# --------------------------------------------------

def create_model():

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

results = []

for fold_number in range(1, FOLD_COUNT + 1):

    print("\n" + "=" * 60)
    print(f"FOLD {fold_number}")
    print("=" * 60)

    fold_dir = CV_DIR / f"fold_{fold_number}"

    X_train_seq = np.load(fold_dir / "X_train_seq.npy")
    X_val_seq = np.load(fold_dir / "X_val_seq.npy")

    y_train = np.load(fold_dir / "y_train.npy")
    y_val = np.load(fold_dir / "y_val.npy")

    # Baseline: final timestep of each window
    X_train_baseline = X_train_seq[:, -1, :]
    X_val_baseline = X_val_seq[:, -1, :]

    # Engineered statistical features
    X_train_engineered = extract_window_features(X_train_seq)
    X_val_engineered = extract_window_features(X_val_seq)

    experiments = {
        "Baseline - Last Timestep": (
            X_train_baseline,
            X_val_baseline
        ),

        "Engineered - Window Statistics": (
            X_train_engineered,
            X_val_engineered
        )
    }

    for experiment_name, (X_train, X_val) in experiments.items():

        model = create_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(
            y_val,
            predictions
        )

        recalls = recall_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0
        )

        qwk = cohen_kappa_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            weights="quadratic"
        )

        results.append({
            "Fold": fold_number,
            "Experiment": experiment_name,
            "Macro F1": macro_f1,
            "Balanced Accuracy": balanced_acc,
            "Recall Class 0": recalls[0],
            "Recall Class 1": recalls[1],
            "Recall Class 2": recalls[2],
            "Quadratic Weighted Kappa": qwk
        })

        print(
            f"{experiment_name}: "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Accuracy={balanced_acc:.4f}"
        )


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "engineered_feature_fold_results.csv",
    index=False
)

metrics = [
    "Macro F1",
    "Balanced Accuracy",
    "Recall Class 0",
    "Recall Class 1",
    "Recall Class 2",
    "Quadratic Weighted Kappa"
]

summary = (
    results_df
    .groupby("Experiment")[metrics]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUTPUT_DIR / "engineered_feature_summary.csv"
)

print("\n" + "=" * 60)
print("STAGE 2: ENGINEERED FEATURE COMPARISON")
print("=" * 60)

print(summary.round(4).to_string())

print("\nResults saved to:", OUTPUT_DIR)