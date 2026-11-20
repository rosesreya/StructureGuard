import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

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
    / "stage0"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_COUNT = 4
WINDOW_SIZE = 8


# --------------------------------------------------
# Load original labels
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

labels = df["Condition Label"].to_numpy(dtype=int)

# A transition at index i means:
# labels[i] != labels[i - 1]

transition_indices = np.where(
    labels[1:] != labels[:-1]
)[0] + 1

print("Total label transitions:", len(transition_indices))


# --------------------------------------------------
# Model
# --------------------------------------------------

def create_model():

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])


# --------------------------------------------------
# D8 analysis
# --------------------------------------------------

all_results = []

for fold_number in range(1, FOLD_COUNT + 1):

    print("\n" + "=" * 60)
    print(f"FOLD {fold_number}")
    print("=" * 60)

    fold_dir = CV_DIR / f"fold_{fold_number}"

    X_train = np.load(fold_dir / "X_train_rows.npy")
    y_train = np.load(fold_dir / "y_train.npy")

    X_val = np.load(fold_dir / "X_val_rows.npy")
    y_val = np.load(fold_dir / "y_val.npy")

    target_indices = np.load(
        fold_dir / "val_target_indices.npy"
    )

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print(
        "Validation accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    for i in range(len(y_val)):

        target_index = int(target_indices[i])

        window_start = target_index - WINDOW_SIZE + 1

        # Transitions occurring inside this window.
        window_transitions = transition_indices[
            (transition_indices > window_start)
            & (transition_indices <= target_index)
        ]

        straddles_transition = len(window_transitions) > 0

        # Distance from target row to nearest transition.
        distance = int(
            np.min(np.abs(transition_indices - target_index))
        ) if len(transition_indices) else -1

        all_results.append({
            "Fold": fold_number,
            "Target Index": target_index,
            "True Label": int(y_val[i]),
            "Predicted Label": int(predictions[i]),
            "Error": int(y_val[i] != predictions[i]),
            "Distance to Nearest Transition": distance,
            "Straddles Transition": straddles_transition,
            "Number of Transitions in Window": len(window_transitions)
        })


# --------------------------------------------------
# Save row-level results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "d8_transition_error_rows.csv",
    index=False
)


# --------------------------------------------------
# Error rates by distance
# --------------------------------------------------

results_df["Distance Group"] = results_df[
    "Distance to Nearest Transition"
].apply(
    lambda distance: str(distance)
    if distance <= 4
    else "5+"
)

distance_summary = (
    results_df
    .groupby("Distance Group")
    .agg(
        Samples=("Error", "count"),
        Errors=("Error", "sum"),
        Error_Rate=("Error", "mean")
    )
    .reset_index()
)

print("\nERROR RATE BY DISTANCE TO NEAREST TRANSITION")
print(distance_summary.to_string(index=False))

distance_summary.to_csv(
    OUTPUT_DIR / "d8_distance_summary.csv",
    index=False
)


# --------------------------------------------------
# Error rates for transition-straddling windows
# --------------------------------------------------

straddle_summary = (
    results_df
    .groupby("Straddles Transition")
    .agg(
        Samples=("Error", "count"),
        Errors=("Error", "sum"),
        Error_Rate=("Error", "mean")
    )
    .reset_index()
)

print("\nERROR RATE BY WINDOW TRANSITION STATUS")
print(straddle_summary.to_string(index=False))

straddle_summary.to_csv(
    OUTPUT_DIR / "d8_straddle_summary.csv",
    index=False
)


# --------------------------------------------------
# Transition count per window
# --------------------------------------------------

count_summary = (
    results_df
    .groupby("Number of Transitions in Window")
    .agg(
        Samples=("Error", "count"),
        Errors=("Error", "sum"),
        Error_Rate=("Error", "mean")
    )
    .reset_index()
)

print("\nERROR RATE BY NUMBER OF TRANSITIONS IN WINDOW")
print(count_summary.to_string(index=False))

count_summary.to_csv(
    OUTPUT_DIR / "d8_transition_count_summary.csv",
    index=False
)


# --------------------------------------------------
# Overall summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("D8 COMPLETED")
print("=" * 60)

print("Validation windows:", len(results_df))
print("Total errors:", results_df["Error"].sum())
print(
    "Overall error rate:",
    round(results_df["Error"].mean(), 4)
)

print(
    "Windows straddling transitions:",
    results_df["Straddles Transition"].sum()
)

print(
    "Fraction straddling transitions:",
    round(results_df["Straddles Transition"].mean(), 4)
)

print("\nResults saved to:", OUTPUT_DIR)