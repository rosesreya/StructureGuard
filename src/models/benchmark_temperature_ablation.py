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

CLASSES = [0, 1, 2]


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
# Cross-validation
# --------------------------------------------------

results = []

for fold_number in range(1, FOLD_COUNT + 1):

    print("\n" + "=" * 60)
    print(f"FOLD {fold_number}")
    print("=" * 60)

    fold_dir = CV_DIR / f"fold_{fold_number}"

    X_train = np.load(fold_dir / "X_train_rows.npy")
    X_val = np.load(fold_dir / "X_val_rows.npy")

    y_train = np.load(fold_dir / "y_train.npy")
    y_val = np.load(fold_dir / "y_val.npy")

    if X_train.shape[1] != 5 or X_val.shape[1] != 5:
        raise ValueError(
            f"Expected five input features, got "
            f"{X_train.shape[1]} and {X_val.shape[1]}"
        )

    experiments = {
        "All Five Features": list(range(5)),
        "Without Temperature": list(range(4))
    }

    for experiment_name, feature_indices in experiments.items():

        X_train_selected = X_train[:, feature_indices]
        X_val_selected = X_val[:, feature_indices]

        model = create_model()

        model.fit(X_train_selected, y_train)

        predictions = model.predict(X_val_selected)

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=CLASSES,
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
            labels=CLASSES,
            average=None,
            zero_division=0
        )

        qwk = cohen_kappa_score(
            y_val,
            predictions,
            labels=CLASSES,
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
            f"Balanced Accuracy={balanced_acc:.4f}, "
            f"Recall C1={recalls[1]:.4f}"
        )


# --------------------------------------------------
# Save fold-level results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "temperature_ablation_fold_results.csv",
    index=False
)


# --------------------------------------------------
# Aggregate results
# --------------------------------------------------

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
    OUTPUT_DIR / "temperature_ablation_summary.csv"
)

print("\n" + "=" * 60)
print("TEMPERATURE ABLATION RESULTS")
print("=" * 60)

print(summary.round(4).to_string())

print("\nSaved results to:", OUTPUT_DIR)