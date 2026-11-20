import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

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

CLASSES = np.array([0, 1, 2])


# --------------------------------------------------
# Class-weight configurations
# --------------------------------------------------

def get_class_weights(y_train):

    counts = np.bincount(y_train, minlength=3)

    total = len(y_train)
    number_of_classes = len(CLASSES)

    sqrt_weights = np.sqrt(
        total / (number_of_classes * counts)
    )

    return {
        "No Weighting": None,

        "Square Root Weighting": {
            int(cls): float(weight)
            for cls, weight in zip(CLASSES, sqrt_weights)
        },

        "Full Balanced Weighting": "balanced"
    }


# --------------------------------------------------
# Model
# --------------------------------------------------

def create_model(class_weight):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
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
    y_train = np.load(fold_dir / "y_train.npy")

    X_val = np.load(fold_dir / "X_val_rows.npy")
    y_val = np.load(fold_dir / "y_val.npy")

    weight_configs = get_class_weights(y_train)

    print("Training class counts:",
          np.bincount(y_train, minlength=3))

    for config_name, class_weight in weight_configs.items():

        model = create_model(class_weight)

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

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
            "Weighting": config_name,
            "Macro F1": macro_f1,
            "Balanced Accuracy": balanced_acc,
            "Recall Class 0": recalls[0],
            "Recall Class 1": recalls[1],
            "Recall Class 2": recalls[2],
            "Quadratic Weighted Kappa": qwk
        })

        print(
            f"{config_name}: "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Accuracy={balanced_acc:.4f}, "
            f"Recall C1={recalls[1]:.4f}"
        )


# --------------------------------------------------
# Save fold-level results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "class_weight_fold_results.csv",
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
    .groupby("Weighting")[metrics]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUTPUT_DIR / "class_weight_summary.csv"
)

print("\n" + "=" * 60)
print("CLASS-WEIGHTING COMPARISON")
print("=" * 60)

print(summary.round(4).to_string())

print("\nSaved results to:", OUTPUT_DIR)