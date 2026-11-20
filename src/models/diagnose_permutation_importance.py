import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from sklearn.metrics import f1_score


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
N_REPEATS = 30

FEATURE_NAMES = [
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "Strain",
    "Temperature"
]


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
# Permutation importance
# --------------------------------------------------

fold_results = []

for fold_number in range(1, FOLD_COUNT + 1):

    print("\n" + "=" * 60)
    print(f"FOLD {fold_number}")
    print("=" * 60)

    fold_dir = CV_DIR / f"fold_{fold_number}"

    X_train = np.load(fold_dir / "X_train_rows.npy")
    y_train = np.load(fold_dir / "y_train.npy")

    X_val = np.load(fold_dir / "X_val_rows.npy")
    y_val = np.load(fold_dir / "y_val.npy")

    if X_train.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Unexpected number of input features.")

    model = create_model()
    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_val)

    baseline_f1 = f1_score(
        y_val,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    print(f"Baseline validation macro F1: {baseline_f1:.4f}")

    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    print("\nFeature importance:")

    for i, feature_name in enumerate(FEATURE_NAMES):

        mean_importance = importance.importances_mean[i]
        std_importance = importance.importances_std[i]

        print(
            f"{feature_name}: "
            f"{mean_importance:.5f} "
            f"+/- {std_importance:.5f}"
        )

        fold_results.append({
            "Fold": fold_number,
            "Feature": feature_name,
            "Importance Mean": mean_importance,
            "Importance Std": std_importance,
            "Baseline Macro F1": baseline_f1
        })


# --------------------------------------------------
# Save fold-level results
# --------------------------------------------------

results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    OUTPUT_DIR / "permutation_importance_fold_results.csv",
    index=False
)


# --------------------------------------------------
# Aggregate across folds
# --------------------------------------------------

summary = (
    results_df
    .groupby("Feature")
    .agg(
        Importance_Mean=("Importance Mean", "mean"),
        Importance_Std_Between_Folds=("Importance Mean", "std"),
        Mean_Repeat_Std=("Importance Std", "mean")
    )
    .sort_values("Importance_Mean", ascending=False)
)

summary.to_csv(
    OUTPUT_DIR / "permutation_importance_summary.csv"
)

print("\n" + "=" * 60)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 60)

print(summary.round(5).to_string())

print("\nSaved results to:", OUTPUT_DIR)