import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

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

OUTPUT_DIR = PROJECT_ROOT / "results" / "vibration" / "stage1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_COUNT = 4

# --------------------------------------------------
# Model definitions
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=3,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}


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

    for model_name, model in models.items():

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

        class_recalls = recall_score(
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

        result = {
            "Fold": fold_number,
            "Model": model_name,
            "Macro F1": macro_f1,
            "Balanced Accuracy": balanced_acc,
            "Recall Class 0": class_recalls[0],
            "Recall Class 1": class_recalls[1],
            "Recall Class 2": class_recalls[2],
            "Quadratic Weighted Kappa": qwk
        }

        results.append(result)

        print(
            f"{model_name}: "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Accuracy={balanced_acc:.4f}"
        )


# --------------------------------------------------
# Save fold-level results
# --------------------------------------------------

results_df = pd.DataFrame(results)

fold_results_path = OUTPUT_DIR / "classical_fold_results.csv"

results_df.to_csv(
    fold_results_path,
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
    .groupby("Model")[metrics]
    .agg(["mean", "std"])
)

summary_path = OUTPUT_DIR / "classical_cv_summary.csv"

summary.to_csv(summary_path)

print("\n" + "=" * 60)
print("CLASSICAL MODEL COMPARISON")
print("=" * 60)

print(summary.round(4).to_string())

print("\nSaved fold results:", fold_results_path)
print("Saved summary:", summary_path)