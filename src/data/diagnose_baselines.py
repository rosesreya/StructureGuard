import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT / "data" / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

FEATURES = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]

TARGET = "Condition Label"

# Load data
df = pd.read_csv(CSV_PATH)

# Chronological 70/15/15 split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

# Models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=3,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# Train and evaluate
for name, model in models.items():

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print(
        "Validation Accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    print(
        "Validation Macro F1:",
        round(f1_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ), 4)
    )

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1, 2]
    ))

print("\nD6 PER-ROW BASELINE DIAGNOSTICS COMPLETED")