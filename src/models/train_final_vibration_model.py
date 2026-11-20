import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vibration"
    / "final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]

LABEL_COLUMN = "Condition Label"

WINDOW_SIZE = 8
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

CLASSES = [0, 1, 2]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

required_columns = FEATURES + [LABEL_COLUMN]

if not all(col in df.columns for col in required_columns):
    raise ValueError("One or more required columns are missing.")

if df[LABEL_COLUMN].isna().any():
    raise ValueError("Target labels contain missing values.")

# Preserve chronological row order.
split_index = int(len(df) * TRAIN_RATIO)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))


# --------------------------------------------------
# Create window-based inputs
# Use the final observation of each window.
# --------------------------------------------------

def create_last_observation_windows(data, window_size):

    values = data[FEATURES].to_numpy(dtype=float)

    windows = np.array([
        values[i:i + window_size]
        for i in range(len(values) - window_size + 1)
    ])

    return windows[:, -1, :]


X_train = create_last_observation_windows(
    train_df, WINDOW_SIZE
)

X_test = create_last_observation_windows(
    test_df, WINDOW_SIZE
)

y_train = train_df[LABEL_COLUMN].to_numpy()[
    WINDOW_SIZE - 1:
]

y_test = test_df[LABEL_COLUMN].to_numpy()[
    WINDOW_SIZE - 1:
]

print("\nTraining input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# --------------------------------------------------
# Create final Random Forest model
# --------------------------------------------------

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    ))
])


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining final Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# Evaluate on test data
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y_test,
    predictions
)

qwk = cohen_kappa_score(
    y_test,
    predictions,
    labels=CLASSES,
    weights="quadratic"
)

print("\n" + "=" * 60)
print("FINAL VIBRATION MODEL TEST RESULTS")
print("=" * 60)

print(f"Accuracy:             {accuracy:.4f}")
print(f"Macro F1:             {macro_f1:.4f}")
print(f"Balanced Accuracy:    {balanced_acc:.4f}")
print(f"Quadratic Weighted Kappa: {qwk:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=CLASSES,
    zero_division=0
))

print("Confusion Matrix:")

print(confusion_matrix(
    y_test,
    predictions,
    labels=CLASSES
))


# --------------------------------------------------
# Save model and predictions
# --------------------------------------------------

model_path = OUTPUT_DIR / "final_vibration_rf.joblib"

joblib.dump(model, model_path)

prediction_results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

prediction_results.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

metrics_df = pd.DataFrame([{
    "Accuracy": accuracy,
    "Macro F1": macro_f1,
    "Balanced Accuracy": balanced_acc,
    "Quadratic Weighted Kappa": qwk
}])

metrics_df.to_csv(
    OUTPUT_DIR / "test_metrics.csv",
    index=False
)

print("\nSaved final model:", model_path)
print("Saved predictions and metrics:", OUTPUT_DIR)