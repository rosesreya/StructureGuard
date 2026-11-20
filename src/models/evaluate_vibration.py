import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vibration"
    / "window_8"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vibration_lstm_best.keras"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "vibration"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load untouched test dataset
# --------------------------------------------------

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Test dataset shape:", X_test.shape)
print("Test labels shape:", y_test.shape)


# --------------------------------------------------
# 3. Load selected model
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("\nSelected vibration LSTM loaded successfully.")


# --------------------------------------------------
# 4. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# Generate predictions
y_prob = model.predict(X_test, verbose=0)

y_pred = np.argmax(y_prob, axis=1)


# --------------------------------------------------
# 5. Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nFINAL TEST RESULTS")
print("=" * 40)

print(f"Test Loss:       {test_loss:.4f}")
print(f"Test Accuracy:   {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall:    {recall:.4f}")
print(f"Macro F1-Score:  {f1:.4f}")


# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 8. Classification report
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Class 0", "Class 1", "Class 2"],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "Metric": [
        "Test Loss",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score"
    ],
    "Score": [
        test_loss,
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(
    RESULTS_DIR / "evaluation_metrics.csv",
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual Class 0", "Actual Class 1", "Actual Class 2"],
    columns=["Predicted Class 0", "Predicted Class 1", "Predicted Class 2"]
).to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)

with open(
    RESULTS_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)

print("\nEvaluation files saved successfully.")
print("Results directory:", RESULTS_DIR)