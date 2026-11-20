import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "vibration" / "window_8"
)

MODEL_PATH = (
    PROJECT_ROOT / "models"
    / "vibration_lstm_best.keras"
)

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

model = tf.keras.models.load_model(MODEL_PATH)

def evaluate(name, X):

    probabilities = model.predict(X, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    print(
        "Macro F1:",
        round(f1_score(
            y_val,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ), 4)
    )


# Normal sequence
evaluate("Normal LSTM", X_val)

# Experiment 1: Shuffle temporal order
rng = np.random.default_rng(42)

X_shuffled = X_val.copy()

for i in range(len(X_shuffled)):
    order = rng.permutation(X_shuffled.shape[1])
    X_shuffled[i] = X_shuffled[i, order, :]

evaluate("Shuffled Time Order", X_shuffled)

# Experiment 2: Keep only final observation
X_last = np.repeat(
    X_val[:, -1:, :],
    repeats=X_val.shape[1],
    axis=1
)

evaluate("Last Observation Repeated", X_last)

print("\nD7 ABLATION DIAGNOSTICS COMPLETED")