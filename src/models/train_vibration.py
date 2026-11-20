import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from src.models.vibration_lstm import build_vibration_lstm


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "vibration"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZES = [8, 16, 32]

EPOCHS = 50
BATCH_SIZE = 32
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# 3. Custom callback for validation macro F1
# --------------------------------------------------

class ValidationMacroF1(tf.keras.callbacks.Callback):

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):

        predictions = self.model.predict(
            self.X_val,
            verbose=0
        )

        y_pred = np.argmax(predictions, axis=1)

        macro_f1 = f1_score(
            self.y_val,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )

        logs = logs or {}
        logs["val_macro_f1"] = macro_f1

        print(
            f"\nEpoch {epoch + 1}: "
            f"Validation Macro F1 = {macro_f1:.4f}"
        )


# --------------------------------------------------
# 4. Run experiments for each window size
# --------------------------------------------------

experiment_results = []

for time_steps in WINDOW_SIZES:

    print("\n" + "=" * 60)
    print(f"TRAINING LSTM WITH WINDOW SIZE: {time_steps}")
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Dataset directory for this window size
    window_dir = DATA_DIR / f"window_{time_steps}"

    # Model output paths
    MODEL_PATH = (
        MODEL_DIR / f"vibration_lstm_window_{time_steps}.keras"
    )

    HISTORY_PATH = (
        MODEL_DIR / f"vibration_history_window_{time_steps}.csv"
    )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    X_train = np.load(window_dir / "X_train.npy")
    y_train = np.load(window_dir / "y_train.npy")

    X_val = np.load(window_dir / "X_val.npy")
    y_val = np.load(window_dir / "y_val.npy")

    print("\nDataset shapes:")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)

    # --------------------------------------------------
    # Check class distribution
    # --------------------------------------------------

    print("\nTraining class distribution:")

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    # --------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    print("\nClass weights:")
    print(class_weights)

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = build_vibration_lstm(
        time_steps=time_steps,
        n_features=5,
        n_classes=3
    )

    model.summary()

    # --------------------------------------------------
    # Define callbacks
    # --------------------------------------------------

    macro_f1_callback = ValidationMacroF1(
        X_val,
        y_val
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            macro_f1_callback,
            early_stopping,
            model_checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    # --------------------------------------------------
    # Record experiment results
    # --------------------------------------------------

    best_epoch = int(
        np.argmax(history.history["val_macro_f1"])
    )

    best_macro_f1 = float(
        history.history["val_macro_f1"][best_epoch]
    )

    best_val_loss = float(
        history.history["val_loss"][best_epoch]
    )

    result = {
        "Window Size": time_steps,
        "Best Epoch": best_epoch + 1,
        "Best Validation Macro F1": best_macro_f1,
        "Validation Loss": best_val_loss,
        "Model Path": str(MODEL_PATH)
    }

    experiment_results.append(result)

    print("\nExperiment completed.")
    print(result)


# --------------------------------------------------
# 5. Compare experiments
# --------------------------------------------------

results_df = pd.DataFrame(experiment_results)

results_df = results_df.sort_values(
    by="Best Validation Macro F1",
    ascending=False
)

comparison_path = MODEL_DIR / "vibration_experiment_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)

print("\n" + "=" * 60)
print("EXPERIMENT COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# --------------------------------------------------
# 6. Select best configuration
# --------------------------------------------------

best_result = results_df.iloc[0]

best_window = int(best_result["Window Size"])

best_model_source = (
    MODEL_DIR / f"vibration_lstm_window_{best_window}.keras"
)

best_model_destination = (
    MODEL_DIR / "vibration_lstm_best.keras"
)

import shutil

shutil.copy2(
    best_model_source,
    best_model_destination
)

print("\nSelected configuration:")
print("Window size:", best_window)
print(
    "Validation Macro F1:",
    best_result["Best Validation Macro F1"]
)

print("\nBest model saved at:", best_model_destination)
print("Comparison saved at:", comparison_path)