import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    recall_score,
    cohen_kappa_score
)

from src.models.vibration_lstm import build_vibration_lstm


# --------------------------------------------------
# 1. Configuration
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
    / "stage1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_COUNT = 4
SEEDS = [42, 123, 2024, 7, 99]

WINDOW_SIZE = 8
N_FEATURES = 5
N_CLASSES = 3

EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --------------------------------------------------
# 3. Validation macro F1 callback
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

        if logs is not None:
            logs["val_macro_f1"] = macro_f1

        print(
            f"\nEpoch {epoch + 1}: "
            f"Validation Macro F1 = {macro_f1:.4f}"
        )


# --------------------------------------------------
# 4. Run cross-validation
# --------------------------------------------------

all_results = []

for fold_number in range(1, FOLD_COUNT + 1):

    fold_dir = CV_DIR / f"fold_{fold_number}"

    X_train = np.load(fold_dir / "X_train_seq.npy")
    y_train = np.load(fold_dir / "y_train.npy")

    X_val = np.load(fold_dir / "X_val_seq.npy")
    y_val = np.load(fold_dir / "y_val.npy")

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}")
    print("=" * 70)

    for seed in SEEDS:

        print("\n" + "-" * 60)
        print(f"FOLD {fold_number} | SEED {seed}")
        print("-" * 60)

        tf.keras.backend.clear_session()
        gc.collect()

        set_seed(seed)

        model = build_vibration_lstm(
            time_steps=WINDOW_SIZE,
            n_features=N_FEATURES,
            n_classes=N_CLASSES
        )

        macro_f1_callback = ValidationMacroF1(
            X_val,
            y_val
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                macro_f1_callback,
                early_stopping
            ],
            verbose=1
        )

        # --------------------------------------------------
        # Evaluate restored best weights
        # --------------------------------------------------

        probabilities = model.predict(
            X_val,
            verbose=0
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

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

        best_epoch = int(
            np.argmax(history.history["val_macro_f1"]) + 1
        )

        result = {
            "Fold": fold_number,
            "Seed": seed,
            "Window Size": WINDOW_SIZE,
            "Best Epoch": best_epoch,
            "Macro F1": macro_f1,
            "Balanced Accuracy": balanced_acc,
            "Recall Class 0": class_recalls[0],
            "Recall Class 1": class_recalls[1],
            "Recall Class 2": class_recalls[2],
            "Quadratic Weighted Kappa": qwk
        }

        all_results.append(result)

        print("\nRun result:")
        print(result)

        del model
        gc.collect()


# --------------------------------------------------
# 5. Save individual run results
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

fold_results_path = (
    OUTPUT_DIR / "lstm_fold_seed_results.csv"
)

results_df.to_csv(
    fold_results_path,
    index=False
)


# --------------------------------------------------
# 6. Aggregate by fold
# --------------------------------------------------

metrics = [
    "Macro F1",
    "Balanced Accuracy",
    "Recall Class 0",
    "Recall Class 1",
    "Recall Class 2",
    "Quadratic Weighted Kappa"
]

fold_summary = (
    results_df
    .groupby("Fold")[metrics]
    .mean()
)

fold_summary_path = (
    OUTPUT_DIR / "lstm_fold_summary.csv"
)

fold_summary.to_csv(fold_summary_path)


# --------------------------------------------------
# 7. Overall summary
# --------------------------------------------------

overall_summary = results_df[metrics].agg(
    ["mean", "std"]
)

overall_path = (
    OUTPUT_DIR / "lstm_overall_summary.csv"
)

overall_summary.to_csv(overall_path)

print("\n" + "=" * 70)
print("LSTM CROSS-VALIDATION SUMMARY")
print("=" * 70)

print("\nFold-wise mean performance:")
print(fold_summary.round(4).to_string())

print("\nOverall performance across all runs:")
print(overall_summary.round(4).to_string())

print("\nSaved run results:", fold_results_path)
print("Saved fold summary:", fold_summary_path)
print("Saved overall summary:", overall_path)

print("\nAll LSTM cross-validation runs completed.")