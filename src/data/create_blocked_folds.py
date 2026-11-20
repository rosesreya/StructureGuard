import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vibration"
    / "blocked_folds.csv"
)

N_FOLDS = 4
BLOCK_SIZE = 200
PURGE_GAP = 8

# --------------------------------------------------
# Load labels
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)
labels = df["Condition Label"].to_numpy()

n_rows = len(df)

# --------------------------------------------------
# Generate expanding-window folds
# --------------------------------------------------

folds = []

for fold in range(N_FOLDS):

    validation_start = (fold + 1) * BLOCK_SIZE
    validation_end = min(
        validation_start + BLOCK_SIZE,
        n_rows
    )

    # Exclude the purge gap immediately before validation.
    training_end = validation_start - PURGE_GAP

    train_indices = np.arange(0, training_end)
    val_indices = np.arange(validation_start, validation_end)

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(f"Invalid fold {fold + 1}")

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    print("\n" + "=" * 50)
    print(f"FOLD {fold + 1}")
    print("=" * 50)

    print(
        f"Training rows: {train_indices[0]} "
        f"to {train_indices[-1]}"
    )

    print(
        f"Validation rows: {val_indices[0]} "
        f"to {val_indices[-1]}"
    )

    print("Purged rows:", PURGE_GAP)

    print("\nTraining class distribution:")
    print(pd.Series(train_labels).value_counts().sort_index())

    print("\nValidation class distribution:")
    print(pd.Series(val_labels).value_counts().sort_index())

    # Verify that all three classes occur in both partitions.
    if set(np.unique(train_labels)) != {0, 1, 2}:
        raise ValueError("A training fold is missing a class.")

    if set(np.unique(val_labels)) != {0, 1, 2}:
        raise ValueError("A validation fold is missing a class.")

    folds.append({
        "fold": fold + 1,
        "train_start": int(train_indices[0]),
        "train_end": int(train_indices[-1]),
        "val_start": int(val_indices[0]),
        "val_end": int(val_indices[-1]),
        "purge_gap": PURGE_GAP
    })

# --------------------------------------------------
# Save fold definitions
# --------------------------------------------------

fold_df = pd.DataFrame(folds)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

fold_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 50)
print("FOLD GENERATION COMPLETED")
print("=" * 50)

print(fold_df.to_string(index=False))
print("\nSaved to:", OUTPUT_PATH)