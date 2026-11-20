import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

FOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vibration"
    / "blocked_folds.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vibration"
    / "cv"
)

SENSOR_COLUMNS = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]

WINDOW_SIZE = 8


# --------------------------------------------------
# 2. Load dataset and fold definitions
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)
fold_df = pd.read_csv(FOLD_PATH)

if df["Condition Label"].isnull().any():
    raise ValueError("Target labels contain missing values.")

df["Condition Label"] = df["Condition Label"].astype(int)

print("Dataset shape:", df.shape)
print("Number of folds:", len(fold_df))


# --------------------------------------------------
# 3. Interpolation helper
# --------------------------------------------------

def interpolate_partition(partition):

    partition = partition.copy()

    partition[SENSOR_COLUMNS] = (
        partition[SENSOR_COLUMNS]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    if partition[SENSOR_COLUMNS].isnull().any().any():
        raise ValueError("Missing values remain after interpolation.")

    return partition


# --------------------------------------------------
# 4. Sequence creation helper
# --------------------------------------------------

def create_sequences(X, y, window_size):

    X_sequences = []
    y_sequences = []

    for i in range(len(X) - window_size + 1):

        X_sequences.append(
            X[i:i + window_size]
        )

        y_sequences.append(
            y[i + window_size - 1]
        )

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(y_sequences, dtype=np.int64)
    )


# --------------------------------------------------
# 5. Process each fold independently
# --------------------------------------------------

for _, fold in fold_df.iterrows():

    fold_number = int(fold["fold"])

    train_start = int(fold["train_start"])
    train_end = int(fold["train_end"])

    val_start = int(fold["val_start"])
    val_end = int(fold["val_end"])

    print("\n" + "=" * 60)
    print(f"PROCESSING FOLD {fold_number}")
    print("=" * 60)

    # Extract partitions using inclusive end indices.
    train_df = df.iloc[train_start:train_end + 1].copy()
    val_df = df.iloc[val_start:val_end + 1].copy()

    # --------------------------------------------------
    # Interpolate independently
    # --------------------------------------------------

    train_df = interpolate_partition(train_df)
    val_df = interpolate_partition(val_df)

    X_train = train_df[SENSOR_COLUMNS].to_numpy()
    y_train = train_df["Condition Label"].to_numpy()

    X_val = val_df[SENSOR_COLUMNS].to_numpy()
    y_val = val_df["Condition Label"].to_numpy()

    # --------------------------------------------------
    # Fit scaler on training data only
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --------------------------------------------------
    # Generate sequences independently
    # --------------------------------------------------

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train,
        WINDOW_SIZE
    )

    X_val_seq, y_val_seq = create_sequences(
        X_val_scaled,
        y_val,
        WINDOW_SIZE
    )

    # Original row indices corresponding to sequence targets.
    train_target_indices = np.arange(
        train_start + WINDOW_SIZE - 1,
        train_end + 1
    )

    val_target_indices = np.arange(
        val_start + WINDOW_SIZE - 1,
        val_end + 1
    )

    # Per-row features aligned to the same sequence targets.
    X_train_rows = X_train_scaled[WINDOW_SIZE - 1:]
    X_val_rows = X_val_scaled[WINDOW_SIZE - 1:]

    # --------------------------------------------------
    # Verify alignment
    # --------------------------------------------------

    assert len(X_train_seq) == len(y_train_seq)
    assert len(X_val_seq) == len(y_val_seq)

    assert len(X_train_rows) == len(y_train_seq)
    assert len(X_val_rows) == len(y_val_seq)

    assert len(val_target_indices) == len(y_val_seq)

    # --------------------------------------------------
    # Save fold-specific datasets
    # --------------------------------------------------

    fold_dir = OUTPUT_DIR / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    np.save(fold_dir / "X_train_seq.npy", X_train_seq)
    np.save(fold_dir / "y_train.npy", y_train_seq)

    np.save(fold_dir / "X_val_seq.npy", X_val_seq)
    np.save(fold_dir / "y_val.npy", y_val_seq)

    np.save(fold_dir / "X_train_rows.npy", X_train_rows)
    np.save(fold_dir / "X_val_rows.npy", X_val_rows)

    np.save(
        fold_dir / "train_target_indices.npy",
        train_target_indices
    )

    np.save(
        fold_dir / "val_target_indices.npy",
        val_target_indices
    )

    joblib.dump(
        scaler,
        fold_dir / "scaler.pkl"
    )

    # --------------------------------------------------
    # Report results
    # --------------------------------------------------

    print("\nTraining sequences:", X_train_seq.shape)
    print("Validation sequences:", X_val_seq.shape)

    print("Training aligned rows:", X_train_rows.shape)
    print("Validation aligned rows:", X_val_rows.shape)

    print("\nTraining class distribution:")
    print(pd.Series(y_train_seq).value_counts().sort_index())

    print("\nValidation class distribution:")
    print(pd.Series(y_val_seq).value_counts().sort_index())

    print("\nValidation target row range:")
    print(
        val_target_indices[0],
        "to",
        val_target_indices[-1]
    )

    print("Saved to:", fold_dir)


print("\n" + "=" * 60)
print("BLOCKED CV PREPROCESSING COMPLETED")
print("=" * 60)