import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "vibration"

SENSOR_COLUMNS = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]

WINDOW_SIZES = [8, 16, 32]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

print("\nMissing sensor values before preprocessing:")
print(df[SENSOR_COLUMNS].isnull().sum())

print("\nOriginal class distribution:")
print(df["Condition Label"].value_counts().sort_index())


# --------------------------------------------------
# 3. Validate labels
# --------------------------------------------------

if df["Condition Label"].isnull().any():
    raise ValueError("Target labels contain missing values.")

df["Condition Label"] = df["Condition Label"].astype(int)

expected_classes = {0, 1, 2}
actual_classes = set(df["Condition Label"].unique())

if not actual_classes.issubset(expected_classes):
    raise ValueError(
        f"Unexpected class labels found: {actual_classes}"
    )


# --------------------------------------------------
# 4. Chronological split BEFORE interpolation
# --------------------------------------------------

total_records = len(df)

train_end = int(total_records * TRAIN_RATIO)
val_end = int(total_records * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nChronological split:")
print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)


# --------------------------------------------------
# 5. Handle missing values independently
# --------------------------------------------------

def interpolate_partition(partition, partition_name):

    partition = partition.copy()

    partition[SENSOR_COLUMNS] = (
        partition[SENSOR_COLUMNS]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    remaining_missing = (
        partition[SENSOR_COLUMNS].isnull().sum().sum()
    )

    print(
        f"\nMissing sensor values in {partition_name}:",
        remaining_missing
    )

    if remaining_missing > 0:
        raise ValueError(
            f"Missing values remain in {partition_name}."
        )

    return partition


train_df = interpolate_partition(train_df, "training")
val_df = interpolate_partition(val_df, "validation")
test_df = interpolate_partition(test_df, "testing")


# --------------------------------------------------
# 6. Separate features and labels
# --------------------------------------------------

X_train = train_df[SENSOR_COLUMNS]
y_train = train_df["Condition Label"]

X_val = val_df[SENSOR_COLUMNS]
y_val = val_df["Condition Label"]

X_test = test_df[SENSOR_COLUMNS]
y_test = test_df["Condition Label"]


# --------------------------------------------------
# 7. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

print("\nScaled feature shapes:")
print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Testing:", X_test_scaled.shape)

print("\nTraining feature means after scaling:")
print(X_train_scaled.mean(axis=0))


# --------------------------------------------------
# 8. Create LSTM sequences
# --------------------------------------------------

def create_sequences(X, y, time_steps):

    X_sequences = []
    y_sequences = []

    for i in range(len(X) - time_steps + 1):

        X_sequences.append(
            X[i:i + time_steps]
        )

        # Label corresponds to the final timestep
        y_sequences.append(
            y.iloc[i + time_steps - 1]
        )

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(y_sequences, dtype=np.int64)
    )


# --------------------------------------------------
# 9. Generate datasets for each window size
# --------------------------------------------------

for time_steps in WINDOW_SIZES:

    print("\n" + "=" * 50)
    print(f"PROCESSING WINDOW SIZE: {time_steps}")
    print("=" * 50)

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train,
        time_steps
    )

    X_val_seq, y_val_seq = create_sequences(
        X_val_scaled,
        y_val,
        time_steps
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled,
        y_test,
        time_steps
    )

    print("\nSequence shapes:")
    print("Training:", X_train_seq.shape, y_train_seq.shape)
    print("Validation:", X_val_seq.shape, y_val_seq.shape)
    print("Testing:", X_test_seq.shape, y_test_seq.shape)

    # Display class distribution after sequence generation
    datasets = {
        "Training": y_train_seq,
        "Validation": y_val_seq,
        "Testing": y_test_seq
    }

    for name, labels in datasets.items():

        print(f"\n{name} class distribution:")

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        for label, count in zip(unique, counts):
            print(f"Class {label}: {count}")

    # Save each window size separately
    window_dir = OUTPUT_DIR / f"window_{time_steps}"

    window_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(window_dir / "X_train.npy", X_train_seq)
    np.save(window_dir / "y_train.npy", y_train_seq)

    np.save(window_dir / "X_val.npy", X_val_seq)
    np.save(window_dir / "y_val.npy", y_val_seq)

    np.save(window_dir / "X_test.npy", X_test_seq)
    np.save(window_dir / "y_test.npy", y_test_seq)

    joblib.dump(
        scaler,
        window_dir / "scaler.pkl"
    )

    print("\nSaved window size:", time_steps)
    print("Output directory:", window_dir)


print("\n" + "=" * 50)
print("VIBRATION PREPROCESSING COMPLETED")
print("=" * 50)