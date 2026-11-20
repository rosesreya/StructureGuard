import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT / "data" / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

sensor_columns = [
    "Accel_X (m/s^2)",
    "Accel_Y (m/s^2)",
    "Accel_Z (m/s^2)",
    "Strain (με)",
    "Temp (°C)"
]

print("D4: MISSING VALUES PER SENSOR")
print(df[sensor_columns].isnull().sum())

print("\nMISSING VALUES PER CLASS")

for class_id in [0, 1, 2]:

    class_df = df[df["Condition Label"] == class_id]

    print(f"\nClass {class_id} ({len(class_df)} rows):")

    print(class_df[sensor_columns].isnull().sum())

print("\nMISSING VALUE COUNTS PER SPLIT")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "Training": df.iloc[:train_end],
    "Validation": df.iloc[train_end:val_end],
    "Testing": df.iloc[val_end:]
}

for name, partition in splits.items():

    print(f"\n{name}:")
    print(partition[sensor_columns].isnull().sum())

print("\nInterpolation method in current preprocessing:")
print("Linear interpolation, independently within each split.")
print("limit_direction='both' allows interpolation in both directions.")