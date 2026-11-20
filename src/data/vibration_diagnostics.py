from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)


def main():

    df = pd.read_csv(CSV_PATH)

    sensor_columns = [
        "Accel_X (m/s^2)",
        "Accel_Y (m/s^2)",
        "Accel_Z (m/s^2)",
        "Strain (με)",
        "Temp (°C)"
    ]

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # 1. Timestamp analysis
    print("\n--- TIMESTAMP ANALYSIS ---")

    print("First timestamp:", df["Timestamp"].min())
    print("Last timestamp:", df["Timestamp"].max())

    print(
        "Timestamps in chronological order:",
        df["Timestamp"].is_monotonic_increasing
    )

    print(
        "Duplicate timestamps:",
        df["Timestamp"].duplicated().sum()
    )

    print(
        "Timestamp differences:"
    )

    print(
        df["Timestamp"].diff().value_counts().head(10)
    )

    # 2. Missing values by condition
    print("\n--- MISSING VALUES BY CONDITION ---")

    print(
        df.groupby("Condition Label")[sensor_columns]
        .apply(lambda group: group.isna().sum())
    )

    # 3. Incomplete rows by condition
    print("\n--- INCOMPLETE ROWS BY CONDITION ---")

    df["Has_Missing"] = df[sensor_columns].isna().any(axis=1)

    print(
        df.groupby("Condition Label")["Has_Missing"]
        .agg(["sum", "count"])
    )

    # 4. Missing-value locations
    print("\n--- MISSING VALUE LOCATIONS ---")

    missing_rows = df[df["Has_Missing"]]

    print(
        missing_rows[
            ["Timestamp", "Condition Label"] + sensor_columns
        ].to_string(index=False)
    )

    # 5. Sensor statistics
    print("\n--- SENSOR STATISTICS ---")

    print(df[sensor_columns].describe().round(3))

    # 6. Condition labels
    print("\n--- CONDITION LABELS ---")

    print(
        df["Condition Label"].value_counts().sort_index()
    )

    print("\nVIBRATION DIAGNOSTICS COMPLETED")


if __name__ == "__main__":
    main()