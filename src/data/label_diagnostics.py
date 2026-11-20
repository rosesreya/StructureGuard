from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT / "data" / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)


def main():
    df = pd.read_csv(CSV_PATH)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    labels = df["Condition Label"]

    print("\n--- LABEL TRANSITIONS ---")
    print("Total label transitions:", (labels != labels.shift()).sum() - 1)

    print("\n--- CONSECUTIVE LABEL SEGMENTS ---")

    segment_id = (labels != labels.shift()).cumsum()

    segments = (
        df.assign(Segment=segment_id)
        .groupby("Segment")
        .agg(
            Start=("Timestamp", "first"),
            End=("Timestamp", "last"),
            Label=("Condition Label", "first"),
            Records=("Condition Label", "size")
        )
    )

    print(segments.to_string(index=False))

    print("\n--- CHRONOLOGICAL SPLIT DISTRIBUTION ---")

    split_data = df.copy()

    split_data["Split"] = (
        ["Train"] * 700
        + ["Validation"] * 150
        + ["Test"] * 150
    )

    print(
        pd.crosstab(
            split_data["Split"],
            split_data["Condition Label"],
            normalize="index"
        ).mul(100).round(2)
    )

    print("\n--- LABEL COUNTS BY SPLIT ---")

    print(
        pd.crosstab(
            split_data["Split"],
            split_data["Condition Label"]
        )
    )


if __name__ == "__main__":
    main()