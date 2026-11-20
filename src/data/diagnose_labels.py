import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

labels = df["Condition Label"]

# Same chronological split as preprocessing
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "Training": labels.iloc[:train_end],
    "Validation": labels.iloc[train_end:val_end],
    "Testing": labels.iloc[val_end:]
}

# D2: Label timeline
plt.figure(figsize=(14, 4))
plt.plot(labels.to_numpy(), linewidth=1)
plt.xlabel("Row Index")
plt.ylabel("Condition Label")
plt.title("Condition Label Timeline")
plt.yticks([0, 1, 2])
plt.grid(alpha=0.3)

plot_path = PROJECT_ROOT / "results" / "vibration"
plot_path.mkdir(parents=True, exist_ok=True)

plt.savefig(plot_path / "label_timeline.png",
            dpi=150, bbox_inches="tight")
plt.show()

transitions = (labels != labels.shift()).sum() - 1

print("D2: LABEL TIMELINE")
print("Total rows:", n)
print("Total label transitions:", transitions)

# D3: Contiguous segments per class
print("\nD3: CONTIGUOUS CLASS SEGMENTS")

for split_name, split_labels in splits.items():

    print(f"\n{'=' * 45}")
    print(split_name)
    print("=" * 45)

    segment_ids = (
        split_labels.ne(split_labels.shift())
        .cumsum()
    )

    segments = pd.DataFrame({
        "Label": split_labels.to_numpy(),
        "Segment": segment_ids.to_numpy()
    })

    segment_summary = (
        segments.groupby(["Segment", "Label"])
        .size()
        .reset_index(name="Rows")
    )

    for class_id in [0, 1, 2]:

        class_segments = segment_summary[
            segment_summary["Label"] == class_id
        ]

        print(f"\nClass {class_id}:")
        print("Number of segments:", len(class_segments))
        print("Rows per segment:",
              class_segments["Rows"].tolist())