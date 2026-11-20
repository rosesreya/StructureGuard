import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

labels = df["Condition Label"].to_numpy()

print("=" * 60)
print("CROSS-VALIDATION FEASIBILITY DIAGNOSTIC")
print("=" * 60)

print("\nTotal observations:", len(df))

# Divide the timeline into five equal chronological blocks.
blocks = np.array_split(np.arange(len(df)), 5)

for i, indices in enumerate(blocks, start=1):

    block_labels = labels[indices]

    print(f"\nBlock {i}")
    print("-" * 30)
    print("Row range:", indices[0], "to", indices[-1])
    print("Observations:", len(indices))

    unique, counts = np.unique(
        block_labels,
        return_counts=True
    )

    distribution = {
        int(label): int(count)
        for label, count in zip(unique, counts)
    }

    for label in [0, 1, 2]:
        print(
            f"Class {label}:",
            distribution.get(label, 0)
        )

    transitions = np.sum(
        block_labels[1:] != block_labels[:-1]
    )

    print("Transitions:", transitions)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)