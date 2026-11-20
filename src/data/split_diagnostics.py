from pathlib import Path
from collections import defaultdict
import hashlib
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path):
    sha = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def get_hashes(folder):
    hashes = defaultdict(list)

    if not folder.exists():
        return hashes

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            hashes[file_hash(path)].append(path)

    return hashes


def diagnose_image_splits(dataset, classes):
    print("\n" + "=" * 60)
    print(f"{dataset} SPLIT DIAGNOSTICS")
    print("=" * 60)

    root = DATA_DIR / dataset

    for class_name in classes:
        print(f"\nClass: {class_name}")

        original = get_hashes(root / class_name)
        train = get_hashes(root / "train" / class_name)
        val = get_hashes(root / "val" / class_name)

        original_hashes = set(original)
        train_hashes = set(train)
        val_hashes = set(val)

        print(f"Original unique images: {len(original_hashes)}")
        print(f"Training unique images: {len(train_hashes)}")
        print(f"Validation unique images: {len(val_hashes)}")

        print(
            "Original images also in train:",
            len(original_hashes & train_hashes)
        )

        print(
            "Original images also in val:",
            len(original_hashes & val_hashes)
        )

        overlap = train_hashes & val_hashes

        print(
            "Identical images shared by train and val:",
            len(overlap)
        )

        if overlap:
            print("Examples of train-validation overlap:")

            for digest in list(overlap)[:5]:
                print("  Train:", train[digest][0])
                print("  Val:  ", val[digest][0])


def diagnose_vibration_missing_values():
    print("\n" + "=" * 60)
    print("VIBRATION MISSING-VALUE DIAGNOSTICS")
    print("=" * 60)

    csv_path = (
        DATA_DIR
        / "VIBRATIONAL"
        / "building_health_monitoring_dataset.csv"
    )

    with csv_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    sensor_columns = [
        column for column in reader.fieldnames
        if column != "Timestamp" and column != "Condition Label"
    ]

    incomplete_rows = []

    for index, row in enumerate(rows):
        missing = [
            column for column in sensor_columns
            if not row[column] or not row[column].strip()
        ]

        if missing:
            incomplete_rows.append((index, missing))

    print("Total records:", len(rows))
    print("Rows with at least one missing sensor value:", len(incomplete_rows))

    print("\nFirst 20 incomplete records:")

    for index, missing in incomplete_rows[:20]:
        print(f"Row {index}: Missing {missing}")

    print("\nMissing values per record:")

    missing_pattern_counts = defaultdict(int)

    for _, missing in incomplete_rows:
        pattern = tuple(missing)
        missing_pattern_counts[pattern] += 1

    for pattern, count in missing_pattern_counts.items():
        print(f"{pattern}: {count} rows")


def main():
    diagnose_image_splits(
        "VISUAL",
        ["crack", "no_crack"]
    )

    diagnose_image_splits(
        "THERMAL",
        ["hotspot", "normal"]
    )

    diagnose_vibration_missing_values()


if __name__ == "__main__":
    main()