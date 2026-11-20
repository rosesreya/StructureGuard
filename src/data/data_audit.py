from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib

from PIL import Image


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

VIBRATION_FILE = (
    DATA_DIR
    / "VIBRATIONAL"
    / "building_health_monitoring_dataset.csv"
)


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def get_file_hash(file_path):
    """Calculate SHA-256 hash to identify exact duplicate files."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_images(folder):
    """Find all supported image files recursively."""

    if not folder.exists():
        return []

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(image_path):
    """Read image dimensions, mode and verify file integrity."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            return {
                "status": "Valid",
                "width": image.width,
                "height": image.height,
                "mode": image.mode
            }

    except Exception as error:
        return {
            "status": "Corrupted",
            "error": str(error)
        }


# --------------------------------------------------
# 3. IMAGE DATASET AUDIT
# --------------------------------------------------

def audit_image_dataset(dataset_name):
    print("\n" + "=" * 65)
    print(f"{dataset_name} DATASET AUDIT")
    print("=" * 65)

    dataset_dir = DATA_DIR / dataset_name

    if not dataset_dir.exists():
        print(f"ERROR: Dataset folder not found: {dataset_dir}")
        return

    images = find_images(dataset_dir)

    print(f"\nDataset location: {dataset_dir}")
    print(f"Total image files: {len(images)}")

    if not images:
        print("No supported image files found.")
        return

    dimensions = Counter()
    color_modes = Counter()
    extensions = Counter()
    corrupted_files = []
    hash_groups = defaultdict(list)

    for index, image_path in enumerate(images, start=1):

        relative_path = image_path.relative_to(dataset_dir)

        extensions[image_path.suffix.lower()] += 1

        image_info = inspect_image(image_path)

        if image_info["status"] == "Corrupted":
            corrupted_files.append(relative_path)
            continue

        dimensions[
            (image_info["width"], image_info["height"])
        ] += 1

        color_modes[image_info["mode"]] += 1

        file_hash = get_file_hash(image_path)
        hash_groups[file_hash].append(relative_path)

        if index % 500 == 0:
            print(f"Inspected {index}/{len(images)} images")

    print("\n--- Image Dimensions ---")

    for dimension, count in dimensions.most_common():
        print(f"{dimension}: {count} images")

    print("\n--- Colour Modes ---")

    for mode, count in color_modes.items():
        print(f"{mode}: {count} images")

    print("\n--- File Extensions ---")

    for extension, count in extensions.items():
        print(f"{extension}: {count}")

    print("\n--- Directory-wise Image Counts ---")

    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir():
            child_images = find_images(child)
            print(f"{child.name}: {len(child_images)} images")

    duplicate_groups = [
        paths for paths in hash_groups.values()
        if len(paths) > 1
    ]

    duplicate_extra_files = sum(
        len(group) - 1 for group in duplicate_groups
    )

    print("\n--- Duplicate Analysis ---")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Extra duplicate files: {duplicate_extra_files}")

    for group in duplicate_groups[:10]:
        print("\nDuplicate group:")
        for path in group:
            print(f"  {path}")

    if len(duplicate_groups) > 10:
        print("\nOnly the first 10 duplicate groups are displayed.")

    print("\n--- File Integrity ---")
    print(f"Valid images: {len(images) - len(corrupted_files)}")
    print(f"Corrupted images: {len(corrupted_files)}")

    for path in corrupted_files[:10]:
        print(f"  {path}")


# --------------------------------------------------
# 4. VIBRATION DATASET AUDIT
# --------------------------------------------------

def audit_vibration_dataset():

    print("\n" + "=" * 65)
    print("VIBRATIONAL DATASET AUDIT")
    print("=" * 65)

    if not VIBRATION_FILE.exists():
        print(f"ERROR: CSV not found: {VIBRATION_FILE}")
        return

    with VIBRATION_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        columns = reader.fieldnames or []
        rows = list(reader)

    print(f"\nCSV file: {VIBRATION_FILE.name}")
    print(f"Total rows: {len(rows)}")
    print(f"Total columns: {len(columns)}")

    print("\n--- Column Names ---")

    for column in columns:
        print(column)

    if not rows:
        print("No data rows found.")
        return

    print("\n--- Missing Values ---")

    total_missing = 0

    for column in columns:
        missing = sum(
            1 for row in rows
            if row.get(column) is None
            or row[column].strip() == ""
        )

        total_missing += missing
        print(f"{column}: {missing}")

    print(f"Total missing cells: {total_missing}")

    print("\n--- Condition Label Distribution ---")

    label_column = next(
        (
            column for column in columns
            if column.strip().lower() == "condition label"
        ),
        None
    )

    if label_column:
        label_counts = Counter(
            row[label_column].strip()
            for row in rows
            if row.get(label_column)
            and row[label_column].strip()
        )

        for label, count in sorted(label_counts.items()):
            percentage = (count / len(rows)) * 100
            print(f"Label {label}: {count} ({percentage:.2f}%)")
    else:
        print("Condition Label column not found.")

    print("\n--- Sample Records (First 5) ---")

    for row in rows[:5]:
        print(row)


# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("MULTI-MODAL SHM PROJECT - DATASET AUDIT")
    print("=" * 65)

    print(f"\nProject root: {PROJECT_ROOT}")

    audit_image_dataset("VISUAL")
    audit_image_dataset("THERMAL")
    audit_vibration_dataset()

    print("\n" + "=" * 65)
    print("DATASET AUDIT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()