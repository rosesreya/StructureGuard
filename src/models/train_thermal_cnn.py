import tensorflow as tf
import numpy as np
import json

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "THERMAL"
OUTPUT_DIR = PROJECT_ROOT / "results" / "thermal"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40
SEED = 42

# Load datasets

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

class_names = train_ds.class_names

print("Classes:", class_names)

if val_ds.class_names != class_names:
    raise ValueError("Training and validation class mappings differ.")

# Class weights: compensate for class imbalance.

train_counts = np.bincount(
    np.concatenate([
        labels.numpy().astype(int).reshape(-1)
        for _, labels in train_ds
    ]),
    minlength=2
)

total = train_counts.sum()

class_weights = {
    i: total / (2 * count)
    for i, count in enumerate(train_counts)
}

print("Training class counts:", train_counts)
print("Class weights:", class_weights)

# Normalisation

normalisation = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalisation(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalisation(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# CNN architecture

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Training

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluation

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in val_ds
]).astype(int)

probabilities = model.predict(
    val_ds, verbose=0
).reshape(-1)

y_pred = (probabilities >= 0.5).astype(int)

print("\nThermal CNN Evaluation")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(
    y_true, y_pred, average="macro"
))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save

model.save(OUTPUT_DIR / "thermal_cnn.keras")

np.save(
    OUTPUT_DIR / "validation_probabilities.npy",
    probabilities
)

np.save(
    OUTPUT_DIR / "validation_labels.npy",
    y_true
)

with open(OUTPUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

with open(OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump(history.history, f)

print("\nSaved thermal model and results to:", OUTPUT_DIR)