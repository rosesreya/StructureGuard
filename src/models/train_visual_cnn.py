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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "VISUAL"
OUTPUT_DIR = PROJECT_ROOT / "results" / "visual"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

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

# Keep the same class mapping for validation.
if val_ds.class_names != class_names:
    raise ValueError("Training and validation class mappings differ.")

# Normalise pixel values to [0, 1].
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

# --------------------------------------------------
# CNN architecture
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Training
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in val_ds
]).astype(int)

probabilities = model.predict(val_ds, verbose=0).reshape(-1)
y_pred = (probabilities >= 0.5).astype(int)

print("\nVisual CNN Evaluation")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# --------------------------------------------------
# Save model and results
# --------------------------------------------------

model.save(OUTPUT_DIR / "visual_cnn.keras")

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

print("\nSaved model and evaluation results to:", OUTPUT_DIR)