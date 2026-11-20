import json
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIBRATION_MODEL_PATH = (
    PROJECT_ROOT / "results/vibration/final/final_vibration_rf.joblib"
)

VISUAL_MODEL_PATH = (
    PROJECT_ROOT / "results/visual/visual_cnn.keras"
)

THERMAL_MODEL_PATH = (
    PROJECT_ROOT / "results/thermal/thermal_cnn.keras"
)

VISUAL_CLASSES_PATH = (
    PROJECT_ROOT / "results/visual/class_names.json"
)

THERMAL_CLASSES_PATH = (
    PROJECT_ROOT / "results/thermal/class_names.json"
)

IMAGE_SIZE = (128, 128)

# Initial equal weights.
W_VIBRATION = 1 / 3
W_VISUAL = 1 / 3
W_THERMAL = 1 / 3


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

print("Loading trained models...")

vibration_model = joblib.load(VIBRATION_MODEL_PATH)

visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)

thermal_model = tf.keras.models.load_model(THERMAL_MODEL_PATH)

with open(VISUAL_CLASSES_PATH) as f:
    visual_classes = json.load(f)

with open(THERMAL_CLASSES_PATH) as f:
    thermal_classes = json.load(f)

# Verify class mappings before interpreting probabilities.
if visual_classes != ["crack", "no_crack"]:
    raise ValueError(f"Unexpected visual classes: {visual_classes}")

if thermal_classes != ["hotspot", "normal"]:
    raise ValueError(f"Unexpected thermal classes: {thermal_classes}")

print("All three models loaded successfully.")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


# --------------------------------------------------
# Vibration prediction
# --------------------------------------------------

def predict_vibration(sensor_values):

    features = np.array(
        sensor_values,
        dtype=float
    ).reshape(1, 5)

    probabilities = vibration_model.predict_proba(features)[0]

    classes = vibration_model.named_steps[
        "classifier"
    ].classes_

    probability_by_class = dict(zip(classes, probabilities))

    abnormal_probability = (
        probability_by_class.get(1, 0.0)
        + probability_by_class.get(2, 0.0)
    )

    return float(abnormal_probability)


# --------------------------------------------------
# Visual prediction
# --------------------------------------------------

def predict_visual(image_path):

    image = preprocess_image(image_path)

    # Sigmoid output represents class 1: no_crack.
    no_crack_probability = float(
        visual_model.predict(image, verbose=0)[0][0]
    )

    crack_probability = 1.0 - no_crack_probability

    return crack_probability


# --------------------------------------------------
# Thermal prediction
# --------------------------------------------------

def predict_thermal(image_path):

    image = preprocess_image(image_path)

    # Sigmoid output represents class 1: normal.
    normal_probability = float(
        thermal_model.predict(image, verbose=0)[0][0]
    )

    hotspot_probability = 1.0 - normal_probability

    return hotspot_probability


# --------------------------------------------------
# SSI calculation
# --------------------------------------------------

def calculate_ssi(p_vibration, p_visual, p_thermal):

    ssi = 100 * (
        W_VIBRATION * p_vibration
        + W_VISUAL * p_visual
        + W_THERMAL * p_thermal
    )

    if ssi <= 30:
        risk = "Safe"

    elif ssi <= 70:
        risk = "Moderate Risk"

    else:
        risk = "High Risk"

    return ssi, risk


# --------------------------------------------------
# Main execution
# --------------------------------------------------

if __name__ == "__main__":

    print("\nMULTIMODAL STRUCTURAL HEALTH MONITORING")
    print("-" * 50)

    print("\nEnter five vibration sensor readings:")

    sensor_names = [
        "Accel_X (m/s^2)",
        "Accel_Y (m/s^2)",
        "Accel_Z (m/s^2)",
        "Strain (με)",
        "Temp (°C)"
    ]

    sensor_values = []

    for name in sensor_names:

        value = float(input(f"{name}: "))

        sensor_values.append(value)

    visual_path = input(
        "\nEnter visual image path: "
    ).strip().strip('"')

    thermal_path = input(
        "Enter thermal image path: "
    ).strip().strip('"')

    if not Path(visual_path).is_file():
        raise FileNotFoundError(visual_path)

    if not Path(thermal_path).is_file():
        raise FileNotFoundError(thermal_path)

    # Get probabilities.
    p_vibration = predict_vibration(sensor_values)

    p_visual = predict_visual(visual_path)

    p_thermal = predict_thermal(thermal_path)

    # Calculate SSI.
    ssi, risk = calculate_ssi(
        p_vibration,
        p_visual,
        p_thermal
    )

    # Display results.
    print("\n" + "=" * 50)
    print("MULTIMODAL PREDICTION RESULTS")
    print("=" * 50)

    print(f"Vibration abnormality: {p_vibration:.4f}")
    print(f"Visual crack:          {p_visual:.4f}")
    print(f"Thermal hotspot:       {p_thermal:.4f}")

    print("-" * 50)

    print(f"Structural Stress Index: {ssi:.2f}/100")
    print(f"Risk Category: {risk}")

    print("=" * 50)