from pathlib import Path
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StructureGuard | SHM Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

VIBRATION_RESULTS = RESULTS_DIR / "vibration" / "final"
VISUAL_RESULTS = RESULTS_DIR / "visual"
THERMAL_RESULTS = RESULTS_DIR / "thermal"

VIBRATION_MODEL = VIBRATION_RESULTS / "final_vibration_rf.joblib"
VISUAL_MODEL = VISUAL_RESULTS / "visual_cnn.keras"
THERMAL_MODEL = THERMAL_RESULTS / "thermal_cnn.keras"

VIBRATION_METRICS = VIBRATION_RESULTS / "test_metrics.csv"
VIBRATION_PREDICTIONS = VIBRATION_RESULTS / "test_predictions.csv"


# =========================================================
# 3. SESSION STATE
# =========================================================

if "fusion_result" not in st.session_state:
    st.session_state.fusion_result = None


# =========================================================
# 4. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f1f5f9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #252630;
    }

    section[data-testid="stSidebar"] * {
        color: #f1f5f9;
    }

    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    h1, h2, h3 {
        color: #e2e8f0;
    }

    p, li, label {
        color: #cbd5e1;
    }

    .architecture-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        min-height: 165px;
    }

    .architecture-card h3 {
        color: #60a5fa;
        margin-bottom: 12px;
    }

    .architecture-card p {
        color: #cbd5e1;
        font-size: 14px;
    }

    hr {
        border-color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. EVALUATION DATA FUNCTIONS
# =========================================================

@st.cache_data
def load_vibration_metrics():

    if not VIBRATION_METRICS.exists():
        return None

    df = pd.read_csv(VIBRATION_METRICS)

    if df.empty:
        return None

    return df.iloc[0].to_dict()


@st.cache_data
def load_image_metrics(modality):

    results_dir = (
        VISUAL_RESULTS
        if modality == "visual"
        else THERMAL_RESULTS
    )

    labels_path = results_dir / "validation_labels.npy"
    probabilities_path = results_dir / "validation_probabilities.npy"

    if not labels_path.exists() or not probabilities_path.exists():
        return None

    labels = np.load(labels_path).reshape(-1)
    probabilities = np.load(probabilities_path).reshape(-1)

    if len(labels) != len(probabilities):
        return None

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "labels": labels,
        "predictions": predictions,
    }


@st.cache_data
def load_vibration_predictions():

    if not VIBRATION_PREDICTIONS.exists():
        return None

    return pd.read_csv(VIBRATION_PREDICTIONS)


def get_metric(metrics, key):

    if metrics is None:
        return None

    value = metrics.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def display_percentage(value):

    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def display_decimal(value):

    if value is None:
        return "N/A"

    return f"{value:.4f}"


# =========================================================
# 6. EVALUATION DISPLAY FUNCTIONS
# =========================================================

def get_vibration_prediction_columns(df):

    columns = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    actual_candidates = [
        "y_true", "true", "actual", "actual_label",
        "true_label", "label", "condition_label",
    ]

    predicted_candidates = [
        "y_pred", "pred", "prediction", "predicted",
        "predicted_label", "predicted_class",
    ]

    actual_col = next(
        (columns[c] for c in actual_candidates if c in columns),
        None,
    )

    predicted_col = next(
        (columns[c] for c in predicted_candidates if c in columns),
        None,
    )

    return actual_col, predicted_col


def show_model_summary(y_true, y_pred):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy_score(y_true, y_pred) * 100:.2f}%",
    )

    col2.metric(
        "Macro F1",
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
    )

    col3.metric(
        "Balanced Accuracy",
        f"{balanced_accuracy_score(y_true, y_pred):.4f}",
    )


def show_confusion_matrix(y_true, y_pred, class_names):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    st.dataframe(
        matrix_df.style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    st.caption(
        "Rows represent actual classes; columns represent predicted classes."
    )


def show_classification_metrics(y_true, y_pred, class_names):

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    report_df = report_df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1 Score",
            "support": "Support",
        }
    )

    st.dataframe(
        report_df.round(4),
        use_container_width=True,
    )


def show_training_curves(modality):

    history_path = RESULTS_DIR / modality / "training_history.json"

    if not history_path.exists():
        st.warning(f"Training history not found for {modality}.")
        return

    try:
        with open(history_path, "r") as file:
            history = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read training history: {error}")
        return

    if "history" in history and isinstance(history["history"], dict):
        history = history["history"]

    loss_data = {}

    if "loss" in history:
        loss_data["Training Loss"] = history["loss"]

    if "val_loss" in history:
        loss_data["Validation Loss"] = history["val_loss"]

    accuracy_data = {}

    if "accuracy" in history:
        accuracy_data["Training Accuracy"] = history["accuracy"]

    if "val_accuracy" in history:
        accuracy_data["Validation Accuracy"] = history["val_accuracy"]

    if loss_data:
        st.markdown("#### Training and Validation Loss")
        loss_df = pd.DataFrame(loss_data)
        loss_df.index = loss_df.index + 1
        loss_df.index.name = "Epoch"
        st.line_chart(loss_df)

    if accuracy_data:
        st.markdown("#### Training and Validation Accuracy")
        accuracy_df = pd.DataFrame(accuracy_data)
        accuracy_df.index = accuracy_df.index + 1
        accuracy_df.index.name = "Epoch"
        st.line_chart(accuracy_df)

    if not loss_data and not accuracy_data:
        st.warning("No recognised training or validation metrics were found.")


# =========================================================
# 7. MODEL LOADING
# =========================================================

@st.cache_resource
def load_vibration_model():

    if not VIBRATION_MODEL.exists():
        raise FileNotFoundError(
            f"Vibration model not found: {VIBRATION_MODEL}"
        )

    return joblib.load(VIBRATION_MODEL)


@st.cache_resource
def load_keras_model(modality):

    if modality == "visual":
        model_path = VISUAL_MODEL
    else:
        model_path = THERMAL_MODEL

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    from tensorflow.keras.models import load_model

    return load_model(model_path)


# =========================================================
# 8. INDIVIDUAL PREDICTION FUNCTIONS
# =========================================================

def predict_vibration(sensor_values):

    model = load_vibration_model()

    feature_names = [
        "Accel_X (m/s^2)",
        "Accel_Y (m/s^2)",
        "Accel_Z (m/s^2)",
        "Strain (με)",
        "Temp (°C)",
    ]

    input_df = pd.DataFrame(
        [sensor_values],
        columns=feature_names,
    )

    probabilities = model.predict_proba(input_df)[0]

    predicted_class = int(np.argmax(probabilities))

    class_names = {
        0: "Normal",
        1: "Minor Damage",
        2: "Severe Damage",
    }

    return {
        "class": predicted_class,
        "class_name": class_names[predicted_class],
        "probabilities": probabilities,
    }


def preprocess_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    return image, np.expand_dims(image_array, axis=0)


def predict_image(uploaded_file, modality):

    model = load_keras_model(modality)

    image, image_array = preprocess_image(uploaded_file)

    output = float(
        model.predict(image_array, verbose=0).reshape(-1)[0]
    )

    if modality == "visual":

        probabilities = {
            "Crack": 1.0 - output,
            "No Crack": output,
        }

    else:

        probabilities = {
            "Hotspot": 1.0 - output,
            "Normal": output,
        }

    predicted_class = max(
        probabilities,
        key=probabilities.get,
    )

    return image, probabilities, predicted_class


# =========================================================
# 9. SSI AND REPORT FUNCTIONS
# =========================================================

def calculate_ssi(
    vibration_probability,
    visual_probability,
    thermal_probability,
):

    ssi = 100 * (
        vibration_probability
        + visual_probability
        + thermal_probability
    ) / 3

    if ssi <= 30:
        risk = "Safe"
    elif ssi <= 70:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    return ssi, risk


def create_fusion_report(result):

    rows = [
        ("Assessment Date", result["timestamp"]),
        ("Vibration Abnormality Probability", result["vibration_probability"]),
        ("Visual Crack Probability", result["visual_probability"]),
        ("Thermal Hotspot Probability", result["thermal_probability"]),
        ("Vibration Predicted Class", result["vibration_class"]),
        ("Visual Predicted Class", result["visual_class"]),
        ("Thermal Predicted Class", result["thermal_class"]),
        ("Vibration Weight", 1 / 3),
        ("Visual Weight", 1 / 3),
        ("Thermal Weight", 1 / 3),
        ("Structural Stress Index", result["ssi"]),
        ("Risk Classification", result["risk"]),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Result"])


def generate_insights(result):

    insights = []

    probabilities = {
        "Vibration": result["vibration_probability"],
        "Visual": result["visual_probability"],
        "Thermal": result["thermal_probability"],
    }

    highest_modality = max(
        probabilities,
        key=probabilities.get,
    )

    lowest_modality = min(
        probabilities,
        key=probabilities.get,
    )

    insights.append(
        f"{highest_modality} has the highest abnormality probability "
        f"({probabilities[highest_modality] * 100:.2f}%)."
    )

    insights.append(
        f"{lowest_modality} has the lowest abnormality probability "
        f"({probabilities[lowest_modality] * 100:.2f}%)."
    )

    if result["risk"] == "Safe":
        insights.append(
            "The calculated SSI falls within the defined Safe range."
        )

    elif result["risk"] == "Moderate Risk":
        insights.append(
            "The calculated SSI falls within the defined Moderate Risk range."
        )

    else:
        insights.append(
            "The calculated SSI exceeds 70 and falls within the High Risk range."
        )

    if result["visual_probability"] >= 0.5:
        insights.append(
            "The visual model predicts the presence of a crack."
        )
    else:
        insights.append(
            "The visual model predicts no crack in the uploaded image."
        )

    if result["thermal_probability"] >= 0.5:
        insights.append(
            "The thermal model predicts a hotspot."
        )
    else:
        insights.append(
            "The thermal model predicts a normal thermal pattern."
        )

    if result["vibration_class"] != "Normal":
        insights.append(
            "The vibration classifier predicts a non-normal structural condition."
        )
    else:
        insights.append(
            "The vibration classifier predicts a normal structural condition."
        )

    return insights


def create_text_report(result):

    insights = generate_insights(result)

    report_lines = [
        "STRUCTUREGUARD",
        "MULTIMODAL STRUCTURAL HEALTH ASSESSMENT REPORT",
        "=" * 55,
        "",
        f"Assessment Date: {result['timestamp']}",
        "",
        "1. INDIVIDUAL MODALITY RESULTS",
        "-" * 35,
        f"Vibration abnormality probability: "
        f"{result['vibration_probability'] * 100:.2f}%",
        f"Vibration predicted class: {result['vibration_class']}",
        "",
        f"Visual crack probability: "
        f"{result['visual_probability'] * 100:.2f}%",
        f"Visual predicted class: {result['visual_class']}",
        "",
        f"Thermal hotspot probability: "
        f"{result['thermal_probability'] * 100:.2f}%",
        f"Thermal predicted class: {result['thermal_class']}",
        "",
        "2. STRUCTURAL STRESS INDEX",
        "-" * 35,
        f"SSI: {result['ssi']:.2f} / 100",
        f"Risk Classification: {result['risk']}",
        "",
        "3. AUTOMATED INSIGHTS",
        "-" * 35,
    ]

    for index, insight in enumerate(insights, start=1):
        report_lines.append(f"{index}. {insight}")

    report_lines.extend(
        [
            "",
            "4. METHODOLOGY AND LIMITATIONS",
            "-" * 35,
            "The SSI is calculated using equal weights for vibration, "
            "visual and thermal abnormality probabilities.",
            "",
            "The three modality datasets are independent and are not "
            "paired measurements from the same physical structure.",
            "",
            "Therefore, this result demonstrates scenario-based "
            "multimodal fusion. It is not a validated physical stress "
            "measurement or an independently verified structural diagnosis.",
            "",
            "End of Report",
        ]
    )

    return "\n".join(report_lines)


# =========================================================
# 10. LOAD SAVED EVALUATION RESULTS
# =========================================================

vibration_metrics = load_vibration_metrics()

visual_metrics = load_image_metrics("visual")
thermal_metrics = load_image_metrics("thermal")

vibration_accuracy = get_metric(vibration_metrics, "Accuracy")
vibration_f1 = get_metric(vibration_metrics, "Macro F1")

visual_accuracy = (
    visual_metrics["accuracy"]
    if visual_metrics is not None else None
)

visual_f1 = (
    visual_metrics["macro_f1"]
    if visual_metrics is not None else None
)

thermal_accuracy = (
    thermal_metrics["accuracy"]
    if thermal_metrics is not None else None
)

thermal_f1 = (
    thermal_metrics["macro_f1"]
    if thermal_metrics is not None else None
)


# =========================================================
# 11. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🏗️ STRUCTUREGUARD</h2>
        <p>Multimodal Structural Health Monitoring</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select a page",
        [
            "Project Overview",
            "Model Evaluation",
            "Individual Modality Analysis",
            "Multimodal Fusion",
            "Reports and Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    if VIBRATION_MODEL.exists():
        st.success("Vibration model ready", icon="✅")
    else:
        st.error("Vibration model missing")

    if VISUAL_MODEL.exists():
        st.success("Visual model ready", icon="✅")
    else:
        st.error("Visual model missing")

    if THERMAL_MODEL.exists():
        st.success("Thermal model ready", icon="✅")
    else:
        st.error("Thermal model missing")

    st.divider()

    st.caption("Deep Learning-Based Multi-Modal SHM")


# =========================================================
# 12. PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title("Structural Health Monitoring Dashboard")

    st.markdown(
        "Deep Learning-Based Multi-Modal Structural Health Monitoring"
    )

    st.divider()

    st.subheader("Model Performance Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Vibration Accuracy",
        display_percentage(vibration_accuracy),
    )
    col1.caption("Random Forest · Test data")

    col2.metric(
        "Visual Accuracy",
        display_percentage(visual_accuracy),
    )
    col2.caption("CNN · Validation data")

    col3.metric(
        "Thermal Accuracy",
        display_percentage(thermal_accuracy),
    )
    col3.caption("CNN · Validation data")

    st.subheader("Macro F1 Scores")

    col1, col2, col3 = st.columns(3)

    col1.metric("Vibration Macro F1", display_decimal(vibration_f1))
    col2.metric("Visual Macro F1", display_decimal(visual_f1))
    col3.metric("Thermal Macro F1", display_decimal(thermal_f1))

    st.info(
        "Vibration metrics use the chronological test split. "
        "Visual and thermal metrics use their validation splits. "
        "These results should not be treated as directly equivalent."
    )

    st.divider()

    st.subheader("System Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="architecture-card">
                <h3>Vibration Analysis</h3>
                <p><b>Model:</b> Random Forest</p>
                <p><b>Input:</b> Five sensor measurements</p>
                <p><b>Output:</b> Structural condition probabilities</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="architecture-card">
                <h3>Visual Analysis</h3>
                <p><b>Model:</b> Convolutional Neural Network</p>
                <p><b>Input:</b> Structural images</p>
                <p><b>Output:</b> Crack probability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="architecture-card">
                <h3>Thermal Analysis</h3>
                <p><b>Model:</b> Convolutional Neural Network</p>
                <p><b>Input:</b> Thermal images</p>
                <p><b>Output:</b> Hotspot probability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Dataset Overview")

    dataset_data = {
        "Modality": ["Vibration", "Visual", "Thermal"],
        "Dataset": [
            "Building Health Monitoring Dataset",
            "Concrete Crack Images",
            "Thermal Images",
        ],
        "Training Samples": [
            "850 rows used for final model",
            "3,200 images",
            "346 images",
        ],
        "Evaluation Samples": [
            "143 windows",
            "800 images",
            "87 images",
        ],
        "Model": ["Random Forest", "CNN", "CNN"],
    }

    st.dataframe(
        pd.DataFrame(dataset_data),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Structural Stress Index (SSI)")

    st.markdown(
        "The SSI combines abnormality probabilities from all three modalities."
    )

    st.latex(
        r"SSI = 100 \times "
        r"(w_V P_V + w_I P_I + w_T P_T)"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Vibration Weight", "33.33%")
    col2.metric("Visual Weight", "33.33%")
    col3.metric("Thermal Weight", "33.33%")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("Safe")
        st.caption("SSI ≤ 30")

    with col2:
        st.warning("Moderate Risk")
        st.caption("30 < SSI ≤ 70")

    with col3:
        st.error("High Risk")
        st.caption("SSI > 70")

    st.info(
        "SSI is an experimental composite indicator, not a calibrated "
        "measurement of physical structural stress. The three modality "
        "datasets are independent and are not paired observations "
        "from the same structure."
    )


# =========================================================
# 13. MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    st.markdown(
        "Detailed performance analysis of the three classification models."
    )

    st.divider()

    modality = st.selectbox(
        "Select Model",
        [
            "Vibration Model",
            "Visual CNN",
            "Thermal CNN",
        ],
    )

    st.divider()

    if modality == "Vibration Model":

        st.subheader("Vibration Model — Random Forest")
        st.caption("Evaluation dataset: Final chronological test split")

        vibration_df = load_vibration_predictions()

        if vibration_df is None:

            st.error("Vibration test predictions file was not found.")

        else:

            actual_col, predicted_col = (
                get_vibration_prediction_columns(vibration_df)
            )

            if actual_col is None or predicted_col is None:

                st.error(
                    "Could not identify actual and predicted label columns."
                )

                st.write("Available columns:", list(vibration_df.columns))

            else:

                try:

                    y_true = pd.to_numeric(
                        vibration_df[actual_col],
                        errors="raise",
                    ).astype(int).to_numpy()

                    y_pred = pd.to_numeric(
                        vibration_df[predicted_col],
                        errors="raise",
                    ).astype(int).to_numpy()

                    class_names = [
                        "Normal",
                        "Minor Damage",
                        "Severe Damage",
                    ]

                    if (
                        not np.isin(y_true, [0, 1, 2]).all()
                        or not np.isin(y_pred, [0, 1, 2]).all()
                    ):

                        st.error("Unexpected vibration class labels.")

                    else:

                        show_model_summary(y_true, y_pred)

                        st.divider()
                        st.markdown("### Confusion Matrix")

                        show_confusion_matrix(
                            y_true,
                            y_pred,
                            class_names,
                        )

                        st.markdown("### Class-wise Performance")

                        show_classification_metrics(
                            y_true,
                            y_pred,
                            class_names,
                        )

                except (ValueError, TypeError) as error:

                    st.error(f"Could not process predictions: {error}")

    elif modality == "Visual CNN":

        st.subheader("Visual Model — Crack Detection CNN")
        st.caption("Evaluation dataset: Visual validation split")

        if visual_metrics is None:

            st.error("Visual evaluation files were not found.")

        else:

            y_true = visual_metrics["labels"]
            y_pred = visual_metrics["predictions"]

            class_names = ["Crack", "No Crack"]

            show_model_summary(y_true, y_pred)

            st.divider()
            st.markdown("### Confusion Matrix")

            show_confusion_matrix(y_true, y_pred, class_names)

            st.markdown("### Class-wise Performance")

            show_classification_metrics(
                y_true,
                y_pred,
                class_names,
            )

            st.divider()
            st.markdown("### Training History")

            show_training_curves("visual")

    elif modality == "Thermal CNN":

        st.subheader("Thermal Model — Hotspot Detection CNN")
        st.caption("Evaluation dataset: Thermal validation split")

        if thermal_metrics is None:

            st.error("Thermal evaluation files were not found.")

        else:

            y_true = thermal_metrics["labels"]
            y_pred = thermal_metrics["predictions"]

            class_names = ["Hotspot", "Normal"]

            show_model_summary(y_true, y_pred)

            st.divider()
            st.markdown("### Confusion Matrix")

            show_confusion_matrix(y_true, y_pred, class_names)

            st.markdown("### Class-wise Performance")

            show_classification_metrics(
                y_true,
                y_pred,
                class_names,
            )

            st.divider()
            st.markdown("### Training History")

            show_training_curves("thermal")

    st.divider()

    st.caption(
        "Evaluation results are loaded from saved predictions and "
        "training histories. No models are retrained on this page."
    )


# =========================================================
# 14. INDIVIDUAL MODALITY ANALYSIS
# =========================================================

elif page == "Individual Modality Analysis":

    st.title("Individual Modality Analysis")

    st.markdown(
        "Run the trained models independently to inspect their predictions."
    )

    st.info(
        "These predictions are not combined into the Structural Stress Index."
    )

    selected_modality = st.selectbox(
        "Choose an analysis modality",
        [
            "Vibration Analysis",
            "Visual Crack Analysis",
            "Thermal Analysis",
        ],
    )

    st.divider()

    if selected_modality == "Vibration Analysis":

        st.subheader("Vibration-Based Condition Assessment")

        col1, col2 = st.columns(2)

        with col1:

            accel_x = st.number_input(
                "Acceleration X (m/s²)",
                value=0.15,
                format="%.5f",
            )

            accel_y = st.number_input(
                "Acceleration Y (m/s²)",
                value=0.02,
                format="%.5f",
            )

            accel_z = st.number_input(
                "Acceleration Z (m/s²)",
                value=9.81,
                format="%.5f",
            )

        with col2:

            strain = st.number_input(
                "Strain (με)",
                value=130.0,
                format="%.5f",
            )

            temperature = st.number_input(
                "Temperature (°C)",
                value=25.0,
                format="%.5f",
            )

        if st.button(
            "Analyse Vibration Data",
            type="primary",
            use_container_width=True,
        ):

            try:

                result = predict_vibration(
                    [
                        accel_x,
                        accel_y,
                        accel_z,
                        strain,
                        temperature,
                    ]
                )

                st.divider()
                st.subheader("Prediction Result")

                if result["class"] == 0:
                    st.success(
                        f"Predicted Condition: {result['class_name']}"
                    )
                elif result["class"] == 1:
                    st.warning(
                        f"Predicted Condition: {result['class_name']}"
                    )
                else:
                    st.error(
                        f"Predicted Condition: {result['class_name']}"
                    )

                probability_df = pd.DataFrame(
                    {
                        "Condition": [
                            "Normal",
                            "Minor Damage",
                            "Severe Damage",
                        ],
                        "Probability": result["probabilities"],
                    }
                )

                st.dataframe(
                    probability_df.style.format(
                        {"Probability": "{:.2%}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    probability_df.set_index("Condition")
                )

            except Exception as error:

                st.error(f"Vibration prediction failed: {error}")

    elif selected_modality == "Visual Crack Analysis":

        st.subheader("Visual Crack Detection")

        uploaded_image = st.file_uploader(
            "Upload a structural image",
            type=["jpg", "jpeg", "png"],
            key="visual_upload",
        )

        if uploaded_image is not None:

            try:

                preview = Image.open(uploaded_image).convert("RGB")

                st.image(
                    preview,
                    caption="Uploaded Structural Image",
                    use_container_width=True,
                )

                if st.button(
                    "Analyse Structural Image",
                    type="primary",
                    use_container_width=True,
                ):

                    uploaded_image.seek(0)

                    image, probabilities, prediction = predict_image(
                        uploaded_image,
                        "visual",
                    )

                    st.divider()
                    st.subheader("Prediction Result")

                    if prediction == "Crack":
                        st.warning("Prediction: Crack Detected")
                    else:
                        st.success("Prediction: No Crack Detected")

                    probability_df = pd.DataFrame(
                        {
                            "Class": list(probabilities.keys()),
                            "Probability": list(probabilities.values()),
                        }
                    )

                    st.dataframe(
                        probability_df.style.format(
                            {"Probability": "{:.2%}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.bar_chart(
                        probability_df.set_index("Class")
                    )

            except Exception as error:

                st.error(f"Visual prediction failed: {error}")

    elif selected_modality == "Thermal Analysis":

        st.subheader("Thermal Hotspot Detection")

        uploaded_thermal = st.file_uploader(
            "Upload a thermal image",
            type=["jpg", "jpeg", "png"],
            key="thermal_upload",
        )

        if uploaded_thermal is not None:

            try:

                preview = Image.open(uploaded_thermal).convert("RGB")

                st.image(
                    preview,
                    caption="Uploaded Thermal Image",
                    use_container_width=True,
                )

                if st.button(
                    "Analyse Thermal Image",
                    type="primary",
                    use_container_width=True,
                ):

                    uploaded_thermal.seek(0)

                    image, probabilities, prediction = predict_image(
                        uploaded_thermal,
                        "thermal",
                    )

                    st.divider()
                    st.subheader("Prediction Result")

                    if prediction == "Hotspot":
                        st.warning("Prediction: Hotspot Detected")
                    else:
                        st.success("Prediction: Normal Thermal Pattern")

                    probability_df = pd.DataFrame(
                        {
                            "Class": list(probabilities.keys()),
                            "Probability": list(probabilities.values()),
                        }
                    )

                    st.dataframe(
                        probability_df.style.format(
                            {"Probability": "{:.2%}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.bar_chart(
                        probability_df.set_index("Class")
                    )

            except Exception as error:

                st.error(f"Thermal prediction failed: {error}")

    st.divider()

    st.caption(
        "Individual predictions are model outputs, not independently "
        "verified structural diagnoses."
    )


# =========================================================
# 15. MULTIMODAL FUSION
# =========================================================

elif page == "Multimodal Fusion":

    st.title("Multimodal Structural Health Assessment")

    st.markdown(
        "Combine vibration, visual and thermal model outputs "
        "to calculate the Structural Stress Index."
    )

    st.info(
        "All three modalities are required for the baseline fusion. "
        "Equal weights are used."
    )

    st.divider()

    st.subheader("Step 1 — Vibration Sensor Input")

    col1, col2 = st.columns(2)

    with col1:

        fusion_accel_x = st.number_input(
            "Acceleration X (m/s²)",
            value=0.15,
            format="%.5f",
            key="fusion_accel_x",
        )

        fusion_accel_y = st.number_input(
            "Acceleration Y (m/s²)",
            value=0.02,
            format="%.5f",
            key="fusion_accel_y",
        )

        fusion_accel_z = st.number_input(
            "Acceleration Z (m/s²)",
            value=9.81,
            format="%.5f",
            key="fusion_accel_z",
        )

    with col2:

        fusion_strain = st.number_input(
            "Strain (με)",
            value=130.0,
            format="%.5f",
            key="fusion_strain",
        )

        fusion_temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
            format="%.5f",
            key="fusion_temperature",
        )

    st.divider()

    st.subheader("Step 2 — Upload Structural Image")

    fusion_visual_file = st.file_uploader(
        "Upload visual structural image",
        type=["jpg", "jpeg", "png"],
        key="fusion_visual_upload",
    )

    if fusion_visual_file is not None:
        st.image(
            fusion_visual_file,
            caption="Visual input",
            width=300,
        )

    st.divider()

    st.subheader("Step 3 — Upload Thermal Image")

    fusion_thermal_file = st.file_uploader(
        "Upload thermal image",
        type=["jpg", "jpeg", "png"],
        key="fusion_thermal_upload",
    )

    if fusion_thermal_file is not None:
        st.image(
            fusion_thermal_file,
            caption="Thermal input",
            width=300,
        )

    st.divider()

    st.subheader("Step 4 — Run Multimodal Fusion")

    if st.button(
        "Run Multimodal Analysis",
        type="primary",
        use_container_width=True,
    ):

        if fusion_visual_file is None or fusion_thermal_file is None:

            st.warning(
                "Please upload both a structural image and a thermal image."
            )

        else:

            try:

                sensor_values = [
                    fusion_accel_x,
                    fusion_accel_y,
                    fusion_accel_z,
                    fusion_strain,
                    fusion_temperature,
                ]

                with st.spinner("Running all three models..."):

                    vibration_result = predict_vibration(sensor_values)

                    vibration_probability = float(
                        np.sum(vibration_result["probabilities"][1:])
                    )

                    _, visual_probabilities, visual_prediction = predict_image(
                        fusion_visual_file,
                        "visual",
                    )

                    visual_probability = float(
                        visual_probabilities["Crack"]
                    )

                    _, thermal_probabilities, thermal_prediction = predict_image(
                        fusion_thermal_file,
                        "thermal",
                    )

                    thermal_probability = float(
                        thermal_probabilities["Hotspot"]
                    )

                    ssi, risk = calculate_ssi(
                        vibration_probability,
                        visual_probability,
                        thermal_probability,
                    )

                # Persist result for the Reports page.
                st.session_state.fusion_result = {
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "sensor_values": sensor_values,
                    "vibration_probability": vibration_probability,
                    "visual_probability": visual_probability,
                    "thermal_probability": thermal_probability,
                    "vibration_class": vibration_result["class_name"],
                    "visual_class": visual_prediction,
                    "thermal_class": thermal_prediction,
                    "ssi": ssi,
                    "risk": risk,
                }

                st.success("All three models completed successfully.")

            except Exception as error:

                st.error(f"Multimodal analysis failed: {error}")

    # Display the latest saved fusion result.
    result = st.session_state.fusion_result

    if result is not None:

        st.divider()
        st.subheader("Individual Modality Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Vibration Abnormality",
            f"{result['vibration_probability'] * 100:.2f}%",
        )
        col1.caption(result["vibration_class"])

        col2.metric(
            "Visual Crack Probability",
            f"{result['visual_probability'] * 100:.2f}%",
        )
        col2.caption(result["visual_class"])

        col3.metric(
            "Thermal Hotspot Probability",
            f"{result['thermal_probability'] * 100:.2f}%",
        )
        col3.caption(result["thermal_class"])

        st.divider()
        st.subheader("Structural Stress Index")

        col1, col2 = st.columns([1, 2])

        col1.metric(
            "Final SSI",
            f"{result['ssi']:.2f} / 100",
        )

        with col2:

            if result["risk"] == "Safe":
                st.success(f"SAFE — SSI: {result['ssi']:.2f}")

            elif result["risk"] == "Moderate Risk":
                st.warning(
                    f"MODERATE RISK — SSI: {result['ssi']:.2f}"
                )

            else:
                st.error(
                    f"HIGH RISK — SSI: {result['ssi']:.2f}"
                )

        st.progress(
            min(max(result["ssi"] / 100, 0.0), 1.0)
        )

        contribution_df = pd.DataFrame(
            {
                "Modality": [
                    "Vibration",
                    "Visual",
                    "Thermal",
                ],
                "Abnormality Probability": [
                    result["vibration_probability"],
                    result["visual_probability"],
                    result["thermal_probability"],
                ],
                "Weight": [1 / 3, 1 / 3, 1 / 3],
                "SSI Contribution": [
                    100 * result["vibration_probability"] / 3,
                    100 * result["visual_probability"] / 3,
                    100 * result["thermal_probability"] / 3,
                ],
            }
        )

        st.markdown("### SSI Contribution Breakdown")

        st.dataframe(
            contribution_df.style.format(
                {
                    "Abnormality Probability": "{:.2%}",
                    "Weight": "{:.2%}",
                    "SSI Contribution": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            contribution_df.set_index("Modality")[
                ["SSI Contribution"]
            ]
        )

        st.markdown("### Calculation")

        st.latex(
            rf"SSI = 100 \times "
            rf"\frac{{{result['vibration_probability']:.4f}"
            rf" + {result['visual_probability']:.4f}"
            rf" + {result['thermal_probability']:.4f}}}{{3}}"
        )

        st.caption(
            f"Latest assessment: {result['timestamp']}"
        )

    st.divider()

    st.warning(
        "The three modality datasets are independent and are not paired "
        "measurements from the same physical structure. This is "
        "scenario-based fusion, not a validated physical stress measurement."
    )


# =========================================================
# 16. REPORTS AND INSIGHTS
# =========================================================

elif page == "Reports and Insights":

    st.title("Reports and Insights")

    st.markdown(
        "Review the latest multimodal assessment, understand its "
        "contributions and export the results."
    )

    st.divider()

    result = st.session_state.fusion_result

    if result is None:

        st.warning(
            "No fusion assessment is available yet."
        )

        st.markdown(
            """
            To generate a report:

            1. Open **Multimodal Fusion**.
            2. Enter the vibration sensor measurements.
            3. Upload a structural image.
            4. Upload a thermal image.
            5. Click **Run Multimodal Analysis**.
            6. Return to this page.
            """
        )

    else:

        st.subheader("Latest Assessment")

        st.caption(
            f"Assessment generated: {result['timestamp']}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Structural Stress Index",
            f"{result['ssi']:.2f} / 100",
        )

        col2.metric(
            "Risk Classification",
            result["risk"],
        )

        col3.metric(
            "Modalities Analysed",
            "3 / 3",
        )

        if result["risk"] == "Safe":
            st.success("The calculated SSI is within the Safe range.")

        elif result["risk"] == "Moderate Risk":
            st.warning(
                "The calculated SSI is within the Moderate Risk range."
            )

        else:
            st.error(
                "The calculated SSI is within the High Risk range."
            )

        st.divider()

        # -------------------------------------------------
        # MODALITY SUMMARY
        # -------------------------------------------------

        st.subheader("Individual Modality Summary")

        modality_df = pd.DataFrame(
            {
                "Modality": [
                    "Vibration",
                    "Visual",
                    "Thermal",
                ],
                "Predicted Class": [
                    result["vibration_class"],
                    result["visual_class"],
                    result["thermal_class"],
                ],
                "Abnormality Probability": [
                    result["vibration_probability"],
                    result["visual_probability"],
                    result["thermal_probability"],
                ],
            }
        )

        st.dataframe(
            modality_df.style.format(
                {"Abnormality Probability": "{:.2%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # CONTRIBUTION SUMMARY
        # -------------------------------------------------

        st.divider()

        st.subheader("SSI Contribution Analysis")

        contribution_df = pd.DataFrame(
            {
                "Modality": [
                    "Vibration",
                    "Visual",
                    "Thermal",
                ],
                "SSI Contribution": [
                    100 * result["vibration_probability"] / 3,
                    100 * result["visual_probability"] / 3,
                    100 * result["thermal_probability"] / 3,
                ],
            }
        )

        st.bar_chart(
            contribution_df.set_index("Modality")
        )

        st.dataframe(
            contribution_df.style.format(
                {"SSI Contribution": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # AUTOMATED INSIGHTS
        # -------------------------------------------------

        st.divider()

        st.subheader("Automated Insights")

        insights = generate_insights(result)

        for index, insight in enumerate(insights, start=1):
            st.markdown(f"{index}. {insight}")

        st.caption(
            "These observations are generated from the model outputs "
            "and the defined SSI thresholds. They are not independent "
            "engineering conclusions."
        )

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        st.divider()

        with st.expander("View vibration input measurements"):

            sensor_names = [
                "Acceleration X (m/s²)",
                "Acceleration Y (m/s²)",
                "Acceleration Z (m/s²)",
                "Strain (με)",
                "Temperature (°C)",
            ]

            sensor_df = pd.DataFrame(
                {
                    "Sensor": sensor_names,
                    "Input Value": result["sensor_values"],
                }
            )

            st.dataframe(
                sensor_df,
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------
        # DOWNLOAD REPORTS
        # -------------------------------------------------

        st.divider()

        st.subheader("Export Assessment")

        csv_report = create_fusion_report(result)

        text_report = create_text_report(result)

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                label="Download CSV Report",
                data=csv_report.to_csv(index=False),
                file_name="structureguard_assessment.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:

            st.download_button(
                label="Download Text Report",
                data=text_report,
                file_name="structureguard_assessment.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # -------------------------------------------------
        # RESEARCH LIMITATIONS
        # -------------------------------------------------

        st.divider()

        st.subheader("Research Limitations")

        st.warning(
            """
            - The vibration, visual and thermal datasets are independent.
            - They are not paired observations from the same physical structure.
            - Equal fusion weights are a baseline assumption, not experimentally
              optimised weights.
            - SSI is an experimental composite indicator, not a calibrated
              physical stress measurement.
            - Model predictions require independent engineering verification
              before any real-world structural decision.
            """
        )

        st.caption(
            "StructureGuard | Multimodal Structural Health Monitoring"
        )