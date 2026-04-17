import streamlit as st
import requests
from PIL import Image
import io
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
CM_PATH = BASE_DIR / "outputs" / "confusion_matrix.json"
Y_TRUE_PATH = BASE_DIR / "outputs" / "y_true.npy"
Y_PROBS_PATH = BASE_DIR / "outputs" / "y_probs.npy"

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OncoVision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("👁️ OncoVision AI")
st.markdown("### Breast Cancer Detection with Explainability")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
col1, col2, col3, col4 = st.columns(4)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.write("### Upload Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {
            "file": uploaded_file.getvalue()
        }

        with st.spinner("Analyzing image... Please wait"):
            # Prediction
            try:
                pred_response = requests.post(f"{API_URL}/predict", files=files)
                pred_data = pred_response.json()
                pred_label = pred_data["prediction"]
            except:
                st.error("FastAPI server is not running. Please start it first and try again.")
                st.stop()

        with col2:
            st.write("### Prediction Results: ")
            st.success(f"Prediction: {pred_label}")
            st.info(f"Confidence Score: {pred_data['confidence']*100:.2f}%")

        with col3:
            st.write("### Class Probabilities: ")
            probs = pred_data["probabilities"]
            class_names = ["benign", "malignant", "normal"]

            for i, prob in enumerate(probs):
                label = class_names[i]

                if label == pred_label:
                    st.progress(float(prob), text=f"👉 {label}: {prob*100:.2f}%")
                else:
                    st.progress(float(prob), text=f"{label}: {prob*100:.2f}%")

        with col4:
            # Model Performance
            st.write("### Model Performance: ")

            if METRICS_PATH.exists():
                with open(METRICS_PATH, "r") as f:
                    metrics = json.load(f)

                st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                st.metric("Precision", f"{metrics['precision']*100:.2f}%")
                st.metric("Recall", f"{metrics['recall']*100:.2f}%")
                st.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")
            else:
                st.warning("Metrics file not found. Run evaluation first.")

        st.markdown("---")

        col5, col6, col7 = st.columns(3)

        with col5:
            # Confusion Metrics
            st.write("### Confusion Matrix: ")

            if CM_PATH.exists():
                with open(CM_PATH, "r") as f:
                    cm = np.array(json.load(f))

                class_names = ["benign", "malignant", "normal"]

                fig, ax = plt.subplots()

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax
                )

                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("Actual labels")
                ax.set_title("Confusion Matrix")

                st.pyplot(fig)
            else:
                st.warning("Confusion matrix not found. Run evaluation first.")

        with col6:
            # ROC Curve
            st.write("### ROC Curve: ")

            if Y_TRUE_PATH.exists() and Y_PROBS_PATH.exists():
                y_true = np.load(Y_TRUE_PATH)
                y_probs = np.load(Y_PROBS_PATH)

                n_classes = y_probs.shape[1]
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

                fig, ax = plt.subplots()

                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                    roc_auc = auc(fpr, tpr)

                    ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

                ax.plot([0, 1], [0, 1], linestyle="--")

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()

                st.pyplot(fig)
            else:
                st.warning("ROC data not found. Run evaluation first.")

        with col7:
            # Grad-CAM
            explain_response = requests.post(f"{API_URL}/explain", files=files)

            heatmap = Image.open(io.BytesIO(explain_response.content))

            st.write("### Explainability Heatmap (Grad-CAM): ")
            st.image(heatmap, caption="Grad-CAM", use_container_width=True)