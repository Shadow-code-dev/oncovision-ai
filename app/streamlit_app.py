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

st.title("OncoVision AI")
st.subheader("Breast Cancer Detection with Explainability")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {
            "file": uploaded_file.getvalue()
        }

        # Prediction
        try:
            pred_response = requests.post(f"{API_URL}/predict", files=files)
            pred_data = pred_response.json()
        except:
            st.error("FastAPI server is not running. Please start it first and try again.")
            st.stop()

        st.write("### Prediction: ")
        st.write(pred_data)

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

                ax.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

            ax.plot([0, 1], [0, 1], linestyle="--")

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()

            st.pyplot(fig)
        else:
            st.warning("ROC data not found. Run evaluation first.")

        # Grad-CAM
        explain_response = requests.post(f"{API_URL}/explain", files=files)

        heatmap = Image.open(io.BytesIO(explain_response.content))

        st.write("### Explainability Heatmap (Grad-CAM): ")
        st.image(heatmap, caption="Grad-CAM", use_container_width=True)