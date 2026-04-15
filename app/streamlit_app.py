import streamlit as st
import requests
from PIL import Image
import io

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

        # Grad-CAM
        explain_response = requests.post(f"{API_URL}/explain", files=files)

        heatmap = Image.open(io.BytesIO(explain_response.content))

        st.write("### Explainability Heatmap (Grad-CAM): ")
        st.image(heatmap, caption="Grad-CAM", use_container_width=True)