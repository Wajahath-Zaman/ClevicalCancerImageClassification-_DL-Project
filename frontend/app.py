import os
import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cervical Cytology Classifier",
    page_icon="🔬",
    layout="centered"
)

# --------------------------------------------------
# Application UI
# --------------------------------------------------

st.title("🔬 Cervical Cytology Image Classifier")

st.write(
    "Upload a cervical cytology image to obtain an "
    "experimental classification using a ResNet50 model."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Image Preview")
    st.image(image, use_container_width=True)

    if st.button("Classify Image", type="primary"):

        with st.spinner("Analyzing image..."):

            try:
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                st.success("Classification completed!")

                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Class",
                        result["predicted_class"]
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f'{result["confidence"]}%'
                    )

                st.info(result["disclaimer"])

            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to connect to the prediction API. "
                    "Please ensure the FastAPI backend is running."
                )

            except requests.exceptions.Timeout:
                st.error("The prediction request timed out.")

            except requests.exceptions.HTTPError as error:
                st.error(f"API error: {error}")

            except Exception as error:
                st.error(f"Unexpected error: {error}")

st.divider()

st.caption(
    "Research and educational purposes only. "
    "This application does not provide medical diagnoses."
)