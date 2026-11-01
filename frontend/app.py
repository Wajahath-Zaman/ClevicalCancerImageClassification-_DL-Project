import os
import requests
import streamlit as st
from PIL import Image
from pathlib import Path

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Sample image directory
SAMPLE_DIR = Path(__file__).parent / "sample_images"

# Page configuration
st.set_page_config(
    page_title="Cervical Cytology Classifier",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Cervical Cytology Image Classifier")

st.write(
    "An experimental cervical cytology image classification "
    "application powered by a ResNet50 deep learning model."
)

st.divider()

# Select input method
input_method = st.radio(
    "Choose an image source:",
    ["Sample Image", "Upload Your Own Image"],
    horizontal=True
)

image_bytes = None
image_name = None

# --------------------------------------------------
# OPTION 1: PRELOADED SAMPLE IMAGES
# --------------------------------------------------

if input_method == "Sample Image":

    class_names = ["ASCUS", "HSIL", "LSIL", "SCC", "NILM"]

    selected_class = st.selectbox(
        "Select a diagnostic class:",
        class_names
    )

    class_directory = SAMPLE_DIR / selected_class

    if class_directory.exists():

        sample_images = sorted([
            path for path in class_directory.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if sample_images:

            selected_image = st.selectbox(
                "Select a sample image:",
                sample_images,
                format_func=lambda path: path.name
            )

            image_bytes = selected_image.read_bytes()
            image_name = selected_image.name

        else:
            st.error("No sample images found in this class folder.")

    else:
        st.error(f"Sample directory not found: {class_directory}")

# --------------------------------------------------
# OPTION 2: USER IMAGE UPLOAD
# --------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload a cervical cytology image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_name = uploaded_file.name

# --------------------------------------------------
# IMAGE PREVIEW
# --------------------------------------------------

if image_bytes is not None:

    st.subheader("Image Preview")

    st.image(
        image_bytes,
        caption=image_name,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    if st.button("Classify Image", type="primary"):

        with st.spinner("Analyzing image..."):

            try:

                files = {
                    "file": (
                        image_name,
                        image_bytes,
                        "image/png"
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
                    "Please ensure the backend is running."
                )

            except requests.exceptions.Timeout:

                st.error("The prediction request timed out.")

            except requests.exceptions.HTTPError as error:

                st.error(f"API error: {error}")

            except Exception as error:

                st.error(f"Unexpected error: {error}")

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "Research and educational purposes only. "
    "This application does not provide medical diagnoses."
)