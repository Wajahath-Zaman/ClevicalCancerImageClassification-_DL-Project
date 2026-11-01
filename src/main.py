from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from io import BytesIO

from src.model_loader import CervicalCytologyModel
from src.predict import predict_image


# Initialize FastAPI
app = FastAPI(
    title="Cervical Cytology Classification API",
    description="Deep learning API for cervical cytology image classification using ResNet50.",
    version="1.0.0"
)


# Load model once during application startup
model_instance = CervicalCytologyModel()


@app.get("/")
def home():
    return {
        "message": "Cervical Cytology Classification API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "ResNet50",
        "device": str(model_instance.device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}

    filename = file.filename or ""
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload JPG, JPEG or PNG."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        image = Image.open(BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    result = predict_image(
        model_instance,
        image
    )

    return {
        "filename": filename,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "disclaimer": "Experimental classification output, not a medical diagnosis."
    }