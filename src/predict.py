import torch

from PIL import Image
from src.preprocessing import get_preprocessing


def predict_image(model_instance, image):

    # Load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Apply preprocessing
    preprocess = get_preprocessing(
        model_instance.image_size
    )

    image_tensor = preprocess(image).unsqueeze(0)

    image_tensor = image_tensor.to(model_instance.device)

    # Model inference
    model_instance.model.eval()

    with torch.no_grad():

        outputs = model_instance.model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    predicted_label = model_instance.class_names[
        predicted_class.item()
    ]

    return {
        "predicted_class": predicted_label,
        "confidence": round(confidence.item() * 100, 2)
    }