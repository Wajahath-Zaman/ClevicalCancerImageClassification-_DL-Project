import io

from fastapi.testclient import TestClient
from PIL import Image

from src.main import app

client = TestClient(app)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Cervical Cytology Classification API is running."
    )


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "ResNet50"


def test_predict_endpoint():

    # Generate a synthetic RGB image for API testing
    image = Image.new("RGB", (512, 512), color="white")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": (
                "test_image.png",
                image_bytes,
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] in [
        "ASCUS", "HSIL", "LSIL", "SCC", "NILM"
    ]

    assert 0 <= result["confidence"] <= 100


def test_invalid_file_format():

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"invalid content",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400