import torch
import torch.nn as nn

from torchvision.models import resnet50
from pathlib import Path


# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "resnet50_cervical_cytology_best.pth"
)


class CervicalCytologyModel:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model artifact
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )

        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint["image_size"]

        self.normalization = checkpoint["normalization"]

        # Reconstruct ResNet50
        self.model = resnet50(weights=None)

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            checkpoint["num_classes"]
        )

        # Restore trained weights
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        print("ResNet50 loaded successfully.")
        print("Device:", self.device)
        print("Classes:", self.class_names)
        print("Image size:", self.image_size)