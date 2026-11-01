"""This module will handle image resizing, padding, tensor conversion and normalization."""

from PIL import Image
from torchvision import transforms


class ResizeWithPadding:

    def __init__(self, size):
        self.size = size

    def __call__(self, image):

        width, height = image.size

        scale = min(
            self.size / width,
            self.size / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0)
        )

        x_offset = (self.size - new_width) // 2
        y_offset = (self.size - new_height) // 2

        canvas.paste(resized, (x_offset, y_offset))

        return canvas


def get_preprocessing(image_size=512):

    return transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])