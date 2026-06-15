# src/inference/predictor.py

import torch
import cv2
from PIL import Image

from torchvision import transforms

from src.inference.face_detector import (
    extract_face
)

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def predict_image(
    image_path,
    model,
    device
):

    face = extract_face(image_path)

    if face is None:

        return {
            "prediction": "NO_FACE",
            "fake_probability": 0.0
        }

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        face_rgb
    )

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        fake_probability = (
            probabilities[0][1]
            .item()
        )

        prediction = (
            "FAKE"
            if fake_probability >= 0.5
            else "REAL"
        )

    return {
        "prediction": prediction,
        "fake_probability": fake_probability
    }