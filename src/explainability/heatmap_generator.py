# src/explainability/heatmap_generator.py

import cv2
import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


def generate_heatmap(
    image_path,
    cam,
    device,
    output_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )

    grayscale_cam = cam(
        input_tensor=image_tensor
    )[0]

    rgb_img = np.array(
        image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return output_path