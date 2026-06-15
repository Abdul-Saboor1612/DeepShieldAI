# src/explainability/gradcam.py

from pytorch_grad_cam import GradCAM


def get_gradcam(model):

    target_layers = [
        model.features[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    return cam