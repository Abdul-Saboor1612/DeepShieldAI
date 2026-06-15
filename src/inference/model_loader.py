# src/inference/model_loader.py

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def load_model(model_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = efficientnet_b0(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    model.to(device)

    return model, device