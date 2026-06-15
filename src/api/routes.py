# src/api/routes.py

import os
import shutil

from fastapi import (
    APIRouter,
    UploadFile,
    File
)

from src.inference.model_loader import (
    load_model
)

from src.inference.image_inference import (
    analyze_image
)

from src.inference.video_inference import (
    analyze_video
)

router = APIRouter()

MODEL_PATH = (
    "models/exp002_best_model.pth"
)

model, device = load_model(
    MODEL_PATH
)


@router.get("/")
def home():

    return {
        "message":
        "DeepFakeGuard API Running"
    }

#image prediction
@router.post("/predict/image")
async def predict_image_api(
    file: UploadFile = File(...)
):

    save_path = os.path.join(
        "uploads/images",
        file.filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = analyze_image(
        save_path,
        model,
        device
    )

    return result

#video prediction
@router.post("/predict/video")
async def predict_video_api(
    file: UploadFile = File(...)
):

    save_path = os.path.join(
        "uploads/videos",
        file.filename
    )

    with open(
        save_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = analyze_video(
        save_path,
        model,
        device
    )

    return result