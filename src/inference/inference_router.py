# src/inference/inference_router.py

import os

from src.inference.image_inference import (
    analyze_image
)

from src.inference.video_inference import (
    analyze_video
)


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


def route_inference(
    file_path,
    model,
    device
):

    ext = os.path.splitext(
        file_path
    )[1].lower()

    if ext in IMAGE_EXTENSIONS:

        return analyze_image(
            file_path,
            model,
            device
        )

    elif ext in VIDEO_EXTENSIONS:

        return analyze_video(
            file_path,
            model,
            device
        )

    else:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )