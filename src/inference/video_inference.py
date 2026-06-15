# src/inference/video_inference.py

import os

from src.inference.frame_extractor import (
    extract_frames
)

from src.inference.predictor import (
    predict_image
)

from src.explainability.video_evidence import (
    generate_video_evidence
)


def analyze_video(
    video_path,
    model,
    device
):

    frames_dir = "outputs/frames"

    frame_count = extract_frames(
        video_path,
        frames_dir
    )

    if frame_count == 0:

        return {
            "type": "video",
            "frames_analyzed": 0,
            "fake_frames": 0,
            "fake_probability": 0.0,
            "prediction": "NO_FRAMES",
            "most_suspicious_frame": None,
            "highest_probability": 0.0
        }

    frame_files = sorted([
        f
        for f in os.listdir(frames_dir)
        if f.endswith(".jpg")
    ])

    fake_frames = 0
    probabilities = []

    highest_probability = 0.0
    most_suspicious_frame = None

    for frame_file in frame_files:

        frame_path = os.path.join(
            frames_dir,
            frame_file
        )

        result = predict_image(
            frame_path,
            model,
            device
        )

        probabilities.append(
            result["fake_probability"]
        )

        if result["prediction"] == "FAKE":
            fake_frames += 1

        if (
            result["fake_probability"]
            > highest_probability
        ):

            highest_probability = (
                result["fake_probability"]
            )

            most_suspicious_frame = (
                frame_path
            )

    avg_probability = (
        sum(probabilities)
        / len(probabilities)
    )

    prediction = (
        "FAKE"
        if fake_frames >= len(frame_files) / 2
        else "REAL"
    )
    evidence_path = None

    if most_suspicious_frame:

        evidence_path = (
            generate_video_evidence(
                most_suspicious_frame,
                model,
                device
            )
        )

    return {
        "type": "video",
        "frames_analyzed": len(frame_files),
        "fake_frames": fake_frames,
        "fake_probability": avg_probability,
        "prediction": prediction,
        "most_suspicious_frame":
            most_suspicious_frame,
        "highest_probability":
            highest_probability,
        "evidence_heatmap":
            evidence_path
    }