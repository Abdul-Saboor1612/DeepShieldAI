# src/explainability/video_evidence.py

from src.explainability.gradcam import (
    get_gradcam
)

from src.explainability.heatmap_generator import (
    generate_heatmap
)


def generate_video_evidence(
    frame_path,
    model,
    device
):

    cam = get_gradcam(
        model
    )

    output_path = (
        "outputs/evidence/video_heatmap.jpg"
    )

    generate_heatmap(
        frame_path,
        cam,
        device,
        output_path
    )

    return output_path