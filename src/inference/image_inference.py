# src/inference/image_inference.py

from src.inference.predictor import (
    predict_image
)

from src.explainability.gradcam import (
    get_gradcam
)

from src.explainability.heatmap_generator import (
    generate_heatmap
)


def analyze_image(
    image_path,
    model,
    device
):

    result = predict_image(
        image_path,
        model,
        device
    )

    cam = get_gradcam(
        model
    )

    heatmap_path = (
        "outputs/heatmaps/image_heatmap.jpg"
    )

    generate_heatmap(
        image_path,
        cam,
        device,
        heatmap_path
    )

    return {
        "type": "image",
        "prediction":
            result["prediction"],
        "fake_probability":
            result["fake_probability"],
        "heatmap":
            heatmap_path
    }