## Current Features

- Deepfake image detection
- Deepfake video detection
- Face detection
- Grad-CAM explainability
- Video evidence generation
- FastAPI backend
- Swagger documentation

## API Endpoints

POST /predict/image

POST /predict/video

----
## Trained Models

The trained DeepFakeGuard models are hosted separately as Kaggle datasets.

### Available Models

| Model   | Description                                          |
| ------- | ---------------------------------------------------- |
| EXP_002 | EfficientNet-B0 trained on FaceForensics++           |
| EXP_007 | EfficientNet-B0 trained on FaceForensics++ + CelebDF |

### Download

Kaggle Dataset:

https://www.kaggle.com/datasets/absaboor/deepfakeguard-models

### Usage

Place the downloaded model inside:

```text
models/
```

Example:

```text
models/
├── exp002_best_model.pth
└── exp007_best_model.pth
```

Then update the model path in the configuration or inference script accordingly.
