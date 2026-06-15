# src/api/schemas.py

from pydantic import BaseModel


class PredictionResponse(BaseModel):

    prediction: str

    fake_probability: float

    file_type: str

    heatmap: str | None = None