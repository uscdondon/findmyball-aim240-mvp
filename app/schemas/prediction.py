"""Prediction response schemas."""

from pydantic import BaseModel


class Detection(BaseModel):
    x: int
    y: int
    w: int
    h: int
    confidence: float
    method: str


class PredictionResponse(BaseModel):
    image_path: str
    detections: list[Detection]
    count: int

