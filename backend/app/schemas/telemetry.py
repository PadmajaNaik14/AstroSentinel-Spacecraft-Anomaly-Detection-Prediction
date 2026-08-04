from pydantic import BaseModel


class TelemetryPoint(BaseModel):
    index: int
    value: float
    reconstruction_error: float
    ensemble_score: float
    is_anomaly: bool
    health_state: str
    ground_truth_anomaly: bool
    forecast: list[float] | None = None
    predicted_steps_to_anomaly: int | None = None


class TelemetryResponse(BaseModel):
    chan_id: str
    points: list[TelemetryPoint]


class AnomalyWindow(BaseModel):
    start: int
    end: int
    anomaly_class: str


class AnomalyEventOut(BaseModel):
    id: int
    start_index: int
    end_index: int | None
    peak_score: float
    health_state: str


class AnomaliesResponse(BaseModel):
    chan_id: str
    ground_truth: list[AnomalyWindow]
    predicted: list[AnomalyEventOut]
