from pydantic import BaseModel


class ForecastResponse(BaseModel):
    chan_id: str
    base_index: int
    forecast_values: list[float]
    predicted_steps_to_anomaly: int | None
