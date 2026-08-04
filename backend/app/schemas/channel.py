from pydantic import BaseModel


class ChannelSummary(BaseModel):
    chan_id: str
    spacecraft: str
    anomaly_class: str
    num_values: int
    eval_f1: float


class ChannelDetail(ChannelSummary):
    window_size: int
    forecast_horizon: int
    ae_threshold: float
    eval_precision: float
    eval_recall: float
    forecast_mae: float | None
    trained_at: str
