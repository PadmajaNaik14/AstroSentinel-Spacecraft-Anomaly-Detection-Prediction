from pydantic import BaseModel


class ExplainResponse(BaseModel):
    chan_id: str
    index: int
    feature_names: list[str]
    shap_values: list[float]
    feature_values: list[float]
    reconstruction_error_heatmap: list[float]
