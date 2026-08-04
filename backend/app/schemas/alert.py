from datetime import datetime

from pydantic import BaseModel


class AlertOut(BaseModel):
    id: int
    chan_id: str
    event_id: int | None
    severity: str
    message: str
    acknowledged: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertListResponse(BaseModel):
    items: list[AlertOut]
    total: int
