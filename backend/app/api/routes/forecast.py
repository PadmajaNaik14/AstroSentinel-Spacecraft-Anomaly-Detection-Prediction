from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas.forecast import ForecastResponse
from app.services.forecast_service import get_forecast

router = APIRouter()


@router.get("/api/channels/{chan_id}/forecast", response_model=ForecastResponse)
def forecast(chan_id: str, request: Request, index: int | None = Query(default=None, ge=0)):
    registry = request.app.state.model_registry
    try:
        channel_model = registry.get(chan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel {chan_id}")

    result = get_forecast(channel_model, index=index)
    return ForecastResponse(chan_id=chan_id, **result)
