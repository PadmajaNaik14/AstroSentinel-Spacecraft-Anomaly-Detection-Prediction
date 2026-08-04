from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnomalyEvent
from app.schemas.telemetry import (
    AnomaliesResponse,
    AnomalyEventOut,
    AnomalyWindow,
    TelemetryPoint,
    TelemetryResponse,
)
from app.services import batch_inference, replay_engine
from ml.data.labels import get_anomaly_windows

router = APIRouter()

DEFAULT_WINDOW = 500


@router.get("/api/channels/{chan_id}/telemetry", response_model=TelemetryResponse)
def get_telemetry(
    chan_id: str,
    request: Request,
    start: int | None = Query(default=None, ge=0),
    end: int | None = Query(default=None, ge=0),
):
    registry = request.app.state.model_registry
    try:
        channel_model = registry.get(chan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel {chan_id}")

    n = replay_engine.get_test_data(chan_id).shape[0]
    end = (n - 1) if end is None else min(end, n - 1)
    start = max(0, end - DEFAULT_WINDOW) if start is None else max(0, start)
    if start > end:
        start = end

    points = batch_inference.get_series_slice(channel_model, start, end)
    return TelemetryResponse(chan_id=chan_id, points=[TelemetryPoint(**p) for p in points])


@router.get("/api/channels/{chan_id}/anomalies", response_model=AnomaliesResponse)
def get_anomalies(chan_id: str, request: Request, db: Session = Depends(get_db)):
    registry = request.app.state.model_registry
    try:
        registry.get(chan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel {chan_id}")

    ground_truth = [AnomalyWindow(**w) for w in get_anomaly_windows(chan_id)]
    predicted = (
        db.query(AnomalyEvent)
        .filter(AnomalyEvent.chan_id == chan_id)
        .order_by(AnomalyEvent.start_index)
        .all()
    )
    predicted_out = [
        AnomalyEventOut(
            id=e.id,
            start_index=e.start_index,
            end_index=e.end_index,
            peak_score=e.peak_score,
            health_state=e.health_state,
        )
        for e in predicted
    ]
    return AnomaliesResponse(chan_id=chan_id, ground_truth=ground_truth, predicted=predicted_out)
