"""WS /ws/replay/{chan_id}: per-connection simulated real-time replay of the
channel's test split, streaming live detection + forecast. New anomaly-window
transitions are persisted as AnomalyEvent/Alert rows from the same inference
result sent to the client -- single source of truth, no recomputation on write.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.db.database import SessionLocal
from app.db.models import Alert, AnomalyEvent
from app.services import replay_engine

router = APIRouter()

SEVERITY_BY_HEALTH = {
    "warning": "warning",
    "critical": "critical",
    "failure_risk": "failure_risk",
}


@router.websocket("/ws/replay/{chan_id}")
async def replay_ws(websocket: WebSocket, chan_id: str):
    registry = websocket.app.state.model_registry
    try:
        channel_model = registry.get(chan_id)
    except KeyError:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    tick_seconds = settings.replay_tick_ms / 1000.0

    open_event_id: int | None = None
    was_anomaly = False

    try:
        async for point in replay_engine.stream(channel_model, tick_seconds):
            await websocket.send_json(point)

            db = SessionLocal()
            try:
                anomaly_now = point["is_anomaly"]
                if anomaly_now and not was_anomaly:
                    event = AnomalyEvent(
                        chan_id=chan_id,
                        start_index=point["index"],
                        end_index=None,
                        predicted=True,
                        ground_truth=point["ground_truth_anomaly"],
                        peak_score=point["ensemble_score"],
                        health_state=point["health_state"],
                    )
                    db.add(event)
                    db.commit()
                    db.refresh(event)
                    open_event_id = event.id

                    severity = SEVERITY_BY_HEALTH.get(point["health_state"])
                    if severity:
                        db.add(
                            Alert(
                                chan_id=chan_id,
                                event_id=open_event_id,
                                severity=severity,
                                message=(
                                    f"{chan_id}: anomaly detected at index {point['index']} "
                                    f"(health={point['health_state']}, "
                                    f"score={point['ensemble_score']:.2f})"
                                ),
                            )
                        )
                        db.commit()
                elif anomaly_now and was_anomaly and open_event_id is not None:
                    event = db.get(AnomalyEvent, open_event_id)
                    if event:
                        event.peak_score = max(event.peak_score, point["ensemble_score"])
                        db.commit()
                elif not anomaly_now and was_anomaly and open_event_id is not None:
                    event = db.get(AnomalyEvent, open_event_id)
                    if event:
                        event.end_index = point["index"]
                        db.commit()
                    open_event_id = None
                was_anomaly = anomaly_now
            finally:
                db.close()
    except WebSocketDisconnect:
        pass
