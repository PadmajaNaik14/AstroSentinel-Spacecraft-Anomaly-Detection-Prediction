from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Alert
from app.schemas.alert import AlertListResponse, AlertOut

router = APIRouter()


@router.get("/api/alerts", response_model=AlertListResponse)
def list_alerts(
    chan_id: str | None = None,
    severity: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(Alert)
    if chan_id:
        query = query.filter(Alert.chan_id == chan_id)
    if severity:
        query = query.filter(Alert.severity == severity)
    total = query.count()
    items = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()
    return AlertListResponse(items=[AlertOut.model_validate(i) for i in items], total=total)


@router.post("/api/alerts/{alert_id}/acknowledge", response_model=AlertOut)
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)
    return AlertOut.model_validate(alert)
