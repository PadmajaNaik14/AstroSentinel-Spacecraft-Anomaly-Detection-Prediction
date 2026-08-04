from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Channel(Base):
    __tablename__ = "channels"

    chan_id: Mapped[str] = mapped_column(String, primary_key=True)
    spacecraft: Mapped[str] = mapped_column(String)
    anomaly_class: Mapped[str] = mapped_column(String)
    num_values: Mapped[int] = mapped_column(Integer)


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chan_id: Mapped[str] = mapped_column(String, ForeignKey("channels.chan_id"))
    start_index: Mapped[int] = mapped_column(Integer)
    end_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted: Mapped[bool] = mapped_column(Boolean, default=True)
    ground_truth: Mapped[bool] = mapped_column(Boolean, default=False)
    peak_score: Mapped[float] = mapped_column(Float)
    health_state: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chan_id: Mapped[str] = mapped_column(String, ForeignKey("channels.chan_id"))
    event_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("anomaly_events.id"), nullable=True)
    severity: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(String)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
