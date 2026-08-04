"""AstroSentinel backend entrypoint. Loads all trained model artifacts and
initializes the DB at startup (lifespan) -- fails fast if artifacts are
missing, forcing the training pre-step to actually happen.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import alerts, channels, explain, forecast, health, telemetry
from app.config import settings
from app.db.database import SessionLocal, init_db
from app.db.models import Channel
from app.services.model_registry import ModelRegistry
from app.ws import replay
from ml.config import ARTIFACTS_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    registry = ModelRegistry.load(ARTIFACTS_DIR)
    app.state.model_registry = registry

    db = SessionLocal()
    try:
        for chan_id in registry.list_channel_ids():
            cm = registry.get(chan_id)
            if db.get(Channel, chan_id) is None:
                db.add(
                    Channel(
                        chan_id=chan_id,
                        spacecraft=cm.spacecraft,
                        anomaly_class=cm.anomaly_class,
                        num_values=cm.metadata["test_length"],
                    )
                )
        db.commit()
    finally:
        db.close()

    yield


app = FastAPI(title="AstroSentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(channels.router)
app.include_router(telemetry.router)
app.include_router(forecast.router)
app.include_router(alerts.router)
app.include_router(explain.router)
app.include_router(replay.router)
