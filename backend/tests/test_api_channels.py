"""Integration tests against the real app: real trained artifacts under
ml/artifacts and real staged data under data/raw, not mocks -- the whole
point is verifying the pipeline actually works end-to-end. Requires
`python -m ml.training.train_all` (and scripts/stage_data.py) to have run
first, same as running the app itself.
"""

import os

os.environ.setdefault("DB_URL", "sqlite:///./test_astrosentinel.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_health_and_channels_endpoints():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["channels_loaded"] == 20

        channels = client.get("/api/channels")
        assert channels.status_code == 200
        data = channels.json()
        assert len(data) == 20
        assert "S-1" in {c["chan_id"] for c in data}

        detail = client.get("/api/channels/S-1")
        assert detail.status_code == 200
        detail_data = detail.json()
        assert detail_data["window_size"] == 100
        assert detail_data["forecast_horizon"] == 15

        missing = client.get("/api/channels/NOT-A-CHANNEL")
        assert missing.status_code == 404


def test_telemetry_and_anomalies_endpoints():
    with TestClient(app) as client:
        telemetry = client.get("/api/channels/S-1/telemetry?start=0&end=10")
        assert telemetry.status_code == 200
        points = telemetry.json()["points"]
        assert len(points) == 11
        assert points[0]["index"] == 0

        anomalies = client.get("/api/channels/S-1/anomalies")
        assert anomalies.status_code == 200
        assert len(anomalies.json()["ground_truth"]) >= 1


def test_forecast_and_explain_endpoints():
    with TestClient(app) as client:
        forecast = client.get("/api/channels/S-1/forecast")
        assert forecast.status_code == 200
        assert len(forecast.json()["forecast_values"]) == 15

        explain = client.get("/api/channels/S-1/explain?index=5500")
        assert explain.status_code == 200
        explain_data = explain.json()
        assert len(explain_data["feature_names"]) == 6
        assert len(explain_data["shap_values"]) == 6
