# AstroSentinel — Spacecraft Telemetry Anomaly Detection & Prediction

## Context

AstroSentinel started as a BIT (Bangalore Institute of Technology) CSE major project academic report describing an ambitious, thesis-style scope (LSTM Autoencoder + Isolation Forest + TCN/Transformer forecasting + SHAP explainability, real-time monitoring, RBAC for Operator/Analyst/Admin, SMS/email alerts, deep-space autonomous operation). This document is the actual, buildable engineering plan: a real frontend + backend, containerized with Docker, with clear scope boundaries and open questions instead of silent assumptions.

The dataset is the NASA SMAP/MSL telemetry anomaly benchmark (the "Telemanom" dataset, Hundman et al.), downloaded locally via `scripts/download_dataset.py` (kagglehub). Verified structure:
- `labeled_anomalies.csv` — `chan_id, spacecraft, anomaly_sequences, class, num_values` (list columns are stringified Python lists, parsed via `ast.literal_eval`); 82 channels total across SMAP and MSL.
- `data/data/train/<chan_id>.npy` and `data/data/test/<chan_id>.npy` — shape `(T, 25)` float64 per channel; column 0 = real sensor value, columns 1–24 = one-hot command/event context (constant 0 in train).
- `data/data/2018-05-19_15.00.10/models/*.h5` — legacy pretrained Telemanom reference models, not used.

Confirmed architecture/scope decisions:
1. **Frontend/backend split**: FastAPI backend + React (Vite+TS) frontend, REST + WebSocket, fully decoupled and containerized (not a Streamlit monolith).
2. **ML scope**: LSTM Autoencoder (primary anomaly detector) + Isolation Forest (secondary/ensemble) + SHAP explainability + a **GRU-based multi-step forecaster for failure prediction** (a lighter but genuine deep-learning alternative to a full Transformer/TCN stack).
3. **Channel coverage**: **10 curated channels** (5 SMAP + 5 MSL, mixed point/contextual/mixed anomaly types), verified against the real `labeled_anomalies.csv`. Reduced from an initial 20 to cut training/serving time. Architecture stays generic so more channels are a config change + retrain, not a redesign.

"Real-time monitoring" is reinterpreted honestly: there is no live spacecraft feed, so the backend **simulates** real-time by replaying each channel's held-out, labeled test split over a WebSocket at an accelerated cadence, running live inference (detection + forecast) as it streams.

---

## What AstroSentinel actually is

AstroSentinel monitors spacecraft subsystem telemetry (power, thermal, propulsion, communication — represented here by the SMAP/MSL benchmark's per-subsystem channels) and does two things traditional threshold monitoring can't: **detects** abnormal behavior that doesn't cross a simple fixed limit, and **predicts** it happening a little before it does.

- **Detection**: An **LSTM Autoencoder** learns what "normal" looks like per channel purely from historical (train-split) data, by learning to reconstruct it. When live data doesn't reconstruct well (high reconstruction error), that's the anomaly signal. An **Isolation Forest** cross-checks this using rolling statistics of that error stream as a second, independent opinion.
- **Prediction**: A **GRU-based forecaster** looks at the current window of telemetry and predicts the next ~15 steps ahead. Those forecasted future values are run back through the same trained LSTM Autoencoder to get a *predicted* reconstruction-error trajectory — if that projected error is on track to cross the channel's anomaly threshold within the forecast horizon, the system surfaces a "Failure Prediction" countdown (e.g. "predicted anomaly in ~8 steps") *before* it actually happens. This reuses the detector's own threshold logic instead of inventing a separate definition of "anomalous forecast," so detection and prediction stay consistent with each other.
- **Explainability**: **SHAP** (`TreeExplainer` on the Isolation Forest) explains *why* a moment is anomalous (which rolling-error features drove the decision); a separate reconstruction-error heatmap shows *which timesteps* the autoencoder struggled with.
- **Severity**: a derived health state (Normal → Warning → Critical → Failure Risk) combines the current detector output with the forecaster's countdown.

All of this is exposed through a live dashboard (React) backed by a real API (FastAPI): a user watches simulated "live" telemetry stream in, sees anomalies get flagged and explained as they happen, sees a forward-looking forecast/risk panel, and reviews historical alerts.

---

## Feature scope

### In scope (v1)
- Data pipeline: load NASA SMAP/MSL channels, per-channel train-split normalization, sliding-window generation, ground-truth anomaly label parsing from `labeled_anomalies.csv`
- **LSTM Autoencoder** (PyTorch) trained per-channel on the "normal" train split; reconstructs the value channel from a 100-step window conditioned on the one-hot context columns
- Reconstruction-error → EWMA-smoothed anomaly score → per-channel dynamic threshold (grid-searched `μ + z·σ` on train-split residuals — a documented simplification of Telemanom's full non-parametric pruning)
- **Isolation Forest** secondary detector on 6 engineered rolling-error features; combined with the AE via an OR rule for binary anomaly flags, and a weighted blend for the continuous dashboard score
- **GRU multi-step forecaster** (failure-prediction module): predicts the next `H=15` steps of raw telemetry per channel; forecast is scored through the existing AE+threshold to produce a **predicted-time-to-anomaly**, shown as its own dashboard module (forecast chart + countdown/"no risk in horizon" indicator)
- **SHAP** (`TreeExplainer` on the Isolation Forest's engineered features) for genuine feature-attribution explanations, plus a clearly-separate reconstruction-error-per-timestep heatmap (labeled as such, not mislabeled as SHAP)
- Derived health-state classification (Stable/Normal/Warning/Critical/Failure Risk) combining current detection + forecast countdown
- **10 curated channels** (5 SMAP + 5 MSL, mixed point/contextual/mixed classes — see list below), config-driven and extensible
- FastAPI backend: REST endpoints (channels, telemetry, anomalies, forecast, alerts, explain, health) + WebSocket simulated real-time replay carrying live detection + forecast together, SQLite persistence of anomaly events/alerts
- React (Vite+TS) frontend: dashboard health grid, channel detail view (live chart + anomaly overlay + score gauge + forecast panel + SHAP panel), alerts log page — Recharts, accessible/theme-aware color system
- Docker: separate `backend`/`frontend` Dockerfiles, a one-shot `trainer` service/profile producing all model artifacts, wired together with `docker-compose.yml`
- Clean modular repo layout with gitignored data/artifacts

### Explicitly out of scope (with reasons)
- **Live/real spacecraft integration** — no real telemetry source exists; this is a historical-benchmark replay system by design.
- **SMS/email alerting** — needs third-party provider credentials/cost with no architectural payoff for a local demo; alerts live in the DB + UI only.
- **Multi-role RBAC (Operator/Analyst/Admin)** — no real multi-user need for a single-demo academic project.
- **GPU training clusters** — 10 small per-channel models train fine on a CPU in minutes.
- **Deep-space autonomous operation** — this is a monitoring/analysis dashboard, not onboard flight software.
- **All 82 channels** — a curated, verified 10-channel subset is used for v1; extending coverage is a config change + retrain.
- **Full Transformer / deep dilated TCN forecaster** — replaced by the lighter GRU-based multi-step forecaster, which delivers the same "predict failures before they happen" capability at a fraction of the cost. A heavier model is a valid future upgrade to the same forecast interface.

### Curated 10-channel list (verified against the real `labeled_anomalies.csv`)

| Spacecraft | Channels |
|---|---|
| SMAP (5) | S-1 (point), D-1 (point), P-1 (contextual), A-2 (contextual), T-1 (point+contextual) |
| MSL (5) | T-4 (point), P-11 (point), F-7 (contextual), M-1 (contextual), C-1 (point+contextual) |

Reduced from an original 20-channel set (dropped: E-2, A-1, G-1, P-4, B-1, M-6, S-2, F-4, T-12, D-16) purely to cut training/serving time — no change in method, just fewer channels.

Defined in `ml/ml/config.py::CURATED_CHANNELS`.

---

## Architecture

See `README.md` for the repo layout and run instructions. Summary:

- **`ml/`** — data loading/windowing, LSTM Autoencoder, GRU forecaster, Isolation Forest, thresholding, SHAP, training/evaluation pipeline. Produces `ml/artifacts/<chan_id>/*` + `manifest.json`.
- **`backend/`** — FastAPI app serving REST + WebSocket, SQLite persistence, loads trained artifacts at startup.
- **`frontend/`** — React (Vite+TS) dashboard consuming the backend API.
- **`docker/` + `docker-compose.yml`** — one-shot trainer + backend + frontend containers.

### ML pipeline details
- **Windowing**: 100-step sliding windows; train-split-only normalization for the value column; one-hot context columns used as-is.
- **LSTM Autoencoder**: `LSTM(25→64)→LSTM(64→32)` encoder (latent 32) → decoder reconstructs only the value channel across 100 steps. MSE loss, Adam, trained only on the train split.
- **Threshold**: EWMA-smoothed squared reconstruction error; per-channel `threshold = μ + z·σ` on train-split residuals, `z` grid-searched (2–5) to maximize F1 against `labeled_anomalies.csv`.
- **Isolation Forest**: 6 engineered rolling-error features (current error, rolling mean/std/max, slope, z-score vs. train baseline); combined with AE via OR (binary) + weighted blend (continuous score) in `ml/ml/models/ensemble.py` (single source of truth, imported by backend too).
- **GRU forecaster**: `GRU(25→64)→GRU(64→32)` → `Dense(32→15)` predicting the next 15 raw value steps directly. Forecast is appended to the current window and scored through the trained AE + threshold to derive `predicted_steps_to_anomaly`.
- **SHAP**: `TreeExplainer` on the Isolation Forest's 6 engineered features only — this is the only thing labeled "SHAP" in the UI. LSTM timestep attribution is a separate, explicitly-labeled reconstruction-error heatmap.
- **Health state**: Stable/Normal/Warning/Critical/Failure Risk — derived rule over score vs. threshold, escalated if forecast countdown is near-term, with a persistence rule for the most severe state.
- **Artifacts**: `ml/artifacts/<chan_id>/{lstm_ae.pt, gru_forecaster.pt, isolation_forest.joblib, scaler.joblib, threshold.json, metadata.json}` + `manifest.json`.

### Backend details
- REST: `GET /api/channels`, `/api/channels/{id}`, `/api/channels/{id}/telemetry`, `/api/channels/{id}/anomalies`, `/api/channels/{id}/forecast`, `/api/alerts` (+ `POST /api/alerts/{id}/acknowledge`), `/api/channels/{id}/explain?index=`, `/api/health`.
- WebSocket: `WS /ws/replay/{chan_id}` — per-connection independent async replay of the test split at an accelerated cadence, streaming live detection + forecast; writes `AnomalyEvent`/`Alert` rows on new anomaly transitions.
- DB: SQLite (zero-config, sufficient for this single-demo scale); `Channel`, `AnomalyEvent`, `Alert` tables.
- Model loading: `ModelRegistry` loaded once at startup from `manifest.json`; fails fast if artifacts are missing.

### Frontend details
- Pages: `DashboardPage` (health grid), `ChannelDetailPage` (live chart + gauge + forecast panel + SHAP panel), `AlertsPage`.
- TanStack Query for REST, a small custom hook for WebSocket streaming.
- Recharts for charts; fixed status-color palette reserved for health state only; theme-aware (light/dark) tokens.

### Docker
- `backend/Dockerfile`, `frontend/Dockerfile` (multi-stage Vite build → nginx), `docker/trainer.Dockerfile` (one-shot, `train` compose profile, not part of default `up`).
- Fresh-clone path: `python scripts/download_dataset.py` → `python scripts/stage_data.py` → `docker compose --profile train run --rm trainer` → `docker compose up`.

---

## Open questions / documented assumptions

Proceeding with these defaults; flagged here as easy to revisit:

1. **Auth**: no RBAC (out of scope); assuming fully open access for a local demo (no lightweight password/API key added). Revisit before ever exposing this beyond localhost.
2. **Framework**: PyTorch used for the LSTM Autoencoder and GRU forecaster (smaller backend image, cleaner training loop), diverging from the report's stated TensorFlow/Keras. If the report/viva needs literal Keras references, the architecture translates directly.
3. **Threshold fidelity**: a simplified `μ + z·σ` EWMA threshold (grid-searched per channel) stands in for Telemanom's full non-parametric pruning algorithm. Documented as a deliberate simplification, not a reimplementation of the original paper's exact method.
4. **Forecast horizon**: `H=15` steps ahead, chosen as a reasonable early-warning window. Configurable via `ml/ml/config.py::FORECAST_HORIZON`.
5. **Alert/event retention**: no cleanup policy enforced; `AnomalyEvent`/`Alert` rows accumulate indefinitely in SQLite (fine at demo scale).
