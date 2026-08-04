from fastapi import APIRouter, HTTPException, Request

from app.schemas.channel import ChannelDetail, ChannelSummary

router = APIRouter()


@router.get("/api/channels", response_model=list[ChannelSummary])
def list_channels(request: Request):
    registry = request.app.state.model_registry
    out = []
    for chan_id in registry.list_channel_ids():
        cm = registry.get(chan_id)
        out.append(
            ChannelSummary(
                chan_id=cm.chan_id,
                spacecraft=cm.spacecraft,
                anomaly_class=cm.anomaly_class,
                num_values=cm.metadata["test_length"],
                eval_f1=cm.metadata["eval_f1"],
            )
        )
    return out


@router.get("/api/channels/{chan_id}", response_model=ChannelDetail)
def get_channel(chan_id: str, request: Request):
    registry = request.app.state.model_registry
    try:
        cm = registry.get(chan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel {chan_id}")

    return ChannelDetail(
        chan_id=cm.chan_id,
        spacecraft=cm.spacecraft,
        anomaly_class=cm.anomaly_class,
        num_values=cm.metadata["test_length"],
        eval_f1=cm.metadata["eval_f1"],
        window_size=cm.metadata["window_size"],
        forecast_horizon=cm.metadata["forecast_horizon"],
        ae_threshold=cm.threshold["ae_threshold"],
        eval_precision=cm.metadata["eval_precision"],
        eval_recall=cm.metadata["eval_recall"],
        forecast_mae=cm.metadata["forecast_mae"],
        trained_at=cm.metadata["trained_at"],
    )
