from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas.explain import ExplainResponse
from app.services.shap_service import explain_index

router = APIRouter()


@router.get("/api/channels/{chan_id}/explain", response_model=ExplainResponse)
def explain(chan_id: str, request: Request, index: int = Query(...)):
    registry = request.app.state.model_registry
    try:
        channel_model = registry.get(chan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel {chan_id}")

    result = explain_index(channel_model, index)
    return ExplainResponse(chan_id=chan_id, index=index, **result)
