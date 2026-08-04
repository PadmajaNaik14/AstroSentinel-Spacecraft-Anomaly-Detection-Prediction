from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/health")
def health(request: Request):
    registry = request.app.state.model_registry
    return {"status": "ok", "channels_loaded": len(registry.list_channel_ids())}
