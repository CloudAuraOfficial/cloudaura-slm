from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    client = request.app.state.ollama_client
    connected = await client.is_healthy()
    models = await client.list_models() if connected else []
    return HealthResponse(
        status="healthy" if connected else "degraded",
        ollama_connected=connected,
        models_available=models,
    )
