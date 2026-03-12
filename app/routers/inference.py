from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ModelInfo,
    StructuredRequest,
)

router = APIRouter(prefix="/api", tags=["inference"])


@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request) -> list[ModelInfo]:
    client = request.app.state.ollama_client
    names = await client.list_models()
    models = []
    for name in names:
        try:
            info = await client.get_model_info(name)
            models.append(info)
        except Exception:
            models.append(
                ModelInfo(
                    name=name,
                    size_gb=0,
                    parameter_count="unknown",
                    quantization="unknown",
                    family="unknown",
                )
            )
    return models


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={502: {"model": ErrorResponse}},
)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    client = request.app.state.ollama_client
    messages = [m.model_dump() for m in body.messages]

    try:
        data = await client.chat(
            model=body.model,
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "ollama_error", "message": str(exc), "status_code": 502},
        ) from exc

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    total_duration_ns = data.get("total_duration", 0)
    tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

    return ChatResponse(
        model=body.model,
        content=data.get("message", {}).get("content", ""),
        tokens_generated=eval_count,
        total_duration_ms=round(total_duration_ns / 1_000_000, 2),
        tokens_per_second=round(tps, 2),
    )


@router.post(
    "/structured",
    responses={502: {"model": ErrorResponse}},
)
async def structured_output(request: Request, body: StructuredRequest) -> dict:
    svc = request.app.state.structured_service
    try:
        return await svc.generate(
            model=body.model,
            prompt=body.prompt,
            response_schema=body.response_schema,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "structured_output_error",
                "message": str(exc),
                "status_code": 502,
            },
        ) from exc
