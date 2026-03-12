from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import BenchmarkReport, BenchmarkRequest, ErrorResponse

router = APIRouter(prefix="/api", tags=["benchmark"])


@router.post(
    "/benchmark",
    response_model=BenchmarkReport,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def run_benchmark(request: Request, body: BenchmarkRequest) -> BenchmarkReport:
    svc = request.app.state.benchmark_service
    try:
        return await svc.run_benchmark(
            prompts=body.prompts,
            models=body.models,
            runs_per_prompt=body.runs_per_prompt,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": str(exc), "status_code": 400},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "benchmark_error", "message": str(exc), "status_code": 502},
        ) from exc


@router.get(
    "/benchmark/latest",
    response_model=BenchmarkReport,
    responses={404: {"model": ErrorResponse}},
)
async def get_latest_benchmark(request: Request) -> BenchmarkReport:
    svc = request.app.state.benchmark_service
    report = await svc.get_latest_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": "No benchmark results found. Run a benchmark first.",
                "status_code": 404,
            },
        )
    return report
