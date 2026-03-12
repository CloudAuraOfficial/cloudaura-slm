from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logging import setup_logging
from app.routers import benchmark, health, inference
from app.services.benchmark import BenchmarkService
from app.services.ollama_client import OllamaClient
from app.services.structured import StructuredOutputService


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)

    client = OllamaClient()
    app.state.ollama_client = client
    app.state.benchmark_service = BenchmarkService(client)
    app.state.structured_service = StructuredOutputService()

    yield

    await client.close()


app = FastAPI(
    title="CloudAura SLM — Local Model Benchmarking",
    description="Run, benchmark, and compare small language models entirely offline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(inference.router)
app.include_router(benchmark.router)

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
