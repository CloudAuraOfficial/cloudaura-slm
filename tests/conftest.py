from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import (
    BenchmarkReport,
    BenchmarkSummary,
    ModelBenchmarkResult,
    ModelInfo,
)


@pytest.fixture()
def mock_ollama_client():
    """Mock OllamaClient so tests never contact a real Ollama instance."""
    client = AsyncMock()
    client.is_healthy = AsyncMock(return_value=True)
    client.list_models = AsyncMock(return_value=["phi3:mini", "gemma2:2b"])
    client.get_model_info = AsyncMock(
        return_value=ModelInfo(
            name="phi3:mini",
            size_gb=2.3,
            parameter_count="3.8B",
            quantization="Q4_K_M",
            family="phi3",
        )
    )
    client.chat = AsyncMock(
        return_value={
            "message": {"content": "Hello! I am a language model."},
            "eval_count": 25,
            "eval_duration": 500_000_000,
            "total_duration": 800_000_000,
        }
    )
    client.generate = AsyncMock(
        return_value={
            "response": "Generated text response.",
            "eval_count": 30,
            "eval_duration": 600_000_000,
            "total_duration": 900_000_000,
            "prompt_eval_duration": 100_000_000,
        }
    )
    client.close = AsyncMock()
    return client


@pytest.fixture()
def mock_benchmark_service():
    """Mock BenchmarkService with canned results."""
    svc = AsyncMock()
    svc.run_benchmark = AsyncMock(return_value=sample_benchmark_report())
    svc.get_latest_report = AsyncMock(return_value=sample_benchmark_report())
    return svc


@pytest.fixture()
def mock_structured_service():
    """Mock StructuredOutputService."""
    svc = AsyncMock()
    svc.generate = AsyncMock(return_value={"name": "Alice", "age": 30})
    return svc


@pytest.fixture()
def app(mock_ollama_client, mock_benchmark_service, mock_structured_service):
    """Create a FastAPI app with all services mocked on app.state."""
    from fastapi import FastAPI
    from app.routers import health, inference, benchmark

    test_app = FastAPI()
    test_app.include_router(health.router)
    test_app.include_router(inference.router)
    test_app.include_router(benchmark.router)

    test_app.state.ollama_client = mock_ollama_client
    test_app.state.benchmark_service = mock_benchmark_service
    test_app.state.structured_service = mock_structured_service

    return test_app


@pytest.fixture()
def client(app):
    """Synchronous test client for the FastAPI app."""
    return TestClient(app)


def sample_benchmark_result(
    model: str = "phi3:mini",
    prompt: str = "Test prompt",
) -> ModelBenchmarkResult:
    return ModelBenchmarkResult(
        model=model,
        prompt=prompt,
        response="Sample response text.",
        tokens_generated=30,
        time_to_first_token_ms=100.0,
        total_duration_ms=900.0,
        tokens_per_second=50.0,
        eval_duration_ms=600.0,
    )


def sample_benchmark_report() -> BenchmarkReport:
    results = [
        sample_benchmark_result("phi3:mini", "Prompt A"),
        sample_benchmark_result("phi3:mini", "Prompt B"),
        sample_benchmark_result("gemma2:2b", "Prompt A"),
        sample_benchmark_result("gemma2:2b", "Prompt B"),
    ]
    summaries = [
        BenchmarkSummary(
            model="phi3:mini",
            total_prompts=2,
            avg_tokens_per_second=50.0,
            avg_time_to_first_token_ms=100.0,
            avg_total_duration_ms=900.0,
            avg_tokens_generated=30.0,
            total_tokens=60,
            model_size_gb=2.3,
        ),
        BenchmarkSummary(
            model="gemma2:2b",
            total_prompts=2,
            avg_tokens_per_second=50.0,
            avg_time_to_first_token_ms=100.0,
            avg_total_duration_ms=900.0,
            avg_tokens_generated=30.0,
            total_tokens=60,
            model_size_gb=1.5,
        ),
    ]
    return BenchmarkReport(
        timestamp="2026-03-17T00:00:00+00:00",
        hardware={
            "platform": "Linux",
            "architecture": "x86_64",
            "cpu_count": 4,
            "memory_gb": 16.0,
            "gpu": "none (CPU-only inference)",
        },
        results=results,
        summaries=summaries,
    )
