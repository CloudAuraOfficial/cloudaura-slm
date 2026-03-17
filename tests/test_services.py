import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.models.schemas import (
    BenchmarkReport,
    BenchmarkSummary,
    ModelBenchmarkResult,
    ModelInfo,
)
from app.services.benchmark import BenchmarkService
from app.services.ollama_client import OllamaClient


def _make_response(status_code: int, **kwargs) -> httpx.Response:
    """Create an httpx.Response with a dummy request so raise_for_status() works."""
    request = httpx.Request("POST", "http://fake-ollama:11434/")
    resp = httpx.Response(status_code, request=request, **kwargs)
    return resp


# ---------------------------------------------------------------------------
# OllamaClient unit tests (httpx mocked via pytest-httpx)
# ---------------------------------------------------------------------------

class TestOllamaClientIsHealthy:
    @pytest.mark.anyio
    async def test_returns_true_when_ollama_responds_200(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        mock_response = _make_response(200, text="Ollama is running")
        client._client = AsyncMock()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.is_healthy()

        assert result is True
        client._client.get.assert_called_once_with("/")

    @pytest.mark.anyio
    async def test_returns_false_on_connection_error(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        client._client = AsyncMock()
        client._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await client.is_healthy()

        assert result is False


class TestOllamaClientListModels:
    @pytest.mark.anyio
    async def test_returns_model_names(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        mock_response = _make_response(
            200,
            json={
                "models": [
                    {"name": "phi3:mini", "size": 2_400_000_000},
                    {"name": "gemma2:2b", "size": 1_500_000_000},
                ]
            },
        )
        client._client = AsyncMock()
        client._client.get = AsyncMock(return_value=mock_response)

        names = await client.list_models()

        assert names == ["phi3:mini", "gemma2:2b"]
        assert hasattr(client, "_model_sizes")
        assert client._model_sizes["phi3:mini"] == round(2_400_000_000 / (1024**3), 2)

    @pytest.mark.anyio
    async def test_returns_empty_list_when_no_models(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        mock_response = _make_response(200, json={"models": []})
        client._client = AsyncMock()
        client._client.get = AsyncMock(return_value=mock_response)

        names = await client.list_models()

        assert names == []


class TestOllamaClientGetModelInfo:
    @pytest.mark.anyio
    async def test_returns_model_info(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        mock_response = _make_response(
            200,
            json={
                "details": {
                    "parameter_size": "3.8B",
                    "quantization_level": "Q4_K_M",
                    "family": "phi3",
                }
            },
        )
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)
        client._model_sizes = {"phi3:mini": 2.24}

        info = await client.get_model_info("phi3:mini")

        assert isinstance(info, ModelInfo)
        assert info.name == "phi3:mini"
        assert info.parameter_count == "3.8B"
        assert info.quantization == "Q4_K_M"
        assert info.family == "phi3"
        assert info.size_gb == 2.24


class TestOllamaClientChat:
    @pytest.mark.anyio
    async def test_sends_correct_payload(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        expected_data = {
            "message": {"content": "Hello!"},
            "eval_count": 10,
            "eval_duration": 200_000_000,
            "total_duration": 300_000_000,
        }
        mock_response = _make_response(200, json=expected_data)
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hi"}]
        data = await client.chat(model="phi3:mini", messages=messages)

        assert data["message"]["content"] == "Hello!"
        client._client.post.assert_called_once()
        call_args = client._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "phi3:mini"
        assert payload["stream"] is False


class TestOllamaClientGenerate:
    @pytest.mark.anyio
    async def test_sends_correct_payload(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        expected_data = {
            "response": "Generated text.",
            "eval_count": 20,
            "eval_duration": 400_000_000,
            "total_duration": 600_000_000,
        }
        mock_response = _make_response(200, json=expected_data)
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        data = await client.generate(model="phi3:mini", prompt="Say hello")

        assert data["response"] == "Generated text."
        call_args = client._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "phi3:mini"
        assert payload["prompt"] == "Say hello"
        assert payload["stream"] is False

    @pytest.mark.anyio
    async def test_includes_system_prompt_when_provided(self):
        client = OllamaClient(base_url="http://fake-ollama:11434")
        mock_response = _make_response(200, json={"response": "ok"})
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)

        await client.generate(model="phi3:mini", prompt="Hi", system="Be concise.")

        call_args = client._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["system"] == "Be concise."


# ---------------------------------------------------------------------------
# BenchmarkService unit tests
# ---------------------------------------------------------------------------

class TestBenchmarkServiceComputeSummaries:
    def test_aggregates_correctly_for_single_model(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        results = [
            ModelBenchmarkResult(
                model="phi3:mini",
                prompt="P1",
                response="R1",
                tokens_generated=20,
                time_to_first_token_ms=80.0,
                total_duration_ms=500.0,
                tokens_per_second=40.0,
                eval_duration_ms=400.0,
            ),
            ModelBenchmarkResult(
                model="phi3:mini",
                prompt="P2",
                response="R2",
                tokens_generated=40,
                time_to_first_token_ms=120.0,
                total_duration_ms=700.0,
                tokens_per_second=60.0,
                eval_duration_ms=600.0,
            ),
        ]

        summaries = svc._compute_summaries(results, ["phi3:mini"])

        assert len(summaries) == 1
        s = summaries[0]
        assert s.model == "phi3:mini"
        assert s.total_prompts == 2
        assert s.avg_tokens_per_second == 50.0  # (40 + 60) / 2
        assert s.avg_time_to_first_token_ms == 100.0  # (80 + 120) / 2
        assert s.avg_total_duration_ms == 600.0  # (500 + 700) / 2
        assert s.total_tokens == 60  # 20 + 40
        assert s.avg_tokens_generated == 30.0

    def test_aggregates_multiple_models(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        results = [
            ModelBenchmarkResult(
                model="phi3:mini", prompt="P1", response="R1",
                tokens_generated=20, time_to_first_token_ms=80.0,
                total_duration_ms=500.0, tokens_per_second=40.0, eval_duration_ms=400.0,
            ),
            ModelBenchmarkResult(
                model="gemma2:2b", prompt="P1", response="R2",
                tokens_generated=30, time_to_first_token_ms=90.0,
                total_duration_ms=600.0, tokens_per_second=50.0, eval_duration_ms=500.0,
            ),
        ]

        summaries = svc._compute_summaries(results, ["phi3:mini", "gemma2:2b"])

        assert len(summaries) == 2
        assert summaries[0].model == "phi3:mini"
        assert summaries[1].model == "gemma2:2b"

    def test_skips_model_with_no_results(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        results = [
            ModelBenchmarkResult(
                model="phi3:mini", prompt="P1", response="R1",
                tokens_generated=20, time_to_first_token_ms=80.0,
                total_duration_ms=500.0, tokens_per_second=40.0, eval_duration_ms=400.0,
            ),
        ]

        summaries = svc._compute_summaries(results, ["phi3:mini", "nonexistent"])

        assert len(summaries) == 1
        assert summaries[0].model == "phi3:mini"


class TestBenchmarkServiceBenchmarkSingle:
    @pytest.mark.anyio
    async def test_converts_ollama_response_to_result(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value={
            "response": "Answer text",
            "eval_count": 25,
            "eval_duration": 500_000_000,      # 0.5s
            "total_duration": 800_000_000,      # 0.8s
            "prompt_eval_duration": 100_000_000,  # 0.1s
        })
        svc = BenchmarkService(mock_client)

        result = await svc._benchmark_single("phi3:mini", "Test prompt")

        assert result.model == "phi3:mini"
        assert result.prompt == "Test prompt"
        assert result.response == "Answer text"
        assert result.tokens_generated == 25
        assert result.tokens_per_second == 50.0  # 25 / 0.5
        assert result.total_duration_ms == 800.0  # 800_000_000 / 1_000_000
        assert result.time_to_first_token_ms == 100.0
        assert result.eval_duration_ms == 500.0


class TestBenchmarkServiceRunBenchmark:
    @pytest.mark.anyio
    async def test_raises_value_error_for_missing_models(self):
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(return_value=["phi3:mini"])
        svc = BenchmarkService(mock_client)

        with pytest.raises(ValueError, match="Models not available"):
            await svc.run_benchmark(
                prompts=["Test"],
                models=["nonexistent_model"],
            )

    @pytest.mark.anyio
    async def test_runs_all_prompt_model_combinations(self):
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(return_value=["phi3:mini", "gemma2:2b"])
        mock_client.generate = AsyncMock(return_value={
            "response": "Answer",
            "eval_count": 10,
            "eval_duration": 200_000_000,
            "total_duration": 300_000_000,
            "prompt_eval_duration": 50_000_000,
        })
        svc = BenchmarkService(mock_client)

        with patch.object(svc, "_persist_report"):
            report = await svc.run_benchmark(
                prompts=["P1", "P2"],
                models=["phi3:mini", "gemma2:2b"],
                runs_per_prompt=1,
            )

        # 2 models x 2 prompts x 1 run = 4 results
        assert len(report.results) == 4
        assert len(report.summaries) == 2


class TestBenchmarkServiceGetLatestReport:
    @pytest.mark.anyio
    async def test_returns_none_when_no_directory(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        with patch("app.services.benchmark.settings") as mock_settings:
            mock_settings.results_dir = "/nonexistent/path"
            result = await svc.get_latest_report()

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_directory_empty(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("app.services.benchmark.settings") as mock_settings:
                mock_settings.results_dir = tmpdir
                result = await svc.get_latest_report()

        assert result is None

    @pytest.mark.anyio
    async def test_returns_latest_report_from_disk(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        report_data = {
            "timestamp": "2026-03-17T00:00:00+00:00",
            "hardware": {"platform": "Linux", "cpu_count": 4, "memory_gb": 16.0},
            "results": [],
            "summaries": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write two files; the latest (lexicographically last) should be returned
            with open(os.path.join(tmpdir, "benchmark_20260316_120000.json"), "w") as f:
                json.dump(report_data, f)
            report_data_newer = {**report_data, "timestamp": "2026-03-17T12:00:00+00:00"}
            with open(os.path.join(tmpdir, "benchmark_20260317_120000.json"), "w") as f:
                json.dump(report_data_newer, f)

            with patch("app.services.benchmark.settings") as mock_settings:
                mock_settings.results_dir = tmpdir
                result = await svc.get_latest_report()

        assert result is not None
        assert isinstance(result, BenchmarkReport)
        assert result.timestamp == "2026-03-17T12:00:00+00:00"


class TestBenchmarkServicePersistReport:
    def test_creates_directory_and_writes_json(self):
        mock_client = AsyncMock()
        svc = BenchmarkService(mock_client)

        report = BenchmarkReport(
            timestamp="2026-03-17T00:00:00+00:00",
            hardware={"platform": "Linux"},
            results=[],
            summaries=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results_subdir")
            with patch("app.services.benchmark.settings") as mock_settings:
                mock_settings.results_dir = results_path
                svc._persist_report(report)

            files = os.listdir(results_path)
            assert len(files) == 1
            assert files[0].startswith("benchmark_")
            assert files[0].endswith(".json")

            with open(os.path.join(results_path, files[0])) as f:
                data = json.load(f)
            assert data["timestamp"] == "2026-03-17T00:00:00+00:00"
