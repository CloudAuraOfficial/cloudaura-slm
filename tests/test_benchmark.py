from unittest.mock import AsyncMock

import pytest

from app.models.schemas import BenchmarkRequest


class TestBenchmarkEndpoint:
    def test_run_benchmark_returns_report(self, client, mock_benchmark_service):
        payload = {
            "prompts": ["Explain quantum computing."],
            "models": ["phi3:mini"],
            "runs_per_prompt": 1,
        }

        resp = client.post("/api/benchmark", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert "hardware" in data
        assert "results" in data
        assert "summaries" in data

    def test_benchmark_report_structure(self, client, mock_benchmark_service):
        payload = {"prompts": ["Test prompt"]}

        resp = client.post("/api/benchmark", json=payload)

        data = resp.json()
        # Validate results structure
        for result in data["results"]:
            assert "model" in result
            assert "prompt" in result
            assert "response" in result
            assert "tokens_generated" in result
            assert "tokens_per_second" in result
            assert "time_to_first_token_ms" in result
            assert "total_duration_ms" in result
            assert "eval_duration_ms" in result

        # Validate summaries structure
        for summary in data["summaries"]:
            assert "model" in summary
            assert "total_prompts" in summary
            assert "avg_tokens_per_second" in summary
            assert "avg_time_to_first_token_ms" in summary
            assert "avg_total_duration_ms" in summary
            assert "avg_tokens_generated" in summary
            assert "total_tokens" in summary
            assert "model_size_gb" in summary

    def test_benchmark_hardware_info(self, client, mock_benchmark_service):
        payload = {"prompts": ["Test"]}

        resp = client.post("/api/benchmark", json=payload)

        hw = resp.json()["hardware"]
        assert "platform" in hw
        assert "cpu_count" in hw
        assert "memory_gb" in hw

    def test_benchmark_validation_error_returns_400(self, client, mock_benchmark_service):
        mock_benchmark_service.run_benchmark.side_effect = ValueError(
            "Models not available in Ollama: ['nonexistent']"
        )

        payload = {
            "prompts": ["Test"],
            "models": ["nonexistent"],
        }

        resp = client.post("/api/benchmark", json=payload)
        assert resp.status_code == 400
        data = resp.json()["detail"]
        assert data["error"] == "validation_error"

    def test_benchmark_ollama_error_returns_502(self, client, mock_benchmark_service):
        mock_benchmark_service.run_benchmark.side_effect = RuntimeError("Ollama down")

        payload = {"prompts": ["Test"]}

        resp = client.post("/api/benchmark", json=payload)
        assert resp.status_code == 502
        data = resp.json()["detail"]
        assert data["error"] == "benchmark_error"

    def test_benchmark_uses_default_prompts(self, client, mock_benchmark_service):
        """Sending no prompts should use the schema defaults."""
        resp = client.post("/api/benchmark", json={})

        assert resp.status_code == 200
        call_kwargs = mock_benchmark_service.run_benchmark.call_args.kwargs
        assert len(call_kwargs["prompts"]) == 5

    def test_benchmark_runs_per_prompt_max_validation(self, client):
        payload = {
            "prompts": ["Test"],
            "runs_per_prompt": 10,  # max is 5
        }

        resp = client.post("/api/benchmark", json=payload)
        assert resp.status_code == 422


class TestLatestBenchmarkEndpoint:
    def test_get_latest_returns_report(self, client, mock_benchmark_service):
        resp = client.get("/api/benchmark/latest")

        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert "results" in data
        assert "summaries" in data

    def test_get_latest_returns_404_when_no_reports(self, client, mock_benchmark_service):
        mock_benchmark_service.get_latest_report.return_value = None

        resp = client.get("/api/benchmark/latest")

        assert resp.status_code == 404
        data = resp.json()["detail"]
        assert data["error"] == "not_found"


class TestBenchmarkRequestValidation:
    def test_default_prompts_are_populated(self):
        req = BenchmarkRequest()
        assert len(req.prompts) == 5

    def test_models_default_to_none(self):
        req = BenchmarkRequest()
        assert req.models is None

    def test_runs_per_prompt_default_is_one(self):
        req = BenchmarkRequest()
        assert req.runs_per_prompt == 1

    def test_runs_per_prompt_minimum(self):
        with pytest.raises(Exception):
            BenchmarkRequest(runs_per_prompt=0)

    def test_runs_per_prompt_maximum(self):
        with pytest.raises(Exception):
            BenchmarkRequest(runs_per_prompt=6)
