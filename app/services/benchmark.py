import json
import os
from datetime import datetime, timezone

import structlog

from app.config import settings
from app.models.schemas import (
    BenchmarkReport,
    BenchmarkSummary,
    ModelBenchmarkResult,
)
from app.services.ollama_client import OllamaClient

logger = structlog.get_logger()

NS_TO_MS = 1_000_000


class BenchmarkService:
    def __init__(self, client: OllamaClient) -> None:
        self._client = client

    async def run_benchmark(
        self,
        prompts: list[str],
        models: list[str] | None = None,
        runs_per_prompt: int = 1,
    ) -> BenchmarkReport:
        target_models = models or settings.model_list
        available = await self._client.list_models()

        missing = [m for m in target_models if m not in available]
        if missing:
            raise ValueError(f"Models not available in Ollama: {missing}")

        results: list[ModelBenchmarkResult] = []

        for model in target_models:
            logger.info("benchmark_model_start", model=model, prompts=len(prompts))
            for prompt in prompts:
                for run in range(runs_per_prompt):
                    result = await self._benchmark_single(model, prompt)
                    results.append(result)
                    logger.info(
                        "benchmark_run_complete",
                        model=model,
                        run=run + 1,
                        tps=result.tokens_per_second,
                    )

        summaries = self._compute_summaries(results, target_models)
        hardware = await self._get_hardware_info()

        report = BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware=hardware,
            results=results,
            summaries=summaries,
        )

        self._persist_report(report)
        return report

    async def _benchmark_single(
        self, model: str, prompt: str
    ) -> ModelBenchmarkResult:
        data = await self._client.generate(
            model=model, prompt=prompt, temperature=0.0, max_tokens=512
        )

        total_duration_ns = data.get("total_duration", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        eval_count = data.get("eval_count", 0)

        total_ms = total_duration_ns / NS_TO_MS
        eval_ms = eval_duration_ns / NS_TO_MS
        ttft_ms = prompt_eval_duration_ns / NS_TO_MS
        tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

        return ModelBenchmarkResult(
            model=model,
            prompt=prompt,
            response=data.get("response", ""),
            tokens_generated=eval_count,
            time_to_first_token_ms=round(ttft_ms, 2),
            total_duration_ms=round(total_ms, 2),
            tokens_per_second=round(tps, 2),
            eval_duration_ms=round(eval_ms, 2),
        )

    def _compute_summaries(
        self,
        results: list[ModelBenchmarkResult],
        models: list[str],
    ) -> list[BenchmarkSummary]:
        summaries = []
        for model in models:
            model_results = [r for r in results if r.model == model]
            if not model_results:
                continue

            count = len(model_results)
            total_tokens = sum(r.tokens_generated for r in model_results)

            summaries.append(
                BenchmarkSummary(
                    model=model,
                    total_prompts=count,
                    avg_tokens_per_second=round(
                        sum(r.tokens_per_second for r in model_results) / count, 2
                    ),
                    avg_time_to_first_token_ms=round(
                        sum(r.time_to_first_token_ms for r in model_results) / count, 2
                    ),
                    avg_total_duration_ms=round(
                        sum(r.total_duration_ms for r in model_results) / count, 2
                    ),
                    avg_tokens_generated=round(total_tokens / count, 1),
                    total_tokens=total_tokens,
                    model_size_gb=0,
                )
            )
        return summaries

    async def _get_hardware_info(self) -> dict:
        import platform

        cpu_count = os.cpu_count() or 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem_kb = int(line.split()[1])
                        mem_gb = round(mem_kb / (1024**2), 1)
                        break
                else:
                    mem_gb = 0
        except FileNotFoundError:
            mem_gb = 0

        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cpu_count": cpu_count,
            "memory_gb": mem_gb,
            "gpu": "none (CPU-only inference)",
        }

    def _persist_report(self, report: BenchmarkReport) -> None:
        results_dir = settings.results_dir
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(results_dir, f"benchmark_{ts}.json")
        with open(path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        logger.info("benchmark_report_saved", path=path)

    async def get_latest_report(self) -> BenchmarkReport | None:
        results_dir = settings.results_dir
        if not os.path.isdir(results_dir):
            return None
        files = sorted(
            [f for f in os.listdir(results_dir) if f.endswith(".json")],
            reverse=True,
        )
        if not files:
            return None
        path = os.path.join(results_dir, files[0])
        with open(path) as f:
            data = json.load(f)
        return BenchmarkReport(**data)
