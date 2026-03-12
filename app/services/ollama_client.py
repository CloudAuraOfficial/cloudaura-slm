import httpx
import structlog

from app.config import settings
from app.models.schemas import ModelInfo

logger = structlog.get_logger()


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=httpx.Timeout(300.0)
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def is_healthy(self) -> bool:
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        self._model_sizes: dict[str, float] = {}
        for m in data.get("models", []):
            size_bytes = m.get("size", 0)
            self._model_sizes[m["name"]] = round(size_bytes / (1024**3), 2)
        return [m["name"] for m in data.get("models", [])]

    async def get_model_info(self, model: str) -> ModelInfo:
        resp = await self._client.post("/api/show", json={"name": model})
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details", {})

        size_gb = getattr(self, "_model_sizes", {}).get(model, 0)

        return ModelInfo(
            name=model,
            size_gb=size_gb,
            parameter_count=details.get("parameter_size", "unknown"),
            quantization=details.get("quantization_level", "unknown"),
            family=details.get("family", "unknown"),
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        logger.info("ollama_generate_start", model=model, prompt_len=len(prompt))
        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "ollama_generate_complete",
            model=model,
            eval_count=data.get("eval_count", 0),
            total_duration_ns=data.get("total_duration", 0),
        )
        return data

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        logger.info("ollama_chat_start", model=model, message_count=len(messages))
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "ollama_chat_complete",
            model=model,
            eval_count=data.get("eval_count", 0),
        )
        return data
