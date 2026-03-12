import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, create_model
import structlog

from app.config import settings

logger = structlog.get_logger()


def _build_pydantic_model(schema: dict) -> type[BaseModel]:
    """Dynamically build a Pydantic model from a JSON Schema dict."""
    fields: dict = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    for name, prop in properties.items():
        prop_type = prop.get("type", "string")
        if prop_type == "array":
            item_type = type_map.get(prop.get("items", {}).get("type", "string"), str)
            field_type = list[item_type]  # type: ignore[valid-type]
        else:
            field_type = type_map.get(prop_type, str)

        default = ... if name in required else None
        fields[name] = (field_type, default)

    return create_model("DynamicResponse", **fields)


class StructuredOutputService:
    def __init__(self) -> None:
        base_url = f"{settings.ollama_base_url}/v1"
        self._openai = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self._client = instructor.from_openai(self._openai)

    async def generate(
        self, model: str, prompt: str, response_schema: dict
    ) -> dict:
        response_model = _build_pydantic_model(response_schema)
        logger.info(
            "structured_generate_start",
            model=model,
            schema_fields=list(response_schema.get("properties", {}).keys()),
        )

        result = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
            max_retries=2,
        )

        logger.info("structured_generate_complete", model=model)
        return result.model_dump()
