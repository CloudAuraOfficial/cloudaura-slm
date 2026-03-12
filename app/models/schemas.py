from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class ChatResponse(BaseModel):
    model: str
    content: str
    tokens_generated: int
    total_duration_ms: float
    tokens_per_second: float


class BenchmarkRequest(BaseModel):
    prompts: list[str] = Field(
        default=[
            "Explain quantum computing in 3 sentences.",
            "Write a Python function that checks if a string is a palindrome.",
            "What are the privacy implications of cloud-based AI vs local AI?",
            "Summarize the benefits of edge computing for IoT devices.",
            "Compare REST and GraphQL APIs in a brief paragraph.",
        ]
    )
    models: list[str] | None = None
    runs_per_prompt: int = Field(default=1, ge=1, le=5)


class ModelBenchmarkResult(BaseModel):
    model: str
    prompt: str
    response: str
    tokens_generated: int
    time_to_first_token_ms: float
    total_duration_ms: float
    tokens_per_second: float
    eval_duration_ms: float


class BenchmarkSummary(BaseModel):
    model: str
    total_prompts: int
    avg_tokens_per_second: float
    avg_time_to_first_token_ms: float
    avg_total_duration_ms: float
    avg_tokens_generated: float
    total_tokens: int
    model_size_gb: float


class BenchmarkReport(BaseModel):
    timestamp: str
    hardware: dict
    results: list[ModelBenchmarkResult]
    summaries: list[BenchmarkSummary]


class StructuredRequest(BaseModel):
    model: str
    prompt: str
    response_schema: dict = Field(
        ..., description="JSON Schema describing the expected output structure"
    )


class ModelInfo(BaseModel):
    name: str
    size_gb: float
    parameter_count: str
    quantization: str
    family: str


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    models_available: list[str]


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int
