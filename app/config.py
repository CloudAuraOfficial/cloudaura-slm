from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8002
    log_level: str = "info"
    ollama_base_url: str = "http://ollama:11434"
    benchmark_models: str = "phi3:mini,gemma2:2b,qwen2.5:1.5b"
    results_dir: str = "/app/benchmark_results"

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.benchmark_models.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
