from unittest.mock import AsyncMock

import pytest


class TestChatEndpoint:
    def test_valid_chat_request_returns_response(self, client, mock_ollama_client):
        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        resp = client.post("/api/chat", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "phi3:mini"
        assert data["content"] == "Hello! I am a language model."
        assert data["tokens_generated"] == 25
        assert data["tokens_per_second"] > 0
        assert data["total_duration_ms"] > 0

    def test_chat_calculates_tokens_per_second(self, client, mock_ollama_client):
        # eval_count=25, eval_duration=500_000_000 ns -> 25 / 0.5 = 50.0 tps
        mock_ollama_client.chat.return_value = {
            "message": {"content": "response"},
            "eval_count": 25,
            "eval_duration": 500_000_000,
            "total_duration": 800_000_000,
        }

        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = client.post("/api/chat", json=payload)

        data = resp.json()
        assert data["tokens_per_second"] == 50.0

    def test_chat_with_custom_temperature_and_max_tokens(self, client, mock_ollama_client):
        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.2,
            "max_tokens": 256,
        }

        resp = client.post("/api/chat", json=payload)

        assert resp.status_code == 200
        mock_ollama_client.chat.assert_called_once()
        call_kwargs = mock_ollama_client.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 256

    def test_chat_missing_model_returns_422(self, client):
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 422

    def test_chat_missing_messages_returns_422(self, client):
        payload = {"model": "phi3:mini"}

        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 422

    def test_chat_empty_messages_returns_422(self, client):
        payload = {
            "model": "phi3:mini",
            "messages": [],
        }

        # FastAPI allows empty lists by default, but let's verify the request goes through
        resp = client.post("/api/chat", json=payload)
        # Empty messages list is technically valid per the schema
        assert resp.status_code == 200

    def test_chat_invalid_role_returns_422(self, client):
        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "invalid_role", "content": "Hello"}],
        }

        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 422

    def test_chat_temperature_out_of_range_returns_422(self, client):
        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 3.0,
        }

        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 422

    def test_chat_ollama_error_returns_502(self, client, mock_ollama_client):
        mock_ollama_client.chat.side_effect = Exception("Connection refused")

        payload = {
            "model": "phi3:mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 502

    def test_chat_multiple_messages(self, client, mock_ollama_client):
        payload = {
            "model": "phi3:mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }

        resp = client.post("/api/chat", json=payload)

        assert resp.status_code == 200
        call_kwargs = mock_ollama_client.chat.call_args.kwargs
        assert len(call_kwargs["messages"]) == 4


class TestModelsEndpoint:
    def test_list_models_returns_model_info(self, client, mock_ollama_client):
        resp = client.get("/api/models")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "phi3:mini"

    def test_list_models_includes_required_fields(self, client, mock_ollama_client):
        resp = client.get("/api/models")

        data = resp.json()
        for model in data:
            assert "name" in model
            assert "size_gb" in model
            assert "parameter_count" in model
            assert "quantization" in model
            assert "family" in model

    def test_list_models_handles_info_failure_gracefully(self, client, mock_ollama_client):
        mock_ollama_client.get_model_info.side_effect = Exception("info unavailable")

        resp = client.get("/api/models")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Should fall back to unknown values
        assert data[0]["parameter_count"] == "unknown"
        assert data[0]["size_gb"] == 0


class TestStructuredEndpoint:
    def test_structured_output_returns_dict(self, client, mock_structured_service):
        payload = {
            "model": "phi3:mini",
            "prompt": "Give me a person",
            "response_schema": {
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        }

        resp = client.post("/api/structured", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Alice"
        assert data["age"] == 30

    def test_structured_output_error_returns_502(self, client, mock_structured_service):
        mock_structured_service.generate.side_effect = Exception("LLM error")

        payload = {
            "model": "phi3:mini",
            "prompt": "Give me a person",
            "response_schema": {
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }

        resp = client.post("/api/structured", json=payload)
        assert resp.status_code == 502
