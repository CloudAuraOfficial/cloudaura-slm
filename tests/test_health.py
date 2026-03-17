from unittest.mock import AsyncMock


class TestHealthEndpoint:
    def test_health_returns_healthy_when_ollama_connected(self, client, mock_ollama_client):
        mock_ollama_client.is_healthy.return_value = True
        mock_ollama_client.list_models.return_value = ["phi3:mini", "gemma2:2b"]

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["ollama_connected"] is True
        assert data["models_available"] == ["phi3:mini", "gemma2:2b"]

    def test_health_returns_degraded_when_ollama_disconnected(self, client, mock_ollama_client):
        mock_ollama_client.is_healthy.return_value = False

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ollama_connected"] is False
        assert data["models_available"] == []

    def test_health_returns_empty_models_when_disconnected(self, client, mock_ollama_client):
        mock_ollama_client.is_healthy.return_value = False

        resp = client.get("/health")

        data = resp.json()
        assert isinstance(data["models_available"], list)
        assert len(data["models_available"]) == 0
        # list_models should NOT be called when not connected
        mock_ollama_client.list_models.assert_not_called()

    def test_health_response_has_required_fields(self, client):
        resp = client.get("/health")

        data = resp.json()
        assert "status" in data
        assert "ollama_connected" in data
        assert "models_available" in data
