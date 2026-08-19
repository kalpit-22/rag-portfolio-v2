import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)

def test_read_main():
    """Test that the frontend HTML serves correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Pradhyumn's AI Agent" in response.text

@patch("main.ask_portfolio")
@patch("main.get_temporary_retriever")
def test_chat_endpoint(mock_get_temporary_retriever, mock_ask_portfolio):
    """Test the chat endpoint without hitting real LLM APIs by mocking ask_portfolio."""
    mock_ask_portfolio.return_value = "This is a mocked response from the AI."
    mock_get_temporary_retriever.return_value = None
    
    response = client.post(
        "/api/chat",
        json={
            "query": "What are your core skills?",
            "chat_history": [],
            "session_id": "test-session-123"
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"response": "This is a mocked response from the AI."}
    mock_ask_portfolio.assert_called_once()

