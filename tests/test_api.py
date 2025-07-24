import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pytest
from fastapi.testclient import TestClient
import app_api

client = TestClient(app_api.app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_endpoint(monkeypatch):
    # Mock the retrieval and LLM logic
    monkeypatch.setattr(app_api, "retrieve_chunks", lambda *a, **kw: [("Mock chunk", 0.1)])
    monkeypatch.setattr(app_api, "build_context", lambda chunks: "- \"Mock chunk\"")
    monkeypatch.setattr(app_api, "ask_llm", lambda context, question: "Mock answer.")

    response = client.post("/query", json={"question": "Test?", "top_n": 1})
    # Accept either 200 (if index is loaded) or 400 (if not loaded)
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        data = response.json()
        assert data["question"] == "Test?"
        assert data["retrieved_chunks"] == ["Mock chunk"]
        assert data["llm_response"] == "Mock answer."
    else:
        data = response.json()
        assert "detail" in data and "Index not loaded" in data["detail"] 