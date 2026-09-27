import os
import pytest
from fastapi.testclient import TestClient

# Ensure dummy API keys are present for test environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "test_groq_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "test-ai-interview-coach"

from agents.utils import strip_thinking, get_model_name
from graph import graph, InterviewState
from ws_server import app as ws_app


def test_strip_thinking():
    """Test that <think>...</think> tags are properly removed from model output."""
    raw_text = "<think>Let me evaluate this answer</think>Technical Score: 8/10\nFeedback: Great job."
    cleaned = strip_thinking(raw_text)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Technical Score: 8/10" in cleaned


def test_strip_thinking_no_tags():
    """Test strip_thinking with normal text without tags."""
    plain_text = "What is the difference between a process and a thread?"
    assert strip_thinking(plain_text) == plain_text


def test_get_model_name_default():
    """Test get_model_name returns valid model string."""
    model = get_model_name()
    assert isinstance(model, str)
    assert "qwen" in model.lower() or "llama" in model.lower()


def test_langgraph_compilation():
    """Test that the LangGraph StateGraph compiles and contains required nodes."""
    assert graph is not None
    nodes = graph.nodes
    assert "question" in nodes
    assert "evaluation" in nodes
    assert "report" in nodes


def test_interview_state_structure():
    """Test that InterviewState contains expected keys."""
    sample_state = {
        "mode": "ask",
        "topic": "Projects",
        "job_description": "Software Engineer role",
        "answer": "",
        "resume_context": "",
        "question": "",
        "question_number": 1,
        "feedback": "",
        "report": "",
        "history": []
    }
    assert sample_state["mode"] == "ask"
    assert sample_state["question_number"] == 1
    assert isinstance(sample_state["history"], list)


def test_websocket_server_health():
    """Test FastAPI WebSocket server health endpoint."""
    client = TestClient(ws_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_websocket_server_root():
    """Test root endpoint returns WebSocket route info."""
    client = TestClient(ws_app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["websocket_endpoint"] == "/ws/interview"


def test_websocket_handshake_and_ping():
    """Test WebSocket connection handshake and ping/pong action."""
    client = TestClient(ws_app)
    with client.websocket_connect("/ws/interview") as websocket:
        # 1. Receive connection handshake
        handshake = websocket.receive_json()
        assert handshake["event"] == "connected"
        assert handshake["session_status"] == "ready"

        # 2. Send ping action
        websocket.send_json({"action": "ping", "timestamp": 123456})
        response = websocket.receive_json()
        assert response["event"] == "pong"
        assert response["timestamp"] == 123456
