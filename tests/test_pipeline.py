import os
import pytest
from fastapi.testclient import TestClient

# Ensure dummy API keys are present for test environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "test_groq_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "test-ai-interview-coach"

from agents.utils import strip_thinking, get_model_name
from graph import (
    graph,
    InterviewState,
    initialize_state_node,
    update_state_node,
    check_progress,
    route_entry_point
)
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
    """Test that the LangGraph StateGraph compiles and contains all required state machine nodes."""
    assert graph is not None
    nodes = graph.nodes
    assert "initialize" in nodes
    assert "question" in nodes
    assert "evaluation" in nodes
    assert "report" in nodes
    assert "update_state" in nodes
    assert "final_report" in nodes


def test_interview_state_initialization():
    """Test initialize_state_node resets and configures state."""
    init_res = initialize_state_node({
        "topic": "System Design",
        "job_description": "Staff Engineer"
    })
    assert init_res["topic"] == "System Design"
    assert init_res["job_description"] == "Staff Engineer"
    assert init_res["question_number"] == 1
    assert init_res["history"] == []
    assert init_res["is_complete"] is False


def test_update_state_node():
    """Test update_state_node appends to history and increments question number."""
    mock_state = {
        "question": "Explain Paxos consensus.",
        "answer": "Paxos is a consensus protocol...",
        "feedback": "Technical Score: 9/10",
        "report": "Solid understanding of consensus.",
        "history": []
    }
    updated = update_state_node(mock_state)
    assert len(updated["history"]) == 1
    assert updated["history"][0]["question"] == "Explain Paxos consensus."
    assert updated["history"][0]["answer"] == "Paxos is a consensus protocol..."
    assert updated["question_number"] == 2
    assert updated["answer"] == ""


def test_check_progress_conditional_edge():
    """Test LangGraph check_progress routing logic (< 5 -> question, >= 5 -> final_report)."""
    # Case 1: 0 completed questions
    assert check_progress({"history": []}) == "question"

    # Case 2: 4 completed questions
    assert check_progress({"history": [1, 2, 3, 4]}) == "question"

    # Case 3: 5 completed questions -> routes to final_report
    assert check_progress({"history": [1, 2, 3, 4, 5]}) == "final_report"

    # Case 4: > 5 completed questions
    assert check_progress({"history": [1, 2, 3, 4, 5, 6]}) == "final_report"


def test_route_entry_point():
    """Test entry point routing based on action parameter."""
    assert route_entry_point({"action": "init"}) == "initialize"
    assert route_entry_point({"action": "question"}) == "question"
    assert route_entry_point({"action": "evaluate"}) == "evaluation"
    assert route_entry_point({"action": "advance"}) == "update_state"
    assert route_entry_point({"action": "final_report"}) == "final_report"


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
