import os
import pytest

# Ensure a dummy GROQ_API_KEY is present for testing
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "test_groq_api_key")

from agents.utils import strip_thinking
from graph import graph, InterviewState


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


def test_langgraph_compilation():
    """Test that the LangGraph StateGraph compiles and contains required nodes."""
    assert graph is not None
    # Verify graph has the required nodes
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
