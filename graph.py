from typing import TypedDict, List, Optional, AsyncGenerator, Dict, Any
import os

from langgraph.graph import StateGraph, END

from agents.question_agent import question_agent
from agents.evaluation_agent import evaluation_agent
from agents.report_agent import report_agent


class InterviewState(TypedDict):
    mode: str  # "ask", "evaluate", or "final_report"
    topic: str
    job_description: str
    answer: str
    resume_context: str
    question: str
    question_number: int
    feedback: str
    report: str
    history: List[dict]  # list of {"question", "answer", "feedback"} dicts


# -------------------------
# Routing Logic
# -------------------------

def route_by_mode(state: InterviewState):
    """Route to the correct starting node based on mode."""
    mode = state.get("mode", "ask")
    if mode == "ask":
        return "question"
    elif mode == "evaluate":
        return "evaluation"
    elif mode == "final_report":
        return "report"
    return "question"


# -------------------------
# Build the Graph
# -------------------------

builder = StateGraph(InterviewState)

builder.add_node("question", question_agent)
builder.add_node("evaluation", evaluation_agent)
builder.add_node("report", report_agent)

# Conditional entry point based on mode
builder.set_conditional_entry_point(route_by_mode, {
    "question": "question",
    "evaluation": "evaluation",
    "report": "report"
})

# question mode: generate question then stop
builder.add_edge("question", END)

# evaluate mode: evaluate → report → stop
builder.add_edge("evaluation", "report")
builder.add_edge("report", END)

graph = builder.compile()


# -------------------------
# Helper Functions with LangSmith Tracing & Metadata
# -------------------------

def run_question(topic: str, job_description: str, question_number: int = 1, history: Optional[List[dict]] = None) -> str:
    """
    Run the graph in 'ask' mode to generate a new interview question.
    Traced via LangSmith with enriched tags and metadata.
    """
    state: InterviewState = {
        "mode": "ask",
        "topic": topic,
        "job_description": job_description,
        "question_number": question_number,
        "history": history or [],
        "answer": "",
        "resume_context": "",
        "question": "",
        "feedback": "",
        "report": ""
    }

    config = {
        "run_name": f"question_generation_round_{question_number}",
        "tags": ["interview_coach", "question_agent", topic.lower()],
        "metadata": {
            "topic": topic,
            "question_number": question_number,
            "has_jd": bool(job_description),
            "prior_questions_count": len(history or [])
        }
    }

    result = graph.invoke(state, config=config)
    return result["question"]


def run_evaluation(question: str, answer: str, topic: str, job_description: str = "", resume_context: str = "", history: Optional[List[dict]] = None) -> dict:
    """
    Run the graph in 'evaluate' mode to evaluate an answer and generate a report.
    Traced via LangSmith with enriched tags and metadata.
    """
    state: InterviewState = {
        "mode": "evaluate",
        "topic": topic,
        "job_description": job_description,
        "question": question,
        "answer": answer,
        "resume_context": resume_context,
        "question_number": 0,
        "history": history or [],
        "feedback": "",
        "report": ""
    }

    config = {
        "run_name": "answer_evaluation_pipeline",
        "tags": ["interview_coach", "evaluation_agent", "report_agent", topic.lower()],
        "metadata": {
            "topic": topic,
            "answer_length": len(answer),
            "question": question[:100]
        }
    }

    result = graph.invoke(state, config=config)

    return {
        "feedback": result["feedback"],
        "report": result["report"]
    }


def run_final_report(history: List[dict]) -> str:
    """
    Run the graph in 'final_report' mode to generate a comprehensive interview report.
    Traced via LangSmith with enriched tags and metadata.
    """
    state: InterviewState = {
        "mode": "final_report",
        "topic": "",
        "job_description": "",
        "question": "",
        "answer": "",
        "resume_context": "",
        "question_number": 0,
        "history": history,
        "feedback": "",
        "report": ""
    }

    config = {
        "run_name": "final_interview_summary_report",
        "tags": ["interview_coach", "final_report"],
        "metadata": {
            "total_questions_answered": len(history)
        }
    }

    result = graph.invoke(state, config=config)
    return result["report"]


# -------------------------
# Async & Streaming Helpers for WebSocket Integration
# -------------------------

async def arun_question(topic: str, job_description: str, question_number: int = 1, history: Optional[List[dict]] = None) -> str:
    """Async question runner for WebSocket servers."""
    state: InterviewState = {
        "mode": "ask",
        "topic": topic,
        "job_description": job_description,
        "question_number": question_number,
        "history": history or [],
        "answer": "",
        "resume_context": "",
        "question": "",
        "feedback": "",
        "report": ""
    }

    config = {
        "run_name": f"ws_question_round_{question_number}",
        "tags": ["websocket", "question_agent", topic.lower()],
        "metadata": {"topic": topic, "question_number": question_number}
    }

    result = await graph.ainvoke(state, config=config)
    return result["question"]


async def arun_evaluation(question: str, answer: str, topic: str, job_description: str = "", resume_context: str = "", history: Optional[List[dict]] = None) -> dict:
    """Async evaluation runner for WebSocket servers."""
    state: InterviewState = {
        "mode": "evaluate",
        "topic": topic,
        "job_description": job_description,
        "question": question,
        "answer": answer,
        "resume_context": resume_context,
        "question_number": 0,
        "history": history or [],
        "feedback": "",
        "report": ""
    }

    config = {
        "run_name": "ws_answer_evaluation",
        "tags": ["websocket", "evaluation_agent", "report_agent", topic.lower()],
        "metadata": {"topic": topic, "answer_length": len(answer)}
    }

    result = await graph.ainvoke(state, config=config)
    return {
        "feedback": result["feedback"],
        "report": result["report"]
    }