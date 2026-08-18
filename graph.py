from typing import TypedDict, List, Optional

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

def route_by_mode(state):
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
# Helper Functions for app.py
# -------------------------

def run_question(topic, job_description, question_number=1, history=None):
    """
    Run the graph in 'ask' mode to generate a new interview question.

    Returns the generated question string.
    """
    state = {
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

    result = graph.invoke(state)

    return result["question"]


def run_evaluation(question, answer, topic, job_description, resume_context="", history=None):
    """
    Run the graph in 'evaluate' mode to evaluate an answer
    and generate a per-question report.

    Returns a dict with 'feedback' and 'report'.
    """
    state = {
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

    result = graph.invoke(state)

    return {
        "feedback": result["feedback"],
        "report": result["report"]
    }


def run_final_report(history):
    """
    Run the graph in 'final_report' mode to generate a
    comprehensive interview summary report.

    Returns the report string.
    """
    state = {
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

    result = graph.invoke(state)

    return result["report"]