from typing import TypedDict, List, Optional, AsyncGenerator, Dict, Any
import os

from langgraph.graph import StateGraph, END

from agents.question_agent import question_agent
from agents.evaluation_agent import evaluation_agent
from agents.report_agent import report_agent, generate_question_report, generate_final_report


class InterviewState(TypedDict, total=False):
    """
    Central LangGraph State for the AI Interview Coach state machine.
    Holds the complete lifecycle of an interview session.
    """
    action: str          # "init", "question", "evaluate", "advance", "full_turn", "final_report", "ask"
    mode: str            # backwards compatibility for older tests/ws payloads
    topic: str
    job_description: str
    resume_context: str
    question: str
    question_number: int # 1 to 5
    answer: str          # candidate response
    feedback: str        # evaluation feedback scorecard
    report: str          # per-question performance report
    history: List[dict]  # list of {"question", "answer", "feedback", "report"}
    final_report: str    # comprehensive final assessment
    is_complete: bool    # True when interview reaches >= 5 questions


# -------------------------
# Node Functions
# -------------------------

def initialize_state_node(state: InterviewState) -> dict:
    """Initialize interview state variables."""
    return {
        "topic": state.get("topic", "Projects"),
        "job_description": state.get("job_description", ""),
        "resume_context": state.get("resume_context", ""),
        "question": "",
        "question_number": 1,
        "answer": "",
        "feedback": "",
        "report": "",
        "history": state.get("history", []),
        "final_report": "",
        "is_complete": False
    }


def question_node(state: InterviewState) -> dict:
    """Generate technical question using candidate context."""
    return question_agent(state)


def evaluation_node(state: InterviewState) -> dict:
    """Evaluate candidate answer and provide scoring scorecard."""
    return evaluation_agent(state)


def report_node(state: InterviewState) -> dict:
    """Generate per-question performance summary."""
    return generate_question_report(state)


def update_state_node(state: InterviewState) -> dict:
    """
    Append evaluated turn to interview history and increment question counter.
    """
    history = list(state.get("history", []))
    history.append({
        "question": state.get("question", ""),
        "answer": state.get("answer", ""),
        "feedback": state.get("feedback", ""),
        "report": state.get("report", "")
    })
    next_q_num = len(history) + 1
    return {
        "history": history,
        "question_number": next_q_num,
        "answer": ""
    }


def final_report_node(state: InterviewState) -> dict:
    """Synthesize final 5-round comprehensive assessment."""
    return generate_final_report(state)


# -------------------------
# Routing Logic (Conditional Edges)
# -------------------------

def route_entry_point(state: InterviewState) -> str:
    """Determine initial node based on requested action."""
    action = state.get("action", state.get("mode", "init"))
    if action in ("init", "start"):
        return "initialize"
    elif action in ("question", "ask"):
        return "question"
    elif action in ("evaluate", "full_turn"):
        return "evaluation"
    elif action == "advance":
        return "update_state"
    elif action == "final_report":
        return "final_report"
    return "initialize"


def route_after_report(state: InterviewState) -> str:
    """If user only requested evaluation, stop at END; otherwise proceed to update_state."""
    action = state.get("action", state.get("mode", ""))
    if action == "evaluate":
        return END
    return "update_state"


def check_progress(state: InterviewState) -> str:
    """
    Check interview progress:
    - If < 5 completed rounds in history: loop back to question agent.
    - If >= 5 completed rounds in history: route to final comprehensive report.
    """
    history = state.get("history", [])
    if len(history) < 5:
        return "question"
    return "final_report"


# -------------------------
# Build the LangGraph StateGraph
# -------------------------

builder = StateGraph(InterviewState)

builder.add_node("initialize", initialize_state_node)
builder.add_node("question", question_node)
builder.add_node("evaluation", evaluation_node)
builder.add_node("report", report_node)
builder.add_node("update_state", update_state_node)
builder.add_node("final_report", final_report_node)

# Conditional entry point
builder.set_conditional_entry_point(route_entry_point, {
    "initialize": "initialize",
    "question": "question",
    "evaluation": "evaluation",
    "update_state": "update_state",
    "final_report": "final_report"
})

# Initialize -> Question -> User Input (END)
builder.add_edge("initialize", "question")
builder.add_edge("question", END)

# Evaluation -> Report
builder.add_edge("evaluation", "report")

# Report -> (END if evaluate-only, or update_state if advance/full_turn)
builder.add_conditional_edges("report", route_after_report, {
    END: END,
    "update_state": "update_state"
})

# Update State -> Check Progress: < 5 -> Question, >= 5 -> Final Report
builder.add_conditional_edges("update_state", check_progress, {
    "question": "question",
    "final_report": "final_report"
})

builder.add_edge("final_report", END)

graph = builder.compile()


# -------------------------
# High-Level LangGraph State Machine Interface (Sync)
# -------------------------

def start_interview(topic: str = "Projects", job_description: str = "") -> InterviewState:
    """
    Initialize the interview state in LangGraph and generate Question #1.
    """
    initial_state: InterviewState = {
        "action": "init",
        "topic": topic,
        "job_description": job_description,
        "history": [],
        "question_number": 1,
        "answer": "",
        "feedback": "",
        "report": "",
        "final_report": "",
        "is_complete": False
    }

    config = {
        "run_name": "langgraph_start_interview",
        "tags": ["interview_coach", "langgraph_state", topic.lower()],
        "metadata": {"topic": topic, "has_jd": bool(job_description)}
    }

    return graph.invoke(initial_state, config=config)


def evaluate_answer(current_state: InterviewState, answer: str) -> InterviewState:
    """
    Execute LangGraph evaluation and report agents for the candidate's answer.
    """
    state_to_run = dict(current_state)
    state_to_run["action"] = "evaluate"
    state_to_run["answer"] = answer

    config = {
        "run_name": f"langgraph_evaluate_round_{current_state.get('question_number', 1)}",
        "tags": ["interview_coach", "langgraph_state", "evaluation"],
        "metadata": {
            "question_number": current_state.get("question_number", 1),
            "answer_length": len(answer)
        }
    }

    result = graph.invoke(state_to_run, config=config)
    # Merge result back into state while preserving current question
    updated_state = dict(current_state)
    updated_state["answer"] = answer
    updated_state["feedback"] = result.get("feedback", "")
    updated_state["report"] = result.get("report", "")
    return updated_state


def advance_interview(current_state: InterviewState) -> InterviewState:
    """
    Update LangGraph state and check progress:
    - If < 5: generates next question.
    - If >= 5: generates final assessment report.
    """
    state_to_run = dict(current_state)
    state_to_run["action"] = "advance"

    config = {
        "run_name": f"langgraph_advance_round_{current_state.get('question_number', 1)}",
        "tags": ["interview_coach", "langgraph_state", "advance"],
        "metadata": {
            "current_history_len": len(current_state.get("history", []))
        }
    }

    return graph.invoke(state_to_run, config=config)


def submit_and_advance(current_state: InterviewState, answer: str) -> InterviewState:
    """
    Run full turn in LangGraph: Evaluation -> Report -> Update State -> Check Progress -> Next Question / Final Report.
    """
    state_to_run = dict(current_state)
    state_to_run["action"] = "full_turn"
    state_to_run["answer"] = answer

    config = {
        "run_name": f"langgraph_full_turn_round_{current_state.get('question_number', 1)}",
        "tags": ["interview_coach", "langgraph_state", "full_turn"],
        "metadata": {"answer_length": len(answer)}
    }

    return graph.invoke(state_to_run, config=config)


# -------------------------
# High-Level LangGraph State Machine Interface (Async)
# -------------------------

async def astart_interview(topic: str = "Projects", job_description: str = "") -> InterviewState:
    """Async interview initialization for WebSocket / FastAPI."""
    initial_state: InterviewState = {
        "action": "init",
        "topic": topic,
        "job_description": job_description,
        "history": [],
        "question_number": 1,
        "answer": "",
        "feedback": "",
        "report": "",
        "final_report": "",
        "is_complete": False
    }

    config = {
        "run_name": "ws_langgraph_start_interview",
        "tags": ["websocket", "langgraph_state", topic.lower()],
        "metadata": {"topic": topic}
    }

    return await graph.ainvoke(initial_state, config=config)


async def aevaluate_answer(current_state: InterviewState, answer: str) -> InterviewState:
    """Async answer evaluation for WebSocket / FastAPI."""
    state_to_run = dict(current_state)
    state_to_run["action"] = "evaluate"
    state_to_run["answer"] = answer

    config = {
        "run_name": f"ws_langgraph_evaluate_round_{current_state.get('question_number', 1)}",
        "tags": ["websocket", "langgraph_state", "evaluation"]
    }

    result = await graph.ainvoke(state_to_run, config=config)
    updated_state = dict(current_state)
    updated_state["answer"] = answer
    updated_state["feedback"] = result.get("feedback", "")
    updated_state["report"] = result.get("report", "")
    return updated_state


async def aadvance_interview(current_state: InterviewState) -> InterviewState:
    """Async progress advance for WebSocket / FastAPI."""
    state_to_run = dict(current_state)
    state_to_run["action"] = "advance"

    config = {
        "run_name": f"ws_langgraph_advance_round_{current_state.get('question_number', 1)}",
        "tags": ["websocket", "langgraph_state", "advance"]
    }

    return await graph.ainvoke(state_to_run, config=config)


# -------------------------
# Backwards-Compatibility Wrappers
# -------------------------

def run_question(topic: str, job_description: str, question_number: int = 1, history: Optional[List[dict]] = None) -> str:
    """Backwards compatibility for standalone question generation."""
    state: InterviewState = {
        "action": "question",
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
    return result.get("question", "")


def run_evaluation(question: str, answer: str, topic: str, job_description: str = "", resume_context: str = "", history: Optional[List[dict]] = None) -> dict:
    """Backwards compatibility for standalone evaluation."""
    state: InterviewState = {
        "action": "evaluate",
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
        "feedback": result.get("feedback", ""),
        "report": result.get("report", "")
    }


def run_final_report(history: List[dict]) -> str:
    """Backwards compatibility for standalone final report generation."""
    state: InterviewState = {
        "action": "final_report",
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
    return result.get("final_report", result.get("report", ""))


async def arun_question(topic: str, job_description: str, question_number: int = 1, history: Optional[List[dict]] = None) -> str:
    """Async question runner for legacy WebSocket handlers."""
    state: InterviewState = {
        "action": "question",
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
    result = await graph.ainvoke(state)
    return result.get("question", "")


async def arun_evaluation(question: str, answer: str, topic: str, job_description: str = "", resume_context: str = "", history: Optional[List[dict]] = None) -> dict:
    """Async evaluation runner for legacy WebSocket handlers."""
    state: InterviewState = {
        "action": "evaluate",
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
    result = await graph.ainvoke(state)
    return {
        "feedback": result.get("feedback", ""),
        "report": result.get("report", "")
    }