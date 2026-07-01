from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.question_agent import question_agent
from agents.evaluation_agent import evaluation_agent
from agents.report_agent import report_agent


class InterviewState(TypedDict):
    topic: str
    job_description: str
    answer: str
    resume_context: str
    question: str
    feedback: str
    report: str


builder = StateGraph(InterviewState)

builder.add_node("question", question_agent)
builder.add_node("evaluation", evaluation_agent)
builder.add_node("report", report_agent)

builder.set_entry_point("question")

builder.add_edge("question", "evaluation")
builder.add_edge("evaluation", "report")
builder.add_edge("report", END)

graph = builder.compile()