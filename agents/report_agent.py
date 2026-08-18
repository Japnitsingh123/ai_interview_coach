import os

from dotenv import load_dotenv
from agents.utils import strip_thinking

from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="qwen/qwen3.6-27b",
    temperature=0
)


def report_agent(state):

    mode = state.get("mode", "evaluate")

    # Final report mode — generate cumulative report across all questions
    if mode == "final_report":
        return _generate_final_report(state)

    # Per-question report mode
    return _generate_question_report(state)


def _generate_question_report(state):

    question = state["question"]

    answer = state["answer"]

    feedback = state["feedback"]

    prompt = f"""
You are an experienced technical interviewer.

Generate a concise interview report for this question.

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluation Feedback:
{feedback}

Generate a brief report with:

1. Performance Summary
2. Key Strengths
3. Areas for Improvement
4. Quick Tip for the Candidate

Keep it concise and encouraging.
"""

    response = llm.invoke(prompt)

    return {
        "report": strip_thinking(response.content)
    }


def _generate_final_report(state):

    history = state.get("history", [])

    if not history:
        return {
            "report": "No interview data available for report generation."
        }

    # Build a comprehensive summary from all Q&A pairs
    qa_summary = ""
    for i, item in enumerate(history, 1):
        qa_summary += f"""
--- Question {i} ---
Question: {item.get('question', 'N/A')}
Answer: {item.get('answer', 'N/A')}
Feedback: {item.get('feedback', 'N/A')}
"""

    prompt = f"""
You are an experienced technical interviewer.

Generate a comprehensive FINAL interview report based on ALL the questions and answers below.

{qa_summary}

Generate a professional report with the following sections:

1. Overall Performance Rating (out of 10)
2. Technical Skills Assessment
3. Communication Skills Assessment
4. Top Strengths (across all answers)
5. Key Areas for Improvement
6. Detailed Question-by-Question Summary
7. Final Recommendation (Hire / Consider / Needs Improvement)
8. Personalized Study Plan

Make the report professional, detailed, and actionable.
"""

    response = llm.invoke(prompt)

    return {
        "report": strip_thinking(response.content)
    }