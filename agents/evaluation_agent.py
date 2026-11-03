import os
import streamlit as st
from dotenv import load_dotenv
from agents.utils import strip_thinking, get_model_name, get_groq_llm

load_dotenv()

# Safe initialization for module export & tests
llm = get_groq_llm(temperature=0, max_tokens=500)


def evaluation_agent(state):
    question = state["question"]
    answer = state["answer"]
    resume_context = state.get("resume_context", "")

    prompt = f"""
You are a senior technical interviewer.

Candidate Resume Context:
{resume_context}

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluate the answer based on the candidate's resume and the question asked.

Provide:

1. Technical Score (out of 10)
2. Communication Score (out of 10)
3. Strengths
4. Weaknesses
5. Improvement Suggestions

Keep the response concise and actionable.
"""

    active_llm = get_groq_llm(temperature=0, max_tokens=500)
    response = active_llm.invoke(prompt)

    return {
        "feedback": strip_thinking(response.content)
    }
