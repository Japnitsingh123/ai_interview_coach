import os
import streamlit as st
from dotenv import load_dotenv
from agents.utils import strip_thinking, get_model_name

from langchain_groq import ChatGroq

load_dotenv()

# Support both .env and Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

MODEL_NAME = get_model_name()

# Setting max_tokens=500 stays safely below Groq's 1000 OTPM rate limit
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0,
    max_tokens=500
)


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

    response = llm.invoke(prompt)

    return {
        "feedback": strip_thinking(response.content)
    }