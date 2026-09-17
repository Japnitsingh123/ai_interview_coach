import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.utils import strip_thinking, get_model_name

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


def evaluate_answer(question, answer):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=get_model_name(),
        temperature=0
    )

    prompt = f"""
You are a senior technical interviewer.

Question:
{question}

Candidate Answer:
{answer}

Provide output EXACTLY in this format:

Technical Score: X/10

Feedback:
<feedback>
"""

    response = llm.invoke(prompt)

    return strip_thinking(response.content)