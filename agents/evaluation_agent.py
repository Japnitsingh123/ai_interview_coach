import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)


def evaluation_agent(state):

    question = state["question"]

    answer = state["answer"]

    prompt = f"""
You are a senior technical interviewer.

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluate the answer.

Provide:

1. Technical Score (out of 10)
2. Communication Score (out of 10)
3. Strengths
4. Weaknesses
5. Improvement Suggestions

Keep the response concise.
"""

    response = llm.invoke(prompt)

    state["feedback"] = response.content

    return state