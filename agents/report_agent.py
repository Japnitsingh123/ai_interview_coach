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


def report_agent(state):

    question = state["question"]

    answer = state["answer"]

    feedback = state["feedback"]

    prompt = f"""
You are an experienced technical interviewer.

Generate a final interview report based on the following information.

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluation Feedback:
{feedback}

Generate a concise report with the following sections:

1. Overall Performance
2. Technical Skills
3. Communication Skills
4. Strengths
5. Areas for Improvement
6. Final Recommendation

Keep the report professional and concise.
"""

    response = llm.invoke(prompt)

    state["report"] = response.content

    return state