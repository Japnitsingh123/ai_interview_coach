import os
import streamlit as st
from dotenv import load_dotenv
from agents.utils import strip_thinking, get_model_name, get_groq_llm
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Safe initialization for module export & tests
llm = get_groq_llm(temperature=0.7, max_tokens=300)

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        _retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
    return _retriever


class _LazyRetrieverProxy:
    def invoke(self, *args, **kwargs):
        return get_retriever().invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_retriever(), name)


retriever = _LazyRetrieverProxy()

prompt = PromptTemplate(
    input_variables=[
        "topic",
        "context",
        "job_description",
        "question_number",
        "previous_questions"
    ],
    template="""
You are an experienced technical interviewer.

Candidate Resume:

{context}

Job Description:

{job_description}

This is question {question_number} of 5 in the interview.

Generate ONE interview question.

The question should be relevant to:

1. Candidate Resume
2. Job Description
3. Selected Topic: {topic}

IMPORTANT: Do NOT repeat any of the following previously asked questions:
{previous_questions}

Make the question progressively more challenging as the question number increases.

Only return the interview question.
"""
)


def question_agent(state):
    topic = state["topic"]
    job_description = state["job_description"]
    question_number = state.get("question_number", 1)
    history = state.get("history", [])

    # Build previous questions list for the prompt
    if history:
        previous_questions = "\n".join(
            [f"- {item['question']}" for item in history]
        )
    else:
        previous_questions = "None (this is the first question)"

    try:
        docs = retriever.invoke(topic)
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        ) if docs else "No specific resume context found."
    except Exception:
        context = "Resume context not available."

    final_prompt = prompt.format(
        topic=topic,
        context=context,
        job_description=job_description,
        question_number=question_number,
        previous_questions=previous_questions
    )

    active_llm = get_groq_llm(temperature=0.7, max_tokens=300)
    response = active_llm.invoke(final_prompt)

    return {
        "question": strip_thinking(response.content),
        "resume_context": context
    }
