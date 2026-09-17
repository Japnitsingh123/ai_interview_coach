import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from agents.utils import strip_thinking, get_model_name

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


def generate_question(topic, job_description):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    try:
        docs = retriever.invoke(topic)
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        ) if docs else "No specific resume context found."
    except Exception:
        context = "Resume context not available."

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=get_model_name(),
        temperature=0.7
    )

    prompt = PromptTemplate(
        input_variables=["topic", "context", "job_description"],
        template="""
You are an experienced technical interviewer.

Candidate Resume:

{context}

Job Description:

{job_description}

Generate ONE interview question.

The question should be relevant to:

1. Candidate resume
2. Job description
3. Selected area: {topic}

Only return the question.
"""
    )   

    final_prompt = prompt.format(
        topic=topic,
        context=context,
        job_description=job_description
    )

    response = llm.invoke(final_prompt)

    return strip_thinking(response.content)