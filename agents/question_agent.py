import os

from dotenv import load_dotenv
from agents.utils import strip_thinking

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="qwen/qwen3.6-27b",
    temperature=0.7
)


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

    docs = retriever.invoke(topic)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.format(
        topic=topic,
        context=context,
        job_description=job_description,
        question_number=question_number,
        previous_questions=previous_questions
    )

    response = llm.invoke(final_prompt)

    return {
        "question": strip_thinking(response.content),
        "resume_context": context
    }