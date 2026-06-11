import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_question(topic,job_description):

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

    docs = retriever.invoke(topic)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

    prompt = PromptTemplate(
        input_variables=["topic","context","job_description"],
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

    return response.content