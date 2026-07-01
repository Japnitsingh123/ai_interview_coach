# 🤖 AI Interview Coach



An AI-powered interview preparation platform that leverages **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, and **semantic search** to conduct personalized technical interviews based on a candidate's resume and target job description.



The platform generates context-aware interview questions, supports multi-round mock interviews, evaluates candidate responses, provides AI-generated interview feedback, and includes voice-enabled interaction through Speech-to-Text and Text-to-Speech.



---



# 🚀 Features



### 📄 Resume Knowledge Base



* Upload resumes in PDF format

* Automatically extract and process resume content

* Build a semantic knowledge base using ChromaDB



### 🎯 Job Description Matching



* Paste a target Job Description

* Align interview questions with company requirements

* Generate role-specific interview questions



### 🧠 Personalized Question Generation



* Uses Retrieval-Augmented Generation (RAG)

* Retrieves relevant resume context using semantic search

* Generates interview questions tailored to:



  * Projects

  * Skills

  * Education

  * Experience



### 🎙️ Voice-Enabled Interview



* AI interviewer reads questions using Text-to-Speech (gTTS)

* Candidate answers through voice

* Speech converted to text using Faster-Whisper

* Supports both voice and text-based interviews



### 💬 AI Answer Evaluation



* Evaluates candidate responses using Groq-hosted LLMs

* Provides interviewer-style feedback

* Simulates a technical interview experience



### 📑 Multi-Round Interview



* Conducts a complete mock interview session

* Supports multiple interview rounds

* Tracks candidate responses throughout the interview



### 📊 AI Interview Report



* Generates a final interview summary

* Highlights strengths and improvement areas

* Provides actionable feedback for future preparation



### 🤖 Modular AI Agent Design



* Question Generation Agent

* Answer Evaluation Agent

* Interview Report Agent



The project includes modular LangGraph-compatible agent components to separate interview generation, evaluation, and reporting into independent AI workflows.



---



# 🏗️ Tech Stack



### Frontend



* Streamlit



### Backend



* Python



### LLM



* Groq (Llama 3.3 70B)



### RAG Framework



* LangChain



### Agent Framework



* LangGraph



### Vector Database



* ChromaDB



### Embedding Model



* BAAI/bge-small-en-v1.5



### Speech Processing



* Faster-Whisper (Speech-to-Text)

* gTTS (Text-to-Speech)



### Document Processing



* PyPDFLoader

* RecursiveCharacterTextSplitter



### Environment Management



* Python Dotenv



---



# 📂 Project Structure



```text

AI_INTERVIEW_COACH/

│

├── agents/

│   ├── question_agent.py

│   ├── evaluation_agent.py

│   ├── report_agent.py

│

├── app.py

├── graph.py

├── ingest.py

├── rag.py

├── evaluator.py

├── report.py

├── stt.py

├── tts.py

├── chroma_db/

├── data/

└── requirements.txt

```



---



# 🔄 System Workflow



```text

Resume Upload

        │

        ▼

Resume Processing

        │

        ▼

Chunking & Embedding Generation

        │

        ▼

ChromaDB Vector Store

        │

        ▼

Semantic Retrieval (RAG)

        │

        ▼

Job Description Matching

        │

        ▼

Interview Question Generation

        │

        ▼

Text-to-Speech (AI Interviewer)

        │

        ▼

Candidate Voice/Text Response

        │

        ▼

Speech-to-Text (Whisper)

        │

        ▼

AI Answer Evaluation

        │

        ▼

Interview Feedback

        │

        ▼

Final Interview Report

```



---



# 🤖 AI Agent Workflow



```text

                Resume + Job Description

                           │

                           ▼

                Question Generation Agent

                           │

                           ▼

                  Evaluation Agent

                           │

                           ▼

                 Interview Report Agent

```



---



# ✨ Highlights



* Resume-Aware Interview Generation

* Job Description Matching

* Retrieval-Augmented Generation (RAG)

* Semantic Search using ChromaDB

* Multi-Round Mock Interview

* AI-Based Answer Evaluation

* Voice Interview Support

* Speech-to-Text & Text-to-Speech

* Modular LangGraph Agent Design

* Personalized Interview Reporting



---



# 🛠️ Installation



```bash

git clone <repository-url>



cd AI_INTERVIEW_COACH



pip install -r requirements.txt



streamlit run app.py

```








