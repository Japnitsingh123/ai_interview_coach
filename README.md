# 🤖 AI Interview Coach

An AI-powered mock interview platform that generates **resume-aware interview questions**, evaluates candidate answers, and generates a comprehensive interview report.

The system combines **RAG, LangGraph multi-agent orchestration, LLMs, and voice interaction** to provide an interactive interview experience.

## 🚀 Features

* 📄 **Resume-Aware Interview** — Generates questions using the candidate's resume, selected interview topic, and job description.
* 🔎 **RAG-Based Question Generation** — Retrieves relevant resume chunks using semantic search.
* 🤖 **Multi-Agent Architecture** — Uses separate agents for question generation, answer evaluation, and reporting.
* 🎤 **Voice Interview** — Supports Speech-to-Text and Text-to-Speech.
* 📊 **AI Answer Evaluation** — Evaluates technical knowledge, communication, strengths, weaknesses, and improvement areas.
* 📝 **Interview Reports** — Generates question-wise feedback and a comprehensive final report.
* 📚 **Interview History** — Maintains previous questions, answers, and feedback throughout the interview.
* 📈 **Progressive Difficulty** — Generates increasingly challenging questions across the 5-question interview.

---

## 🏗️ Architecture

```text
                Resume + Job Description
                         │
                         ▼
                  Resume Ingestion
                         │
                         ▼
              Chunking + Embeddings
                         │
                         ▼
                      ChromaDB
                         │
                         ▼
                  ┌───────────────┐
                  │ Question Agent│
                  └───────┬───────┘
                          │
                          ▼
                   Interview Question
                          │
                          ▼
                    Candidate Answer
                          │
                          ▼
                ┌──────────────────┐
                │ Evaluation Agent │
                └─────────┬────────┘
                          │
                          ▼
                   Evaluation Feedback
                          │
                          ▼
                  ┌──────────────┐
                  │ Report Agent │
                  └──────┬───────┘
                         │
                         ▼
                  Interview Report
```

**LangGraph** manages the agent workflow using a shared `InterviewState`.

---

## 🔎 RAG Pipeline

```text
Resume PDF
    ↓
PyPDFLoader
    ↓
Text Chunking
    ↓
Hugging Face Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
Top 3 Resume Chunks
    ↓
LLM + Resume Context + JD + Topic
    ↓
Interview Question
```

Currently, the **selected interview topic** is used as the retrieval query. The retrieved resume context is then combined with the **job description and selected topic** before being passed to the Question Agent.

---

## 🤖 Multi-Agent Workflow

### Question Agent

Generates one interview question using:

* Retrieved resume context
* Job description
* Selected interview topic
* Question number
* Previous interview questions

It also increases difficulty across the five questions and avoids repeating previous questions.

### Evaluation Agent

Evaluates the candidate's answer based on:

* Technical knowledge
* Communication
* Strengths
* Weaknesses
* Improvement suggestions

### Report Agent

Generates:

* Question-level performance reports
* Overall performance rating
* Technical assessment
* Communication assessment
* Strengths and weaknesses
* Final recommendation
* Personalized study plan

---

## 🎤 Voice Interview

```text
AI Question
     ↓
Text-to-Speech
     ↓
Candidate
     ↓
Voice Answer
     ↓
Speech-to-Text
     ↓
Evaluation
```

The application supports both **text and voice-based answers**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Frontend:** Streamlit
* **LLM:** Groq
* **Agent Orchestration:** LangGraph
* **LLM Framework:** LangChain
* **RAG:** ChromaDB, Hugging Face Embeddings
* **Embeddings:** `BAAI/bge-small-en-v1.5`
* **Speech-to-Text:** Faster Whisper
* **Text-to-Speech:** gTTS
* **PDF Processing:** PyPDF
* **Vector Search:** ChromaDB

---

## ⚙️ Setup

### Clone the repository

```bash
git clone <repository-url>
cd ai_interview_coach
```

### Create virtual environment

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### Run

```bash
streamlit run app.py
```

---

## 🔄 Interview Flow

```text
Upload Resume
      ↓
Create Knowledge Base
      ↓
Enter Job Description
      ↓
Select Interview Topic
      ↓
Start Interview
      ↓
Question Agent
      ↓
Candidate Answer
      ↓
Evaluation Agent
      ↓
Report Agent
      ↓
Store Q&A + Feedback in History
      ↓
Next Question
      ↓
After 5 Questions
      ↓
Final Report
```

---

## 🔮 Future Improvements

### 1. LLM-Based Evaluation Metrics

Integrate **RAGAS or similar LLM-based evaluation frameworks** to quantitatively evaluate the quality of generated questions, retrieved context, and responses.

### 2. Improved Retrieval Query

Use both the **Job Description + Selected Topic** as the retrieval query instead of using only the selected topic, enabling more role-specific resume retrieval.

### 3. Layout-Aware / OCR Document Processing

Support **layout-aware document parsing and OCR** to better handle resumes containing tables, multiple columns, images, and complex formatting.

### 4. Scalable Vector Database

Replace ChromaDB with a more scalable vector database such as **FAISS, Pinecone, Weaviate, or Milvus** for larger-scale document storage and retrieval.

---

## 👨‍💻 Author

**Japnit Singh Sawhney**

B.Tech Computer Engineering
Thapar Institute of Engineering & Technology
