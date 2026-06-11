# 🤖 AI Interview Coach

An AI-powered Interview Preparation Platform that simulates personalized technical interviews using Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and Resume-Aware Question Generation.

The system analyzes a candidate's resume, aligns interview questions with a target job description, conducts a multi-round mock interview, evaluates responses, and generates a detailed AI-powered interview report.

## 🚀 Features

### 📄 Resume-Aware Interview Generation

* Upload a resume in PDF format
* Extracts and processes resume content automatically
* Creates a searchable knowledge base using vector embeddings

### 🔍 RAG-Based Question Generation

* Uses ChromaDB as a vector database
* Retrieves relevant resume context through semantic search
* Generates personalized interview questions based on candidate background

### 🎯 Job Description Matching

* Accepts a target Job Description
* Aligns interview questions with company requirements
* Produces role-specific and skill-focused questions

### 🧠 Multi-Round Interview Simulation

* Conducts a complete 5-question mock interview
* Tracks interview progress in real time
* Supports multiple interview domains:

  * Projects
  * Skills
  * Education
  * Experience

### 📊 AI-Powered Answer Evaluation

* Evaluates candidate responses using Groq-hosted LLMs
* Provides technical feedback and performance assessment
* Simulates interviewer-style evaluation

### 📑 Final Interview Report

* Generates a comprehensive interview summary
* Highlights strengths and areas for improvement
* Provides actionable recommendations for preparation

---

## 🏗️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### LLM

* Groq (Llama 3.3 70B)

### RAG Framework

* LangChain

### Embedding Model

* BAAI/bge-small-en-v1.5

### Vector Database

* ChromaDB

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

### Environment Management

* Python Dotenv

---

## 🔄 System Workflow

Resume Upload
→ PDF Processing
→ Chunking
→ Embedding Generation
→ ChromaDB Vector Storage
→ Semantic Retrieval
→ Job Description Matching
→ Personalized Interview Question Generation
→ Candidate Response Evaluation
→ Multi-Round Interview Session
→ Final AI Interview Report

---

## 🎯 Project Highlights

* Retrieval-Augmented Generation (RAG)
* Resume-Based Question Generation
* Job Description Alignment
* Semantic Search with ChromaDB
* LLM-Powered Interview Evaluation
* Multi-Step Interview Simulation
* AI-Generated Performance Reports

This project demonstrates practical applications of Generative AI, RAG pipelines, vector databases, semantic retrieval, and LLM-powered evaluation systems in the recruitment and interview preparation domain.
