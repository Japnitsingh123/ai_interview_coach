import os
import time
import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_resume
from graph import run_question, run_evaluation, run_final_report
from tts import text_to_speech
from streamlit_mic_recorder import mic_recorder
from stt import speech_to_text

load_dotenv()

# -------------------------
# Page Configuration & Setup
# -------------------------
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# LangSmith & WebSocket Configuration in Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ System Status & Config")

    # LangSmith Tracing Status
    ls_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not ls_api_key and hasattr(st, "secrets") and "LANGCHAIN_API_KEY" in st.secrets:
        ls_api_key = st.secrets["LANGCHAIN_API_KEY"]

    ls_enabled = bool(ls_api_key and len(ls_api_key.strip()) > 5)

    if ls_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = ls_api_key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-interview-coach")
        st.success("🟢 **LangSmith Tracing:** Active")
        st.caption(f"Project: `{os.environ['LANGCHAIN_PROJECT']}`")
        st.markdown("[📊 View Traces on LangSmith](https://smith.langchain.com/)")
    else:
        st.info("🟡 **LangSmith Tracing:** Standby")
        user_ls_key = st.text_input("Optional: LangSmith API Key", type="password", placeholder="lsv2_pt_...")
        if user_ls_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = user_ls_key
            os.environ["LANGCHAIN_PROJECT"] = "ai-interview-coach"
            st.success("🟢 LangSmith Connected!")

    st.divider()

    # Real-Time WebSocket Backend Status
    st.subheader("⚡ Real-Time WebSockets")
    ws_port = os.getenv("WS_PORT", "8000")
    st.write("WebSocket Server for Real-Time Streaming:")
    st.code(f"ws://localhost:{ws_port}/ws/interview", language="text")
    st.caption("Start standalone server with: `python ws_server.py`")

st.title("🤖 AI Interview Coach")
st.caption("Powered by Agentic AI — LangGraph Multi-Agent Pipeline | LangSmith Tracing | WebSockets")

# -------------------------
# Session State Initialization
# -------------------------
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "question" not in st.session_state:
    st.session_state.question = ""

if "answers" not in st.session_state:
    st.session_state.answers = []

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []

if "reports" not in st.session_state:
    st.session_state.reports = []

if "history" not in st.session_state:
    st.session_state.history = []

if "voice_answer" not in st.session_state:
    st.session_state.voice_answer = ""

if "topic" not in st.session_state:
    st.session_state.topic = "Projects"

if "job_description" not in st.session_state:
    st.session_state.job_description = ""


# -------------------------
# Streaming Text Generator Helper
# -------------------------
def stream_text(text: str):
    """Yield text chunks with slight delay for realistic typing stream."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


# -------------------------
# Resume Upload & Job Description
# -------------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

    if uploaded_file:
        os.makedirs("data/resumes", exist_ok=True)
        save_path = os.path.join("data/resumes", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("Create Resume Knowledge Base"):
            with st.spinner("Processing & embedding resume into ChromaDB..."):
                chunk_count = ingest_resume(save_path)
            st.success(f"Resume Indexed ({chunk_count} chunks)")

with col2:
    job_description = st.text_area(
        "Paste Job Description (Optional)",
        height=160,
        placeholder="e.g. Senior Python / AI Engineer with LangChain, FastAPI and SQL expertise..."
    )

    topic = st.selectbox(
        "Select Interview Focus Area",
        ["Projects", "Skills", "Education", "Experience"]
    )

st.session_state.topic = topic
st.session_state.job_description = job_description

st.divider()

# -------------------------
# Start Interview Button
# -------------------------
if st.session_state.question_count == 0:
    if st.button("🚀 Start Interview Session", type="primary"):
        st.session_state.question_count = 1
        st.session_state.answers = []
        st.session_state.feedbacks = []
        st.session_state.reports = []
        st.session_state.history = []

        with st.spinner("🤖 Question Agent is generating your first question..."):
            st.session_state.question = run_question(
                st.session_state.topic,
                st.session_state.job_description,
                question_number=1,
                history=[]
            )
        st.rerun()

# -------------------------
# Interview Complete — Final Report
# -------------------------
if st.session_state.question_count > 5:
    st.success("🎉 Interview Completed!")
    st.write(f"Total Questions Answered: {len(st.session_state.answers)} / 5")

    if "final_report" not in st.session_state:
        with st.spinner("🤖 Report Agent is generating your comprehensive final assessment..."):
            st.session_state.final_report = run_final_report(st.session_state.history)

    st.subheader("📊 Comprehensive Interview Report")
    st.markdown(st.session_state.final_report)

    if st.button("🔄 Start New Interview"):
        st.session_state.question_count = 0
        st.session_state.answers = []
        st.session_state.feedbacks = []
        st.session_state.reports = []
        st.session_state.history = []
        if "final_report" in st.session_state:
            del st.session_state["final_report"]
        st.rerun()

    st.stop()

# -------------------------
# Active Question & Answer Section
# -------------------------
if st.session_state.question_count > 0 and st.session_state.question_count <= 5:
    st.info(f"📍 Question {st.session_state.question_count} of 5 — Focus Area: **{st.session_state.topic}**")

    st.subheader("AI Interviewer Question")
    st.write(st.session_state.question)

    # Audio playback of interviewer question
    try:
        audio_file = text_to_speech(st.session_state.question)
        st.audio(audio_file)
    except Exception as e:
        st.caption(f"Audio playback note: {e}")

    st.markdown("### 🎙️ Your Response")
    st.write("Record your voice or type your answer below:")

    audio = mic_recorder(
        start_prompt="🎤 Click to Start Recording",
        stop_prompt="⏹ Stop Recording",
        key="recorder"
    )

    if audio:
        with open("answer.webm", "wb") as f:
            f.write(audio["bytes"])

        try:
            with st.spinner("Transcribing speech with Faster-Whisper..."):
                text = speech_to_text("answer.webm")
            st.session_state.voice_answer = text
            st.success("Voice transcribed successfully!")
        except Exception as e:
            st.error(f"Transcription Error: {e}")

    answer = st.text_area(
        "Candidate Answer",
        value=st.session_state.voice_answer,
        height=180,
        placeholder="Type or dictate your answer here..."
    )

    col_eval, col_next = st.columns([1, 1])

    with col_eval:
        if st.button("⚡ Evaluate Answer", type="primary"):
            if not answer.strip():
                st.error("Please provide an answer before evaluation.")
            else:
                with st.spinner("🤖 Evaluation & Report Agents are analyzing your answer with LangGraph..."):
                    result = run_evaluation(
                        st.session_state.question,
                        answer,
                        st.session_state.topic,
                        st.session_state.job_description,
                        history=st.session_state.history
                    )

                feedback = result["feedback"]
                report = result["report"]

                st.session_state.answers.append(answer)
                st.session_state.feedbacks.append(feedback)
                st.session_state.reports.append(report)

                st.session_state.history.append({
                    "question": st.session_state.question,
                    "answer": answer,
                    "feedback": feedback
                })

                st.subheader("📝 Evaluation Feedback")
                st.markdown(feedback)

                st.subheader("📋 Question Report")
                st.markdown(report)

    with col_next:
        if st.button("➡️ Next Question"):
            st.session_state.question_count += 1
            st.session_state.voice_answer = ""

            if st.session_state.question_count <= 5:
                with st.spinner("🤖 Question Agent is generating your next question..."):
                    st.session_state.question = run_question(
                        st.session_state.topic,
                        st.session_state.job_description,
                        question_number=st.session_state.question_count,
                        history=st.session_state.history
                    )
            st.rerun()