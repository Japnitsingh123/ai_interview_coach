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

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Coach — Autonomous Mock Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1E1E2F 0%, #2D2D44 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
    }

    /* Badges & Status Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-primary {
        background: rgba(96, 165, 250, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    .badge-success {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-purple {
        background: rgba(167, 139, 250, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }

    /* Question Card */
    .question-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #334155;
        border-left: 5px solid #60A5FA;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .question-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .question-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC;
        line-height: 1.6;
    }

    /* Feedback Card */
    .feedback-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }

    /* Pulsing Green Dot Animation */
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Sidebar Clean Enhancements */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: System Status & Tracing
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Architecture")

    # LangSmith Tracing Panel
    ls_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not ls_api_key and hasattr(st, "secrets") and "LANGCHAIN_API_KEY" in st.secrets:
        ls_api_key = st.secrets["LANGCHAIN_API_KEY"]

    ls_enabled = bool(ls_api_key and len(ls_api_key.strip()) > 5)

    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        if ls_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = ls_api_key
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-interview-coach")
            st.markdown('<div><span class="pulse-dot"></span> <b>LangSmith Observability</b></div>', unsafe_allow_html=True)
            st.caption(f"Traces active for: `{os.environ['LANGCHAIN_PROJECT']}`")
            st.markdown("[📊 Open Tracing Dashboard](https://smith.langchain.com/)")
        else:
            st.markdown("🟡 **LangSmith Tracing: Standby**")
            user_ls_key = st.text_input("Enter LangSmith API Key", type="password", placeholder="lsv2_pt_...")
            if user_ls_key:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = user_ls_key
                os.environ["LANGCHAIN_PROJECT"] = "ai-interview-coach"
                st.success("🟢 Connected to LangSmith!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Real-Time WebSocket Status
    with st.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div><span class="pulse-dot"></span> <b>Real-Time WebSockets</b></div>', unsafe_allow_html=True)
        ws_port = os.getenv("WS_PORT", "8000")
        st.caption(f"Duplex Streaming Port: `{ws_port}`")
        st.code(f"ws://localhost:{ws_port}/ws/interview", language="text")
        st.caption("Start standalone server: `python ws_server.py`")
        st.markdown('</div>', unsafe_allow_html=True)

    # Candidate Prep Tips
    with st.expander("💡 Candidate Success Tips"):
        st.markdown("""
        * **STAR Technique:** Structure answers by *Situation, Task, Action, Result*.
        * **Be Specific:** Mention concrete metrics, technologies, and trade-offs.
        * **Speak Clearly:** Use the voice recorder or type concise technical answers.
        """)

    # Reset Session
    if st.session_state.get("question_count", 0) > 0:
        st.divider()
        if st.button("🔄 Reset Interview Session", use_container_width=True):
            st.session_state.question_count = 0
            st.session_state.answers = []
            st.session_state.feedbacks = []
            st.session_state.reports = []
            st.session_state.history = []
            if "final_report" in st.session_state:
                del st.session_state["final_report"]
            st.rerun()

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🤖 AI Technical Interview Coach</div>
    <p class="hero-subtitle">Agentic Multi-Round Technical Preparation powered by LangGraph, ChromaDB, Groq LLM & Whisper</p>
    <div style="margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-pill badge-primary">✨ LangGraph Pipeline</span>
        <span class="badge-pill badge-success">🎙️ Voice & Faster-Whisper</span>
        <span class="badge-pill badge-purple">⚡ Groq LPU Accelerated</span>
        <span class="badge-pill badge-primary">🔍 LangSmith Tracing</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Step 1: Resume Setup & Configuration (Before Interview)
# ---------------------------------------------------------
if st.session_state.question_count == 0:
    st.markdown("### 📋 1. Setup Your Interview Context")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("#### 📄 Candidate Resume")
        uploaded_file = st.file_uploader(
            "Upload your Resume (PDF format)",
            type=["pdf"],
            help="Upload your technical resume to generate context-aware questions tailored to your experience."
        )

        if uploaded_file:
            os.makedirs("data/resumes", exist_ok=True)
            save_path = os.path.join("data/resumes", uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ Uploaded: **{uploaded_file.name}**")

            if st.button("⚡ Index Resume in ChromaDB Vector Store", use_container_width=True):
                with st.spinner("Chunking and generating semantic embeddings with BAAI/bge-small..."):
                    chunk_count = ingest_resume(save_path)
                st.success(f"🎉 Knowledge Base Ready ({chunk_count} semantic chunks indexed)")

    with col2:
        st.markdown("#### 🎯 Target Role & Focus Area")
        job_description = st.text_area(
            "Target Job Description (Optional)",
            height=130,
            placeholder="Paste target job requirements (e.g. Backend Software Engineer, ML Engineer with Python, LangChain, Distributed Systems)..."
        )

        topic = st.selectbox(
            "Select Interview Core Focus",
            ["Projects", "Skills", "Education", "Experience"],
            help="Choose the primary category for technical question generation."
        )

    st.session_state.topic = topic
    st.session_state.job_description = job_description

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start 5-Round Mock Interview", type="primary", use_container_width=True):
        st.session_state.question_count = 1
        st.session_state.answers = []
        st.session_state.feedbacks = []
        st.session_state.reports = []
        st.session_state.history = []

        with st.spinner("🤖 Question Agent is analyzing your profile and crafting your first question..."):
            st.session_state.question = run_question(
                st.session_state.topic,
                st.session_state.job_description,
                question_number=1,
                history=[]
            )
        st.rerun()

# ---------------------------------------------------------
# Step 2: Active Technical Interview (Rounds 1 to 5)
# ---------------------------------------------------------
elif st.session_state.question_count >= 1 and st.session_state.question_count <= 5:
    # Visual Progress Bar
    progress_val = int((st.session_state.question_count - 1) / 5 * 100)
    col_prog1, col_prog2 = st.columns([4, 1])
    with col_prog1:
        st.progress(progress_val / 100, text=f"Interview Progress: Round {st.session_state.question_count} of 5")
    with col_prog2:
        st.markdown(f'<div style="text-align:right;"><span class="badge-pill badge-primary">Topic: {st.session_state.topic}</span></div>', unsafe_allow_html=True)

    # Question Display Card
    st.markdown(f"""
    <div class="question-card">
        <div class="question-header">🤖 AI Interviewer &bull; Question {st.session_state.question_count} of 5</div>
        <div class="question-text">{st.session_state.question}</div>
    </div>
    """, unsafe_allow_html=True)

    # Audio Player for Voice Questions
    try:
        audio_file = text_to_speech(st.session_state.question)
        st.audio(audio_file)
    except Exception as e:
        pass

    st.markdown("---")

    # Response Section
    st.markdown("#### 🎙️ Your Response")
    st.caption("Answer through voice recording or type your technical response below:")

    col_mic, col_status = st.columns([1, 2])
    with col_mic:
        audio = mic_recorder(
            start_prompt="🎤 Start Recording Answer",
            stop_prompt="⏹ Stop & Transcribe",
            key=f"recorder_{st.session_state.question_count}"
        )

    if audio:
        with open("answer.webm", "wb") as f:
            f.write(audio["bytes"])

        try:
            with st.spinner("Transcribing speech with Faster-Whisper..."):
                text = speech_to_text("answer.webm")
            st.session_state.voice_answer = text
            st.success("✅ Voice transcribed successfully!")
        except Exception as e:
            st.error(f"Transcription error: {e}")

    answer = st.text_area(
        "Candidate Answer Text",
        value=st.session_state.voice_answer,
        height=180,
        placeholder="Structure your answer with technical details, architecture decisions, and metrics...",
        key=f"answer_input_{st.session_state.question_count}"
    )

    # Action Buttons
    col_eval, col_next = st.columns([1, 1], gap="medium")

    with col_eval:
        if st.button("⚡ Evaluate Answer", type="primary", use_container_width=True):
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

                # Display Feedback Cards
                st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
                st.markdown("### 📝 Evaluation Feedback & Scoring")
                st.markdown(feedback)
                st.markdown("---")
                st.markdown("### 📋 Question Performance Report")
                st.markdown(report)
                st.markdown('</div>', unsafe_allow_html=True)

    with col_next:
        if st.button("➡️ Advance to Next Question", use_container_width=True):
            st.session_state.question_count += 1
            st.session_state.voice_answer = ""

            if st.session_state.question_count <= 5:
                with st.spinner("🤖 Question Agent is crafting your next interview question..."):
                    st.session_state.question = run_question(
                        st.session_state.topic,
                        st.session_state.job_description,
                        question_number=st.session_state.question_count,
                        history=st.session_state.history
                    )
            st.rerun()

# ---------------------------------------------------------
# Step 3: Interview Completed — Final Comprehensive Assessment
# ---------------------------------------------------------
elif st.session_state.question_count > 5:
    st.markdown("""
    <div style="text-align: center; padding: 24px; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 16px; margin-bottom: 24px;">
        <h2 style="color: #34D399; margin: 0;">🎉 Interview Session Completed!</h2>
        <p style="color: #94A3B8; margin-top: 8px;">All 5 technical interview rounds successfully completed and analyzed.</p>
    </div>
    """, unsafe_allow_html=True)

    # Generate final report if not present
    if "final_report" not in st.session_state:
        with st.spinner("🤖 Report Agent is compiling your comprehensive evaluation across all 5 questions..."):
            st.session_state.final_report = run_final_report(st.session_state.history)

    # Metrics Overview
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Questions Attempted", len(st.session_state.answers))
    with metric_col2:
        st.metric("Focus Domain", st.session_state.topic)
    with metric_col3:
        st.metric("Session Status", "Evaluated ✅")

    st.markdown("### 📊 Comprehensive Performance Report")
    st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_down, col_restart = st.columns([1, 1], gap="medium")
    with col_down:
        st.download_button(
            label="📥 Download Interview Assessment (.md)",
            data=st.session_state.final_report,
            file_name="interview_performance_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_restart:
        if st.button("🔄 Start a New Interview Session", type="primary", use_container_width=True):
            st.session_state.question_count = 0
            st.session_state.answers = []
            st.session_state.feedbacks = []
            st.session_state.reports = []
            st.session_state.history = []
            if "final_report" in st.session_state:
                del st.session_state["final_report"]
            st.rerun()