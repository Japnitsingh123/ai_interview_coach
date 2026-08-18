import os
import streamlit as st

from ingest import ingest_resume
from graph import run_question, run_evaluation, run_final_report
from tts import text_to_speech
from streamlit_mic_recorder import mic_recorder
from stt import speech_to_text

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖"
)

st.title("🤖 AI Interview Coach")
st.caption("Powered by Agentic AI — LangGraph Pipeline")

# -------------------------
# Session State
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
# Resume Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=200
)

topic = st.selectbox(
    "Select Interview Area",
    [
        "Projects",
        "Skills",
        "Education",
        "Experience"
    ]
)

# Persist topic and JD in session state
st.session_state.topic = topic
st.session_state.job_description = job_description

if uploaded_file:

    os.makedirs(
        "data/resumes",
        exist_ok=True
    )

    save_path = os.path.join(
        "data/resumes",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Resume Uploaded")

    if st.button("Create Knowledge Base"):

        with st.spinner("Processing Resume..."):

            chunk_count = ingest_resume(
                save_path
            )

        st.success(
            f"Resume Processed Successfully ({chunk_count} chunks)"
        )

# -------------------------
# Start Interview
# -------------------------

if st.button("Start Interview"):

    st.session_state.question_count = 1

    st.session_state.answers = []

    st.session_state.feedbacks = []

    st.session_state.reports = []

    st.session_state.history = []

    with st.spinner("🤖 Agent is generating your first question..."):

        st.session_state.question = run_question(
            st.session_state.topic,
            st.session_state.job_description,
            question_number=1,
            history=[]
        )

# -------------------------
# Interview Complete — Final Report
# -------------------------

if st.session_state.question_count > 5:

    st.success("🎉 Interview Completed!")

    st.write(
        f"Questions Attempted: {len(st.session_state.answers)}"
    )

    # Generate final comprehensive report via the agentic graph
    if "final_report" not in st.session_state:

        with st.spinner(
            "🤖 Report Agent is generating your comprehensive final report..."
        ):

            st.session_state.final_report = run_final_report(
                st.session_state.history
            )

    st.subheader("📊 Comprehensive Interview Report")

    st.markdown(st.session_state.final_report)

    st.stop()

# -------------------------
# Current Question
# -------------------------

if st.session_state.question_count > 0:

    st.info(
        f"Question {st.session_state.question_count} of 5"
    )

    st.subheader("Interview Question")

    st.write(
        st.session_state.question
    )

    audio_file = text_to_speech(
        st.session_state.question
    )

    st.audio(audio_file)

    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹ Stop Recording",
        key="recorder"
    )

    if audio:

        with open(
            "answer.webm",
            "wb"
        ) as f:

            f.write(
                audio["bytes"]
            )

        try:

            text = speech_to_text(
                "answer.webm"
            )

            st.session_state.voice_answer = text

            st.success(
                "Voice converted to text!"
            )

        except Exception as e:

            st.error(
                f"Transcription Error: {e}"
            )

    answer = st.text_area(
        "Your Answer",
        value=st.session_state.voice_answer,
        height=200
    )

    if st.button("Evaluate Answer"):

        if not answer.strip():

            st.error(
                "Please enter an answer."
            )

        else:

            with st.spinner(
                "🤖 Evaluation & Report Agents are analyzing your answer..."
            ):

                result = run_evaluation(
                    st.session_state.question,
                    answer,
                    st.session_state.topic,
                    st.session_state.job_description,
                    history=st.session_state.history
                )

            feedback = result["feedback"]
            report = result["report"]

            st.session_state.answers.append(
                answer
            )

            st.session_state.feedbacks.append(
                feedback
            )

            st.session_state.reports.append(
                report
            )

            # Track Q&A history for the graph agents
            st.session_state.history.append({
                "question": st.session_state.question,
                "answer": answer,
                "feedback": feedback
            })

            st.subheader(
                "📝 Evaluation Feedback"
            )

            st.markdown(
                feedback
            )

            st.subheader(
                "📋 Question Report"
            )

            st.markdown(
                report
            )

    if st.button("Next Question"):

        st.session_state.question_count += 1

        st.session_state.voice_answer = ""

        if st.session_state.question_count <= 5:

            with st.spinner(
                "🤖 Question Agent is generating your next question..."
            ):

                st.session_state.question = run_question(
                    st.session_state.topic,
                    st.session_state.job_description,
                    question_number=st.session_state.question_count,
                    history=st.session_state.history
                )

        st.rerun()