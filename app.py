import os
import streamlit as st

from ingest import ingest_resume
from rag import generate_question
from evaluator import evaluate_answer

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖"
)

st.title("🤖 AI Interview Coach")

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

    st.session_state.question = generate_question(
        topic,
        job_description
    )

# -------------------------
# Interview Complete
# -------------------------

if st.session_state.question_count > 5:

    st.success("🎉 Interview Completed!")

    st.write(
        f"Questions Attempted: {len(st.session_state.answers)}"
    )

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

    answer = st.text_area(
        "Your Answer",
        height=200,
        key=f"answer_{st.session_state.question_count}"
    )

    if st.button("Evaluate Answer"):

        if not answer.strip():

            st.error(
                "Please enter an answer."
            )

        else:

            with st.spinner(
                "Evaluating..."
            ):

                feedback = evaluate_answer(
                    st.session_state.question,
                    answer
                )

            st.session_state.answers.append(
                answer
            )

            st.session_state.feedbacks.append(
                feedback
            )

            st.subheader(
                "Interview Feedback"
            )

            st.write(
                feedback
            )

    if st.button("Next Question"):

        st.session_state.question_count += 1

        if st.session_state.question_count <= 5:

            st.session_state.question = generate_question(
                topic,
                job_description
            )

        st.rerun()