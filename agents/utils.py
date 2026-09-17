import os
import re
import streamlit as st

DEFAULT_MODEL = "qwen/qwen3.8-27b"


def get_model_name():
    """
    Get Groq model name from environment or Streamlit secrets.
    Defaults to 'qwen/qwen3.8-27b'.
    """
    model = os.getenv("GROQ_MODEL")
    if not model and hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
        model = st.secrets["GROQ_MODEL"]

    if not model:
        return DEFAULT_MODEL

    model = model.strip()

    # Automatically map older/typo names to the working Groq endpoint
    if "qwen3.6" in model.lower() or "llama" in model.lower():
        return DEFAULT_MODEL

    return model


def strip_thinking(text):
    """
    Strip <think>...</think> blocks from model output.
    Qwen and reasoning models emit thinking blocks that should not be shown to the user.
    """
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )
    return cleaned.strip()