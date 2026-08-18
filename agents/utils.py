import re


def strip_thinking(text):
    """
    Strip <think>...</think> blocks from model output.
    Qwen and other reasoning models emit thinking blocks
    that should not be shown to the user.
    """
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )
    return cleaned.strip()
