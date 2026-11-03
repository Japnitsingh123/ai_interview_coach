_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
    return _model


class _LazyModelProxy:
    def __getattr__(self, name):
        return getattr(get_model(), name)


model = _LazyModelProxy()


def speech_to_text(audio_path):
    m = get_model()
    segments, _ = m.transcribe(
        audio_path
    )

    text = ""
    for segment in segments:
        text += segment.text + " "

    return text
