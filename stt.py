_model = None


def _patch_av():
    """Patch av.open to remove metadata_errors for PyAV 18+ compatibility."""
    try:
        import av
        if not hasattr(av, "_patched_metadata_errors"):
            _orig_open = av.open

            def _safe_av_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return _orig_open(*args, **kwargs)

            av.open = _safe_av_open
            av._patched_metadata_errors = True
    except Exception:
        pass


def get_model():
    global _model
    if _model is None:
        _patch_av()
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
    _patch_av()
    m = get_model()
    segments, _ = m.transcribe(
        audio_path
    )

    text = ""
    for segment in segments:
        text += segment.text + " "

    return text.strip()
