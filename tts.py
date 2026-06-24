from gtts import gTTS


def text_to_speech(text):

    tts = gTTS(
        text=text,
        lang="en"
    )

    audio_file = "question.mp3"

    tts.save(audio_file)

    return audio_file