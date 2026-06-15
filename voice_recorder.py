# Voice recording is only available on local deployment
# On cloud (Hugging Face Spaces), this module is disabled gracefully

VOICE_AVAILABLE = False
whisper_model   = None
SAMPLE_RATE     = 16000

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import whisper
    import tempfile
    import os

    print("Loading Whisper model...")
    whisper_model   = whisper.load_model("base")
    VOICE_AVAILABLE = True
    print("Whisper ready!")

except Exception as e:
    print(f"Voice recording not available: {e}")
    VOICE_AVAILABLE = False


def record_and_transcribe(duration=30):
    if not VOICE_AVAILABLE:
        return None, "Voice journaling is only available on local deployment."

    try:
        import numpy as np
        import soundfile as sf
        import tempfile
        import os

        audio = sd.rec(int(duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if np.max(np.abs(audio)) < 0.01:
            return None, "No audio detected. Please check your microphone."

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)

        result = whisper_model.transcribe(
            tmp.name, language="en", fp16=False, verbose=False)
        os.unlink(tmp.name)

        text = result["text"].strip()
        return (text, None) if text else (None, "Could not transcribe.")
    except Exception as e:
        return None, f"Recording error: {str(e)}"