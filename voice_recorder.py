import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import tempfile
import os

# Load Whisper model once at module level (singleton)
# 'base' model = 150MB, fast, good accuracy for clear speech
# Options: tiny(75MB), base(150MB), small(500MB), medium(1.5GB)
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper ready!")

SAMPLE_RATE = 16000  # Whisper requires 16kHz audio

def record_audio(duration=30):
    """
    Record audio from microphone for given duration.
    Returns numpy array of audio data.
    duration: max seconds to record (user can stop early)
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is complete
    return audio.flatten()

def save_audio_temp(audio_data):
    """Save audio numpy array to a temporary WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_data, SAMPLE_RATE)
    return tmp.name

def transcribe_audio(audio_path):
    """
    Transcribe audio file using OpenAI Whisper.
    Returns transcribed text string.
    """
    try:
        result = whisper_model.transcribe(
            audio_path,
            language="en",        # Force English (change to None for auto-detect)
            fp16=False,           # Use fp32 for CPU compatibility
            verbose=False
        )
        text = result["text"].strip()
        return text if text else None
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except Exception:
            pass

def record_and_transcribe(duration=30):
    """
    Master function — record audio then transcribe.
    Returns transcribed text or None if failed.
    """
    try:
        audio_data = record_audio(duration)
        # Check if audio has actual sound (not silence)
        if np.max(np.abs(audio_data)) < 0.01:
            return None, "No audio detected. Please check your microphone."
        audio_path = save_audio_temp(audio_data)
        text       = transcribe_audio(audio_path)
        if not text:
            return None, "Could not transcribe. Please speak clearly and try again."
        return text, None
    except Exception as e:
        return None, f"Recording error: {str(e)}"