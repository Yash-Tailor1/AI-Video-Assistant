import os
import requests
import whisper

from dotenv import load_dotenv

load_dotenv()


# =========================
# Whisper Configuration
# =========================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


# =========================
# Sarvam Configuration
# =========================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_TRANSLATE_URL = (
    "https://indus.sarvam.ai/api-playground/speech-to-text"
)

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "Saaras v3"
)


# =========================
# Whisper Model
# =========================

def load_model():
    global _model

    if _model is None:
        print("Loading Whisper model...")

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded successfully.")

    return _model


# =========================
# Whisper Transcription
# =========================

def transcribe_chunk_whisper(
    chunk_path: str,
    translate: bool = False
) -> str:

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(
        chunk_path,
        task=task
    )

    return result["text"]


# =========================
# Sarvam Transcription
# =========================

from sarvamai import SarvamAI


def transcribe_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in .env"
        )

    client = SarvamAI(
        api_subscription_key=SARVAM_API_KEY
    )

    with open(chunk_path, "rb") as audio_file:

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="codemix"
        )

    return response.transcript


# =========================
# Single Chunk Router
# =========================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Route one chunk to Whisper or Sarvam.

    english  -> Whisper
    hinglish -> Sarvam
    """

    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# =========================
# Transcribe All Chunks
# =========================

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    full_transcript = ""

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

        print(
            f"Transcription completed "
            f"for chunk {i + 1}."
        )

    return full_transcript.strip()