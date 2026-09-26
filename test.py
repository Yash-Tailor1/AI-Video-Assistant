from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all


load_dotenv()


source = "https://youtu.be/VfNluJsP1fY"

language = "hinglish"


# Sarvam REST API supports max 30 seconds
chunks = process_input(
    source,
    chunk_minutes=0.49
)


transcript = transcribe_all(
    chunks,
    language=language
)


print("\n=== TRANSCRIPT ===\n")
print(transcript)