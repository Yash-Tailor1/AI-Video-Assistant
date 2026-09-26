import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio as MP3."""

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info_dict)
        filename = os.path.splitext(filename)[0] + ".mp3"

        return filename


def convert_to_wav(input_path: str) -> str:
    """Convert audio to 16kHz mono WAV."""

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path

def chunk_audio(wav_path: str, chunk_minutes: float = 10) -> list:
    """Split WAV into chunks."""

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = int(chunk_minutes * 60 * 1000)

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):
        chunk = audio[start:start + chunk_ms]

        chunk_path = (
            f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        )

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def create_original_wav(input_path: str) -> str:
    """Create original WAV without changing sample rate/channels."""

    output_path = (
        os.path.splitext(input_path)[0] + "_original.wav"
    )

    audio = AudioSegment.from_file(input_path)

    audio.export(output_path, format="wav")

    return output_path


def process_input(source: str, chunk_minutes: float = 10) -> list:

    if source.startswith("http://") or source.startswith("https://"):

        print("Detected YouTube URL. Downloading audio...")

        mp3_path = download_youtube_audio(source)

        print(f"Audio downloaded: {mp3_path}")

        wav_path = convert_to_wav(mp3_path)

        print(f"Audio converted: {wav_path}")

    else:

        print("Detected local file.")

        wav_path = convert_to_wav(source)

        print(f"Audio converted: {wav_path}")

    print("Chunking audio...")

    chunks = chunk_audio(
        wav_path,
        chunk_minutes=chunk_minutes
    )

    print(
        f"Audio ready - {len(chunks)} chunk(s) created."
    )

    return chunks


if __name__ == "__main__":

    source = "https://youtu.be/HDqd8oJxlG8"

    chunks = process_input(source)

    print("\nFiles created:")

    for chunk in chunks:
        print(chunk)

