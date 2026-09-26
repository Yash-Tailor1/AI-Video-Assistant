from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def run_pipeline(
    source: str,
    language: str = "english",
) -> dict:

    print("\n" + "=" * 60)
    print("STARTING AI VIDEO ASSISTANT")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Process audio
    # --------------------------------------------------

    print("\n[1/7] Processing audio...")

    if language.lower() == "hinglish":
        # Sarvam normal STT API has 30-second limit
        chunk_minutes = 0.49
    else:
        chunk_minutes = 10

    chunks = process_input(
        source,
        chunk_minutes=chunk_minutes,
    )

    print(
        f"Audio ready - "
        f"{len(chunks)} chunk(s) created."
    )

    # --------------------------------------------------
    # 2. Transcription
    # --------------------------------------------------

    print("\n[2/7] Transcribing audio...")

    transcript = transcribe_all(
        chunks,
        language=language,
    )

    print(
        "\nRaw transcription "
        f"(first 300 characters):\n"
        f"{transcript[:300]}"
    )

    # --------------------------------------------------
    # 3. Generate title
    # --------------------------------------------------

    print("\n[3/7] Generating title...")

    try:
        title = generate_title(transcript)

    except Exception as e:
        print(f"Title generation failed: {e}")
        title = "Untitled Meeting"

    # --------------------------------------------------
    # 4. Generate summary
    # --------------------------------------------------

    print("\n[4/7] Generating summary...")

    try:
        summary = summarize(transcript)

    except Exception as e:
        print(f"Summary generation failed: {e}")
        summary = "Summary could not be generated."

    # --------------------------------------------------
    # 5. Extract action items
    # --------------------------------------------------

    print("\n[5/7] Extracting action items...")

    try:
        action_items = extract_action_items(transcript)

    except Exception as e:
        print(f"Action item extraction failed: {e}")
        action_items = "No action items found."

    # --------------------------------------------------
    # 6. Extract decisions and questions
    # --------------------------------------------------

    print("\n[6/7] Extracting decisions and questions...")

    try:
        decisions = extract_key_decisions(transcript)

    except Exception as e:
        print(f"Decision extraction failed: {e}")
        decisions = "No key decisions found."

    try:
        questions = extract_questions(transcript)

    except Exception as e:
        print(f"Question extraction failed: {e}")
        questions = "No open questions found."

    # --------------------------------------------------
    # 7. Build RAG
    # --------------------------------------------------

    print("\n[7/7] Building RAG chain...")

    try:
        rag_chain = build_rag_chain(transcript)

    except Exception as e:
        print(f"RAG creation failed: {e}")
        rag_chain = None

    print("\n" + "=" * 60)
    print("AI VIDEO ASSISTANT PIPELINE COMPLETED")
    print("=" * 60)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    # --------------------------------------------------
    # CLI input
    # --------------------------------------------------

    source = input(
        "\nEnter a YouTube URL or local file path: "
    ).strip()

    language = input(
        "Language (english/hinglish): "
    ).strip().lower() or "english"

    result = run_pipeline(
        source,
        language,
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)

    print("\n========== TITLE ==========")
    print(result["title"])

    print("\n========== SUMMARY ==========")
    print(result["summary"])

    print("\n========== ACTION ITEMS ==========")
    print(result["action_items"])

    print("\n========== KEY DECISIONS ==========")
    print(result["key_decisions"])

    print("\n========== OPEN QUESTIONS ==========")
    print(result["open_questions"])

    print("\n" + "=" * 60)

    # --------------------------------------------------
    # Phase 2 - Chat with meeting using RAG
    # --------------------------------------------------

    rag_chain = result["rag_chain"]

    if rag_chain is not None:

        print(
            "\nChat with your meeting "
            "(type 'exit' to quit)\n"
        )

        while True:

            question = input("You: ").strip()

            if question.lower() in [
                "exit",
                "quit",
                "q",
            ]:

                print("Goodbye!")
                break

            if not question:
                continue

            try:

                answer = ask_question(
                    rag_chain,
                    question,
                )

                print(
                    f"\nAssistant: {answer}\n"
                )

            except Exception as e:

                print(
                    f"\nRAG error: {e}\n"
                )

    else:

        print(
            "\nRAG is unavailable, "
            "so meeting chat cannot be started."
        )