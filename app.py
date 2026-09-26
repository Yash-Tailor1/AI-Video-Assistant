import os
import streamlit as st
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


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎥 AI Video Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Transform YouTube videos or local audio/video files "
    "into transcripts, summaries and actionable insights."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    input_type = st.radio(
        "Input type",
        [
            "YouTube URL",
            "Local File",
        ],
    )

    language = st.selectbox(
        "Transcription language",
        [
            "english",
            "hinglish",
        ],
    )

    st.divider()

    st.markdown("### Pipeline")

    st.markdown(
        """
        1. 🎵 Audio processing
        2. 📝 Transcription
        3. 🏷️ Title generation
        4. 📋 Summary
        5. ✅ Action items
        6. 🎯 Decisions & questions
        7. 🔎 RAG knowledge base
        """
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📥 Input")

source = None

if input_type == "YouTube URL":

    source = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

else:

    uploaded_file = st.file_uploader(
        "Upload audio/video",
        type=[
            "mp3",
            "wav",
            "m4a",
            "mp4",
            "webm",
            "mov",
            "avi",
        ],
    )


# ============================================================
# PROCESS BUTTON
# ============================================================

process_button = st.button(
    "🚀 Analyze",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if process_button:

    # ---------------------------------------------
    # Validate input
    # ---------------------------------------------

    if input_type == "YouTube URL":

        if not source:

            st.error("Please enter a YouTube URL.")

            st.stop()

    else:

        if uploaded_file is None:

            st.error("Please upload an audio/video file.")

            st.stop()

        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(
            "uploads",
            uploaded_file.name,
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        source = file_path

    # ---------------------------------------------
    # Progress
    # ---------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    try:

        # ====================================================
        # STEP 1 - AUDIO PROCESSING
        # ====================================================

        status.info("🎵 Processing audio...")

        progress.progress(10)

        if language.lower() == "hinglish":

            chunk_minutes = 0.49

        else:

            chunk_minutes = 10

        chunks = process_input(
            source,
            chunk_minutes=chunk_minutes,
        )

        progress.progress(20)

        # ====================================================
        # STEP 2 - TRANSCRIPTION
        # ====================================================

        status.info("📝 Transcribing audio...")

        transcript = transcribe_all(
            chunks,
            language=language,
        )

        progress.progress(40)

        # ====================================================
        # STEP 3 - TITLE
        # ====================================================

        status.info("🏷️ Generating title...")

        try:

            title = generate_title(
                transcript
            )

        except Exception as e:

            title = "Untitled Meeting"

            st.warning(
                f"Title generation failed: {e}"
            )

        progress.progress(50)

        # ====================================================
        # STEP 4 - SUMMARY
        # ====================================================

        status.info("📋 Generating summary...")

        try:

            summary = summarize(
                transcript
            )

        except Exception as e:

            summary = (
                "Summary could not be generated."
            )

            st.warning(
                f"Summary generation failed: {e}"
            )

        progress.progress(60)

        # ====================================================
        # STEP 5 - ACTION ITEMS
        # ====================================================

        status.info(
            "✅ Extracting action items..."
        )

        try:

            action_items = extract_action_items(
                transcript
            )

        except Exception as e:

            action_items = (
                "No action items found."
            )

            st.warning(
                f"Action item extraction failed: {e}"
            )

        progress.progress(70)

        # ====================================================
        # STEP 6 - DECISIONS & QUESTIONS
        # ====================================================

        status.info(
            "🎯 Extracting decisions and questions..."
        )

        try:

            decisions = extract_key_decisions(
                transcript
            )

        except Exception as e:

            decisions = (
                "No key decisions found."
            )

            st.warning(
                f"Decision extraction failed: {e}"
            )

        try:

            questions = extract_questions(
                transcript
            )

        except Exception as e:

            questions = (
                "No open questions found."
            )

            st.warning(
                f"Question extraction failed: {e}"
            )

        progress.progress(80)

        # ====================================================
        # STEP 7 - RAG
        # ====================================================

        status.info(
            "🔎 Building meeting knowledge base..."
        )

        try:

            rag_chain = build_rag_chain(
                transcript
            )

        except Exception as e:

            rag_chain = None

            st.warning(
                f"RAG creation failed: {e}"
            )

        progress.progress(100)

        status.success(
            "✅ Analysis completed successfully!"
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        st.session_state.result = {

            "title": title,

            "transcript": transcript,

            "summary": summary,

            "action_items": action_items,

            "key_decisions": decisions,

            "open_questions": questions,

            "chunks": chunks,

        }

        st.session_state.rag_chain = rag_chain

        st.session_state.chat_history = []

    except Exception as e:

        st.error(
            f"❌ Pipeline failed:\n\n{e}"
        )

        st.stop()


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result:

    st.divider()

    # ========================================================
    # TITLE
    # ========================================================

    st.header("📌 " + result["title"])


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📋 Summary",
            "✅ Action Items",
            "🎯 Decisions",
            "❓ Questions",
            "📝 Transcript",
            "💬 Ask AI",
        ]
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    with tab1:

        st.subheader("Meeting Summary")

        st.write(
            result["summary"]
        )


    # ========================================================
    # ACTION ITEMS
    # ========================================================

    with tab2:

        st.subheader("Action Items")

        st.write(
            result["action_items"]
        )


    # ========================================================
    # DECISIONS
    # ========================================================

    with tab3:

        st.subheader("Key Decisions")

        st.write(
            result["key_decisions"]
        )


    # ========================================================
    # QUESTIONS
    # ========================================================

    with tab4:

        st.subheader("Open Questions")

        st.write(
            result["open_questions"]
        )


    # ========================================================
    # TRANSCRIPT
    # ========================================================

    with tab5:

        st.subheader("Full Transcript")

        st.text_area(
            "Transcript",
            value=result["transcript"],
            height=500,
        )

        st.download_button(
            label="⬇️ Download Transcript",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )


    # ========================================================
    # CHAT WITH MEETING
    # ========================================================

    with tab6:

        st.subheader(
            "💬 Chat with your meeting"
        )

        rag_chain = st.session_state.rag_chain

        if rag_chain is None:

            st.warning(
                "RAG is unavailable."
            )

        else:

            # Display history

            for message in st.session_state.chat_history:

                if message["role"] == "user":

                    with st.chat_message("user"):

                        st.write(
                            message["content"]
                        )

                else:

                    with st.chat_message("assistant"):

                        st.write(
                            message["content"]
                        )

            # Chat input

            question = st.chat_input(
                "Ask something about the video..."
            )

            if question:

                # User message

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                with st.chat_message("user"):

                    st.write(question)

                # AI response

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Thinking..."
                    ):

                        try:

                            answer = ask_question(
                                rag_chain,
                                question,
                            )

                        except Exception as e:

                            answer = (
                                f"RAG error: {e}"
                            )

                        st.write(answer)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.divider()

    st.subheader("📥 Export")

    report = f"""
AI VIDEO ASSISTANT REPORT
=========================

TITLE
-----
{result["title"]}


SUMMARY
-------
{result["summary"]}


ACTION ITEMS
------------
{result["action_items"]}


KEY DECISIONS
-------------
{result["key_decisions"]}


OPEN QUESTIONS
--------------
{result["open_questions"]}


TRANSCRIPT
----------
{result["transcript"]}
"""

    st.download_button(
        label="⬇️ Download Full Report",
        data=report,
        file_name="ai_video_assistant_report.txt",
        mime="text/plain",
        use_container_width=True,
    )