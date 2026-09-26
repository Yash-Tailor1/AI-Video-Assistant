import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0,
    )


def build_chain(system_prompt: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all actionable items.\n\n"
        "For each action item include:\n"
        "- Task Description\n"
        "- Owner (who is responsible)\n\n"
        "Format the result as a numbered list.\n"
        "If none are found, say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all key decisions made.\n\n"
        "Format the result as a numbered list.\n"
        "If none are found, say 'No key decisions found.'"
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all unresolved questions "
        "or topics requiring follow-up.\n\n"
        "Format the result as a numbered list.\n"
        "If none are found, say 'No unresolved questions found.'"
    )

    return chain.invoke(transcript)