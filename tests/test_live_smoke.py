"""Optional smoke test for a locally running Ollama model."""

import os

import pytest
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_OLLAMA_TESTS") != "1",
    reason="Set RUN_LIVE_OLLAMA_TESTS=1 to contact the local Ollama service.",
)
def test_live_ollama_can_answer_from_chat_context(chat_module, monkeypatch):
    model = ChatOllama(
        model=chat_module.CHAT_MODEL,
        temperature=0,
        num_ctx=chat_module.NUM_CTX,
        num_predict=64,
    )
    document = Document(
        page_content="The return window is 30 calendar days from delivery.",
        metadata={"source": "smoke-test.md", "section_title": "Returns"},
    )
    monkeypatch.setattr(
        chat_module, "retrieve", lambda query: [(document, 0.9)]
    )
    monkeypatch.setattr(
        chat_module,
        "build_context",
        lambda results: "[Source 1: Returns]\n" + document.page_content,
    )
    monkeypatch.setattr(chat_module, "DEBUG", False)
    monkeypatch.setattr(chat_module, "answer_llm", model)

    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert "30" in answer
    assert sources == ["Returns"]
