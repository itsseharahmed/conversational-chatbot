"""Regression tests for unique deterministic/error-path behavior.

Real end-to-end behavior is covered by test_live_*.py.
"""

from types import SimpleNamespace
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

def test_retrieve_filters_by_relevance_threshold(chat_module, monkeypatch):
    relevant = Document(page_content="Return policy.", metadata={"source": "returns.md"})
    borderline = Document(page_content="Shipping policy.", metadata={"source": "shipping.md"})
    calls = []

    def similarity_search(query, k):
        calls.append((query, k))
        return [(relevant, 0.8), (borderline, 0.59)]

    monkeypatch.setattr(
        chat_module,
        "vectorstore",
        SimpleNamespace(
            similarity_search_with_relevance_scores=similarity_search
        ),
    )

    assert chat_module.retrieve("return policy") == [(relevant, 0.8)]
    assert calls == [("return policy", chat_module.RETRIEVAL_K)]


def test_prompt_templates_render_runtime_values(chat_module):
    rewrite = chat_module.render_prompt(
        "rewrite_question",
        history="user: How long does shipping take?",
        question="How long does it take?",
    )
    gate = chat_module.render_prompt(
        "scope_gate",
        scope_policy="Shipping and returns are in scope.",
        context="Standard shipping takes 3-7 days.",
        history="user: Shipping question",
        question="How long does shipping take?",
    )
    answer = chat_module.render_prompt(
        "answer",
        rules="Use calm, clear language.",
        context="[Source 1] Shipping takes 3-7 days.",
        history="user: How long does delivery take?",
        question="How long does it take?",
    )

    assert "How long does it take?" in rewrite
    assert "How long does shipping take?" in rewrite
    assert "Shipping and returns are in scope." in gate
    assert "Standard shipping takes 3-7 days." in gate
    assert "ONLY the context below" in answer
    assert "Use calm, clear language." in answer
    assert "[Source 1] Shipping takes 3-7 days." in answer
    assert "user: How long does delivery take?" in answer
    assert "How long does it take?" in answer
    assert "$" not in rewrite + gate + answer
    assert "The return window length and the event it starts from are whatever the context states" in answer
    assert "If it has arrived" in answer
    assert "ask when it was delivered" in answer.casefold()
    assert "earlier assistant replies are not evidence about the customer's order" in answer.casefold()
    assert "include the key steps, conditions, and limitations" in answer.casefold()
    assert "still an in-scope support question" in answer.casefold()
    assert "disconnecting power only if safe" in answer.casefold()


def test_build_context_numbers_sources_and_obeys_character_limit(
    chat_module, monkeypatch
):
    monkeypatch.setattr(chat_module, "MAX_CONTENT_CHARS", 80)
    documents = [
        (
            Document(
                page_content="A" * 100,
                metadata={"section_title": "Returns"},
            ),
            0.9,
        ),
        (
            Document(
                page_content="This second document is truncated.",
                metadata={"section_title": "Shipping"},
            ),
            0.8,
        ),
    ]

    context = chat_module.build_context(documents)

    assert len(context) == 80
    assert context.startswith("[Source 1: Returns]")
    assert "[Source 2: Shipping]" not in context


def test_preserve_policy_caveats_from_cited_evidence(chat_module):
    document = Document(
        page_content=(
            "A refund or replacement is not automatic for a missing parcel."
        ),
        metadata={"section_title": "Shipping"},
    )
    results = [(document, 0.9)]

    answer = chat_module.preserve_policy_caveats(
        "Contact support with your order number.", [1], results
    )

    assert answer.endswith(
        "A refund or replacement is not automatic for a missing parcel."
    )
    assert chat_module.preserve_policy_caveats(
        answer, [1], results
    ) == answer
    assert chat_module.preserve_policy_caveats(
        "Answer.", [], results
    ) == "Answer."


def test_citations_resolve_sources_and_strip_from_display(chat_module):
    results = [
        (
            Document(
                page_content="Returns.",
                metadata={
                    "document_title": "Store",
                    "section_title": "Returns",
                },
            ),
            0.9,
        ),
        (
            Document(
                page_content="Shipping.",
                metadata={"source": "shipping.md"},
            ),
            0.8,
        ),
    ]

    assert chat_module.cited_sources(
        "See [Source 1] and [Sources 2].", results
    ) == [1, 2]
    assert chat_module.cited_sources("[Source 8]", results) == []
    assert chat_module.strip_citations(
        "Answer. [Source 1] Details. [Sources 1, 2]"
    ) == "Answer. Details."
    assert chat_module.split_answer("Delivery takes 3-7 days. SOURCES: 1") == (
        "Delivery takes 3-7 days.",
        {1},
    )
    assert chat_module.split_answer("Answer.\nSOURCES: none") == (
        "Answer.",
        set(),
    )


def test_missing_source_line_gets_one_formatting_retry(chat_module, monkeypatch):
    document = Document(
        page_content="Standard delivery takes 3-7 business days.",
        metadata={"section_title": "Shipping"},
    )
    responses = iter(
        [
            SimpleNamespace(content="Standard delivery takes 3-7 business days."),
            SimpleNamespace(content="Standard delivery takes 3-7 business days. SOURCES: 1"),
        ]
    )
    monkeypatch.setattr(chat_module, "USE_SCOPE_GATE", False)
    monkeypatch.setattr(chat_module, "retrieve", lambda query: [(document, 0.9)])
    monkeypatch.setattr(
        chat_module, "build_context", lambda results: "[Source 1: Shipping]\n" + document.page_content
    )
    monkeypatch.setattr(
        chat_module.answer_llm, "invoke", lambda messages: next(responses)
    )

    answer, sources = chat_module._answer_question(
        "How long does standard delivery take?", []
    )

    assert answer == "Standard delivery takes 3-7 business days."
    assert sources == [1]


@pytest.mark.parametrize(
    "rewritten",
    [
        "",
        "x" * 301,
        "What is the price of a television?",
    ],
)
def test_rewrite_question_falls_back_on_unusable_model_output(
    chat_module, monkeypatch, rewritten
):
    question = "How long does it take?"
    monkeypatch.setattr(
        chat_module.llm,
        "invoke",
        lambda messages: SimpleNamespace(content=rewritten),
    )

    assert chat_module.rewrite_question(
        question, [("user", "How long does shipping take?")]
    ) == question


def test_rewrite_question_uses_valid_standalone_output(chat_module, monkeypatch):
    monkeypatch.setattr(
        chat_module.llm,
        "invoke",
        lambda messages: SimpleNamespace(
            content='"How long does shipping take?"'
        ),
    )

    assert chat_module.rewrite_question(
        "How long does it take?",
        [("user", "How long does shipping take?")],
    ) == "How long does shipping take?"

