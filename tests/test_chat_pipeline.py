"""Tests for retrieval, prompt gates, citations, and real local Chroma wiring."""

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


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("ANSWERABLE", True),
        ("NO_EVIDENCE", False),
        ("OUT_OF_SCOPE", False),
        ("YES", False),
        ("ANSWERABLE with explanation", False),
    ],
)
def test_scope_gate_interprets_model_verdict(
    chat_module, monkeypatch, verdict, expected
):
    prompts = []
    monkeypatch.setattr(
        chat_module.gate_llm,
        "invoke",
        lambda messages: prompts.append(messages) or SimpleNamespace(content=verdict),
    )

    assert chat_module.is_answerable(
        "What is the return window?",
        [("user", "A prior question")],
        "Returns are accepted within 30 days.",
    ) is expected
    assert "Returns are accepted within 30 days." in str(prompts[0])
    assert "OUT_OF_SCOPE only for a clearly unrelated topic" in str(prompts[0])


def test_scope_gate_no_verdict_prevents_answer_generation(
    chat_module, monkeypatch
):
    document = Document(
        page_content="Returns follow the published policy.",
        metadata={"source": "returns.md"},
    )
    monkeypatch.setattr(
        chat_module,
        "retrieve",
        lambda query: [(document, 0.99)],
    )
    monkeypatch.setattr(
        chat_module,
        "build_context",
        lambda results: "Returns follow the published policy.",
    )
    monkeypatch.setattr(
        chat_module.gate_llm,
        "invoke",
        lambda messages: SimpleNamespace(content="OUT_OF_SCOPE"),
    )
    monkeypatch.setattr(
        chat_module.answer_llm,
        "invoke",
        lambda messages: pytest.fail("Answer model called after gate refusal"),
    )

    answer, sources = chat_module._answer_question("What is the return policy?", [])

    assert answer == chat_module.REFUSAL
    assert sources == []


def test_scope_gate_yes_allows_grounded_answer(chat_module, monkeypatch):
    document = Document(
        page_content="Returns are accepted within 30 calendar days.",
        metadata={"source": "returns.md", "section_title": "Returns"},
    )
    monkeypatch.setattr(
        chat_module, "retrieve", lambda query: [(document, 0.65)]
    )
    monkeypatch.setattr(
        chat_module,
        "build_context",
        lambda results: "[Source 1: Returns]\n" + document.page_content,
    )
    monkeypatch.setattr(
        chat_module.gate_llm,
        "invoke",
        lambda messages: SimpleNamespace(content="ANSWERABLE"),
    )
    monkeypatch.setattr(
        chat_module.answer_llm,
        "invoke",
        lambda messages: SimpleNamespace(
            content="Returns are accepted within 30 days. [Source 1]"
        ),
    )

    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert "30 days" in answer
    assert sources == [1]


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


def test_rewrite_question_rejects_unrelated_topic(chat_module, monkeypatch):
    question = "How long does it take?"
    monkeypatch.setattr(
        chat_module.llm,
        "invoke",
        lambda messages: SimpleNamespace(
            content="What is the price of a television?"
        ),
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


class KeywordEmbeddings(Embeddings):
    """Deterministic vectors let Chroma integration tests run without Ollama."""

    @staticmethod
    def _vector(text):
        return [1.0, 0.0] if "return" in text.lower() else [0.4, 0.916515]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_real_chroma_retrieval_and_chat_pipeline(
    chat_module, monkeypatch, tmp_path
):
    documents = [
        Document(
            page_content="Most items can be returned within 30 calendar days.",
            metadata={"source": "returns.md", "section_title": "Returns"},
        ),
        Document(
            page_content="Standard shipping takes three to seven business days.",
            metadata={"source": "shipping.md", "section_title": "Shipping"},
        ),
    ]
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=KeywordEmbeddings(),
        collection_name="chat-test",
        persist_directory=str(tmp_path / "chroma"),
    )
    monkeypatch.setattr(chat_module, "vectorstore", vectorstore)
    monkeypatch.setattr(chat_module, "DEBUG", False)
    monkeypatch.setattr(
        chat_module.gate_llm,
        "invoke",
        lambda messages: SimpleNamespace(content="ANSWERABLE"),
    )
    monkeypatch.setattr(
        chat_module.answer_llm,
        "invoke",
        lambda messages: SimpleNamespace(
            content="Most items can be returned within 30 days. [Source 1]"
        ),
    )

    results = chat_module.retrieve("What is the return window?")
    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert len(results) == 1
    assert results[0][0].metadata["source"] == "returns.md"
    assert "30 days" in answer
    assert sources == [1]
