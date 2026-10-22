"""Regression tests for unique deterministic/error-path behavior.

Real end-to-end behavior is covered by test_live_*.py.
"""

import pytest
from types import SimpleNamespace
from chat_safety import SAFETY_WORDS


def test_normalize_text(chat_module):
    assert chat_module.normalize_text("That’s ALL!!") == "thats all"


def test_missing_citation_is_refused(chat_module, monkeypatch):
    monkeypatch.setattr(chat_module, "USE_SCOPE_GATE", False)
    doc = SimpleNamespace(
        page_content="Returns are accepted within 30 days.",
        metadata={"source": "returns.md", "title": "Returns"},
    )

    monkeypatch.setattr(chat_module, "REQUIRE_CITATIONS", True)
    monkeypatch.setattr(
        chat_module, "rewrite_question", lambda question, history: question
    )
    monkeypatch.setattr(
        chat_module, "retrieve", lambda query: [(doc, 0.9)]
    )
    monkeypatch.setattr(
        chat_module,
        "build_context",
        lambda results: "Returns are accepted within 30 days.",
    )
    monkeypatch.setattr(
        chat_module.answer_llm,
        "invoke",
        #simulate a plausible answer that omits the required citation.
        lambda messages: SimpleNamespace(
            content="Returns are accepted within 30 days."
        ),
    )

    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert answer == chat_module.REFUSAL
    assert sources == []


def test_optional_missing_citation_does_not_claim_sources(chat_module, monkeypatch):
    doc = SimpleNamespace(
        page_content="Returns are accepted within 30 days.",
        metadata={"source": "returns.md", "title": "Returns"},
    )

    monkeypatch.setattr(chat_module, "REQUIRE_CITATIONS", False)
    monkeypatch.setattr(
        chat_module, "rewrite_question", lambda question, history: question
    )
    monkeypatch.setattr(chat_module, "retrieve", lambda query: [(doc, 0.9)])
    monkeypatch.setattr(
        chat_module, "build_context", lambda results: "Returns are accepted within 30 days."
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
            content="Returns are accepted within 30 days."
        ),
    )

    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert "30 days" in answer
    assert sources == []


def test_followup_reuses_previous_question(chat_module, monkeypatch):
    calls = []

    monkeypatch.setattr(
        chat_module, "awaiting_answer", lambda history: True
    )
    monkeypatch.setattr(
        chat_module,
        "last_real_question",
        lambda history: "What is the return window?",
    )
    monkeypatch.setattr(
        chat_module,
        "retrieve",
        lambda query: calls.append(query) or [],
    )

    answer, sources = chat_module._answer_question(
        "yes",
        [
            ("user", "What is the return window?"),
            ("assistant", "Would you like more details?"),
        ],
    )

    assert calls == ["What is the return window?"]
    assert answer == chat_module.REFUSAL
    assert sources == []


def test_safety_notice_fills_missing_hazard_guidance(chat_module, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "_answer_question",
        lambda question, history: ("Stop using the kettle.", [1]),
    )

    answer, sources = chat_module.answer_question(
        "My kettle is sparking", []
    )

    assert answer.startswith(chat_module.SAFETY_NOTICE)
    assert "disconnect power only if it is safe" in answer
    assert "emergency services" in answer
    assert sources == [1]


def test_credit_card_number_is_masked(chat_module, monkeypatch):
    received = []

    monkeypatch.setattr(
        chat_module,
        "_answer_question",
        lambda question, history: (
            received.append(question) or "I can help with billing.",
            [],
        ),
    )

    answer, sources = chat_module.answer_question(
        "My card number is 4111 1111 1111 1111", []
    )

    assert received == []
    assert "4111 1111 1111 1111" not in answer
    assert "Please don't share passwords" in answer
    assert sources == []


def test_password_is_not_forwarded(chat_module, monkeypatch):
    received = []

    monkeypatch.setattr(
        chat_module,
        "_answer_question",
        lambda question, history: (
            received.append(question) or chat_module.REFUSAL,
            [],
        ),
    )

    answer, sources = chat_module.answer_question(
        "My password is SuperSecret123!", []
    )

    assert received == []
    assert "SuperSecret123!" not in answer
    assert "Please don't share passwords" in answer
    assert sources == []


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ("My CVV is 123", "123"),
        ("My one-time code is 123456", "123456"),
        ("My recovery code: 12345678", "12345678"),
        ("My PIN is 1234", "1234"),
    ],
)
def test_security_codes_are_masked(chat_module, message, secret):
    masked = chat_module.mask_sensitive(message)

    assert secret not in masked
    assert "[code removed]" in masked


def test_sensitive_input_adds_warning_to_non_refusal(chat_module, monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "_answer_question",
        lambda question, history: ("I can help with sign-in.", [1]),
    )

    answer, sources = chat_module.answer_question(
        "I cannot sign in to my account because my password is SuperSecret123!",
        [],
    )

    assert "Please don't share passwords" in answer
    assert "I can help with sign-in." in answer
    assert sources == [1]


def test_invalid_citation_number_is_not_accepted(chat_module, monkeypatch):
    monkeypatch.setattr(chat_module, "USE_SCOPE_GATE", False)
    doc = SimpleNamespace(
        page_content="Returns are accepted within 30 days.",
        metadata={"source": "returns.md", "title": "Returns"},
    )

    monkeypatch.setattr(chat_module, "REQUIRE_CITATIONS", True)
    monkeypatch.setattr(
        chat_module, "rewrite_question", lambda question, history: question
    )
    monkeypatch.setattr(
        chat_module, "retrieve", lambda query: [(doc, 0.9)]
    )
    monkeypatch.setattr(
        chat_module, "build_context",
        lambda results: "[Source 1] Returns are accepted within 30 days.",
    )
    monkeypatch.setattr(
        chat_module.answer_llm,
        "invoke",
        lambda messages: SimpleNamespace(
            content="Returns are accepted within 30 days. [Source 99]"
        ),
    )

    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert answer == chat_module.REFUSAL
    assert "[Source 99]" not in answer
    assert sources == []


def test_retrieval_exception_propagates_to_caller(chat_module, monkeypatch):
    def broken_retrieval(query):
        raise RuntimeError("Vector database unavailable")

    monkeypatch.setattr(chat_module, "retrieve", broken_retrieval)

    with pytest.raises(RuntimeError, match="Vector database unavailable"):
        chat_module._answer_question(
            "What is the return policy?", []
        )


def test_empty_model_response_does_not_crash(chat_module, monkeypatch):
    monkeypatch.setattr(chat_module, "USE_SCOPE_GATE", False)
    doc = SimpleNamespace(
        page_content="Returns are accepted within 30 days.",
        metadata={"source": "returns.md", "title": "Returns"},
    )

    monkeypatch.setattr(
        chat_module, "rewrite_question", lambda question, history: question
    )
    monkeypatch.setattr(
        chat_module, "retrieve", lambda query: [(doc, 0.9)]
    )
    monkeypatch.setattr(
        chat_module, "build_context",
        lambda results: "[Source 1] Returns are accepted within 30 days.",
    )
    monkeypatch.setattr(
        chat_module.answer_llm, "invoke",
        lambda messages: SimpleNamespace(content=""),
    )

    answer, sources = chat_module._answer_question(
        "What is the return window?", []
    )

    assert answer == chat_module.REFUSAL
    assert sources == []


from chat_safety import SAFETY_WORDS


def test_electrocute_keyword_regression():
    """Regression: electrocute must trigger the safety keyword regex."""

    assert SAFETY_WORDS.search("I might electrocute myself.")
    assert SAFETY_WORDS.search("There is an electrocution risk.")
    assert SAFETY_WORDS.search("Could this kettle electrocute someone?")

    assert not SAFETY_WORDS.search("electocue")
