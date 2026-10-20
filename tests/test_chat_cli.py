"""CLI behavior tests use scripted input and never contact a model."""

from types import SimpleNamespace

import pytest


def test_cli_commands_and_successful_turn(chat_module, monkeypatch, capsys):
    inputs = iter(["", "/debug", "/sources", "/clear", "Where is my order?", "/exit"])
    events = []

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: events.append("init"))
    monkeypatch.setattr(chat_module, "clear_history", lambda: events.append("clear"))
    monkeypatch.setattr(chat_module, "load_history", lambda: [])
    monkeypatch.setattr(
        chat_module,
        "answer_question",
        lambda question, history: ("Your order is on its way. [Source 1]", ["Shipping"]),
    )
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content: events.append((role, content)),
    )
    monkeypatch.setattr(
        chat_module,
        "vectorstore",
        SimpleNamespace(
            get=lambda **kwargs: {
                "metadatas": [{"section_title": "Shipping"}]
            }
        ),
    )

    chat_module.main()

    output = capsys.readouterr().out
    assert "Debug mode:" in output
    assert "Shipping" in output
    assert "Conversation history cleared." in output
    assert "Your order is on its way." in output
    assert "Source 1" not in output
    assert events == [
        "init",
        "clear",
        ("user", "Where is my order?"),
        ("assistant", "Your order is on its way."),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("connection refused"), "could not reach Ollama"),
        (RuntimeError("vector store unavailable"), "Error: vector store unavailable"),
    ],
)
def test_cli_reports_turn_errors_without_saving(
    chat_module, monkeypatch, capsys, error, expected
):
    inputs = iter(["What is the return policy?", "/exit"])
    saved = []

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: None)
    monkeypatch.setattr(chat_module, "load_history", lambda: [])

    def fail_to_answer(question, history):
        raise error

    monkeypatch.setattr(chat_module, "answer_question", fail_to_answer)
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content: saved.append((role, content)),
    )

    chat_module.main()

    assert expected in capsys.readouterr().out
    assert saved == []


def test_cli_persists_masked_sensitive_input(chat_module, monkeypatch):
    inputs = iter(["My password is SecretValue123!", "/exit"])
    saved = []

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: None)
    monkeypatch.setattr(chat_module, "load_history", lambda: [])
    monkeypatch.setattr(
        chat_module,
        "answer_question",
        lambda question, history: ("I can help with sign-in.", []),
    )
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content: saved.append((role, content)),
    )

    chat_module.main()

    assert saved[0][0] == "user"
    assert "SecretValue123!" not in saved[0][1]
    assert "[password removed]" in saved[0][1]
    assert saved[1] == ("assistant", "I can help with sign-in.")


@pytest.mark.parametrize("non_answer", ["refusal", "greeting", "closing"])
def test_cli_does_not_save_non_answers(chat_module, monkeypatch, non_answer):
    inputs = iter(["Hello", "/exit"])
    saved = []
    answers = {
        "refusal": chat_module.REFUSAL,
        "greeting": chat_module.GREETING_REPLY,
        "closing": chat_module.CLOSING_REPLY,
    }

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: None)
    monkeypatch.setattr(chat_module, "load_history", lambda: [])
    monkeypatch.setattr(
        chat_module,
        "answer_question",
        lambda question, history: (answers[non_answer], []),
    )
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content: saved.append((role, content)),
    )

    chat_module.main()

    assert saved == []
