"""Regression tests for unique deterministic/error-path behavior.

Real end-to-end behavior is covered by test_live_*.py.
"""

from types import SimpleNamespace
import uuid
import pytest

def test_cli_commands_and_successful_turn(chat_module, monkeypatch, capsys):
    assert chat_module.DEBUG is False
    inputs = iter(["", "/debug", "/sources", "/clear", "Where is my order?", "/exit"])
    events = []
    session_id = "fresh-session"

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: events.append("init"))
    monkeypatch.setattr(
        chat_module,
        "clear_history",
        lambda active_session: events.append(("clear", active_session)),
    )
    monkeypatch.setattr(
        chat_module,
        "load_history",
        lambda active_session: events.append(("load", active_session)) or [],
    )
    monkeypatch.setattr(
        chat_module,
        "answer_question",
        lambda question, history: ("Your order is on its way. [Source 1]", [1, 2]),
    )
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content, active_session: events.append(
            (role, content, active_session)
        ),
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

    chat_module.main(session_id)

    output = capsys.readouterr().out
    assert f"Session: {session_id}" in output
    assert "Debug mode: ON" in output
    assert "Shipping" in output
    assert "Conversation history cleared." in output
    assert "Your order is on its way." in output
    assert "Source 1" in output
    assert "Sources: 1, 2" in output
    assert events == [
        "init",
        ("clear", session_id),
        ("load", session_id),
        ("user", "Where is my order?", session_id),
        ("assistant", "Your order is on its way. [Source 1]", session_id),
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
    monkeypatch.setattr(chat_module, "load_history", lambda session_id: [])

    def fail_to_answer(question, history):
        raise error

    monkeypatch.setattr(chat_module, "answer_question", fail_to_answer)
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content, session_id: saved.append((role, content, session_id)),
    )

    chat_module.main()

    assert expected in capsys.readouterr().out
    assert saved == []


def test_cli_persists_masked_sensitive_input(chat_module, monkeypatch):
    inputs = iter(["My password is SecretValue123!", "/exit"])
    saved = []

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: None)
    monkeypatch.setattr(chat_module, "load_history", lambda session_id: [])
    monkeypatch.setattr(
        chat_module,
        "answer_question",
        lambda question, history: ("I can help with sign-in.", []),
    )
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content, session_id: saved.append((role, content, session_id)),
    )

    chat_module.main()

    assert saved[0][0] == "user"
    assert "SecretValue123!" not in saved[0][1]
    assert "[password removed]" in saved[0][1]
    assert saved[0][2] == saved[1][2]
    assert saved[1][:2] == ("assistant", "I can help with sign-in.")


@pytest.mark.parametrize("non_answer", ["refusal", "greeting", "closing"])
def test_cli_persists_canned_turns_for_history_filtering(
    chat_module, monkeypatch, non_answer
):
    inputs = iter(["Hello", "/exit"])
    saved = []
    answers = {
        "refusal": chat_module.REFUSAL,
        "greeting": chat_module.GREETING_REPLY,
        "closing": chat_module.CLOSING_REPLY,
    }

    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: None)
    monkeypatch.setattr(chat_module, "load_history", lambda session_id: [])
    monkeypatch.setattr(
        chat_module,
        "answer_question",
        lambda question, history: (answers[non_answer], []),
    )
    monkeypatch.setattr(
        chat_module,
        "save_message",
        lambda role, content, session_id: saved.append((role, content, session_id)),
    )

    chat_module.main()

    assert saved == [
        ("user", "Hello", saved[0][2]),
        ("assistant", answers[non_answer], saved[0][2]),
    ]


def test_cli_session_argument_and_fresh_default(chat_module, monkeypatch, capsys):
    assert chat_module.parse_args(["--session", "resume-me"]).session_id == "resume-me"
    assert chat_module.parse_args([]).session_id is None

    session_ids = iter(
        [
            uuid.UUID("00000000-0000-0000-0000-000000000001"),
            uuid.UUID("00000000-0000-0000-0000-000000000002"),
        ]
    )
    monkeypatch.setattr(chat_module.uuid, "uuid4", lambda: next(session_ids))
    inputs = iter(["/exit", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(chat_module, "init_history", lambda: None)

    chat_module.main()
    first_output = capsys.readouterr().out
    chat_module.main()
    second_output = capsys.readouterr().out

    assert "Session: 00000000-0000-0000-0000-000000000001" in first_output
    assert "Session: 00000000-0000-0000-0000-000000000002" in second_output

