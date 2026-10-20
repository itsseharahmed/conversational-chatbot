"""Tests for SQLite history isolation, ordering, retention, and deletion."""


def test_history_round_trip_and_session_isolation(chat_module, monkeypatch, tmp_path):
    database = tmp_path / "history.sqlite3"
    monkeypatch.setattr(chat_module, "HISTORY_DB", database)
    chat_module.init_history()

    chat_module.save_message("user", "Question one", session_id="first")
    chat_module.save_message("assistant", "Answer one", session_id="first")
    chat_module.save_message("user", "Other session", session_id="second")

    assert chat_module.load_history("first") == [
        ("user", "Question one"),
        ("assistant", "Answer one"),
    ]
    assert chat_module.load_history("second") == [
        ("user", "Other session"),
    ]


def test_history_keeps_only_the_most_recent_messages(
    chat_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(chat_module, "HISTORY_DB", tmp_path / "history.sqlite3")
    chat_module.init_history()

    for index in range(chat_module.HISTORY_TURNS * 2 + 2):
        chat_module.save_message("user", str(index))

    history = chat_module.load_history()
    assert len(history) == chat_module.HISTORY_TURNS * 2
    assert history[0] == ("user", "2")
    assert history[-1] == ("user", str(chat_module.HISTORY_TURNS * 2 + 1))


def test_clear_history_only_deletes_the_requested_session(
    chat_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(chat_module, "HISTORY_DB", tmp_path / "history.sqlite3")
    chat_module.init_history()
    chat_module.save_message("user", "Keep me", session_id="keep")
    chat_module.save_message("user", "Delete me", session_id="clear")

    chat_module.clear_history("clear")

    assert chat_module.load_history("clear") == []
    assert chat_module.load_history("keep") == [("user", "Keep me")]
