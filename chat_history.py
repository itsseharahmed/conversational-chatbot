import sqlite3
from contextlib import contextmanager
from pathlib import Path


DEFAULT_HISTORY_DB = Path(__file__).resolve().parent / "data" / "history.sqlite3"
HISTORY_TURNS = 5


@contextmanager
def _db_connection(database_path):
    conn = sqlite3.connect(database_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_history(database_path=DEFAULT_HISTORY_DB):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with _db_connection(database_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages (session_id, id)
        """)


def load_history(database_path=DEFAULT_HISTORY_DB, session_id="default"):
    with _db_connection(database_path) as conn:
        rows = conn.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, HISTORY_TURNS * 2)).fetchall()

    rows.reverse()
    return rows


def save_message(database_path, role, content, session_id="default"):
    with _db_connection(database_path) as conn:
        conn.execute("""
            INSERT INTO messages (session_id, role, content)
            VALUES (?, ?, ?)
        """, (session_id, role, content))


def clear_history(database_path=DEFAULT_HISTORY_DB, session_id="default"):
    with _db_connection(database_path) as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
