"""Actual CLI smoke test: real chat process, retrieval, gate and model."""
import subprocess
import sys
from pathlib import Path
import pytest
from test_live_chatbot import live_chat

pytestmark = pytest.mark.live


def test_cli_answers_real_return_question(live_chat):
    root = live_chat.BASE_DIR
    result = subprocess.run(
        [sys.executable, str(root / "chat.py"), "--session", "pytest-live-cli"],
        input="What is the return window?\n/exit\n",
        text=True, capture_output=True, cwd=root, timeout=240,
    )
    assert result.returncode == 0, result.stderr
    assert "30" in result.stdout and "delivery" in result.stdout.lower(), result.stdout
