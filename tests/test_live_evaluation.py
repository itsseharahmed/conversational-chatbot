"""Run the full evaluation once against the real chatbot."""

from pathlib import Path

import pytest

from evaluate_chatbot import (
    load_cases,
    run_evaluation,
    answer_with_source_labels,
)
from test_live_chatbot import live_chat

pytestmark = [pytest.mark.live]

ROOT = Path(__file__).resolve().parents[1]


def test_real_evaluation_cases(live_chat):
    cases = load_cases(ROOT / "eval_cases.json")

    results = run_evaluation(
        cases,
        lambda question, history: answer_with_source_labels(
            live_chat, question, history
        ),
    )

    failures = [
        f"{item['id']}: {item['errors']} | {item['answer']}"
        for item in results
        if not item["passed"]
    ]

    assert not failures, "\n".join(failures)
