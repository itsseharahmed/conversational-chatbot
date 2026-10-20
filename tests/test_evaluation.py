"""Offline tests for the live evaluation case format and grading rules."""

import json

import pytest

from evaluate_chatbot import grade_case, load_cases, run_evaluation


def test_grade_case_checks_grounding_and_expected_source():
    case = {
        "id": "return-policy",
        "outcome": "answer",
        "required_groups": [["30"], ["delivery"]],
        "source_contains": "Returns",
    }

    assert grade_case(
        case,
        "Returns are accepted within 30 days of delivery.",
        ["Returns"],
    ) == []

    errors = grade_case(case, "Returns are accepted within 90 days.", [])
    assert "missing expected fact: 30" in errors
    assert "missing expected fact: delivery" in errors
    assert "missing expected source: Returns" in errors


def test_grade_case_distinguishes_expected_refusal():
    case = {"id": "out-of-scope", "outcome": "refusal"}

    assert grade_case(
        case, "I can't find that in the information available to me.", []
    ) == []
    assert grade_case(case, "The answer is 42.", []) == [
        "expected the standard refusal"
    ]


def test_grade_case_checks_forbidden_claims():
    case = {
        "id": "return-policy",
        "outcome": "answer",
        "forbidden_terms": ["guaranteed refund"],
    }

    assert grade_case(case, "A guaranteed refund is available.", []) == [
        "contains forbidden claim: guaranteed refund"
    ]


def test_load_cases_rejects_invalid_case_shapes(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"id": "missing-question"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="needs an id and a question"):
        load_cases(cases_path)


def test_run_evaluation_records_passes_failures_and_exceptions():
    cases = [
        {"id": "known", "question": "known?", "outcome": "answer"},
        {"id": "broken", "question": "broken?", "outcome": "answer"},
    ]

    def answer_question(question, history):
        if question == "broken?":
            raise RuntimeError("model unavailable")
        return "Here is an answer.", ["Source"]

    results = run_evaluation(cases, answer_question)

    assert results[0]["passed"] is True
    assert results[1]["passed"] is False
    assert results[1]["errors"] == ["RuntimeError: model unavailable"]
