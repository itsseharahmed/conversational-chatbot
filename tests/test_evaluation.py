"""Offline tests for the live evaluation case format and grading rules."""

import json

import pytest

from evaluate_chatbot import (
    grade_case,
    load_cases,
    run_evaluation,
    answer_with_source_labels,
    source_labels_for_citations,
)


def test_source_labels_for_citations_maps_current_numeric_source_format():
    class Document:
        def __init__(self, metadata):
            self.metadata = metadata

    retrieved = [
        (Document({"section_title": "Shipping"}), 0.8),
        (Document({"section_title": "Returns"}), 0.7),
    ]

    labels = source_labels_for_citations(
        [2, 1, 3],
        retrieved,
        lambda metadata: metadata["section_title"],
    )

    assert labels == ["Returns", "Shipping"]


def test_answer_with_source_labels_uses_the_retrieval_that_answered_question():
    class Document:
        metadata = {"section_title": "Returns"}

    retrieved = [(Document(), 0.8)]
    queries = []

    class Chat:
        source_label = staticmethod(lambda metadata: metadata["section_title"])

        @staticmethod
        def retrieve(query):
            queries.append(query)
            return retrieved

        @staticmethod
        def answer_question(question, history):
            Chat.retrieve("rewritten query")
            return "Answer.", [1]

    answer, labels = answer_with_source_labels(Chat, "original query", [])

    assert answer == "Answer."
    assert labels == ["Returns"]
    assert queries == ["rewritten query"]
    assert Chat.retrieve("another query") == retrieved
    assert queries[-1] == "another query"


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
