"""Run a repeatable answer-quality check against the configured local chatbot."""

import argparse
import json
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = BASE_DIR / "eval_cases.json"


def load_cases(path):
    with Path(path).open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a non-empty JSON list.")

    for case in cases:
        if not isinstance(case, dict) or not case.get("id") or not case.get("question"):
            raise ValueError("Each evaluation case needs an id and a question.")
        if case.get("outcome") not in {"answer", "refusal"}:
            raise ValueError(
                f"Evaluation case {case.get('id')!r} has an invalid outcome."
            )
        if any(
            not isinstance(group, list) or not group or
            any(not isinstance(term, str) or not term for term in group)
            for group in case.get("required_groups", [])
        ):
            raise ValueError(
                f"Evaluation case {case['id']!r} has an invalid required_groups value."
            )

    return cases


def grade_case(case, answer, sources):
    """Apply transparent, human-readable checks to one generated answer."""
    errors = []
    normalized_answer = answer.casefold()

    if case["outcome"] == "refusal":
        if "i can't find that" not in normalized_answer:
            errors.append("expected the standard refusal")
    elif "i can't find that" in normalized_answer:
        errors.append("unexpected refusal for an answerable question")

    for group in case.get("required_groups", []):
        if not any(term.casefold() in normalized_answer for term in group):
            errors.append(f"missing expected fact: {' or '.join(group)}")

    for term in case.get("forbidden_terms", []):
        if term.casefold() in normalized_answer:
            errors.append(f"contains forbidden claim: {term}")

    expected_source = case.get("source_contains")
    if expected_source and not any(
        expected_source.casefold() in source.casefold() for source in sources
    ):
        errors.append(f"missing expected source: {expected_source}")

    return errors


def run_evaluation(cases, answer_question):
    """Run every case and collect outputs, source labels, timings, and failures."""
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            answer, sources = answer_question(case["question"], [])
            errors = grade_case(case, answer, sources)
            result = {
                "id": case["id"],
                "passed": not errors,
                "answer": answer,
                "sources": sources,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "errors": errors,
            }
        except Exception as exc:
            result = {
                "id": case["id"],
                "passed": False,
                "answer": "",
                "sources": [],
                "duration_seconds": round(time.perf_counter() - started, 3),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate configured chatbot answers against local quality checks."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="JSON evaluation case file (default: eval_cases.json).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write the full results as JSON.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)

    # Import only when executing the evaluation so the checker itself is testable offline.
    import chat

    results = run_evaluation(cases, chat.answer_question)
    for result in results:
        label = "PASS" if result["passed"] else "FAIL"
        print(
            f"{label} {result['id']} "
            f"({result['duration_seconds']:.3f}s)"
        )
        for error in result["errors"]:
            print(f"  - {error}")

    passed = sum(result["passed"] for result in results)
    print(f"\nEvaluation: {passed}/{len(results)} cases passed.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Detailed results written to {args.json_out}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
