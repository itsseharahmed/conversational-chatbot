from pathlib import Path
from string import Template


BASE_DIR = Path(__file__).resolve().parent
SCOPE_FILE = BASE_DIR / "scope.txt"
RULES_FILE = BASE_DIR / "rules.txt"
PROMPT_DIR = BASE_DIR / "prompts"


def _read_policy(path):
    if not path.is_file():
        raise FileNotFoundError(f"Policy file not found: {path}")
    return path.read_text(encoding="utf-8")


SCOPE_POLICY = _read_policy(SCOPE_FILE)
RULES_POLICY = _read_policy(RULES_FILE)


def _load_prompt_template(name):
    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


PROMPT_TEMPLATES = {
    "rewrite_question": _load_prompt_template("rewrite_question.txt"),
    "scope_gate": _load_prompt_template("scope_gate.txt"),
    "answer": _load_prompt_template("answer.txt"),
}


def render_prompt(name, **values):
    return Template(PROMPT_TEMPLATES[name]).substitute(**values)
