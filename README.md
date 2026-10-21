# Northstar Home Support Chatbot

A customer support chatbot for a fictional online store. It answers questions **only** from an approved knowledge base, refuses anything else, and runs fully locally with [Ollama](https://ollama.com).

> Northstar Home, its policies, URLs and phone number are invented for this demo.

## How it works

1. The question is rewritten into a standalone query if it is a follow-up.
2. The knowledge base is searched. If nothing is relevant enough, the bot refuses.
3. A scope and evidence check allows an answer only when the model returns `ANSWERABLE`.
4. The model writes the answer using only the retrieved text, following `rules.txt`.
5. The chat is saved to a local history database, with card numbers and passwords masked.

Anything it can't answer gets a standard refusal that points to a human agent.

## Project structure

```
chatbot/
├── chat.py                 # chatbot orchestration and entry point
├── chat_prompts.py         # prompt and policy loading
├── chat_history.py         # SQLite conversation storage
├── chat_safety.py          # sensitive-data masking and safety guidance
├── chat_citations.py       # source labels and citation parsing
├── ingest.py               # builds the searchable database from the knowledge base
├── embeddings.py           # embedding helper shared by ingest.py and chat.py
├── scope.txt               # approved topics
├── rules.txt               # tone and safety rules
├── prompts/                # question-rewrite, scope-check and answer prompts
├── requirements.txt        # Python dependencies
├── documents/
│   └── northstar_knowledge_base.md
└── data/                   # created automatically (database and chat history)
```

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

2. Create a virtual environment and install the requirements:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

```bash
python ingest.py     # build the knowledge base (re-run whenever the document changes)
python chat.py       # start the chat
python chat.py --session customer-123  # resume a named conversation
```

Each launch without `--session` creates a fresh conversation ID, preventing prior-run history from leaking into a new chat. The ID is printed at startup. Use `--session <id>` to resume it later; `/clear` clears only the active session. Chat commands: `/exit`, `/clear`, `/sources` (list indexed sections), `/debug` (show scores and why a question was refused). When enabled, answer citations are displayed as source numbers, such as `Sources: 1, 2`.

## Tests

Activate the virtual environment from the setup instructions, then run:

```bash
python -m pytest
```

The default suite is offline: model calls are mocked, history uses temporary
SQLite files, and Chroma retrieval uses a temporary database with deterministic
test embeddings. To run the optional Ollama smoke test, start Ollama, make sure
the configured chat model is installed, then run this in PowerShell:

```powershell
$env:RUN_LIVE_OLLAMA_TESTS = "1"
python -m pytest -m live
```

### Answer-quality evaluation

The tests validate code paths; they do not establish that a model consistently
answers policy questions correctly. Run the separate evaluation against the
configured knowledge base and local Ollama model after rebuilding the index:

```powershell
python ingest.py
python evaluate_chatbot.py --json-out evaluation-results.json
```

The evaluation checks a small set of representative in-scope and out-of-scope
questions for required facts, expected refusal behavior, and source labels. Add
reviewed cases to `eval_cases.json` whenever policies or prompts change. Review
the generated answers and results manually; keyword checks are a regression
signal, not a semantic correctness guarantee.

## Production readiness

This repository is a local demo, not a production service. Before exposing it
to customers, an owner should review and approve the evaluation cases and
thresholds, test the actual deployment environment and failure recovery, and
arrange monitoring, access control, data-retention, backup, and incident
response policies. Chat history is stored in a local SQLite database and is
not encrypted by this application; protect the host and its files accordingly.
Safety-sensitive answers require human review and escalation procedures.

## Changing the knowledge base

1. Edit `documents/northstar_knowledge_base.md` (use `#` to `####` headings, ideally one customer question per heading).
2. Run `python ingest.py` again.

For a different topic, also update `scope.txt`, `rules.txt`, `DOCUMENT_FILE` in `ingest.py`, and `REFUSAL` / `GREETING_REPLY` in `chat.py`, then run `/clear`. Retune `MIN_RELEVANCE_SCORE` in `chat.py` with `/debug`, since scores depend on the content.

## Troubleshooting

- **"Knowledge base not found":** run `python ingest.py` first.
- **Connection error:** make sure Ollama is running (`ollama serve`).
- **Good questions get refused:** turn on `/debug` to see which step refused. If it is the first check, lower `MIN_RELEVANCE_SCORE` in `chat.py`.

## Limitations

- It uses a small local model, so answers can be less polished or inconsistent.
- It cannot see live orders, payments or accounts, only general policy information.
