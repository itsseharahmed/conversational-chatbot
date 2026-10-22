#main program: searches vector base, runs scope checks, generates answers, stores history and provides CLI

import argparse
from pathlib import Path
import re
import uuid
from langchain_ollama import ChatOllama
from embeddings import PrefixedOllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import chat_citations
import chat_history
import chat_safety
import chat_prompts
import os


#paths
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Path(
    os.environ.get(
        "CHATBOT_DB_DIR",
        BASE_DIR / "data" / "chroma"
    )
)
HISTORY_DB = BASE_DIR / "data" / "history.sqlite3"

#models
CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION = "approved_knowledge"

#model settings
NUM_CTX = 8192               #context window size for the chat model
MAX_OUTPUT_TOKENS = 1024     #maximum number of tokens to generate in the output
HISTORY_TURNS = chat_history.HISTORY_TURNS
RETRIEVAL_K = 5              #chunks fetched per question
MIN_RELEVANCE_SCORE = 0.60   #minimum relevance score for retrieved chunks 
MAX_CONTENT_CHARS = 14000    #maximum retrieved text sent to the model

#if False, the scope and evidence check (gate 2) is skipped, useful for finding out if it rejects too much
USE_SCOPE_GATE = True

#if True, an answer with no valid [Source N] tag is refused (cheap hallucination check, but small models often forget tags)
#if False, answers without valid citations are kept without a source list
REQUIRE_CITATIONS = False

#if True, cited [Source N] numbers are shown to the user
#if False, they are hidden from the customer, but the citation check still runs behind the scenes
SHOW_SOURCES = True

DEBUG = False              #enable/disable debug logging (type /debug in the chat to toggle), off by default so customers never see internals

#standard refusal, it also points the customer to a human (update the contact details if they change)
REFUSAL = (
    "I can't find that in the Northstar Home information available to me. "
    "I can help with products, orders, shipping, returns, billing, accounts and warranty. "
    "For anything else, please contact Northstar Home Support at help@northstarhome.example "
    "or 1-800-555-0148 (Monday-Friday, 9:00 a.m.-6:00 p.m. Eastern Time)."
)

#reply for greetings and thanks, handled in code so no model call is needed
GREETING_REPLY = (
    "Hi, I'm Northstar Home Support. I can help with products, orders, shipping, "
    "returns, billing, accounts and warranty. How can I help?"
)

#reply for thanks, "no" and goodbyes
CLOSING_REPLY = "You're welcome! If you need anything else, just ask."

#lowercases and removes apostrophes and punctuation, so "That\u2019s all!!" and "that's  all" compare equal
def normalize_text(text):
    text = text.lower().replace("\u2019", "'").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())

#messages that are treated as greetings instead of questions
GREETINGS = {normalize_text(t) for t in(
    "hi", "hello", "hey", "hi there", "hello there",
    "good morning", "good afternoon", "good evening"
)}

#short replies that say "yes" to the assistant's last message
CONFIRMATIONS = {normalize_text(t) for t in (
    "yes", "yeah", "yep", "yes please", "sure", "please", "ok please", "ok", "okay"
)}

#short replies that say "no" (a goodbye, unless the assistant just asked a question)
NEGATIONS = {"no", "nope"}

#replies that are only an answer to the assistant's last question
SHORT_REPLIES = CONFIRMATIONS | NEGATIONS

#a message made only of these words, with at least one anchor word, ends the topic ("thanks a lot", "ok great, bye")
CLOSER_ANCHORS = {"thanks", "thank", "thx", "ty", "bye", "goodbye", "cheers", "ok", "okay", "appreciate", "thats"}
CLOSER_FILLER = CLOSER_ANCHORS | {
    "alright", "great", "perfect", "cool", "awesome", "nice", "good", "got", "it", "you", "so", "very", "much",
    "a", "lot", "that", "all", "i", "needed", "no", "nope", "nothing", "else", "for", "your", "the", "help",
    "day", "take", "care", "see", "later",
}

#a bare "no" is a goodbye too (the caller decides whether it answers a pending question instead)
def is_closer(normalized):
    if normalized in NEGATIONS:
        return True
    words = normalized.split()
    return (
        0 < len(words) <= 7
        and all(w in CLOSER_FILLER for w in words)
        and any(w in CLOSER_ANCHORS for w in words)
    )

SCOPE_POLICY = chat_prompts.SCOPE_POLICY
RULES_POLICY = chat_prompts.RULES_POLICY
PROMPT_TEMPLATES = chat_prompts.PROMPT_TEMPLATES
render_prompt = chat_prompts.render_prompt


if not DB_DIR.exists():
    raise RuntimeError("Knowledge base not found. Please run ingest.py first.")

#deterministic model for rewriting questions
llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.0,
    num_ctx=NUM_CTX,
    num_predict=MAX_OUTPUT_TOKENS,
)

#deterministic model for the YES/NO gate, only a few tokens are needed so it is faster
gate_llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.0,
    num_ctx=NUM_CTX,
    num_predict=5,
)

#slightly more flexible model for writing the final answer
answer_llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0.0,
    num_ctx=NUM_CTX,
    num_predict=MAX_OUTPUT_TOKENS,
)

#embeddings model used to turn questions into vectors
embeddings = PrefixedOllamaEmbeddings(model=EMBEDDING_MODEL)

#connect to the existing chroma collection
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
    collection_metadata={"hnsw:space": "cosine"},
)

if not vectorstore.get(limit=1)["ids"]:
    raise RuntimeError("Knowledge base is empty. Please run ingest.py first.")


#SQLite history operations remain exposed here for compatibility with the CLI and tests.
def init_history():
    chat_history.init_history(HISTORY_DB)


def load_history(session_id="default"):
    return chat_history.load_history(HISTORY_DB, session_id)


def save_message(role, content, session_id="default"):
    chat_history.save_message(HISTORY_DB, role, content, session_id)


def clear_history(session_id="default"):
    chat_history.clear_history(HISTORY_DB, session_id)


CARD_NUMBER = chat_safety.CARD_NUMBER
SECURITY_CODE = chat_safety.SECURITY_CODE
PASSWORD = chat_safety.PASSWORD
REMOVED_PLACEHOLDER = chat_safety.REMOVED_PLACEHOLDER
SAFETY_WORDS = chat_safety.SAFETY_WORDS
SAFETY_NOTICE = chat_safety.SAFETY_NOTICE
SENSITIVE_WARNING = chat_safety.SENSITIVE_WARNING
SENSITIVE_HELP = chat_safety.SENSITIVE_HELP
SENSITIVE_ONLY_REPLY = chat_safety.SENSITIVE_ONLY_REPLY
mask_sensitive = chat_safety.mask_sensitive

#format recent history as plain text for the rewrite and gate prompts
def format_recent(history, limit=4):
    return "\n".join(f"{role}: {content}" for role, content in history[-limit:])


def to_messages(history):
    messages = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


HEADER_KEYS = chat_citations.HEADER_KEYS
source_label = chat_citations.source_label
SOURCES_LINE = chat_citations.SOURCES_LINE
INLINE_TAG = chat_citations.INLINE_TAG
split_answer = chat_citations.split_answer
cited_sources = chat_citations.cited_sources
strip_citations = chat_citations.strip_citations


#retrieval
def retrieve(query):
    #search for the query in the vectorstore
    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=RETRIEVAL_K
    )

    if DEBUG:
        print("\n[debug] retrieval scores:")
        for doc, score in results:
            print(f"  {score:.3f}  {source_label(doc.metadata)}")

    #keep only chunks that are relevant enough
    return [(doc, score) for doc, score in results if score >= MIN_RELEVANCE_SCORE]


#combine the retrieved chunks into one text block with numbered source labels
def build_context(results):
    blocks = []
    used_chars = 0

    for index, (doc, _score) in enumerate(results, start=1):
        #stop once the character limit has been reached
        remaining_chars = MAX_CONTENT_CHARS - used_chars
        if remaining_chars <= 0:
            break

        block = f"[Source {index}: {source_label(doc.metadata)}]\n{doc.page_content.strip()}"
        block = block[:remaining_chars]
        blocks.append(block)
        used_chars += len(block)

    return "\n\n----\n\n".join(blocks)


def preserve_policy_caveats(answer, sources, results):
    """Retain explicit limitations from the evidence the model cited."""
    caveat_pattern = re.compile(
        r"\b(?:not automatic|not guaranteed|not a guarantee|not complete until|"
        r"not confirmed until|cannot be guaranteed|not promised)\b",
        re.IGNORECASE,
    )
    answer_lower = answer.casefold()
    missing_caveats = []

    for number in sources:
        if not 1 <= number <= len(results):
            continue
        document, _score = results[number - 1]
        for sentence in re.split(r"(?<=[.!?])\s+", document.page_content.strip()):
            sentence = sentence.strip()
            if (
                sentence
                and caveat_pattern.search(sentence)
                and sentence.casefold() not in answer_lower
            ):
                missing_caveats.append(sentence)

    if missing_caveats:
        return f"{answer.rstrip()} {' '.join(missing_caveats)}"
    return answer


def rewrite_question(question, history):
    #the model decides whether the message depends on earlier context, not a regex
    customer_history = [
        (role, content) for role, content in history if role == "user"
    ]
    if not customer_history:
        return question

    prompt = render_prompt(
        "rewrite_question",
        history=format_recent(customer_history),
        question=question,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    #the model sometimes echoes the "Query:" label from the examples or adds explanation lines, keep the first line only
    lines = [line.strip() for line in str(response.content).splitlines() if line.strip()]
    rewritten = re.sub(r"^query\s*:\s*", "", lines[0], flags=re.IGNORECASE).strip().strip('"') if lines else ""

    if not rewritten or len(rewritten) > 300:
        return question

    #A rewrite must preserve the current message's numbers and cannot replace them
    #with a number that only appeared in an earlier turn.
    latest_numbers = set(re.findall(r"\d+", question))
    rewritten_numbers = set(re.findall(r"\d+", rewritten))
    if not latest_numbers <= rewritten_numbers or not rewritten_numbers <= (
        latest_numbers | set(re.findall(r"\d+", format_recent(customer_history)))
    ):
        if DEBUG:
            print(f"[debug] rewrite rejected, it changed or invented a number: {rewritten}")
        return question

    meaningful_words = {
        word
        for word in re.findall(r"[a-z]{3,}", question.lower())
        if word not in {"the", "and", "for", "from", "was", "were", "does", "did", "can"}
    }
    rewritten_words = set(re.findall(r"[a-z]{3,}", rewritten.lower()))
    if meaningful_words and not meaningful_words.intersection(rewritten_words):
        if DEBUG:
            print(f"[debug] rewrite rejected, it changed the topic: {rewritten}")
        return question

    if DEBUG:
        print(f"[debug] rewritten query: {rewritten}")
    return rewritten

#scope and evidence check
def is_answerable(question, history, context):
    #nothing to check if no evidence was retrieved
    if not context.strip():
        return False

    recent = format_recent(history)

    gate_prompt = render_prompt(
        "scope_gate",
        scope_policy=SCOPE_POLICY,
        context=context,
        history=recent,
        question=question,
    )

    response = gate_llm.invoke([
        SystemMessage(content=
                        "You are a Northstar Home scope and evidence classifier. "
                        "Apply the approved scope policy and classification rules "
                        "in the user message. Do not reject an in-scope customer "
                        "safety question merely because it mentions sensitive "
                        "information. Return exactly one of: ANSWERABLE, "
                        "NO_EVIDENCE, OUT_OF_SCOPE."
                    ),
        HumanMessage(content=gate_prompt),
    ])

    print(f"\n[DEBUG] Scope gate raw response: {response.content!r}")

    verdict = str(response.content).strip().upper()

    if DEBUG:
        print(f"[debug] gate verdict: {verdict!r}")

    #Only the documented positive verdict passes; unknown or malformed output fails closed.
    return verdict == "ANSWERABLE"


#canned replies are not model answers: they never count as a pending question and are kept out of the model's context
#(the safety notice is put in front of the refusal, so a refusal can also be the end of the text)
def is_canned(answer):
    return answer in (GREETING_REPLY, CLOSING_REPLY, SENSITIVE_ONLY_REPLY) or answer.endswith(REFUSAL)


#the history the models see: turns that ended in a canned reply (refusals, greetings, thanks) are dropped together with
#the question they answered, so they do not confuse query rewriting
#they are still stored, so awaiting_answer always looks at what the customer really saw last
def model_history(history):
    kept = []
    for role, content in history:
        if role == "assistant" and is_canned(content):
            if kept and kept[-1][0] == "user":
                kept.pop()
            continue
        kept.append((role, content))
    return kept


#true when the assistant's last message asked the customer something, so a bare "yes" or "no" is an answer to it
#(the greeting ends with "How can I help?", that is not a question about an earlier topic)
def awaiting_answer(history):
    for role, content in reversed(history):
        if role == "assistant":
            return "?" in content and not is_canned(content)
    return False


#the customer's most recent real question, skipping bare "yes" or "no" replies
def last_real_question(history):
    for role, content in reversed(history):
        if role == "user" and normalize_text(content) not in SHORT_REPLIES:
            return content
    return None


#public entry point: masks sensitive details first so no model, search or log ever sees them,
#then adds the safety notice or the sensitive-data warning where needed
def answer_question(question, history):
    masked = mask_sensitive(question)

    #nothing but a secret was sent ("card 1245 ..."): there is no question left, so do not let the models guess one
    #(a safety word keeps the normal path, so the safety notice is still shown)
    leftover = normalize_text(REMOVED_PLACEHOLDER.sub("", masked))
    if masked != question and len(leftover.split()) <= 4 and not SAFETY_WORDS.search(masked):
        return SENSITIVE_ONLY_REPLY, []

    answer, sources = _answer_question(masked, history)

    if SAFETY_WORDS.search(masked):
        answer_text = answer.casefold()
        has_stop_guidance = any(
            phrase in answer_text for phrase in ("stop using", "do not use", "don't use")
        )
        has_safe_power_guidance = (
            any(phrase in answer_text for phrase in ("disconnect power", "unplug"))
            and any(
                phrase in answer_text
                for phrase in ("only if safe", "if safe", "if it is safe")
            )
        )
        has_emergency_guidance = "emergency services" in answer_text
        if not (
            has_stop_guidance
            and has_safe_power_guidance
            and has_emergency_guidance
        ):
            answer = f"{SAFETY_NOTICE}\n\n{answer}"

    if masked != question:
        #a plain refusal, greeting or thanks is replaced by the warning, a safety notice must stay
        if answer in (REFUSAL, GREETING_REPLY, CLOSING_REPLY):
            return SENSITIVE_ONLY_REPLY, []
        answer = f"{SENSITIVE_WARNING}\n\n{answer}"

    return answer, sources


#answer generation
def _answer_question(question, history):
    #answer greetings and closers directly, they are not questions for the knowledge base
    normalized = normalize_text(question)
    if normalized in GREETINGS:
        return GREETING_REPLY, []

    #this must look at the stored history, the filtered one below no longer contains the canned replies
    pending_question = awaiting_answer(history)
    history = model_history(history)

    #a short "yes" or "no" answers the assistant's last question, so it cannot be rewritten into a question
    #when the assistant just asked something, yes/ok/no answer it; otherwise they end the chat
    is_confirmation = pending_question and normalized in SHORT_REPLIES
    if is_closer(normalized) and not is_confirmation:
        return CLOSING_REPLY, []
    #search for the customer's previous real question instead and let the history provide the context
    #with no question pending, a bare "yes" is just an ordinary (unanswerable) message
    if is_confirmation:
        search_query = next(
            (
                content
                for role, content in reversed(history)
                if role == "user" and normalize_text(content) not in SHORT_REPLIES
            ),
            question,
        )
        if DEBUG:
            print(f"[debug] confirmation, searching with the previous question: {search_query}")
    elif normalized in CONFIRMATIONS:
        #nothing is pending, so there is nothing for "yes" to refer to
        search_query = question
    else:
        #rewrite follow-ups so references like "it" are resolved before searching
        search_query = rewrite_question(question, history)

    #gate 1: retrieval relevance
    results = retrieve(search_query)
    if not results:
        if DEBUG:
            print("[debug] refused at gate 1: no chunk scored above MIN_RELEVANCE_SCORE")
        return REFUSAL, []

    context = build_context(results)

    if USE_SCOPE_GATE and not is_answerable(search_query, history, context):
        if DEBUG:
            print("[debug] refused at gate 2: scope/evidence check was not ANSWERABLE")
        return REFUSAL, []
    
    system_prompt = render_prompt(
        "answer",
        rules=RULES_POLICY,
        context=context,
        history=format_recent(history),
        question=search_query,
    )

    messages = [
        SystemMessage(content=system_prompt),
        *to_messages(history),
        HumanMessage(content=search_query),
    ]

    response = answer_llm.invoke(messages)
    raw_answer = str(response.content).strip()
    answer, cited = split_answer(raw_answer)
    sources = cited_sources(cited, results)

    if answer and not sources:
        citation_reminder = HumanMessage(
            content=(
                "Revise your previous answer without changing its factual content. "
                "End with a SOURCES line containing the numbers of the context sources "
                "that support the answer, or SOURCES: none if none support it."
            )
        )
        revised_response = answer_llm.invoke(
            [*messages, AIMessage(content=raw_answer), citation_reminder]
        )
        revised_raw = str(revised_response.content).strip()
        revised_answer, revised_cited = split_answer(revised_raw)
        revised_sources = cited_sources(revised_cited, results)
        if revised_answer and revised_sources:
            answer, sources = revised_answer, revised_sources

    #the model refused, or returned nothing
    if not answer or REFUSAL in answer:
        if DEBUG:
            print("[debug] refused by the answer model (empty or returned the refusal text)")
        return REFUSAL, []

    answer = preserve_policy_caveats(answer, sources, results)
    if DEBUG:
        print(f"[debug] cited sources: {sources}")

    #An answer without a valid citation is refused only when citation enforcement is enabled.
    if not sources:
        if REQUIRE_CITATIONS:
            if DEBUG:
                print(f"[debug] refused at citation check, raw answer was:\n{answer}")
            return REFUSAL, []

    return answer, sources
   
#CLI
def main(session_id=None):
    global DEBUG

    session_id = session_id or str(uuid.uuid4())
    init_history()

    print("=" * 58)
    print("Northstar Home Support (demo chatbot)")
    print(f"Session: {session_id}")
    print("Commands: /exit, /clear, /sources, /debug")
    print("=" * 58)

    while True:
        #read the next question, exit cleanly on ctrl+c or ctrl+d
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        #ignore empty input
        if not question:
            continue

        command = question.lower()

        #exit the program
        if command in ("/exit", "/quit"):
            print("Goodbye.")
            break

        #delete the saved conversation
        if command == "/clear":
            clear_history(session_id)
            print("Conversation history cleared.")
            continue

        #turn debug output on or off
        if command == "/debug":
            DEBUG = not DEBUG
            print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
            continue

        #list every section that is indexed in the knowledge base
        if command == "/sources":
            records = vectorstore.get(include=["metadatas"])
            labels = sorted({
                source_label(meta) for meta in records["metadatas"] if meta
            })
            print("\nIndexed sections:")
            for label in labels:
                print(f"- {label}")
            continue

        history = load_history(session_id)

        try:
            answer, sources = answer_question(question, history)
        except KeyboardInterrupt:
            #ctrl+c while the model is working cancels this question only, nothing is saved
            print("\nCancelled. Ask another question or type /exit.")
            continue
        except Exception as exc:
            #connection problems are most likely caused by ollama not running
            if "connect" in str(exc).lower():
                print("\nError: could not reach Ollama. Is `ollama serve` running?")
            else:
                print(f"\nError: {exc}")
            continue

        display = answer

        #show the sources below the answer (only when SHOW_SOURCES is on)
        if SHOW_SOURCES and sources:
            display += "\n\nSources: " + ", ".join(str(source) for source in sources)

        print(f"\nAssistant: {display}")

        #every turn is saved, so a later "yes" or "no" is judged against what the customer really saw last
        #canned replies are filtered out of what the models see by model_history()
        save_message("user", mask_sensitive(question), session_id)
        save_message("assistant", answer, session_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the Northstar Home support chatbot.")
    parser.add_argument(
        "--session",
        dest="session_id",
        help="resume a named conversation; without this option a fresh session starts",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args().session_id)