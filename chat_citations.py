import re


HEADER_KEYS = (
    "document_title",
    "section_title",
    "subsection_title",
    "subsubsection_title",
)


def source_label(metadata):
    parts = [metadata[key] for key in HEADER_KEYS if metadata.get(key)]
    return " > ".join(parts) if parts else metadata.get("source", "Unknown source")


SOURCES_LINE = re.compile(
    r"(?im)(?:^|[ \t])[*_`]*sources?\s*:[ \t*_`]*(?P<nums>none|(?:\[?[ \t]*(?:source[ \t]*)?\d+[ \t]*\]?[ \t,;]*)+)[ \t*_`]*$",
    re.IGNORECASE | re.MULTILINE,
)
INLINE_TAG = re.compile(
    r"[ \t]*\[\s*sources?\s*:?\s*(?P<nums>\d+(?:\s*[,;&]\s*(?:source\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)


def split_answer(raw):
    nums = set()
    for match in SOURCES_LINE.finditer(raw):
        nums.update(int(number) for number in re.findall(r"\d+", match.group("nums")))
    for match in INLINE_TAG.finditer(raw):
        nums.update(int(number) for number in re.findall(r"\d+", match.group("nums")))
    text = INLINE_TAG.sub("", SOURCES_LINE.sub("", raw))
    return re.sub(r"\n{3,}", "\n\n", text).strip(), nums


def cited_sources(citations, results):
    if isinstance(citations, str):
        _answer, numbers = split_answer(citations)
    else:
        numbers = set(citations)
    return sorted(number for number in numbers if 1 <= number <= len(results))


def strip_citations(text):
    return split_answer(text)[0]
