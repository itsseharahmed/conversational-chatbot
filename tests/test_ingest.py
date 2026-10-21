"""Knowledge-base parsing and chunk annotation tests."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import ingest


def test_load_knowledge_base_parses_headings_and_source(
    monkeypatch, tmp_path
):
    source = tmp_path / "knowledge.md"
    source.write_text(
        "# Store\n## Returns\n### Damaged item\nContact support for help.",
        encoding="utf-8",
    )
    monkeypatch.setattr(ingest, "DOCUMENT_FILE", source)
    monkeypatch.setattr(ingest, "BASE_DIR", tmp_path)

    sections = ingest.load_knowledge_base()

    assert len(sections) == 1
    assert sections[0].page_content == "Contact support for help."
    assert sections[0].metadata["document_title"] == "Store"
    assert sections[0].metadata["section_title"] == "Returns"
    assert sections[0].metadata["subsection_title"] == "Damaged item"
    assert sections[0].metadata["source"] == "knowledge.md"
    assert sections[0].metadata["source_type"] == "markdown"


def test_load_knowledge_base_rejects_missing_or_empty_files(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.md"
    monkeypatch.setattr(ingest, "DOCUMENT_FILE", missing)

    with pytest.raises(FileNotFoundError, match="Knowledge base file not found"):
        ingest.load_knowledge_base()

    empty = tmp_path / "empty.md"
    empty.write_text(" \n", encoding="utf-8")
    monkeypatch.setattr(ingest, "DOCUMENT_FILE", empty)

    with pytest.raises(ValueError, match="Knowledge base file is empty"):
        ingest.load_knowledge_base()


def test_add_breadcrumbs_updates_text_and_metadata():
    chunks = [
        Document(
            page_content="Returns are accepted.",
            metadata={
                "document_title": "Store",
                "section_title": "Returns",
            },
        ),
        Document(page_content="Unheaded content.", metadata={}),
    ]

    result = ingest.add_breadcrumbs(chunks)

    assert result is chunks
    assert chunks[0].metadata["breadcrumb"] == "Store > Returns"
    assert chunks[0].page_content == "Store > Returns\n\nReturns are accepted."
    assert chunks[1].page_content == "Unheaded content."
    assert "breadcrumb" not in chunks[1].metadata


def test_ingest_main_rebuilds_collection_without_live_ollama(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "knowledge.md"
    source.write_text("# Store\n## Returns\nReturn items within 30 days.", encoding="utf-8")
    database = tmp_path / "chroma"
    captured = {}

    class FakeCollection:
        def delete_collection(self):
            captured["deleted_old_collection"] = True

    class FakeChroma:
        def __init__(self, **kwargs):
            captured["open_kwargs"] = kwargs

        def delete_collection(self):
            captured["deleted_old_collection"] = True

        @classmethod
        def from_documents(cls, **kwargs):
            captured["documents"] = kwargs["documents"]
            captured["embedding"] = kwargs["embedding"]
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                get=lambda: {"ids": [f"chunk-{i}" for i, _ in enumerate(kwargs["documents"])]}
            )

    embedding_calls = []

    class FakeEmbeddings:
        def embed_query(self, text):
            embedding_calls.append(text)
            return [1.0]

    embedding = FakeEmbeddings()
    monkeypatch.setattr(ingest, "DOCUMENT_FILE", source)
    monkeypatch.setattr(ingest, "BASE_DIR", tmp_path)
    monkeypatch.setattr(ingest, "DB_DIR", database)
    monkeypatch.setattr(
        ingest, "PrefixedOllamaEmbeddings", lambda model: embedding
    )
    monkeypatch.setattr(ingest, "Chroma", FakeChroma)

    ingest.main()

    assert database.exists()
    assert embedding_calls == ["test"]
    assert captured["deleted_old_collection"] is True
    assert captured["embedding"] is embedding
    assert captured["documents"]
    assert captured["documents"][0].metadata["breadcrumb"] == "Store > Returns"
    assert "created successfully" in capsys.readouterr().out
