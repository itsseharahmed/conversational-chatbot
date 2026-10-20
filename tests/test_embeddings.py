"""Prefix behavior is tested by capturing inputs before any Ollama call."""

from langchain_ollama import OllamaEmbeddings

from embeddings import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    PrefixedOllamaEmbeddings,
)


def test_document_embeddings_add_search_document_prefix(monkeypatch):
    captured = []
    monkeypatch.setattr(
        OllamaEmbeddings,
        "embed_documents",
        lambda self, texts: captured.extend(texts) or [[1.0] for _ in texts],
    )
    embedder = PrefixedOllamaEmbeddings(model="unused-in-test")

    assert embedder.embed_documents(["return policy", "shipping policy"]) == [
        [1.0],
        [1.0],
    ]
    assert captured == [
        f"{DOCUMENT_PREFIX}return policy",
        f"{DOCUMENT_PREFIX}shipping policy",
    ]


def test_query_embeddings_add_search_query_prefix(monkeypatch):
    captured = []
    monkeypatch.setattr(
        OllamaEmbeddings,
        "embed_query",
        lambda self, text: captured.append(text) or [1.0],
    )
    embedder = PrefixedOllamaEmbeddings(model="unused-in-test")

    assert embedder.embed_query("return policy") == [1.0]
    assert captured == [f"{QUERY_PREFIX}return policy"]
