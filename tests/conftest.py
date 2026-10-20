import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def chat_module():
    class FakeVectorStore:
        def get(self, **kwargs):
            return {"ids": ["test-document"], "metadatas": []}

        def similarity_search_with_relevance_scores(self, query, k):
            raise AssertionError("A test must configure retrieval explicitly")

    class FakeModel:
        def invoke(self, messages):
            raise AssertionError("A test must configure model responses explicitly")

    fake_vectorstore = FakeVectorStore()
    original_exists = Path.exists
    local_database = Path(__file__).resolve().parents[1] / "data" / "chroma"

    def exists_without_local_database(path):
        if path == local_database:
            return True
        return original_exists(path)

    # Import chat without opening the user's Chroma database or contacting Ollama.
    with (
        patch.object(Path, "exists", exists_without_local_database),
        patch("langchain_chroma.Chroma", return_value=fake_vectorstore),
        patch("langchain_ollama.ChatOllama", side_effect=lambda **kwargs: FakeModel()),
        patch(
            "embeddings.PrefixedOllamaEmbeddings",
            return_value=SimpleNamespace(),
        ),
    ):
        module = importlib.import_module("chat")

    return module
