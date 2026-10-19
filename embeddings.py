#shared embeddings: adds the task prefixes that nomic-embed-text was trained with

#for creating embeddings with ollama
from langchain_ollama import OllamaEmbeddings

#prefix for text that is stored in the database (document chunks)
DOCUMENT_PREFIX = "search_document: "

#prefix for text that is used to search the database (user questions)
QUERY_PREFIX = "search_query: "


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    #embeds chunks with the document prefix
    def embed_documents(self, texts):
        return super().embed_documents([f"{DOCUMENT_PREFIX}{text}" for text in texts])

    #embeds questions with the query prefix
    def embed_query(self, text):
        return super().embed_query(f"{QUERY_PREFIX}{text}")