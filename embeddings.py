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
    #OllamaEmbeddings.embed_query is implemented as embed_documents([text])[0], so calling super().embed_query here
    #would go through the override above and add the document prefix too ("search_document: search_query: ...")
    #calling the parent's embed_documents directly sends exactly one prefix
    def embed_query(self, text):
        return super().embed_documents([f"{QUERY_PREFIX}{text}"])[0]

    #async versions follow the same rule, otherwise they would silently send no prefix or two
    async def aembed_documents(self, texts):
        return await super().aembed_documents([f"{DOCUMENT_PREFIX}{text}" for text in texts])

    async def aembed_query(self, text):
        return (await super().aembed_documents([f"{QUERY_PREFIX}{text}"]))[0]