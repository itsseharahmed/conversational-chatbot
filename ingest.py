#for file and directory paths 
from pathlib import Path

#for reading and coverting text files into document objects
from langchain_community.document_loaders import TextLoader

#for splitting documents into smaller chunks
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

#for creating embeddings with the nomic-embed-text task prefixes (see embeddings.py)
from embeddings import PrefixedOllamaEmbeddings

#for creating a vector database using Chroma
from langchain_chroma import Chroma

#get the absolute path of the current file and its parent directory
BASE_DIR = Path(__file__).resolve().parent

#define the path to the knowledge base document and the directory for the Chroma database
DOCUMENT_FILE = BASE_DIR / "documents" / "northstar_knowledge_base.md"
DB_DIR = BASE_DIR / "data" / "chroma"

#define the chroma collection name and the ollama embedding model to be used
COLLECTION = "approved_knowledge"
EMBEDDING_MODEL = "nomic-embed-text"

#metadata keys that hold the markdown headers
HEADER_KEYS = (
    "document_title",
    "section_title",
    "subsection_title",
    "subsubsection_title",
)

def load_knowledge_base():
    #check if the knowledge base document exists before attempting to load it
    if not DOCUMENT_FILE.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {DOCUMENT_FILE}")
    
    #load the knowledge base document
    loader = TextLoader(str(DOCUMENT_FILE), encoding="utf-8",)
    
    #load the document into a list of Document objects
    documents = loader.load()

    #check if the loaded documents are empty or contain only whitespace, and raise an error if so
    if not documents or not documents[0].page_content.strip():
        raise ValueError(f"Knowledge base file is empty: {DOCUMENT_FILE}")

    #preserve markdown headers as metadata for citations
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "document_title"), ("##", "section_title"), ("###", "subsection_title"), ("####", "subsubsection_title"),],
        strip_headers=True,
    )

    sections = markdown_splitter.split_text(documents[0].page_content)

    source = DOCUMENT_FILE.relative_to(BASE_DIR).as_posix()

    for doc in sections:
        doc.metadata["source"] = source
        doc.metadata["source_type"] = "markdown"

    return sections

def add_breadcrumbs(chunks):
    #prepends a heading breadcrumb like "Title > Section > Subsection" to every chunk
    for chunk in chunks:
        parts = [chunk.metadata[key] for key in HEADER_KEYS if chunk.metadata.get(key)]

        #chunks without any headers are left as they are
        if not parts:
            continue

        breadcrumb = " > ".join(parts)

        #keep a copy in the metadata and put the breadcrumb at the start of the text
        chunk.metadata["breadcrumb"] = breadcrumb
        chunk.page_content = f"{breadcrumb}\n\n{chunk.page_content}"

    return chunks

def main():
    print("Loading knowledge base...")
    documents = load_knowledge_base()

    #split sections into smaller chunks for better embedding and retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150,add_start_index=True,)

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the knowledge base. Please check the document content and splitting parameters.")

    #add the heading breadcrumb to every chunk after splitting so it does not count towards the chunk size
    chunks = add_breadcrumbs(chunks)

    print(f"Loaded {len(documents)} sections from the knowledge base.")
    print(f"Created {len(chunks)} chunks for embedding.")

    #the embeddings class adds the "search_document: " prefix to every chunk automatically
    print(f"Loading embeddings model: '{EMBEDDING_MODEL}'...")
    embeddings = PrefixedOllamaEmbeddings(model=EMBEDDING_MODEL)

    DB_DIR.mkdir(parents=True, exist_ok=True)

    try: 
        old_db = Chroma(persist_directory=str(DB_DIR), collection_name=COLLECTION, embedding_function=embeddings,)
        old_db.delete_collection()
        print(f"Removed previous collection.")
    except Exception as e:
        print(f"No previous collection removed ({e})")

    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR), collection_name=COLLECTION, collection_metadata={"hnsw:space": "cosine"},)

    print(f"Knowledge base created successfully.")
    print(f"Source file: {DOCUMENT_FILE.name}")
    print(f"Indexed {len(chunks)} chunks into the Chroma database.")
    print(f"Chroma database directory: {DB_DIR}")
    print(f"Collection name: {COLLECTION}")

if __name__ == "__main__":
    main()