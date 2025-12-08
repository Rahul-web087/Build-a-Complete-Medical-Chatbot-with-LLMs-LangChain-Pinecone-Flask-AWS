
from pathlib import Path
from typing import List

# ----------------- safe imports with fallbacks -----------------
# Document loaders (PyPDFLoader, DirectoryLoader)
try:
    # preferred new location
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
except Exception:
    try:
        # fallback to older location
        from langchain.document_loaders import PyPDFLoader, DirectoryLoader
    except Exception as e:
        raise ImportError(
            "Could not import PyPDFLoader/DirectoryLoader. "
            "Install langchain-community or use a compatible langchain."
        ) from e

# Text splitter (RecursiveCharacterTextSplitter)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as e:
        raise ImportError(
            "Could not import RecursiveCharacterTextSplitter. "
            "Install langchain-text-splitters or use a compatible langchain."
        ) from e

# Embeddings (HuggingFaceEmbeddings) - updated import
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception as e:
    raise ImportError(
        "Could not import HuggingFaceEmbeddings. "
        "Install langchain-huggingface with `pip install -U langchain-huggingface`."
    ) from e

# Document type (langchain_core.documents or langchain.schema)
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        # tiny fallback dataclass if neither is available (keeps typing)
        from dataclasses import dataclass

        @dataclass
        class Document:
            page_content: str
            metadata: dict

# ----------------- helper functions -----------------

def load_pdf_file(data: str) -> List[Document]:
    """
    Load all PDF files from `data` directory (recursively). Returns list of Documents.
    """
    data_path = Path(data)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {data!r}")

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",      # recursive match for all .pdf files
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given Documents, return new Documents keeping only page_content and metadata['source'].
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


def text_split(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter (returns list of Documents).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Return HuggingFaceEmbeddings instance (requires langchain-huggingface).
    """
    return HuggingFaceEmbeddings(model_name=model_name)
