# #Extract Data From the PDF File
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from typing import List
# from langchain.schema import Document



# src/helper.py
# Compatible helper for loading PDFs, splitting text and creating embeddings
# Works with both old and new langchain package layouts.

#
# from pathlib import Path
# from typing import List
#
# # ----------------- safe imports with fallbacks -----------------
# # Document loaders (PyPDFLoader, DirectoryLoader)
# try:
#     # preferred new location
#     from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# except Exception:
#     try:
#         # fallback to older location
#         from langchain.document_loaders import PyPDFLoader, DirectoryLoader
#     except Exception as e:
#         raise ImportError(
#             "Could not import PyPDFLoader/DirectoryLoader. "
#             "Install langchain-community or use a compatible langchain."
#         ) from e
#
# # Text splitter (RecursiveCharacterTextSplitter)
# try:
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
# except Exception:
#     try:
#         from langchain_text_splitters import RecursiveCharacterTextSplitter
#     except Exception as e:
#         raise ImportError(
#             "Could not import RecursiveCharacterTextSplitter. "
#             "Install langchain-text-splitters or use a compatible langchain."
#         ) from e
#
# # Embeddings (HuggingFaceEmbeddings) - optional fallback
# try:
#     from langchain_community.embeddings import HuggingFaceEmbeddings
# except Exception:
#     try:
#         from langchain.embeddings import HuggingFaceEmbeddings
#     except Exception:
#         HuggingFaceEmbeddings = None  # Will raise later if used without installed package
#
#
# # Document type (langchain_core.documents or langchain.schema)
# try:
#     from langchain_core.documents import Document
# except Exception:
#     try:
#         from langchain.schema import Document
#     except Exception:
#         # tiny fallback dataclass if neither is available (keeps typing)
#         from dataclasses import dataclass
#
#         @dataclass
#         class Document:
#             page_content: str
#             metadata: dict
#
# # ----------------- helper functions -----------------
#
# def load_pdf_file(data: str) -> List[Document]:
#     """
#     Load all PDF files from `data` directory (recursively). Returns list of Documents.
#     """
#     data_path = Path(data)
#     if not data_path.exists() or not data_path.is_dir():
#         raise FileNotFoundError(f"Directory not found: {data!r}")
#
#     loader = DirectoryLoader(
#         str(data_path),
#         glob="**/*.pdf",      # recursive match for all .pdf files
#         loader_cls=PyPDFLoader
#     )
#     documents = loader.load()
#     return documents
#
#
# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     """
#     Given Documents, return new Documents keeping only page_content and metadata['source'].
#     """
#     minimal_docs: List[Document] = []
#     for doc in docs:
#         src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={"source": src}
#             )
#         )
#     return minimal_docs
#
#
# def text_split(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
#     """
#     Split documents into chunks using RecursiveCharacterTextSplitter (returns list of Documents).
#     """
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = splitter.split_documents(documents)
#     return chunks
#
#
# def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#     """
#     Return HuggingFaceEmbeddings instance (requires langchain_community or compatible langchain).
#     """
#     if HuggingFaceEmbeddings is None:
#         raise RuntimeError("HuggingFaceEmbeddings is not available. Install langchain_community or proper embeddings package.")
#     return HuggingFaceEmbeddings(model_name=model_name)
#
#
# def load_pdf_file(data):
#     loader = DirectoryLoader(
#         data,
#         glob='*.pdf',
#         loader_cls=PyPDFLoader
#     )
#     documents = loader.load()
#     return documents
#
#
#
# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     """
#     Given a list of Document objects, return a new list of Document objects
#     containing only 'source' in metadata and the original page_content.
#     """
#     minimal_docs: List[Document] = []
#
#     for doc in docs:
#         src = doc.metadata.get("source")
#         minimal_docs.append(
#             Document(
#                 page_content = doc.page_content,
#                 metadata = {"source": src}
#             )
#         )
#
#     return minimal_docs
#
#
# # Split the document into smaller chunks
#
# def text_split(minimal_docs):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=20,
#     )
#     text_chunk =text_splitter.split_documents(minimal_docs)
#     return text_chunk
#
#
#
# #Download the Embeddings From HUGGINGFACE
# def download_huggingface_embeddings():
#     embeddings = HuggingFaceEmbeddings(
#         model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     )
#     return embeddings
#
#
# def download_hugging_face_embeddings():
#     return None




#
# from pathlib import Path
# from typing import List
#
# # ----------------- safe imports with fallbacks -----------------
# # Document loaders (PyPDFLoader, DirectoryLoader)
# try:
#     # preferred new location
#     from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# except Exception:
#     try:
#         # fallback to older location
#         from langchain.document_loaders import PyPDFLoader, DirectoryLoader
#     except Exception as e:
#         raise ImportError(
#             "Could not import PyPDFLoader/DirectoryLoader. "
#             "Install langchain-community or use a compatible langchain."
#         ) from e
#
# # Text splitter (RecursiveCharacterTextSplitter)
# try:
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
# except Exception:
#     try:
#         from langchain_text_splitters import RecursiveCharacterTextSplitter
#     except Exception as e:
#         raise ImportError(
#             "Could not import RecursiveCharacterTextSplitter. "
#             "Install langchain-text-splitters or use a compatible langchain."
#         ) from e
#
# # Embeddings (HuggingFaceEmbeddings) - optional fallback
# try:
#     from langchain_community.embeddings import HuggingFaceEmbeddings
# except Exception:
#     try:
#         from langchain.embeddings import HuggingFaceEmbeddings
#     except Exception:
#         HuggingFaceEmbeddings = None  # Will raise later if used without installed package
#
# # Document type (langchain_core.documents or langchain.schema)
# try:
#     from langchain_core.documents import Document
# except Exception:
#     try:
#         from langchain.schema import Document
#     except Exception:
#         # tiny fallback dataclass if neither is available (keeps typing)
#         from dataclasses import dataclass
#
#         @dataclass
#         class Document:
#             page_content: str
#             metadata: dict
#
# # ----------------- helper functions -----------------
#
# def load_pdf_file(data: str) -> List[Document]:
#     """
#     Load all PDF files from `data` directory (recursively). Returns list of Documents.
#     """
#     data_path = Path(data)
#     if not data_path.exists() or not data_path.is_dir():
#         raise FileNotFoundError(f"Directory not found: {data!r}")
#
#     loader = DirectoryLoader(
#         str(data_path),
#         glob="**/*.pdf",      # recursive match for all .pdf files
#         loader_cls=PyPDFLoader
#     )
#     documents = loader.load()
#     return documents
#
#
# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     """
#     Given Documents, return new Documents keeping only page_content and metadata['source'].
#     """
#     minimal_docs: List[Document] = []
#     for doc in docs:
#         src = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={"source": src}
#             )
#         )
#     return minimal_docs
#
#
# def text_split(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
#     """
#     Split documents into chunks using RecursiveCharacterTextSplitter (returns list of Documents).
#     """
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = splitter.split_documents(documents)
#     return chunks
#
#
# def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#     """
#     Return HuggingFaceEmbeddings instance (requires langchain_community or compatible langchain).
#     """
#     if HuggingFaceEmbeddings is None:
#         raise RuntimeError("HuggingFaceEmbeddings is not available. Install langchain_community or proper embeddings package.")
#     return HuggingFaceEmbeddings(model_name=model_name)















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