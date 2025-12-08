# store_index.py 
from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

# modern public pinecone SDK imports
from pinecone import Pinecone, ServerlessSpec

# LangChain <-> Pinecone helper (package name: langchain-pinecone)
# Depending on your installed version this import location may vary.
# Try this first; if it fails, see notes below.
try:
    from langchain_pinecone import PineconeVectorStore
except Exception:
    # fallback - newer langchain integrates vectorstores differently;
    # we'll raise a clear error later if the import isn't available.
    PineconeVectorStore = None

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment (check .env).")
if not OPENAI_API_KEY:
    # Not strictly required for indexing, but warn if missing
    print("Warning: OPENAI_API_KEY not found in environment.")

# (optional) set in env for libraries that expect it:
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 1) Load PDFs
extracted_data = load_pdf_file(data="data/")       # returns list[Document]
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)              # returns list[Document]

# 2) Create embeddings object
embeddings = download_hugging_face_embeddings()    # must return embedding instance
if embeddings is None:
    raise RuntimeError("Embeddings instance not available. Check helper.download_hugging_face_embeddings()")

# 3) Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"      
dimension = 384                     

# create index 
# modern Pinecone client: use try/except because exact API can vary by version
try:
    existing = pc.list_indexes()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
except Exception as e:
    # some pinecone SDK versions have different list/create API - try alternate approach
    try:
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
    except Exception:
        raise RuntimeError("Failed to verify/create Pinecone index: " + str(e))

# get index handle
index = pc.Index(index_name)

# 4) Upsert documents to Pinecone via LangChain vectorstore
if PineconeVectorStore is None:
    raise RuntimeError(
        "langchain_pinecone.PineconeVectorStore not available. "
        "Install 'langchain-pinecone' or use a different vectorstore (Chroma/FAISS) "
        "or adapt to your langchain version."
    )

# Use the vectorstore convenience function. Signature may vary by version.
# This is a common pattern: from_documents(..., embedding=embeddings, index_name=...)
try:
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )
    print("Uploaded", len(text_chunks), "chunks to Pinecone index:", index_name)
except TypeError as te:
    # signature mismatch — try an alternate argument name
    try:
        docsearch = PineconeVectorStore.from_documents(
            text_chunks,
            embedding=embeddings,
            index_name=index_name,
        )
        print("Uploaded (alternate signature) ", len(text_chunks), "chunks to Pinecone.")
    except Exception as e:
        raise RuntimeError("Failed to create PineconeVectorStore: " + str(e))
except Exception as e:
    raise RuntimeError("Failed to upload documents to Pinecone: " + str(e))
