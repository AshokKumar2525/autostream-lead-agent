"""
retriever.py - RAG Retriever Setup

Loads the AutoStream knowledge base from a local markdown file,
splits it into chunks, embeds with HuggingFace sentence-transformers,
and stores in a FAISS vector store for retrieval.
"""

import os
import sys
import io
import warnings
import logging
import contextlib

# ── Suppress ALL warnings from HuggingFace / transformers / safetensors ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_retriever():
    """
    Build and return a FAISS-based retriever from the knowledge base.

    Steps:
        1. Load the autostream_kb.md file using TextLoader.
        2. Split documents into chunks (chunk_size=500, overlap=50).
        3. Create embeddings using all-MiniLM-L6-v2 model.
        4. Build a FAISS vector store from the chunks.
        5. Return a retriever with k=2 (top 2 most relevant chunks).

    Returns:
        A LangChain retriever object for semantic search.
    """
    # Resolve the path to the knowledge base file
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_base",
        "autostream_kb.md"
    )

    # Load the markdown knowledge base
    loader = TextLoader(kb_path, encoding="utf-8")
    documents = loader.load()

    # Split into smaller chunks for better retrieval precision
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings using a lightweight sentence-transformer model
    # Redirect stderr to suppress HuggingFace/safetensors console warnings
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    # Build the FAISS vector store from document chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Return a retriever configured to fetch top-2 results
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    return retriever
