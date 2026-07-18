"""
rag/vector_store.py

Builds a FAISS vector store from the hospital policy documents and persists
it to disk so it doesn't need to be rebuilt on every app start-up. If an
index already exists at settings.faiss_index_dir, it is loaded instead of
being regenerated.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from rag.document_loader import load_and_split_policies
from utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE_NAME = "index.faiss"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the sentence-transformers embedding model used across the app."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """
    Load the persisted FAISS index if present, otherwise build it from the
    policy documents and persist it for next time.

    Args:
        force_rebuild: if True, ignores any existing index and rebuilds it
            (useful after policy documents change).
    """
    index_dir = settings.faiss_index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    index_ready = (index_dir / _INDEX_FILE_NAME).exists()

    if index_ready and not force_rebuild:
        logger.info("Loading existing FAISS index from %s", index_dir)
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("Building FAISS index from policy documents...")
    chunks = load_and_split_policies()
    if not chunks:
        raise ValueError("No policy document chunks were produced; check policies/ folder.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(index_dir))
    logger.info("FAISS index built and saved to %s", index_dir)
    return vector_store
