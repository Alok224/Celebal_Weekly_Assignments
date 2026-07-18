"""
rag/retriever.py

Wraps the FAISS vector store in a retriever configured with the project's
default top-k setting.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS

from config import settings


def get_retriever(vector_store: FAISS, k: int | None = None) -> BaseRetriever:
    """Return a similarity-search retriever over the given FAISS store."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.retrieval_top_k},
    )
