"""
rag/document_loader.py

Loads the already-provided hospital policy documents (policies/*.txt) and
splits them into overlapping chunks ready for embedding.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def load_policy_documents(policies_dir: Path | None = None) -> list[Document]:
    """Load every .txt policy file, tagging each Document with its source file."""
    directory = policies_dir or settings.policies_dir

    if not directory.exists():
        raise FileNotFoundError(f"Policies directory not found: {directory}")

    loader = DirectoryLoader(
        str(directory),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()

    for doc in documents:
        source_path = Path(doc.metadata.get("source", "unknown"))
        doc.metadata["source"] = source_path.name
        doc.metadata["policy_name"] = source_path.stem.replace("_", " ").title()

    logger.info("Loaded %d policy document(s) from %s", len(documents), directory)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def load_and_split_policies(policies_dir: Path | None = None) -> list[Document]:
    """Convenience helper combining load + split in one call."""
    documents = load_policy_documents(policies_dir)
    return split_documents(documents)
