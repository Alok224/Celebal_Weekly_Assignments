"""
agents/rag_agent.py

Retrieval-augmented agent that answers policy questions using the FAISS
vector store built from the hospital policy documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from prompts.rag_prompts import RAG_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SourceChunk:
    """A single retrieved chunk, shaped for display in the UI."""

    policy_name: str
    source_file: str
    content: str


@dataclass
class RAGAgentResult:
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    error: str | None = None


class RAGAgent:
    """Thin, explicit retrieve-then-generate pipeline (no hidden agent loop)."""

    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm

    def _format_context(self, documents: list[Document]) -> str:
        blocks = []
        for i, doc in enumerate(documents, start=1):
            policy_name = doc.metadata.get("policy_name", "Policy")
            blocks.append(f"[{i}] ({policy_name})\n{doc.page_content}")
        return "\n\n".join(blocks)

    def answer(self, question: str) -> RAGAgentResult:
        try:
            documents = self.retriever.invoke(question)
            if not documents:
                return RAGAgentResult(
                    answer="I couldn't find any relevant policy information for that question.",
                )

            context = self._format_context(documents)
            prompt_messages = RAG_PROMPT.invoke({"context": context, "question": question})
            response = self.llm.invoke(prompt_messages)

            sources = [
                SourceChunk(
                    policy_name=doc.metadata.get("policy_name", "Policy"),
                    source_file=doc.metadata.get("source", "unknown"),
                    content=doc.page_content,
                )
                for doc in documents
            ]
            return RAGAgentResult(answer=response.content, sources=sources)
        except Exception as exc:  # noqa: BLE001
            logger.error("RAG agent execution failed: %s", exc)
            return RAGAgentResult(
                answer="I couldn't retrieve an answer from the policy documents due to an error.",
                error=str(exc),
            )
