"""
agents/orchestrator.py

The Orchestrator classifies every incoming user query with an LLM (not
keyword matching) and routes it to either the SQL Agent (patient/hospital
data) or the RAG Agent (hospital policy documents).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel

from agents.rag_agent import RAGAgent, RAGAgentResult
from agents.sql_agent import SQLAgentResult, run_sql_agent
from prompts.classifier_prompts import CLASSIFIER_PROMPT, RouteDecision
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrchestratorResult:
    route: Literal["SQL", "RAG"]
    route_reason: str
    answer: str
    generated_sql: list[str]
    sources: list
    error: str | None = None


class Orchestrator:
    def __init__(self, classifier_llm: BaseChatModel, sql_agent, rag_agent: RAGAgent):
        self.classifier = classifier_llm.with_structured_output(RouteDecision)
        self.sql_agent = sql_agent
        self.rag_agent = rag_agent

    def classify(self, query: str) -> RouteDecision:
        prompt_messages = CLASSIFIER_PROMPT.invoke({"query": query})
        decision: RouteDecision = self.classifier.invoke(prompt_messages)
        logger.info("Routed query to %s (%s)", decision.route, decision.reason)
        return decision

    def handle_query(self, query: str) -> OrchestratorResult:
        decision = self.classify(query)

        if decision.route == "SQL":
            result: SQLAgentResult = run_sql_agent(self.sql_agent, query)
            return OrchestratorResult(
                route="SQL",
                route_reason=decision.reason,
                answer=result.answer,
                generated_sql=result.generated_sql,
                sources=[],
                error=result.error,
            )

        result: RAGAgentResult = self.rag_agent.answer(query)
        return OrchestratorResult(
            route="RAG",
            route_reason=decision.reason,
            answer=result.answer,
            generated_sql=[],
            sources=result.sources,
            error=result.error,
        )
