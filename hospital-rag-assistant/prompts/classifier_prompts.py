"""
prompts/classifier_prompts.py

Prompt and structured-output schema used by the Orchestrator agent to decide
whether an incoming query should be routed to the SQL Agent or the RAG Agent.
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured routing decision returned by the classifier LLM."""

    route: Literal["SQL", "RAG"] = Field(
        description=(
            "'SQL' if the query asks about specific patient records, "
            "admissions, billing amounts, doctors, hospitals, or any data "
            "that lives in the hospital database. 'RAG' if the query asks "
            "about hospital policies, rules, procedures, or general "
            "guidance found in policy documents."
        )
    )
    reason: str = Field(description="One short sentence explaining the decision.")


CLASSIFIER_SYSTEM_PROMPT = """\
You are an intent classification component inside a hospital assistant
application. Classify the user's query into exactly one route:

- SQL: the query needs facts from structured patient/hospital data
  (e.g. patient counts, ages, billing amounts, admission dates, doctors,
  hospitals, insurance providers, medical conditions of specific patients).
- RAG: the query needs information from hospital policy documents
  (e.g. admission rules, discharge process, billing policy, insurance
  policy, emergency care policy).

Always choose exactly one route, even if the query is ambiguous — pick the
closer match.
"""

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFIER_SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)
