"""
agents/sql_agent.py

Builds a LangChain agent (langchain.agents.create_agent, running on
LangGraph under the hood) wired to the hospital_db PostgreSQL database via
SQLDatabaseToolkit. The agent can list tables, inspect schema, write SQL,
execute it, and self-correct on errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel

from prompts.sql_prompts import SQL_AGENT_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

# The toolkit's SQL execution tool is named "sql_db_query" in langchain_community.
SQL_QUERY_TOOL_NAME = "sql_db_query"


@dataclass
class SQLAgentResult:
    """Normalized result returned to the orchestrator / UI layer."""

    answer: str
    generated_sql: list[str] = field(default_factory=list)
    error: str | None = None


def build_sql_agent(llm: BaseChatModel, db: SQLDatabase):
    """Create the SQL agent bound to the given database and LLM."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SQL_AGENT_SYSTEM_PROMPT,
    )
    logger.info("SQL agent initialized with %d tool(s)", len(tools))
    return agent


def _extract_generated_sql(messages: list) -> list[str]:
    """Pull every SQL string the agent actually executed from the message trace."""
    queries: list[str] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name == SQL_QUERY_TOOL_NAME:
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                query = (args or {}).get("query")
                if query:
                    queries.append(query)
    return queries


def run_sql_agent(agent, query: str) -> SQLAgentResult:
    """Invoke the SQL agent on a natural-language query and normalize the output."""
    try:
        result = agent.invoke({"messages": [("user", query)]})
        messages = result.get("messages", [])
        final_answer = messages[-1].content if messages else ""
        generated_sql = _extract_generated_sql(messages)
        return SQLAgentResult(answer=final_answer, generated_sql=generated_sql)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as an error state
        logger.error("SQL agent execution failed: %s", exc)
        return SQLAgentResult(
            answer="I couldn't retrieve that information due to a database error.",
            error=str(exc),
        )
