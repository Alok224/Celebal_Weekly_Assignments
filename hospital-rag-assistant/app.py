"""
app.py

Streamlit entry point for the RAG-Based Healthcare Query Assistant.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from langchain_groq import ChatGroq

from agents.orchestrator import Orchestrator
from agents.rag_agent import RAGAgent
from agents.sql_agent import build_sql_agent
from config import settings
from database.db_connector import check_connection, get_sql_database
from rag.retriever import get_retriever
from rag.vector_store import build_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Hospital RAG Assistant",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_orchestrator() -> Orchestrator:
    """Build every component once per server process and cache it."""
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
    )

    db = get_sql_database()
    sql_agent = build_sql_agent(llm, db)

    vector_store = build_vector_store()
    retriever = get_retriever(vector_store)
    rag_agent = RAGAgent(retriever=retriever, llm=llm)

    return Orchestrator(classifier_llm=llm, sql_agent=sql_agent, rag_agent=rag_agent)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("🏥 Hospital RAG Assistant")
        st.caption("Multi-agent assistant for patient data & hospital policies.")

        st.subheader("System Status")
        db_ok = check_connection()
        st.write("PostgreSQL:", " Connected" if db_ok else " Unavailable")
        st.write("Vector Store:", f" {settings.faiss_index_dir.name}/")
        st.write("LLM (Groq):", f" {settings.groq_model}")

        st.divider()
        st.subheader("Ask about...")
        st.markdown(
            "- **Patient data** → routed to the SQL Agent\n"
            "- **Hospital policies** → routed to the RAG Agent\n"
        )
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


def render_message(role: str, content: dict) -> None:
    with st.chat_message(role):
        st.markdown(content["answer"])

        if role != "assistant":
            return

        badge = "🗄️ SQL Agent" if content["route"] == "SQL" else "📄 RAG Agent"
        st.caption(f"Detected agent: **{badge}**  ·  _{content['route_reason']}_")

        if content.get("generated_sql"):
            with st.expander("Generated SQL"):
                for query in content["generated_sql"]:
                    st.code(query, language="sql")

        if content.get("sources"):
            with st.expander(f"Retrieved documents ({len(content['sources'])})"):
                for chunk in content["sources"]:
                    st.markdown(f"**{chunk.policy_name}** · `{chunk.source_file}`")
                    st.text(chunk.content)
                    st.divider()

        if content.get("error"):
            st.error(f"Underlying error: {content['error']}")


def main() -> None:
    render_sidebar()

    st.title("Hospital Query Assistant")
    st.caption("Ask about patient records or hospital policies in plain English.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        render_message(role, content)

    user_query = st.chat_input("e.g. 'How many patients were admitted for Diabetes?'")
    if not user_query:
        return

    st.session_state.chat_history.append(("user", {"answer": user_query}))
    render_message("user", {"answer": user_query})

    try:
        orchestrator = load_orchestrator()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialize the assistant: {exc}")
        logger.error("Initialization error: %s", exc)
        return

    with st.spinner("Thinking..."):
        result = orchestrator.handle_query(user_query)

    assistant_content = {
        "answer": result.answer,
        "route": result.route,
        "route_reason": result.route_reason,
        "generated_sql": result.generated_sql,
        "sources": result.sources,
        "error": result.error,
    }
    st.session_state.chat_history.append(("assistant", assistant_content))
    render_message("assistant", assistant_content)


if __name__ == "__main__":
    main()
