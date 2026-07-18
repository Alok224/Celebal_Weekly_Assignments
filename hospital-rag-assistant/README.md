# Hospital RAG Assistant

A multi-agent, RAG-based healthcare query assistant for hospital staff. A
single chat interface lets staff ask questions in plain English; an
LLM-based orchestrator routes each query to the right specialist agent:

- **SQL Agent** → structured patient/admission/billing data in PostgreSQL
- **RAG Agent** → hospital policy documents (admission, billing, discharge,
  emergency, insurance), retrieved from a FAISS vector store

## Architecture

```
User query
    │
    ▼
Orchestrator (LLM intent classifier)
    │
    ├── "SQL"  → SQL Agent → SQLDatabaseToolkit → PostgreSQL (hospital_db)
    │
    └── "RAG"  → RAG Agent → FAISS retriever → policy chunks → Groq LLM
```

## Project layout

```
hospital-rag-assistant/
├── app.py                  # Streamlit chat UI
├── config.py                # Central settings (env vars, paths)
├── requirements.txt
├── .env.example
├── database/
│   └── db_connector.py      # SQLDatabase wired to existing hospital_db
├── rag/
│   ├── document_loader.py    # Load + split policy .txt files
│   ├── vector_store.py       # Build/persist/load FAISS index
│   └── retriever.py          # Top-k retriever
├── agents/
│   ├── sql_agent.py          # NL → SQL agent (create_agent + SQLDatabaseToolkit)
│   ├── rag_agent.py          # Retrieve-then-generate policy Q&A
│   └── orchestrator.py       # LLM-based routing between the two agents
├── prompts/                  # System prompts for each agent
├── utils/logger.py           # Shared logging config
├── faiss_index/              # Persisted vector index (generated on first run)
├── policies/                 # Hospital policy documents (already provided)
└── notebooks/                 # Original dataset-preparation notebook
```

## Prerequisites (already completed, not rebuilt by this project)

- `hospital_db` PostgreSQL database, populated and normalized with tables
  `patients`, `admissions`, `medical_records`, `insurance`
  (see `notebooks/dataset_preparation.ipynb`)
- Hospital policy documents in `policies/`
- A free [Groq API key](https://console.groq.com/)

## Setup

```bash
cd hospital-rag-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env with your DB_PASSWORD and GROQ_API_KEY

streamlit run app.py
```

On first run, the app builds the FAISS index from `policies/` and saves it
to `faiss_index/`. Subsequent runs load the saved index directly.

## How routing works

`agents/orchestrator.py` sends every query to the Groq LLM with a structured
output schema (`RouteDecision`) asking it to classify the query as `SQL` or
`RAG`. This is a genuine LLM classification step, not keyword matching, so
it generalizes to phrasing the keyword approach would miss.

## Notes on safety

- The SQL agent's system prompt (`prompts/sql_prompts.py`) explicitly
  restricts it to read-only queries and caps result size.
- `database/db_connector.py` limits the agent's visible tables to the four
  hospital tables via `include_tables`.

## Extending the project

- Swap `GROQ_MODEL` in `.env` to any other Groq-hosted model.
- Add more policy `.txt`/`.pdf` files to `policies/` and call
  `build_vector_store(force_rebuild=True)` to re-index.
- Add more SQL guardrails (row-level security, query allow-lists) before
  connecting this to a production database with real patient data.
