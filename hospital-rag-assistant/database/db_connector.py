"""
database/db_connector.py

Thin wrapper around the already-built and normalized `hospital_db` PostgreSQL
database (patients, admissions, medical_records, insurance tables — see
notebooks/dataset_preparation.ipynb for how they were created).

This module does NOT recreate the schema. It only exposes:
  * a LangChain `SQLDatabase` object for the SQL agent to use, and
  * a lightweight health-check function used at app start-up.
"""

from __future__ import annotations

from langchain_community.utilities import SQLDatabase

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


ALLOWED_TABLES = ["patients", "admissions", "medical_records", "insurance"]


def get_sql_database() -> SQLDatabase:
    """
    Build and return a LangChain SQLDatabase bound to the existing hospital_db.

    Raises:
        RuntimeError: if the connection cannot be established, with the
        underlying driver error preserved in the message for debugging.
    """
    try:
        db = SQLDatabase.from_uri(
            settings.sqlalchemy_uri,
            include_tables=ALLOWED_TABLES,
            sample_rows_in_table_info=2,
        )
        logger.info("Connected to PostgreSQL database '%s' at %s:%s",
                     settings.db_name, settings.db_host, settings.db_port)
        return db
    except Exception as exc:  # noqa: BLE001 - surfaced to caller intentionally
        logger.error("Failed to connect to PostgreSQL: %s", exc)
        raise RuntimeError(
            "Could not connect to hospital_db. Check DB_HOST/DB_PORT/DB_NAME/"
            "DB_USER/DB_PASSWORD in your .env file."
        ) from exc


def check_connection() -> bool:
    """Quick boolean health check used by the Streamlit sidebar."""
    try:
        db = get_sql_database()
        db.run("SELECT 1;")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Database health check failed: %s", exc)
        return False
