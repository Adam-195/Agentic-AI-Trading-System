"""
graph/checkpointer.py
LangGraph checkpoint persistence using SQLite.
"""

import logging
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def get_checkpointer() -> MemorySaver:
    """
    Return an in-memory checkpointer for LangGraph.
    Sufficient for single-session persistence; swap for SQLite in production.
    """
    logger.info("Initialising in-memory checkpointer")
    return MemorySaver()


def make_thread_id(ticker: str, run_at: str) -> dict:
    thread_id = f"{ticker}_{run_at}"
    return {"configurable": {"thread_id": thread_id}}