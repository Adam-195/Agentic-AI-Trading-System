"""
graph/trading_graph.py
LangGraph state machine for the trading agent.

Graph flow:
  [fetch_data] → [research] → [reason] → <conditional> → [execute] or [hold]
                                                               ↓
                                                       [update_portfolio]
                                                               ↓
                                                            [END]
"""

import logging
from typing import TypedDict, Optional, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END

from agents.research_agent import research_ticker, format_research_for_prompt
from agents.reasoning_agent import make_decision
from agents.execution_agent import execute_trade
from models.portfolio import Portfolio
from models.trade import Trade
from tools.market_data import get_market_data
from graph.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Graph State
# Shared dict passed between every node — each node reads and updates it
# ------------------------------------------------------------------

class TradingState(TypedDict):
    # Input
    ticker: str

    # Populated by fetch_data node
    market_data: Optional[dict]

    # Populated by research node
    research: Optional[dict]

    # Populated by reason node
    trade: Optional[dict]           # Serialised Trade dict
    action: Optional[str]           # "buy" | "sell" | "hold"

    # Populated by execute/hold node
    executed: bool
    error: Optional[str]

    # Carried through — portfolio is shared across all tickers in a run
    portfolio_snapshot: Optional[dict]

    # Metadata
    run_at: str


# ------------------------------------------------------------------
# Node functions
# Each takes TradingState, returns a partial TradingState update
# ------------------------------------------------------------------

def fetch_data_node(state: TradingState) -> dict:
    """Fetch live market data for the ticker."""
    ticker = state["ticker"]
    logger.info(f"[{ticker}] Node: fetch_data")

    market_data = get_market_data(ticker)
    if market_data is None:
        return {
            "market_data": None,
            "error": f"Failed to fetch market data for {ticker}",
        }

    return {"market_data": market_data, "error": None}


def research_node(state: TradingState) -> dict:
    """Gather news and sentiment to build a full research summary."""
    ticker = state["ticker"]
    logger.info(f"[{ticker}] Node: research")

    if state.get("error"):
        logger.warning(f"[{ticker}] Skipping research — earlier error: {state['error']}")
        return {"research": None}

    research = research_ticker(ticker)
    if research is None:
        return {
            "research": None,
            "error": f"Research failed for {ticker}",
        }

    # Inject market data already fetched (avoids a second API call)
    research["market_data"] = state["market_data"]
    return {"research": research}


def reason_node(state: TradingState, portfolio: Portfolio) -> dict:
    """Call the LLM to produce a trade decision."""
    ticker = state["ticker"]
    logger.info(f"[{ticker}] Node: reason")

    if state.get("error") or state.get("research") is None:
        logger.warning(f"[{ticker}] Skipping reasoning — no research available.")
        return {"trade": None, "action": "hold"}

    current_price = state["market_data"]["price"]
    trade = make_decision(state["research"], portfolio, current_price)

    if trade is None:
        return {
            "trade": None,
            "action": "hold",
            "error": f"Reasoning failed for {ticker}",
        }

    return {
        "trade": trade.to_dict(),
        "action": trade.action,
    }


def execute_node(state: TradingState, portfolio: Portfolio) -> dict:
    """Execute the trade via Alpaca and update the portfolio."""
    ticker = state["ticker"]
    logger.info(f"[{ticker}] Node: execute — action={state.get('action')}")

    trade_dict = state.get("trade")
    if trade_dict is None:
        return {"executed": False, "error": "No trade to execute"}

    # Reconstruct Trade object from dict
    trade = Trade(
        ticker=trade_dict["ticker"],
        action=trade_dict["action"],
        quantity=trade_dict["quantity"],
        price=trade_dict["price"],
        rationale=trade_dict["rationale"],
        confidence=trade_dict["confidence"],
    )

    success = execute_trade(trade, portfolio)
    return {
        "executed": success,
        "portfolio_snapshot": {
            "cash_balance": portfolio.cash_balance,
            "positions": dict(portfolio.positions),
        },
    }


def hold_node(state: TradingState, portfolio: Portfolio) -> dict:
    """Log a hold decision without placing any order."""
    ticker = state["ticker"]
    logger.info(f"[{ticker}] Node: hold")

    trade_dict = state.get("trade")
    if trade_dict:
        trade = Trade(
            ticker=trade_dict["ticker"],
            action="hold",
            quantity=0,
            price=state["market_data"]["price"] if state.get("market_data") else 0,
            rationale=trade_dict.get("rationale", "No action taken."),
            confidence=trade_dict.get("confidence", 0.0),
        )
        portfolio.log_hold(trade)
        portfolio.save()

    return {
        "executed": True,
        "portfolio_snapshot": {
            "cash_balance": portfolio.cash_balance,
            "positions": dict(portfolio.positions),
        },
    }


# ------------------------------------------------------------------
# Conditional edge — routes after reason node
# ------------------------------------------------------------------

def should_trade(state: TradingState) -> str:
    """Route to execute if action is buy/sell, otherwise hold."""
    action = state.get("action", "hold")
    error = state.get("error")

    if error:
        logger.warning(f"Routing to hold due to error: {error}")
        return "hold"

    if action in ("buy", "sell"):
        return "execute"

    return "hold"


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_graph(portfolio: Portfolio, use_checkpointer: bool = True) -> any:
    """
    Compile and return the LangGraph trading workflow.
    Portfolio is injected via closures so nodes can share state.
    Checkpointer gives the agent persistent memory across restarts.
    """

    # Wrap nodes that need portfolio access in closures
    def _reason(state): return reason_node(state, portfolio)
    def _execute(state): return execute_node(state, portfolio)
    def _hold(state): return hold_node(state, portfolio)

    workflow = StateGraph(TradingState)

    # Register nodes
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("research", research_node)
    workflow.add_node("reason", _reason)
    workflow.add_node("execute", _execute)
    workflow.add_node("hold", _hold)

    # Define edges
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "research")
    workflow.add_edge("research", "reason")

    # Conditional routing after reasoning
    workflow.add_conditional_edges(
        "reason",
        should_trade,
        {
            "execute": "execute",
            "hold": "hold",
        },
    )

    # Both execute and hold lead to END
    workflow.add_edge("execute", END)
    workflow.add_edge("hold", END)

    if use_checkpointer:
        checkpointer = get_checkpointer()
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()
