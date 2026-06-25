"""
tests/test_trading_graph.py
Tests for the LangGraph trading state machine.
All external calls are mocked — no API keys needed.
Run with: pytest test/test_trading_graph.py -v
"""

import json
from unittest.mock import patch, MagicMock
import pytest

from graph.trading_graph import (
    TradingState,
    fetch_data_node,
    research_node,
    should_trade,
    build_graph,
)
from models.portfolio import Portfolio


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_state(**overrides) -> TradingState:
    base = TradingState(
        ticker="AAPL",
        market_data=None,
        research=None,
        trade=None,
        action=None,
        executed=False,
        error=None,
        portfolio_snapshot=None,
        run_at="2024-01-01T00:00:00",
    )
    base.update(overrides)
    return base


MOCK_MARKET_DATA = {
    "ticker": "AAPL",
    "price": 180.0,
    "volume_24h": 55_000_000,
    "change_24h_pct": 1.5,
    "source": "yfinance",
}

MOCK_RESEARCH = {
    "ticker": "AAPL",
    "market_data": MOCK_MARKET_DATA,
    "news_summary": "Apple hits record high.",
    "sentiment": {"label": "positive", "positive": 0.8, "negative": 0.1, "neutral": 0.1, "confidence": 0.8, "article_count": 3},
    "sentiment_summary": "Sentiment: POSITIVE",
}

MOCK_TRADE_DICT = {
    "ticker": "AAPL",
    "action": "buy",
    "quantity": 5.0,
    "price": 180.0,
    "rationale": "Strong positive sentiment and upward momentum.",
    "confidence": 0.78,
    "timestamp": "2024-01-01T00:00:00",
    "executed": False,
    "order_id": None,
    "total_value": 900.0,
}


# ------------------------------------------------------------------
# fetch_data_node tests
# ------------------------------------------------------------------

class TestFetchDataNode:
    @patch("graph.trading_graph.get_market_data", return_value=MOCK_MARKET_DATA)
    def test_returns_market_data_on_success(self, mock_md):
        state = make_state(ticker="AAPL")
        result = fetch_data_node(state)
        assert result["market_data"]["price"] == 180.0
        assert result["error"] is None

    @patch("graph.trading_graph.get_market_data", return_value=None)
    def test_returns_error_on_failure(self, mock_md):
        state = make_state(ticker="AAPL")
        result = fetch_data_node(state)
        assert result["market_data"] is None
        assert "error" in result
        assert result["error"] is not None


# ------------------------------------------------------------------
# research_node tests
# ------------------------------------------------------------------

class TestResearchNode:
    @patch("graph.trading_graph.research_ticker", return_value=MOCK_RESEARCH)
    def test_returns_research_on_success(self, mock_research):
        state = make_state(ticker="AAPL", market_data=MOCK_MARKET_DATA)
        result = research_node(state)
        assert result["research"] is not None
        assert result["research"]["ticker"] == "AAPL"

    def test_skips_if_prior_error(self):
        state = make_state(ticker="AAPL", error="Earlier failure")
        result = research_node(state)
        assert result["research"] is None

    @patch("graph.trading_graph.research_ticker", return_value=None)
    def test_sets_error_on_research_failure(self, mock_research):
        state = make_state(ticker="AAPL", market_data=MOCK_MARKET_DATA)
        result = research_node(state)
        assert result["research"] is None
        assert result["error"] is not None


# ------------------------------------------------------------------
# should_trade (conditional edge) tests
# ------------------------------------------------------------------

class TestShouldTrade:
    def test_buy_routes_to_execute(self):
        state = make_state(action="buy")
        assert should_trade(state) == "execute"

    def test_sell_routes_to_execute(self):
        state = make_state(action="sell")
        assert should_trade(state) == "execute"

    def test_hold_routes_to_hold(self):
        state = make_state(action="hold")
        assert should_trade(state) == "hold"

    def test_error_routes_to_hold(self):
        state = make_state(action="buy", error="Something went wrong")
        assert should_trade(state) == "hold"

    def test_none_action_routes_to_hold(self):
        state = make_state(action=None)
        assert should_trade(state) == "hold"


# ------------------------------------------------------------------
# build_graph tests
# ------------------------------------------------------------------

class TestBuildGraph:
    def test_graph_compiles(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_graph.json")
        graph = build_graph(portfolio)
        assert graph is not None

    @patch("graph.trading_graph.execute_trade", return_value=True)
    @patch("graph.trading_graph.make_decision")
    @patch("graph.trading_graph.research_ticker", return_value=MOCK_RESEARCH)
    @patch("graph.trading_graph.get_market_data", return_value=MOCK_MARKET_DATA)
    def test_full_buy_run(self, mock_md, mock_research, mock_decision, mock_execute):
        from models.trade import Trade
        mock_trade = Trade(
            ticker="AAPL",
            action="buy",
            quantity=5,
            price=180.0,
            rationale="Strong momentum",
            confidence=0.8,
        )
        mock_decision.return_value = mock_trade

        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_graph_run.json")
        graph = build_graph(portfolio)

        result = graph.invoke({
            "ticker": "AAPL",
            "market_data": None,
            "research": None,
            "trade": None,
            "action": None,
            "executed": False,
            "error": None,
            "portfolio_snapshot": None,
            "run_at": "2024-01-01T00:00:00",
        })

        assert result["action"] == "buy"
        assert result["executed"] is True

    @patch("graph.trading_graph.make_decision")
    @patch("graph.trading_graph.research_ticker", return_value=MOCK_RESEARCH)
    @patch("graph.trading_graph.get_market_data", return_value=MOCK_MARKET_DATA)
    def test_full_hold_run(self, mock_md, mock_research, mock_decision):
        from models.trade import Trade
        mock_trade = Trade(
            ticker="AAPL",
            action="hold",
            quantity=0,
            price=180.0,
            rationale="Uncertain conditions",
            confidence=0.45,
        )
        mock_decision.return_value = mock_trade

        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_graph_hold.json")
        graph = build_graph(portfolio)

        result = graph.invoke({
            "ticker": "AAPL",
            "market_data": None,
            "research": None,
            "trade": None,
            "action": None,
            "executed": False,
            "error": None,
            "portfolio_snapshot": None,
            "run_at": "2024-01-01T00:00:00",
        })

        assert result["action"] == "hold"
