"""
tests/test_portfolio.py
Unit tests for the Portfolio class.
Run with: pytest tests/test_portfolio.py -v
"""

import pytest
from datetime import datetime

from models.portfolio import Portfolio, InsufficientFundsError, InsufficientPositionError
from models.trade import Trade


def make_trade(ticker="AAPL", action="buy", quantity=10.0, price=150.0, confidence=0.8):
    return Trade(
        ticker=ticker,
        action=action,
        quantity=quantity,
        price=price,
        rationale="Test trade",
        confidence=confidence,
    )


class TestPortfolioBuy:
    def test_buy_reduces_cash(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        trade = make_trade(quantity=10, price=100.0)
        portfolio.buy(trade)
        assert portfolio.cash_balance == pytest.approx(9000.0)

    def test_buy_updates_position(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        trade = make_trade(quantity=5, price=200.0)
        portfolio.buy(trade)
        assert portfolio.positions["AAPL"] == 5.0

    def test_buy_insufficient_funds_raises(self):
        portfolio = Portfolio(initial_cash=100.0, state_file="/tmp/test_portfolio.json")
        trade = make_trade(quantity=10, price=200.0)
        with pytest.raises(InsufficientFundsError):
            portfolio.buy(trade)

    def test_buy_multiple_updates_avg_price(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        portfolio.buy(make_trade(quantity=10, price=200.0))
        # Average should be (10*100 + 10*200) / 20 = 150
        assert portfolio.avg_buy_price["AAPL"] == pytest.approx(150.0)


class TestPortfolioSell:
    def test_sell_increases_cash(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        portfolio.sell(make_trade(action="sell", quantity=5, price=120.0))
        assert portfolio.cash_balance == pytest.approx(9000.0 + 600.0)

    def test_sell_reduces_position(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        portfolio.sell(make_trade(action="sell", quantity=4, price=100.0))
        assert portfolio.positions["AAPL"] == 6.0

    def test_sell_clears_position_on_full_exit(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        portfolio.sell(make_trade(action="sell", quantity=10, price=100.0))
        assert "AAPL" not in portfolio.positions

    def test_sell_more_than_held_raises(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=5, price=100.0))
        with pytest.raises(InsufficientPositionError):
            portfolio.sell(make_trade(action="sell", quantity=10, price=100.0))


class TestPortfolioValuation:
    def test_get_total_value(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        # Cash = 9000, position value at current price 150 = 1500
        total = portfolio.get_total_value({"AAPL": 150.0})
        assert total == pytest.approx(10500.0)

    def test_get_pnl_positive(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        pnl = portfolio.get_pnl({"AAPL": 150.0})
        assert pnl["AAPL"] == pytest.approx(500.0)

    def test_get_pnl_negative(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade(quantity=10, price=100.0))
        pnl = portfolio.get_pnl({"AAPL": 80.0})
        assert pnl["AAPL"] == pytest.approx(-200.0)


class TestTradeHistory:
    def test_trade_logged_after_buy(self):
        portfolio = Portfolio(initial_cash=10000.0, state_file="/tmp/test_portfolio.json")
        portfolio.buy(make_trade())
        assert len(portfolio.trade_history) == 1
        assert portfolio.trade_history[0].executed is True
