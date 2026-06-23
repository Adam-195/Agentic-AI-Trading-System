"""
models/portfolio.py
OOP Portfolio class — tracks cash, positions, trade history, and P&L.
Persists state to JSON so the agent remembers across restarts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.trade import Trade
from config import config

logger = logging.getLogger(__name__)


class InsufficientFundsError(Exception):
    pass


class InsufficientPositionError(Exception):
    pass


class Portfolio:
    def __init__(
        self,
        initial_cash: float = config.INITIAL_CASH_BALANCE,
        state_file: str = config.PORTFOLIO_STATE_FILE,
    ) -> None:
        self.cash_balance: float = initial_cash
        self.positions: dict[str, float] = {}       # ticker → quantity held
        self.avg_buy_price: dict[str, float] = {}   # ticker → average cost basis
        self.trade_history: list[Trade] = []
        self._state_file = Path(state_file)

    # ------------------------------------------------------------------
    # Core actions
    # ------------------------------------------------------------------

    def buy(self, trade: Trade) -> None:
        """Execute a buy, deducting cash and updating position."""
        cost = trade.total_value
        if cost > self.cash_balance:
            raise InsufficientFundsError(
                f"Cannot buy {trade.quantity} {trade.ticker} @ ${trade.price:.2f} "
                f"(cost ${cost:.2f}) — only ${self.cash_balance:.2f} available."
            )

        prev_qty = self.positions.get(trade.ticker, 0.0)
        prev_avg = self.avg_buy_price.get(trade.ticker, 0.0)

        new_qty = prev_qty + trade.quantity
        # Weighted average cost basis
        self.avg_buy_price[trade.ticker] = (
            (prev_qty * prev_avg + trade.quantity * trade.price) / new_qty
        )
        self.positions[trade.ticker] = new_qty
        self.cash_balance -= cost

        trade.executed = True
        self.trade_history.append(trade)
        logger.info(f"BUY executed: {trade}")

    def sell(self, trade: Trade) -> None:
        """Execute a sell, adding cash and reducing position."""
        held = self.positions.get(trade.ticker, 0.0)
        if trade.quantity > held:
            raise InsufficientPositionError(
                f"Cannot sell {trade.quantity} {trade.ticker} — only {held} held."
            )

        self.positions[trade.ticker] = held - trade.quantity
        if self.positions[trade.ticker] == 0:
            del self.positions[trade.ticker]
            del self.avg_buy_price[trade.ticker]

        self.cash_balance += trade.total_value
        trade.executed = True
        self.trade_history.append(trade)
        logger.info(f"SELL executed: {trade}")

    def log_hold(self, trade: Trade) -> None:
        """Record a hold decision without changing positions."""
        trade.executed = True
        self.trade_history.append(trade)
        logger.info(f"HOLD logged: {trade}")

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def get_position_value(self, current_prices: dict[str, float]) -> float:
        """Total market value of all open positions."""
        return sum(
            qty * current_prices.get(ticker, 0.0)
            for ticker, qty in self.positions.items()
        )

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        return self.cash_balance + self.get_position_value(current_prices)

    def get_pnl(self, current_prices: dict[str, float]) -> dict[str, float]:
        """Unrealised P&L per position."""
        pnl = {}
        for ticker, qty in self.positions.items():
            current = current_prices.get(ticker, 0.0)
            cost = self.avg_buy_price.get(ticker, 0.0)
            pnl[ticker] = (current - cost) * qty
        return pnl

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist portfolio state to JSON."""
        state = {
            "cash_balance": self.cash_balance,
            "positions": self.positions,
            "avg_buy_price": self.avg_buy_price,
            "trade_history": [t.to_dict() for t in self.trade_history],
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Portfolio state saved to {self._state_file}")

    def load(self) -> bool:
        """Load portfolio state from JSON. Returns True if loaded."""
        if not self._state_file.exists():
            logger.info("No saved portfolio found — starting fresh.")
            return False

        with open(self._state_file) as f:
            state = json.load(f)

        self.cash_balance = state["cash_balance"]
        self.positions = state["positions"]
        self.avg_buy_price = state["avg_buy_price"]
        # Trade history is logged but not reconstructed as Trade objects
        logger.info(
            f"Portfolio loaded: ${self.cash_balance:.2f} cash, "
            f"{len(self.positions)} open positions."
        )
        return True

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self, current_prices: Optional[dict[str, float]] = None) -> str:
        prices = current_prices or {}
        lines = [
            "=" * 50,
            "PORTFOLIO SUMMARY",
            "=" * 50,
            f"Cash:        ${self.cash_balance:,.2f}",
        ]
        if self.positions:
            lines.append("\nOpen Positions:")
            for ticker, qty in self.positions.items():
                price = prices.get(ticker, 0.0)
                value = qty * price
                pnl = (price - self.avg_buy_price.get(ticker, 0.0)) * qty
                lines.append(
                    f"  {ticker:<10} {qty:.4f} units  "
                    f"@ ${price:.2f}  value=${value:,.2f}  P&L=${pnl:+,.2f}"
                )
        else:
            lines.append("No open positions.")

        if prices:
            total = self.get_total_value(prices)
            lines.append(f"\nTotal Value:  ${total:,.2f}")

        lines.append("=" * 50)
        return "\n".join(lines)
