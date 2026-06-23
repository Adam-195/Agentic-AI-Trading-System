"""
agents/execution_agent.py
Executes trade decisions via Alpaca's paper trading API.
Updates the Portfolio object after each trade.
"""

import logging
from typing import Optional

import requests

from config import config
from models.trade import Trade
from models.portfolio import Portfolio

logger = logging.getLogger(__name__)

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": config.ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}

# Crypto tickers need a different symbol format for Alpaca
ALPACA_CRYPTO_MAP = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "BNB-USD": "BNB/USD",
}


def to_alpaca_symbol(ticker: str) -> str:
    """Convert our ticker format to Alpaca's expected format."""
    return ALPACA_CRYPTO_MAP.get(ticker, ticker)


def place_order(trade: Trade) -> Optional[str]:
    """
    Submit a buy or sell order to Alpaca paper trading.
    Returns the Alpaca order ID on success, None on failure.
    """
    symbol = to_alpaca_symbol(trade.ticker)
    is_crypto = trade.ticker in ALPACA_CRYPTO_MAP

    payload = {
        "symbol": symbol,
        "qty": str(trade.quantity),
        "side": trade.action,   # "buy" or "sell"
        "type": "market",
        "time_in_force": "gtc" if is_crypto else "day",
    }

    try:
        response = requests.post(
            f"{config.ALPACA_BASE_URL}/v2/orders",
            headers=ALPACA_HEADERS,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        order = response.json()
        order_id = order.get("id")
        logger.info(f"Alpaca order placed: {order_id} — {trade.action} {trade.quantity} {symbol}")
        return order_id

    except requests.HTTPError as e:
        logger.error(f"Alpaca order failed [{response.status_code}]: {response.text}")
        return None
    except requests.RequestException as e:
        logger.error(f"Alpaca request error: {e}")
        return None


def get_account() -> Optional[dict]:
    """Fetch current Alpaca paper account info (cash, buying power etc.)."""
    try:
        response = requests.get(
            f"{config.ALPACA_BASE_URL}/v2/account",
            headers=ALPACA_HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Alpaca account: {e}")
        return None


def execute_trade(trade: Trade, portfolio: Portfolio) -> bool:
    """
    Main entry point: place the order, update the portfolio, return success bool.
    Handles buy, sell, and hold actions.
    """
    if trade.action == "hold":
        portfolio.log_hold(trade)
        portfolio.save()
        return True

    # Place the order with Alpaca
    order_id = place_order(trade)
    if order_id is None:
        logger.error(f"Order placement failed for {trade.ticker} — portfolio not updated.")
        return False

    trade.order_id = order_id

    # Update local portfolio state
    try:
        if trade.action == "buy":
            portfolio.buy(trade)
        elif trade.action == "sell":
            portfolio.sell(trade)
    except (Exception) as e:
        logger.error(f"Portfolio update failed after order {order_id}: {e}")
        return False

    portfolio.save()
    return True
