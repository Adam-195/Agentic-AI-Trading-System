"""
tools/market_data.py
Fetches live price and volume data.
- Stocks: yfinance
- Crypto: CoinGecko public API (no key required)
"""

import logging
from typing import Optional

import yfinance as yf
import requests

logger = logging.getLogger(__name__)

# Map common crypto tickers to CoinGecko IDs
COINGECKO_IDS: dict[str, str] = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "BNB-USD": "binancecoin",
}

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"


def is_crypto(ticker: str) -> bool:
    return ticker in COINGECKO_IDS


def get_crypto_price(ticker: str) -> Optional[dict]:
    coin_id = COINGECKO_IDS.get(ticker)
    if not coin_id:
        logger.warning(f"No CoinGecko mapping for {ticker}")
        return None

    try:
        response = requests.get(
            COINGECKO_URL,
            params={
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json().get(coin_id, {})

        return {
            "ticker": ticker,
            "price": data.get("usd", 0.0),
            "volume_24h": data.get("usd_24h_vol", 0.0),
            "change_24h_pct": data.get("usd_24h_change", 0.0),
            "source": "coingecko",
        }
    except requests.RequestException as e:
        logger.error(f"CoinGecko request failed for {ticker}: {e}")
        return None


def get_stock_price(ticker: str) -> Optional[dict]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        price = info.last_price
        prev_close = info.previous_close
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

        hist = stock.history(period="1d")
        volume = int(hist["Volume"].iloc[-1]) if not hist.empty else 0

        return {
            "ticker": ticker,
            "price": round(price, 4),
            "volume_24h": volume,
            "change_24h_pct": round(change_pct, 4),
            "source": "yfinance",
        }
    except Exception as e:
        logger.error(f"yfinance request failed for {ticker}: {e}")
        return None


def get_market_data(ticker: str) -> Optional[dict]:
    """
    Main entry point. Routes to the right data source based on ticker type.
    Returns a normalised dict or None on failure.
    """
    if is_crypto(ticker):
        return get_crypto_price(ticker)
    return get_stock_price(ticker)
