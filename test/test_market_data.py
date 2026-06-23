"""
tests/test_market_data.py
Unit tests for market data fetching tools.
Uses mocking so tests run offline without hitting real APIs.
Run with: pytest tests/test_market_data.py -v
"""

from unittest.mock import patch, MagicMock
import pytest

from tools.market_data import get_market_data, is_crypto


class TestIsCrypto:
    def test_btc_is_crypto(self):
        assert is_crypto("BTC-USD") is True

    def test_eth_is_crypto(self):
        assert is_crypto("ETH-USD") is True

    def test_aapl_is_not_crypto(self):
        assert is_crypto("AAPL") is False

    def test_msft_is_not_crypto(self):
        assert is_crypto("MSFT") is False


class TestGetMarketData:
    @patch("tools.market_data.get_crypto_price")
    def test_routes_crypto_to_coingecko(self, mock_crypto):
        mock_crypto.return_value = {"ticker": "BTC-USD", "price": 60000.0}
        result = get_market_data("BTC-USD")
        mock_crypto.assert_called_once_with("BTC-USD")

    @patch("tools.market_data.get_stock_price")
    def test_routes_stock_to_yfinance(self, mock_stock):
        mock_stock.return_value = {"ticker": "AAPL", "price": 180.0}
        result = get_market_data("AAPL")
        mock_stock.assert_called_once_with("AAPL")

    @patch("tools.market_data.get_stock_price")
    def test_returns_none_on_failure(self, mock_stock):
        mock_stock.return_value = None
        result = get_market_data("AAPL")
        assert result is None

    @patch("tools.market_data.get_crypto_price")
    def test_result_has_required_keys(self, mock_crypto):
        mock_crypto.return_value = {
            "ticker": "ETH-USD",
            "price": 3000.0,
            "volume_24h": 1_000_000.0,
            "change_24h_pct": 2.5,
            "source": "coingecko",
        }
        result = get_market_data("ETH-USD")
        for key in ["ticker", "price", "volume_24h", "change_24h_pct"]:
            assert key in result
