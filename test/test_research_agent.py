"""
tests/test_research_agent.py
Tests for the research agent — fully mocked, no real API calls.
Run with: pytest tests/test_research_agent.py -v
"""

from unittest.mock import patch, MagicMock
import pytest

from agents.research_agent import research_ticker, format_research_for_prompt


MOCK_MARKET_DATA = {
    "ticker": "AAPL",
    "price": 180.0,
    "volume_24h": 55_000_000,
    "change_24h_pct": 1.5,
    "source": "yfinance",
}

MOCK_ARTICLES = [
    {"title": "Apple reports record earnings", "description": "Strong Q4.", "source": "Reuters", "published_at": "2024-01-01", "url": ""},
    {"title": "iPhone sales beat expectations", "description": "Analysts surprised.", "source": "Bloomberg", "published_at": "2024-01-01", "url": ""},
]

MOCK_SENTIMENT = {
    "label": "positive",
    "positive": 0.82,
    "negative": 0.05,
    "neutral": 0.13,
    "confidence": 0.82,
    "article_count": 2,
}


class TestResearchTicker:
    @patch("agents.research_agent.score_headlines", return_value=MOCK_SENTIMENT)
    @patch("agents.research_agent.fetch_news", return_value=MOCK_ARTICLES)
    @patch("agents.research_agent.get_market_data", return_value=MOCK_MARKET_DATA)
    def test_returns_research_dict(self, mock_md, mock_news, mock_sentiment):
        result = research_ticker("AAPL")
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert "market_data" in result
        assert "sentiment" in result
        assert "news_summary" in result

    @patch("agents.research_agent.get_market_data", return_value=None)
    def test_returns_none_on_market_data_failure(self, mock_md):
        result = research_ticker("AAPL")
        assert result is None

    @patch("agents.research_agent.score_headlines", return_value=MOCK_SENTIMENT)
    @patch("agents.research_agent.fetch_news", return_value=MOCK_ARTICLES)
    @patch("agents.research_agent.get_market_data", return_value=MOCK_MARKET_DATA)
    def test_sentiment_included(self, mock_md, mock_news, mock_sentiment):
        result = research_ticker("AAPL")
        assert result["sentiment"]["label"] == "positive"

    @patch("agents.research_agent.score_headlines", return_value=MOCK_SENTIMENT)
    @patch("agents.research_agent.fetch_news", return_value=MOCK_ARTICLES)
    @patch("agents.research_agent.get_market_data", return_value=MOCK_MARKET_DATA)
    def test_format_for_prompt_contains_key_info(self, mock_md, mock_news, mock_sentiment):
        result = research_ticker("AAPL")
        formatted = format_research_for_prompt(result)
        assert "AAPL" in formatted
        assert "180" in formatted
        assert "SENTIMENT" in formatted
        assert "NEWS" in formatted


class TestFormatResearchForPrompt:
    def test_format_structure(self):
        research = {
            "ticker": "BTC-USD",
            "market_data": {
                "price": 62000.0,
                "change_24h_pct": -2.1,
                "volume_24h": 30_000_000_000,
            },
            "sentiment_summary": "Sentiment: NEGATIVE",
            "news_summary": "1. [Reuters] Bitcoin falls amid regulation fears",
        }
        formatted = format_research_for_prompt(research)
        assert "BTC-USD" in formatted
        assert "62,000" in formatted
        assert "NEGATIVE" in formatted
