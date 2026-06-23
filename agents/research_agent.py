"""
agents/research_agent.py
Gathers market data, news, and sentiment for a single ticker.
Returns a structured research summary ready for the reasoning agent.
"""

import logging
from typing import Optional

from langchain.tools import tool

from tools.market_data import get_market_data
from tools.news_fetcher import fetch_news, headlines_as_text
from tools.sentiment import score_headlines, sentiment_summary

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# LangChain tool definitions
# These are wrapped so the LLM can call them directly in Phase 3
# ------------------------------------------------------------------

@tool
def tool_get_market_data(ticker: str) -> dict:
    """
    Fetch the current price, 24h volume, and 24h price change for a ticker.
    Works for both stocks (e.g. AAPL, MSFT) and crypto (e.g. BTC-USD, ETH-USD).
    """
    result = get_market_data(ticker)
    if result is None:
        return {"error": f"Could not fetch market data for {ticker}"}
    return result


@tool
def tool_fetch_news(ticker: str) -> str:
    """
    Fetch the 5 most recent financial news headlines for a ticker.
    Returns a formatted plain-text summary of headlines and descriptions.
    """
    articles = fetch_news(ticker, max_articles=5)
    return headlines_as_text(articles)


@tool
def tool_get_sentiment(ticker: str) -> dict:
    """
    Run FinBERT sentiment analysis on recent news for a ticker.
    Returns positive/negative/neutral scores and an overall label.
    """
    articles = fetch_news(ticker, max_articles=5)
    headlines = [a["title"] for a in articles if a.get("title")]
    return score_headlines(headlines)


# ------------------------------------------------------------------
# Research agent — calls tools in sequence and returns a summary
# ------------------------------------------------------------------

def research_ticker(ticker: str) -> Optional[dict]:
    """
    Run full research pipeline for one ticker.
    Returns a structured dict ready to pass into the reasoning agent.
    """
    logger.info(f"Researching {ticker}...")

    # 1. Market data
    market_data = get_market_data(ticker)
    if market_data is None:
        logger.error(f"Failed to fetch market data for {ticker} — skipping.")
        return None

    # 2. News
    articles = fetch_news(ticker, max_articles=5)
    news_text = headlines_as_text(articles)

    # 3. Sentiment
    headlines = [a["title"] for a in articles if a.get("title")]
    sentiment = score_headlines(headlines)
    sentiment_text = sentiment_summary(sentiment)

    research = {
        "ticker": ticker,
        "market_data": market_data,
        "news_summary": news_text,
        "sentiment": sentiment,
        "sentiment_summary": sentiment_text,
    }

    logger.info(
        f"{ticker} research complete — "
        f"price=${market_data['price']:.2f}, "
        f"sentiment={sentiment['label']}"
    )
    return research


def format_research_for_prompt(research: dict) -> str:
    """
    Format a research dict as a clean text block for injection into an LLM prompt.
    """
    md = research["market_data"]
    return f"""
ASSET: {research["ticker"]}

MARKET DATA:
  Price:          ${md["price"]:,.4f}
  24h Change:     {md["change_24h_pct"]:+.2f}%
  24h Volume:     {md["volume_24h"]:,.0f}

SENTIMENT ANALYSIS:
  {research["sentiment_summary"]}

RECENT NEWS:
{research["news_summary"]}
""".strip()
