"""
tools/news_fetcher.py
Fetches recent financial news headlines for a given ticker using NewsAPI.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import config

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Clean ticker for search (BTC-USD → Bitcoin, etc.)
TICKER_SEARCH_TERMS: dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "Binance",
}


def get_search_term(ticker: str) -> str:
    return TICKER_SEARCH_TERMS.get(ticker, ticker.replace("-USD", ""))


def fetch_news(ticker: str, max_articles: int = 5) -> list[dict]:
    """
    Fetch recent news headlines for a ticker.
    Returns a list of dicts with title, description, url, publishedAt.
    """
    search_term = get_search_term(ticker)
    from_date = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")

    try:
        response = requests.get(
            NEWSAPI_URL,
            params={
                "q": search_term,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "from": from_date,
                "apiKey": config.NEWS_API_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])

        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            }
            for a in articles
            if a.get("title")  # filter out null titles
        ]

    except requests.RequestException as e:
        logger.error(f"NewsAPI request failed for {ticker}: {e}")
        return []


def headlines_as_text(articles: list[dict]) -> str:
    """Format articles as a plain-text summary for LLM prompts."""
    if not articles:
        return "No recent news found."

    lines = []
    for i, article in enumerate(articles, 1):
        lines.append(f"{i}. [{article['source']}] {article['title']}")
        if article.get("description"):
            lines.append(f"   {article['description'][:200]}")
    return "\n".join(lines)
