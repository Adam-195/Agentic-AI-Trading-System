"""
config.py
Centralised configuration — loads from .env and exposes typed settings.
All other modules import from here; never call os.getenv() directly elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- API Keys ---
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )

    # --- Agent Settings ---
    WATCHLIST: list[str] = os.getenv(
        "WATCHLIST", "AAPL,MSFT,BTC-USD,ETH-USD"
    ).split(",")
    RUN_INTERVAL_MINUTES: int = int(os.getenv("RUN_INTERVAL_MINUTES", "60"))
    INITIAL_CASH_BALANCE: float = float(os.getenv("INITIAL_CASH_BALANCE", "10000.00"))

    # --- Model ---
    LLM_MODEL: str = "claude-sonnet-4-6"
    SENTIMENT_MODEL: str = "ProsusAI/finbert"

    # --- Paths ---
    PORTFOLIO_STATE_FILE: str = "data/portfolio_state.json"
    LOG_DIR: str = "logs"

    @classmethod
    def validate(cls) -> None:
        """Raise early if critical keys are missing."""
        missing = []
        for key in ["ANTHROPIC_API_KEY", "NEWS_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
            if not getattr(cls, key):
                missing.append(key)
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your keys."
            )


config = Config()
