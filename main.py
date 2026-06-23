"""
main.py
Entry point for the trading agent.
Run once:     python main.py
Run on loop:  python main.py --loop
"""

import argparse
import logging
import time
from datetime import datetime

import schedule

from config import config
from models.portfolio import Portfolio
from agents.research_agent import research_ticker
from agents.reasoning_agent import make_decision
from agents.execution_agent import execute_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{config.LOG_DIR}/agent.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_agent() -> None:
    """Single run of the agent across the full watchlist."""
    logger.info(f"--- Agent run started at {datetime.utcnow().isoformat()} ---")

    # Portfolio is loaded/initialised here
    # Graph import deferred until Phase 5
    portfolio = Portfolio()
    portfolio.load()

    logger.info(f"Watchlist: {config.WATCHLIST}")

    results = []
    for ticker in config.WATCHLIST:
        logger.info(f"\n{'='*50}\nProcessing {ticker}\n{'='*50}")

        # 1. Research
        research = research_ticker(ticker)
        if research is None:
            logger.warning(f"Skipping {ticker} — research failed.")
            continue

        # 2. Reason
        current_price = research["market_data"]["price"]
        trade = make_decision(research, portfolio, current_price)
        if trade is None:
            logger.warning(f"Skipping {ticker} — reasoning failed.")
            continue

        # 3. Execute
        success = execute_trade(trade, portfolio)
        results.append({
            "ticker": ticker,
            "action": trade.action,
            "confidence": trade.confidence,
            "success": success,
        })

    # Summary
    logger.info(f"\n{'='*50}\nRUN SUMMARY\n{'='*50}")
    for r in results:
        status = "✓" if r["success"] else "✗"
        logger.info(
            f"{status} {r['ticker']}: {r['action'].upper()} "
            f"(confidence={r['confidence']:.0%})"
        )
    logger.info(portfolio.summary())
    logger.info("--- Agent run complete ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic AI Trading System")
    parser.add_argument(
        "--loop",
        action="store_true",
        help=f"Run on a schedule every {config.RUN_INTERVAL_MINUTES} minutes",
    )
    args = parser.parse_args()

    config.validate()

    if args.loop:
        logger.info(f"Starting scheduled loop every {config.RUN_INTERVAL_MINUTES} minutes.")
        run_agent()  # Run immediately on start
        schedule.every(config.RUN_INTERVAL_MINUTES).minutes.do(run_agent)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_agent()


if __name__ == "__main__":
    main()
