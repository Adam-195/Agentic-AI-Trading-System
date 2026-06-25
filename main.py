"""
main.py
Entry point for the trading agent.
Run once:     python main.py
Run on loop:  python main.py --loop
"""

import argparse
import logging
import os
import time
from datetime import datetime

import schedule

from config import config
from models.portfolio import Portfolio
from graph.trading_graph import build_graph

os.makedirs(config.LOG_DIR, exist_ok=True)

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
    """Single run of the agent across the full watchlist via LangGraph."""
    run_at = datetime.utcnow().isoformat()
    logger.info(f"--- Agent run started at {run_at} ---")

    portfolio = Portfolio()
    portfolio.load()

    # Build the LangGraph workflow (portfolio injected via closure)
    graph = build_graph(portfolio)

    logger.info(f"Watchlist: {config.WATCHLIST}")

    results = []
    for ticker in config.WATCHLIST:
        logger.info(f"\n{'='*50}\nProcessing {ticker}\n{'='*50}")

        # Initial state for this ticker
        initial_state = {
            "ticker": ticker,
            "market_data": None,
            "research": None,
            "trade": None,
            "action": None,
            "executed": False,
            "error": None,
            "portfolio_snapshot": None,
            "run_at": run_at,
        }

        try:
            final_state = graph.invoke(initial_state)
            results.append({
                "ticker": ticker,
                "action": final_state.get("action", "unknown"),
                "executed": final_state.get("executed", False),
                "error": final_state.get("error"),
            })
        except Exception as e:
            logger.error(f"Graph run failed for {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "action": "error",
                "executed": False,
                "error": str(e),
            })

    # Summary
    logger.info(f"\n{'='*50}\nRUN SUMMARY\n{'='*50}")
    for r in results:
        if r["error"]:
            logger.info(f"✗ {r['ticker']}: ERROR — {r['error']}")
        else:
            status = "✓" if r["executed"] else "✗"
            logger.info(f"{status} {r['ticker']}: {r['action'].upper()}")

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
        run_agent()
        schedule.every(config.RUN_INTERVAL_MINUTES).minutes.do(run_agent)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_agent()


if __name__ == "__main__":
    main()
