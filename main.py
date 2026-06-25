"""
main.py
Entry point for the trading agent.

Usage:
  python main.py              # Single run
  python main.py --loop       # Run on schedule (interval set in .env)
  python main.py --dry-run    # Research and reason, but don't place orders
"""

import argparse
import logging
import time
from datetime import datetime, timezone

import schedule

from config import config
from models.portfolio import Portfolio
from graph.trading_graph import build_graph
from graph.checkpointer import make_thread_id
from utils.logger import setup_logging

import os
os.makedirs("data", exist_ok=True)

setup_logging()
logger = logging.getLogger(__name__)


def print_banner() -> None:
    logger.info("=" * 60)
    logger.info("   AGENTIC AI TRADING SYSTEM")
    logger.info(f"   Watchlist : {', '.join(config.WATCHLIST)}")
    logger.info(f"   Interval  : every {config.RUN_INTERVAL_MINUTES} minutes")
    logger.info(f"   Model     : {config.LLM_MODEL}")
    logger.info("=" * 60)


def run_agent(dry_run: bool = False) -> None:
    """Single run of the agent across the full watchlist via LangGraph."""
    run_at = datetime.now(timezone.utc).isoformat()
    logger.info(f"\n--- Agent run started at {run_at} ---")

    if dry_run:
        logger.info("DRY RUN MODE — no orders will be placed")

    # Load portfolio (creates fresh one if none saved)
    portfolio = Portfolio()
    portfolio.load()

    # Build the LangGraph workflow with persistent checkpointing
    graph = build_graph(portfolio, use_checkpointer=not dry_run)

    results = []
    for ticker in config.WATCHLIST:
        logger.info(f"\n{'='*50}\nProcessing {ticker}\n{'='*50}")

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

        # Each ticker gets its own thread ID for checkpointing
        config_dict = make_thread_id(ticker, run_at) if not dry_run else {}

        try:
            final_state = graph.invoke(initial_state, config=config_dict)
            results.append({
                "ticker": ticker,
                "action": final_state.get("action", "unknown"),
                "executed": final_state.get("executed", False),
                "error": final_state.get("error"),
            })

        except Exception as e:
            logger.error(f"Graph run failed for {ticker}: {e}", exc_info=True)
            results.append({
                "ticker": ticker,
                "action": "error",
                "executed": False,
                "error": str(e),
            })

    _print_summary(results, portfolio)


def _print_summary(results: list[dict], portfolio: Portfolio) -> None:
    logger.info(f"\n{'='*60}")
    logger.info("RUN SUMMARY")
    logger.info(f"{'='*60}")

    for r in results:
        if r["error"]:
            logger.info(f"  ✗ {r['ticker']:<10} ERROR     — {r['error']}")
        else:
            status = "✓" if r["executed"] else "~"
            logger.info(f"  {status} {r['ticker']:<10} {r['action'].upper():<8}")

    logger.info(portfolio.summary())
    logger.info("--- Agent run complete ---\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=f"Run on a schedule every {config.RUN_INTERVAL_MINUTES} minutes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Research and reason but do not place any orders",
    )
    args = parser.parse_args()

    config.validate()
    print_banner()

    dry_run = args.dry_run

    if args.loop:
        logger.info(
            f"Starting scheduled loop — running every {config.RUN_INTERVAL_MINUTES} minutes. "
            "Press Ctrl+C to stop."
        )
        run_agent(dry_run=dry_run)  # Run immediately on start
        schedule.every(config.RUN_INTERVAL_MINUTES).minutes.do(
            run_agent, dry_run=dry_run
        )
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
    else:
        run_agent(dry_run=dry_run)


if __name__ == "__main__":
    main()
