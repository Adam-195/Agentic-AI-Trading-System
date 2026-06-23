# Agentic AI Trading System

An autonomous AI agent that researches financial assets, reasons about market conditions using an LLM, and executes paper trades.
Built with LangChain, LangGraph, FinBERT, and the Alpaca API.

## Architecture

```
[Watchlist] → [Fetch Market Data] → [Fetch News]
                                         ↓
                               [FinBERT Sentiment]
                                         ↓
                               [LLM Reasoning Node]
                                         ↓
                              ┌──────────┴──────────┐
                          [Buy/Sell]             [Hold]
                              ↓                     ↓
                       [Execute Trade]        [Log Decision]
                              ↓                     ↓
                              └──────────┬──────────┘
                                  [Update Portfolio]
                                         ↓
                                   [Wait / Loop]
```

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | Claude (Anthropic) via LangChain |
| Sentiment analysis | FinBERT (ProsusAI/finbert) |
| Stock data | yfinance |
| Crypto data | CoinGecko API |
| News | NewsAPI |
| Paper trading | Alpaca Markets |

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo>
cd trading-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 5. Run once
python main.py

# 6. Run on a loop (every N minutes, set in .env)
python main.py --loop
```

## API Keys Required

| Service | Purpose | Free tier |
|---|---|---|
| [Anthropic](https://console.anthropic.com) | LLM reasoning | Yes |
| [NewsAPI](https://newsapi.org) | Financial headlines | Yes (100 req/day) |
| [Alpaca](https://alpaca.markets) | Paper trading | Yes (unlimited) |
| CoinGecko | Crypto prices | Yes (no key needed) |

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

## Project Structure

```
trading-agent/
├── agents/
│   ├── research_agent.py      # News + sentiment gathering
│   ├── reasoning_agent.py     # LLM decision making
│   └── execution_agent.py     # Trade execution
├── graph/
│   └── trading_graph.py       # LangGraph state machine
├── models/
│   ├── portfolio.py            # Portfolio state management
│   └── trade.py                # Trade dataclass
├── tools/
│   ├── market_data.py          # Price fetching
│   ├── news_fetcher.py         # NewsAPI integration
│   └── sentiment.py            # FinBERT sentiment scoring
├── tests/
│   ├── test_portfolio.py
│   └── test_market_data.py
├── main.py                     # Entry point
├── config.py                   # Centralised settings
├── requirements.txt
└── .env.example
```

## Build Phases

- [x] Phase 1 — Project structure, config, models, tools
- [ ] Phase 2 — LangChain tool wrappers
- [ ] Phase 3 — LangGraph agent graph
- [ ] Phase 4 — Alpaca paper trading execution
- [ ] Phase 5 — Scheduling and state persistence
- [ ] Phase 6 — Polish, error handling, logging
