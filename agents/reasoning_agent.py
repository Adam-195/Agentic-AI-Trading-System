"""
agents/reasoning_agent.py
Uses Claude to reason about a research summary and produce a trade decision.
Returns a structured Trade object.
"""

import json
import logging
import re
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from config import config
from models.trade import Trade
from models.portfolio import Portfolio

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# LLM setup
# ------------------------------------------------------------------

def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=config.LLM_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=1024,
        temperature=0.2,  # Low temp for consistent, structured decisions
    )


# ------------------------------------------------------------------
# Reasoning prompt
# ------------------------------------------------------------------

REASONING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a disciplined quantitative trading analyst. Your job is to analyse 
market data and news sentiment for a single asset and make a clear trading decision.

You must respond with ONLY valid JSON — no preamble, no explanation outside the JSON.

Your response must follow this exact schema:
{{
  "action": "buy" | "sell" | "hold",
  "quantity": <number — how many units to trade, 0 if hold>,
  "confidence": <float 0.0 to 1.0>,
  "rationale": "<2-3 sentence explanation of your reasoning>",
  "risk_level": "low" | "medium" | "high"
}}

Guidelines:
- Only recommend buy if confidence > 0.6
- Only recommend sell if you hold a position and conditions have deteriorated
- Consider position size relative to available cash (never risk more than 20% on one trade)
- Quantity for crypto can be fractional (e.g. 0.05 BTC), stocks must be whole numbers
- If uncertain, default to hold""",
    ),
    (
        "human",
        """Here is the current research for this asset:

{research_summary}

CURRENT PORTFOLIO CONTEXT:
  Cash available:     ${cash_balance:,.2f}
  Current position:   {current_position} units held
  Avg buy price:      ${avg_buy_price:.4f}

Based on this data, what is your trading decision?""",
    ),
])


# ------------------------------------------------------------------
# Reasoning agent
# ------------------------------------------------------------------

def parse_decision(raw: str) -> Optional[dict]:
    """Extract and validate JSON from LLM response."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        decision = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}\nRaw: {raw}")
        return None

    required = {"action", "quantity", "confidence", "rationale", "risk_level"}
    if not required.issubset(decision.keys()):
        logger.error(f"LLM response missing required fields. Got: {decision.keys()}")
        return None

    if decision["action"] not in ("buy", "sell", "hold"):
        logger.error(f"Invalid action in LLM response: {decision['action']}")
        return None

    return decision


def make_decision(
    research: dict,
    portfolio: Portfolio,
    current_price: float,
) -> Optional[Trade]:
    """
    Call the LLM with research + portfolio context.
    Returns a Trade object or None on failure.
    """
    ticker = research["ticker"]
    llm = get_llm()
    chain = REASONING_PROMPT | llm

    current_position = portfolio.positions.get(ticker, 0.0)
    avg_buy_price = portfolio.avg_buy_price.get(ticker, 0.0)

    from agents.research_agent import format_research_for_prompt
    research_text = format_research_for_prompt(research)

    logger.info(f"Requesting trade decision for {ticker} from LLM...")

    try:
        response = chain.invoke({
            "research_summary": research_text,
            "cash_balance": portfolio.cash_balance,
            "current_position": current_position,
            "avg_buy_price": avg_buy_price,
        })
        raw = response.content
    except Exception as e:
        logger.error(f"LLM call failed for {ticker}: {e}")
        return None

    decision = parse_decision(raw)
    if decision is None:
        return None

    trade = Trade(
        ticker=ticker,
        action=decision["action"],
        quantity=float(decision["quantity"]),
        price=current_price,
        rationale=decision["rationale"],
        confidence=float(decision["confidence"]),
    )

    logger.info(
        f"Decision for {ticker}: {trade.action.upper()} "
        f"{trade.quantity} units @ ${trade.price:.4f} "
        f"(confidence={trade.confidence:.0%}, risk={decision['risk_level']})"
    )
    logger.info(f"Rationale: {trade.rationale}")

    return trade
