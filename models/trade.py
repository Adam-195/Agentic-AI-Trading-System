"""
models/trade.py
Dataclass representing a single trade decision and its execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


Action = Literal["buy", "sell", "hold"]


@dataclass
class Trade:
    ticker: str
    action: Action
    quantity: float
    price: float
    rationale: str
    confidence: float  # 0.0 - 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    executed: bool = False
    order_id: str | None = None

    @property
    def total_value(self) -> float:
        return self.quantity * self.price

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "total_value": self.total_value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "executed": self.executed,
            "order_id": self.order_id,
        }

    def __str__(self) -> str:
        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] "
            f"{self.action.upper()} {self.quantity} {self.ticker} "
            f"@ ${self.price:.2f} (confidence: {self.confidence:.0%})"
        )
