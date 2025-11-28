# core/risk_manager.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "safe": {
        "max_risk_per_trade_pct": 0.5,
        "max_daily_loss_pct": 2.0,
        "max_daily_trades": 3,
        "max_portfolio_risk_pct": 3.0,
        "max_leverage": 4.0,
    },
    "normal": {
        "max_risk_per_trade_pct": 1.5,
        "max_daily_loss_pct": 5.0,
        "max_daily_trades": 6,
        "max_portfolio_risk_pct": 8.0,
        "max_leverage": 8.0,
    },
    "aggressif": {
        "max_risk_per_trade_pct": 2.0,
        "max_daily_loss_pct": 10.0,
        "max_daily_trades": 10,
        "max_portfolio_risk_pct": 12.0,
        "max_leverage": 10.0,
    },
}


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_daily_trades: int
    max_portfolio_risk_pct: float
    max_leverage: float


def get_risk_limits(profile: str) -> RiskLimits:
    if profile not in RISK_PROFILES:
        raise ValueError(f"Profil de risque inconnu: {profile}")
    return RiskLimits(**RISK_PROFILES[profile])


class RiskManager:
    def __init__(self, starting_equity: float, limits: RiskLimits):
        self.starting_equity = float(starting_equity)
        self.equity = float(starting_equity)
        self.limits = limits
        self.trades_taken = 0
        self.realized_pnl = 0.0

    @property
    def daily_pnl_pct(self) -> float:
        return (self.realized_pnl / self.starting_equity) * 100.0 if self.starting_equity > 0 else 0.0

    @property
    def reached_daily_loss_limit(self) -> bool:
        return self.daily_pnl_pct <= -self.limits.max_daily_loss_pct

    @property
    def reached_trade_limit(self) -> bool:
        return self.trades_taken >= self.limits.max_daily_trades

    def can_open_new_trade(self) -> bool:
        return not (self.reached_trade_limit or self.reached_daily_loss_limit)

    def compute_position_size(self, stop_distance_points: float, tick_value: float) -> float:
        if stop_distance_points <= 0 or tick_value <= 0:
            return 0.0

        risk_eur = (self.limits.max_risk_per_trade_pct / 100.0) * self.equity
        unit_risk = stop_distance_points * tick_value
        if unit_risk <= 0:
            return 0.0

        position_size = risk_eur / unit_risk
        return round(max(position_size, 0.0), 2)

    def register_closed_trade(self, pnl_eur: float) -> None:
        self.trades_taken += 1
        self.realized_pnl += pnl_eur
        self.equity = self.starting_equity + self.realized_pnl

    def summary_dict(self) -> Dict[str, float]:
        return {
            "starting_equity": self.starting_equity,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "trades_taken": self.trades_taken,
            "max_daily_loss_pct": self.limits.max_daily_loss_pct,
            "max_trades": self.limits.max_daily_trades,
        }
