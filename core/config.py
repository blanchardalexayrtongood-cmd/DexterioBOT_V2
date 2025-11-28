from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Dict

StrategyMode = Literal["tyler", "fxalex", "fusion"]
RiskProfile   = Literal["safe", "normal", "aggressive"]
BotMode       = Literal["backtest", "monitoring", "live"]

DEFAULT_RISK = {
    "safe": {
        "max_risk_per_trade_pct": 1.0,
        "max_daily_loss_pct": 5.0,
        "max_daily_trades": 4,
        "max_portfolio_risk_pct": 5.0,
        "max_leverage": 6.0,
    },
    "normal": {
        "max_risk_per_trade_pct": 2.0,
        "max_daily_loss_pct": 10.0,
        "max_daily_trades": 6,
        "max_portfolio_risk_pct": 10.0,
        "max_leverage": 8.0,
    },
    "aggressive": {
        "max_risk_per_trade_pct": 3.0,
        "max_daily_loss_pct": 15.0,
        "max_daily_trades": 8,
        "max_portfolio_risk_pct": 15.0,
        "max_leverage": 12.0,
    }
}

@dataclass
class BotConfig:
    mode: BotMode
    strategy_mode: StrategyMode
    risk_profile: RiskProfile
    symbols: List[str]
    timeframe: str
    lookback_days: int
    initial_capital: float
    session: dict
    discord: dict
    risk_params: Dict[str, float]

def load_config(path: str) -> BotConfig:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        raw = json.load(f)

    mode = raw.get("mode", "backtest")
    strategy_mode = raw.get("strategy_mode", "fusion")
    risk_obj = raw.get("risk", {})
    profile = risk_obj.get("profile", "normal")
    default_risk = DEFAULT_RISK.get(profile, DEFAULT_RISK["normal"])

    # fusion params fixes + overrides utilisateur
    risk_params = {**default_risk, **risk_obj}

    return BotConfig(
        mode=mode,
        strategy_mode=strategy_mode,
        risk_profile=profile,
        symbols=raw.get("symbols", ["ES=F", "NQ=F"]),
        timeframe=raw.get("timeframe", "1m"),
        lookback_days=int(raw.get("lookback_days", 6)),
        initial_capital=float(raw.get("initial_capital", 1000.0)),
        session=raw.get("session", {}),
        discord=raw.get("discord", {}),
        risk_params=risk_params
    )
