# core/risk.py

import json
from typing import Tuple, Dict


# Profils de risque prédéfinis
RISK_PROFILES = {
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
    },
}


def compute_stop_distance(price: float, stop_price: float, direction: str) -> float:
    """
    Retourne la distance du stop loss (SL) selon la direction.
    """
    if direction == "long":
        return max(price - stop_price, 0.0)
    else:
        return max(stop_price - price, 0.0)


def position_size(
    initial_capital: float,
    price: float,
    stop_distance: float,
    risk_pct: float,
    max_leverage: float,
) -> Tuple[float, float, float, float]:
    """
    Calcule la taille de position brute.
    Retourne : (taille, montant du risque, risque effectif en %, levier)
    """
    if stop_distance <= 0 or initial_capital <= 0 or price <= 0:
        return 0.0, 0.0, 0.0, 0.0

    risk_amount = (risk_pct / 100.0) * initial_capital
    size = risk_amount / stop_distance

    notional = abs(price * size)
    leverage = notional / initial_capital

    if max_leverage > 0 and leverage > max_leverage:
        scale = max_leverage / leverage
        size *= scale
        risk_amount *= scale
        leverage = max_leverage

    effective_risk_pct = (risk_amount / initial_capital * 100.0)
    return size, risk_amount, effective_risk_pct, leverage


def take_profit_targets(price: float, stop_distance: float, direction: str) -> Tuple[float, float]:
    """
    Renvoie les niveaux de TP classiques : 1R et 3R.
    """
    if direction == "long":
        tp1 = price + stop_distance
        tp2 = price + 3 * stop_distance
    else:
        tp1 = price - stop_distance
        tp2 = price - 3 * stop_distance
    return tp1, tp2


def position_size_vol_adjusted(
    capital: float,
    price: float,
    stop_distance: float,
    atr: float,
    base_risk_pct: float,
    max_leverage: float
) -> Tuple[float, float, float, float]:
    """
    Ajuste le sizing selon la volatilité (ATR).
    """
    if atr <= 0 or stop_distance <= 0:
        return 0.0, 0.0, 0.0, 0.0

    risk_vol_ratio = stop_distance / atr
    scaling_factor = 1.0 / max(1.0, risk_vol_ratio)
    adj_risk_pct = base_risk_pct * scaling_factor

    return position_size(capital, price, stop_distance, adj_risk_pct, max_leverage)


def load_custom_risk_profile(path: str) -> Dict[str, float]:
    """
    Charge un profil personnalisé de risque à partir d'un fichier JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_risk_efficiency(r_multiple: float, leverage: float) -> float:
    """
    Indicateur de qualité du trade : rendement/levier.
    """
    return r_multiple / leverage if leverage > 0 else 0.0


def sizing_quality_tag(leverage: float, max_leverage: float) -> str:
    """
    Qualité du sizing en fonction du levier utilisé.
    """
    if leverage >= 0.9 * max_leverage:
        return "overexposed"
    elif leverage <= 0.3 * max_leverage:
        return "underleveraged"
    else:
        return "balanced"
