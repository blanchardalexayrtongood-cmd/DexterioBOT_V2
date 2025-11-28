from datetime import datetime, time, date
from typing import List


# Killzones par défaut (heure NY)
KILLZONES = [
    (time(9, 50),  time(10, 10)),   # Morning
    (time(13, 50), time(14, 10)),   # Afternoon
]


def is_in_killzone(ts: datetime, zones: List[tuple] = KILLZONES) -> bool:
    """
    Vérifie si le timestamp est dans une killzone donnée (timezone NY).
    """
    t = ts.time()
    return any(start <= t < end for start, end in zones)


def is_in_blackout(ts: datetime) -> bool:
    """
    True si le timestamp est dans une période de blackout (illiquidité extrême).
    Par défaut : après 21h30 ou avant 03h00 heure New York.
    """
    t = ts.tz_convert("America/New_York").time()
    return t >= time(21, 30) or t <= time(3, 0)


def daily_loss_exceeded(
    daily_pnl: float,
    initial_capital: float,
    max_daily_loss_pct: float,
) -> bool:
    """
    True si la perte du jour excède le seuil max autorisé.
    """
    allowed_loss = -initial_capital * (max_daily_loss_pct / 100.0)
    return daily_pnl <= allowed_loss


def daily_trades_exceeded(
    daily_trades: int,
    max_daily_trades: int,
) -> bool:
    """
    True si le nombre de trades dépasse la limite journalière.
    """
    return daily_trades >= max_daily_trades


def is_blackout_day(ts: datetime, blackout_dates: List[date]) -> bool:
    """
    True si le jour est un blackout (pas de trading souhaité ce jour-là).
    """
    return ts.date() in blackout_dates
