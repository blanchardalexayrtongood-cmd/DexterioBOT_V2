# core/session_handler.py

from datetime import datetime, time, date
from typing import List, Optional

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

import pandas as pd
from pytz import timezone
from analysis.daytype_classifier import classify_day_type

def compute_daily_bias(df: pd.DataFrame, current_date: date, allow_guess: bool = True) -> Optional[str]:
    """
    Détermine le biais quotidien ('long', 'short' ou None) pour la session de trading du jour current_date.
    Analyse les sessions précédentes (Asie, Londres) en identifiant les manipulations de liquidité et la direction probable.
    Si allow_guess=True, on anticipe le biais même après une consolidation (cas 1) en devinant le sens du faux départ.
    """
    # Assure que l'index est en timezone New York pour segmenter correctement
    if df.index.tz is None:
        df_ny = df.copy()
        df_ny.index = df_ny.index.tz_localize("America/New_York")
    else:
        df_ny = df.copy()
        df_ny.index = df_ny.index.tz_convert("America/New_York")

    # Jour de trading précédent (ignore week-ends)
    prev_date = current_date - pd.Timedelta(days=1)
    while prev_date.weekday() >= 5:  # 5 = Samedi, 6 = Dimanche
        prev_date = prev_date - pd.Timedelta(days=1)
    prev_trading_date = prev_date

    # Données de la veille (jour précédent), session Asie (veille 18h à jour 3h), session Londres (jour 3h à 9h30)
    prev_day_df = df_ny[df_ny.index.date == prev_trading_date]
    asia_start = timezone("America/New_York").localize(datetime.combine(prev_trading_date, time(18, 0)))
    asia_end = timezone("America/New_York").localize(datetime.combine(current_date, time(3, 0)))
    asia_df = df_ny[(df_ny.index >= asia_start) & (df_ny.index < asia_end)]
    london_start = timezone("America/New_York").localize(datetime.combine(current_date, time(3, 0)))
    london_end = timezone("America/New_York").localize(datetime.combine(current_date, time(9, 30)))
    london_df = df_ny[(df_ny.index >= london_start) & (df_ny.index < london_end)]

    if london_df.empty:
        return None  # pas de session Londres (jour férié ou marché fermé)

    # Niveaux de liquidité clés
    prev_day_high = float(prev_day_df["high"].max()) if not prev_day_df.empty else None
    prev_day_low = float(prev_day_df["low"].min()) if not prev_day_df.empty else None
    asia_high = float(asia_df["high"].max()) if not asia_df.empty else None
    asia_low = float(asia_df["low"].min()) if not asia_df.empty else None
    london_high = float(london_df["high"].max())
    london_low = float(london_df["low"].min())

    # Détection de manipulation (stop-run) pendant Londres
    broke_up = False
    broke_down = False
    if prev_day_high is not None and london_high >= prev_day_high:
        broke_up = True
    if prev_day_low is not None and london_low <= prev_day_low:
        broke_down = True
    if asia_high is not None and london_high >= asia_high:
        broke_up = True
    if asia_low is not None and london_low <= asia_low:
        broke_down = True

    bias_direction: Optional[str] = None

    # Cas 1: Londres en range (aucune manipulation franche)
    if not broke_up and not broke_down:
        if not allow_guess:
            return None  # biais indéterminé si on ne devine pas après consolidation
        # On anticipe le sens du faux mouvement initial à New York
        try:
            current_price = float(df_ny.loc[london_end]["close"])
        except Exception:
            df_before_open = df_ny[df_ny.index < london_end]
            current_price = float(df_before_open.iloc[-1]["close"]) if not df_before_open.empty else float(london_df.iloc[-1]["close"])
        nearest_up_dist = float("inf")
        nearest_down_dist = float("inf")
        for level in [prev_day_high, asia_high]:
            if level is not None:
                dist = level - current_price
                if dist >= 0 and dist < nearest_up_dist:
                    nearest_up_dist = dist
        for level in [prev_day_low, asia_low]:
            if level is not None:
                dist = current_price - level
                if dist >= 0 and dist < nearest_down_dist:
                    nearest_down_dist = dist
        if nearest_up_dist < nearest_down_dist:
            bias_direction = "short"   # plus proche du haut -> probablement une manipulation haussière puis baisse
        elif nearest_down_dist < nearest_up_dist:
            bias_direction = "long"    # plus proche du bas -> probablement une manipulation baissière puis hausse
        else:
            bias_direction = None

    else:
        # Cas 2 ou 3: une manipulation a eu lieu pendant Londres
        try:
            london_day_type, _ = classify_day_type(london_df)
        except Exception:
            london_day_type = None
        if str(london_day_type) == "trending_bull":
            # Cas 3 haussier: Londres a manipulé un plus bas et fortement inversé à la hausse
            bias_direction = "long"
        elif str(london_day_type) == "trending_bear":
            # Cas 3 baissier: Londres a manipulé un plus haut et fortement inversé à la baisse
            bias_direction = "short"
        else:
            # Cas 2: manipulation sans vrai trend derrière pendant Londres
            if broke_up and not broke_down:
                bias_direction = "short"   # Londres a fait un stop-run haussier puis s'est essoufflé -> biais baissier
            elif broke_down and not broke_up:
                bias_direction = "long"    # Londres a fait un stop-run baissier puis s'est essoufflé -> biais haussier
            else:
                bias_direction = None

    return bias_direction
