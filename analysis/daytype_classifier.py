from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


class DayType(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_CHOPPY = "range_choppy"


@dataclass
class DayTypeFeatures:
    day_return_pct: float
    range_pct: float
    close_position: float
    body_range_ratio: float
    direction_persistence: float
    wick_ratio: float
    late_momentum: float
    volume_mean: Optional[float] = None
    zscore_range: Optional[float] = None
    midday_spike: Optional[bool] = None


def _compute_intraday_features(df: pd.DataFrame, range_history: Optional[list] = None) -> DayTypeFeatures:
    if df.empty:
        raise ValueError("DayType classifier: df intraday vide")

    o = float(df["open"].iloc[0])
    h = float(df["high"].max())
    l = float(df["low"].min())
    c = float(df["close"].iloc[-1])

    day_return_pct = (c - o) / o * 100.0
    range_abs = max(h - l, 1e-9)
    range_pct = range_abs / o * 100.0
    close_position = (c - l) / range_abs

    body = (df["close"] - df["open"]).abs()
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body_range_ratio = (body / candle_range).replace([np.inf, -np.inf], np.nan).mean()
    if pd.isna(body_range_ratio):
        body_range_ratio = 0.0

    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    wick_ratio = ((upper_wick + lower_wick) / candle_range).replace([np.inf, -np.inf], np.nan).mean()
    if pd.isna(wick_ratio):
        wick_ratio = 0.0

    close_diff = df["close"].diff().fillna(0.0)
    direction = np.sign(close_diff)
    local_trend = direction.rolling(window=5, min_periods=1).mean()
    same_direction = np.sign(local_trend) == direction
    direction_persistence = same_direction.mean()
    if pd.isna(direction_persistence):
        direction_persistence = 0.0

    end_quarter = df.iloc[int(len(df) * 0.75):]
    late_momentum = end_quarter["close"].diff().sum() / o * 100.0

    volume_mean = df["volume"].mean() if "volume" in df.columns else None

    zscore_range = None
    if range_history and len(range_history) > 5:
        mean_range = np.mean(range_history)
        std_range = np.std(range_history)
        if std_range > 0:
            zscore_range = (range_abs - mean_range) / std_range

    midday_idx = int(len(df) / 2)
    midpoint = df.iloc[midday_idx - 2:midday_idx + 3]
    ret = midpoint["close"].pct_change().abs().sum()
    midday_spike = ret > 0.01

    return DayTypeFeatures(
        day_return_pct=day_return_pct,
        range_pct=range_pct,
        close_position=close_position,
        body_range_ratio=float(body_range_ratio),
        direction_persistence=float(direction_persistence),
        wick_ratio=float(wick_ratio),
        late_momentum=float(late_momentum),
        volume_mean=float(volume_mean) if volume_mean is not None else None,
        zscore_range=float(zscore_range) if zscore_range is not None else None,
        midday_spike=midday_spike,
    )


def classify_day_type(
    df_intraday: pd.DataFrame,
    range_history: Optional[list] = None,
    min_range_pct: float = 0.3,
) -> Tuple[DayType, Dict[str, float]]:
    feats = _compute_intraday_features(df_intraday, range_history)

    if feats.range_pct < min_range_pct or feats.body_range_ratio < 0.3 or feats.wick_ratio > 0.7:
        day_type = DayType.RANGE_CHOPPY
    elif feats.day_return_pct > 0 and feats.close_position > 0.6 and feats.late_momentum > 0:
        day_type = DayType.TRENDING_BULL
    elif feats.day_return_pct < 0 and feats.close_position < 0.4 and feats.late_momentum < 0:
        day_type = DayType.TRENDING_BEAR
    else:
        day_type = DayType.RANGE_CHOPPY

    metrics: Dict[str, float] = feats.__dict__
    return day_type, metrics


def describe_day_type(day_type: DayType, metrics: Dict[str, float]) -> str:
    base = (
        f"day_return={metrics.get('day_return_pct', 0):+.2f}% | "
        f"range={metrics.get('range_pct', 0):.2f}% | "
        f"close_pos={metrics.get('close_position', 0):.2f}"
    )
    if day_type == DayType.TRENDING_BULL:
        return f"Trending Bullish Day ({base})"
    if day_type == DayType.TRENDING_BEAR:
        return f"Trending Bearish Day ({base})"
    return f"Range / Choppy Day ({base})"
