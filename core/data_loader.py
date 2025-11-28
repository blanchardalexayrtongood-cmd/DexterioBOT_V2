# core/data_loader.py

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close"]
CACHE_DIR = "cache"
CACHE_TTL_MINUTES = 30  # durée de validité du cache


def clamp_lookback(timeframe: str, lookback_days: int) -> int:
    if timeframe == "1m" and lookback_days > 6:
        return 6
    return lookback_days


def adjust_timezone(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df


def validate_ohlc_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"[!] {symbol}: Missing columns: {', '.join(missing)}")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[!] {symbol}: Index not sorted")

    gaps = df.index.to_series().diff().dt.total_seconds().dropna()
    if gaps.empty or gaps.max() > 60 * 10:
        logger.warning(f"[~] {symbol}: Detected time gaps in OHLC data")

    return df


def cache_key(symbol: str, timeframe: str, lookback_days: int) -> str:
    key = f"{symbol}_{timeframe}_{lookback_days}"
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.csv")


def load_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    modified_time = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - modified_time > timedelta(minutes=CACHE_TTL_MINUTES):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def save_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path)


def download_ohlc(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    try:
        lookback_days = clamp_lookback(timeframe, lookback_days)
        cache_path = cache_key(symbol, timeframe, lookback_days)
        cached = load_cache(cache_path)
        if cached is not None and not cached.empty:
            logger.info(f"[↻] Loaded {symbol} ({timeframe}) from cache ({len(cached)} bars)")
            return cached

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days + 1)
        logger.info(f"[+] Downloading {symbol} ({timeframe}) from {start.date()} to {end.date()}")

        df = yf.download(
            symbol,
            interval=timeframe,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if df.empty or len(df) < 10:
            raise RuntimeError(f"[!] No or insufficient data for {symbol} ({timeframe})")

        df = df.rename(columns=str.lower)
        df = adjust_timezone(df)
        df = validate_ohlc_df(df, symbol)
        df.dropna(inplace=True)

        save_cache(df, cache_path)
        logger.info(f"[✓] Downloaded and cached {len(df)} bars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"[✗] Failed to load data for {symbol}: {e}")
        raise


def download_batch_ohlc(symbols: List[str], timeframe: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    results = {}
    for symbol in symbols:
        try:
            results[symbol] = download_ohlc(symbol, timeframe, lookback_days)
        except Exception as e:
            logger.warning(f"[!] Skipping {symbol}: {e}")
    return results
