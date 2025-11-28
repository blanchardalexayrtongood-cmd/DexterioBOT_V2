# engine.py

import math
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from core.structure import detect_swings, is_structure_clean
from core.risk import compute_stop_distance, position_size_vol_adjusted, take_profit_targets, sizing_quality_tag
from core.session_handler import is_in_killzone, daily_loss_exceeded, daily_trades_exceeded, is_in_blackout
from core.journal import BacktestStats, Trade
from models.setup_scorer import SetupContext, score_setup_cjr, should_trade_es, should_trade_nq
from analysis.daytype_classifier import classify_day_type

def scalar(value):
    if isinstance(value, (pd.Series, pd.DataFrame, list, tuple, np.ndarray)):
        try:
            return float(value.iloc[0] if hasattr(value, "iloc") else value[0])
        except Exception:
            return None
    return float(value)

def reset_prev(prev1_high, prev1_low, high, low):
    return high, low, prev1_high, prev1_low

def should_skip_time(ts, capital, in_kz, current_date, daily_pnl, daily_trades, risk_cfg, killzone_only):
    return (
        daily_loss_exceeded(daily_pnl[current_date], capital, risk_cfg.get("max_daily_loss_pct", 100.0)) or
        daily_trades_exceeded(daily_trades[current_date], risk_cfg.get("max_daily_trades", 999)) or
        (killzone_only and not in_kz) or
        is_in_blackout(ts)
    )

def classify_market_regime(df, window=20):
    atr = df["high"].rolling(window).max() - df["low"].rolling(window).min()
    regime = []
    for i in range(len(df)):
        rng = df["high"].iloc[i] - df["low"].iloc[i]
        a = atr.iloc[i]
        if pd.isna(a):
            regime.append("unknown")
        elif rng < a * 1.2:
            regime.append("range")
        elif rng > a * 2:
            regime.append("expansion")
        else:
            regime.append("trend")
    return regime

def run_strategy_on_symbol(
    symbol: str,
    df: pd.DataFrame,
    strategy_mode: str,
    risk_cfg: Dict[str, float],
    initial_capital: float,
    peer_df: Optional[pd.DataFrame] = None,
    killzone_only: bool = True,
    use_scorer_filters: bool = True,
) -> BacktestStats:
    if df.empty:
        return BacktestStats(symbol=symbol, initial_capital=initial_capital, final_capital=initial_capital)

    df = df.copy()
    df.index = pd.to_datetime(df.index)  # 🔧 Assure que l'index est bien en datetime
    df.sort_index(inplace=True)
    strategy_mode = strategy_mode.lower()

    try:
        df_day = df[df.index.date == df.index[-1].date()]
        day_type, _ = classify_day_type(df_day)
    except Exception:
        day_type = "unknown"

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["market_regime"] = classify_market_regime(df)
    df = df.join(detect_swings(df))

    peer_ema9 = peer_ema21 = None
    if peer_df is not None:
        peer_df = peer_df.copy().sort_index()
        peer_close = peer_df["close"].reindex(df.index).ffill()
        peer_ema9 = peer_close.ewm(span=9, adjust=False).mean()
        peer_ema21 = peer_close.ewm(span=21, adjust=False).mean()

    capital = float(initial_capital)
    trades: List[Trade] = []
    equity_times, equity_values, drawdowns = [], [], []
    max_capital = capital
    daily_pnl, daily_trades, daily_r = {}, {}, {}

    prev1_high = prev1_low = prev2_high = prev2_low = None

    for i, ts in enumerate(df.index):
        row = df.iloc[i]
        price, high, low, open_price = map(scalar, (row["close"], row["high"], row["low"], row["open"]))
        current_date = ts.date()
        daily_pnl.setdefault(current_date, 0.0)
        daily_trades.setdefault(current_date, 0)
        daily_r.setdefault(current_date, 0.0)

        if prev1_high is None:
            prev1_high, prev1_low = high, low
            continue

        in_kz = is_in_killzone(ts)
        if should_skip_time(ts, initial_capital, in_kz, current_date, daily_pnl, daily_trades, risk_cfg, killzone_only):
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
            continue

        ema9 = scalar(df["ema9"].iloc[i])
        ema21 = scalar(df["ema21"].iloc[i])
        if math.isnan(ema9) or math.isnan(ema21):
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
            continue

        bias = "long" if ema9 > ema21 else "short"
        bos_long = price > prev1_high
        bos_short = price < prev1_low
        fvg_bullish = low > prev2_high if prev2_high is not None else False
        fvg_bearish = high < prev2_low if prev2_low is not None else False
        sweep_bullish = high > prev1_high and price < prev1_high if prev1_high else False
        sweep_bearish = low < prev1_low and price > prev1_low if prev1_low else False

        direction = None
        smt_tag = ""

        if bias == "long" and (bos_long or sweep_bullish):
            direction = "long"
        elif bias == "short" and (bos_short or sweep_bearish):
            direction = "short"

        if direction is None:
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
            continue

        environment_choppy = (prev1_high - prev1_low) / price < 0.001 if price else False

        if symbol == "NQ=F" and peer_ema9 is not None and peer_ema21 is not None:
            peer_e9 = scalar(peer_ema9.get(ts, float("nan")))
            peer_e21 = scalar(peer_ema21.get(ts, float("nan")))
            if not math.isnan(peer_e9) and not math.isnan(peer_e21):
                es_bias_direction = "long" if peer_e9 > peer_e21 else "short"
                if es_bias_direction != direction:
                    prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                    continue
                smt_tag = f"es_{es_bias_direction}"

        confluence_count = sum([
            (direction == "long" and sweep_bullish) or (direction == "short" and sweep_bearish),
            (direction == "long" and bos_long) or (direction == "short" and bos_short),
            (direction == "long" and fvg_bullish) or (direction == "short" and fvg_bearish),
        ])
        multi_signal = confluence_count >= 2

        if symbol == "ES=F":
            if confluence_count < 1 or environment_choppy:
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue
            if (direction == "long" and price < open_price) or (direction == "short" and price > open_price):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue

        if symbol == "NQ=F":
            if not (sweep_bullish and bos_long) and direction == "long":
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue
            if not (sweep_bearish and bos_short) and direction == "short":
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue
            if environment_choppy:
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue

        structure_clean = is_structure_clean(df.loc[:ts, ["swing_high", "swing_low", "swing_high_val", "swing_low_val"]])

        setup_ctx = SetupContext(
            symbol=symbol,
            direction=direction,
            has_htf_bias_confluence=True,
            has_liquidity_sweep=(sweep_bullish if direction == "long" else sweep_bearish),
            has_bos_in_direction=(bos_long if direction == "long" else bos_short),
            uses_fvg=(fvg_bullish if direction == "long" else fvg_bearish),
            in_killzone=in_kz,
            environment_choppy=environment_choppy,
            es_bias_direction=smt_tag,
            es_structure_clean=structure_clean,
            multi_signal_confirmed=multi_signal,
        )

        score_obj = score_setup_cjr(setup_ctx)
        if use_scorer_filters:
            if symbol == "ES=F" and not should_trade_es(score_obj):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue
            if symbol == "NQ=F" and not should_trade_nq(score_obj, daily_r[current_date], -2.0):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
                continue

        stop_price = prev1_low if direction == "long" else prev1_high
        if (direction == "long" and stop_price >= price) or (direction == "short" and stop_price <= price):
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
            continue

        stop_dist = compute_stop_distance(price, stop_price, direction)
        if stop_dist <= 0:
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
            continue

        atr = scalar(df["high"].rolling(14).max().iloc[i] - df["low"].rolling(14).min().iloc[i])
        size, risk_amount, risk_pct_eff, leverage = position_size_vol_adjusted(
            capital, price, stop_dist, atr,
            risk_cfg.get("max_risk_per_trade_pct", 1.0),
            risk_cfg.get("max_leverage", 5.0),
        )
        if size <= 0 or risk_amount <= 0:
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)
            continue

        sizing_tag = sizing_quality_tag(leverage, risk_cfg.get("max_leverage", 5.0))
        tp1_price, tp_price = take_profit_targets(price, stop_dist, direction)

        exit_price = price
        exit_time = ts
        tp1_hit = tp2_hit = False
        r_multiple = pnl = 0.0
        equity_before_trade = capital

        for j in range(i + 1, len(df.index)):
            ts2 = df.index[j]
            row2 = df.iloc[j]
            high2, low2 = map(scalar, (row2["high"], row2["low"]))

            stop_hit = low2 <= stop_price if direction == "long" else high2 >= stop_price
            tp2_hit_fg = high2 >= tp_price if direction == "long" else low2 <= tp_price
            tp1_cross = high2 >= tp1_price if direction == "long" else low2 <= tp1_price

            if tp1_cross and not tp1_hit:
                tp1_hit = True
            if stop_hit:
                exit_price = stop_price
                exit_time = ts2
                break
            elif tp2_hit_fg:
                exit_price = tp_price
                exit_time = ts2
                tp2_hit = True
                break
        else:
            exit_price = scalar(df["close"].iloc[-1])
            exit_time = df.index[-1]

        pnl = (exit_price - price) * size if direction == "long" else (price - exit_price) * size
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
        pnl_pct = pnl / equity_before_trade * 100.0 if equity_before_trade else 0.0

        capital += pnl
        max_capital = max(max_capital, capital)
        drawdown = 100 * (max_capital - capital) / max_capital if max_capital else 0
        drawdowns.append(drawdown)

        daily_pnl[current_date] += pnl
        daily_r[current_date] += r_multiple
        daily_trades[current_date] += 1

        equity_times.append(exit_time)
        equity_values.append(capital)

        trades.append(Trade(
            symbol=symbol,
            direction=direction,
            entry_time=ts,
            exit_time=exit_time,
            entry_price=price,
            exit_price=exit_price,
            stop_price=stop_price,
            tp_price=tp_price,
            size=size,
            risk_amount=risk_amount,
            risk_pct=risk_pct_eff,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            leverage=leverage,
            htf_bias=bias,
            strategy_mode=strategy_mode,
            smt_tag=smt_tag,
            context=sizing_tag,
            tp1_price=tp1_price,
            tp1_hit=tp1_hit,
            tp2_hit=tp2_hit,
            grade=score_obj.grade,
            setup_score=score_obj.score,
            day_type=day_type,
        ))

        prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(prev1_high, prev1_low, high, low)

    equity_series = pd.Series(equity_values, index=pd.to_datetime(equity_times)) if equity_times else pd.Series(dtype=float)
    drawdown_series = pd.Series(drawdowns, index=pd.to_datetime(equity_times)) if equity_times else pd.Series(dtype=float)

    stats = BacktestStats(
        symbol=symbol,
        initial_capital=initial_capital,
        final_capital=capital,
        trades=trades,
        equity_curve=equity_series,
    )
    stats.drawdown_curve = drawdown_series
    stats.max_drawdown_pct = drawdown_series.max() if not drawdown_series.empty else 0.0

    grade_dist: Dict[str, int] = {}
    r_by_grade: Dict[str, float] = {}
    for t in trades:
        g = getattr(t, "grade", "")
        if g:
            grade_dist[g] = grade_dist.get(g, 0) + 1
            r_by_grade[g] = r_by_grade.get(g, 0.0) + t.r_multiple

    stats.grade_distribution = grade_dist
    stats.r_by_grade = r_by_grade

    return stats
