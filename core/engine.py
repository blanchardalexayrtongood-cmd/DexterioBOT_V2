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
        daily_loss_exceeded(daily_pnl[current_date], capital, risk_cfg.get("max_daily_loss_pct", 100.0))
        or daily_trades_exceeded(daily_trades[current_date], risk_cfg.get("max_daily_trades", 999))
        or (killzone_only and not in_kz)
        or is_in_blackout(ts)
    )


def classify_market_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Classe le régime de marché de façon vectorisée.

    - 'unknown'   : pas assez d'historique pour le rolling (NaN)
    - 'range'     : range < 1.2 * ATR_window
    - 'expansion' : range > 2 * ATR_window
    - 'trend'     : entre les deux
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    rng = high - low
    roll_high = high.rolling(window).max()
    roll_low = low.rolling(window).min()
    atr = roll_high - roll_low

    cond_unknown = atr.isna()
    cond_range = rng < atr * 1.2
    cond_expansion = rng > atr * 2

    # np.where peut renvoyer un array 2D -> on aplatit
    regime = np.where(
        cond_unknown,
        "unknown",
        np.where(
            cond_range,
            "range",
            np.where(cond_expansion, "expansion", "trend"),
        ),
    )

    regime = np.array(regime).ravel()   # 🔥 correction ici

    return pd.Series(regime, index=df.index)


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

    # 🔧 Conversion robuste de l'index en datetime
    idx = pd.to_datetime(df.index, errors="coerce")

    # Si rien n'est convertible → on arrête proprement
    if idx.isna().all():
        raise ValueError(f"[!] Aucun index datetime valide dans les données pour {symbol}")

    # Si certaines lignes sont invalides (ex: 'Ticker'), on les vire
    if idx.isna().any():
        mask = idx.notna()
        df = df.loc[mask].copy()
        idx = idx[mask]

    df.index = idx
    df.sort_index(inplace=True)
    strategy_mode = strategy_mode.lower()

    try:
        df_day = df[df.index.date == df.index[-1].date()]
        day_type, _ = classify_day_type(df_day)
    except Exception:
        day_type = "unknown"

    #    # EMAs & régime
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["market_regime"] = classify_market_regime(df)

    # ---- Swings (robuste) ----
    try:
        swings = detect_swings(df)

        # Si même longueur mais index différent → on force le même index
        if len(swings.index) == len(df.index) and not swings.index.equals(df.index):
            swings = swings.copy()
            swings.index = df.index

        df = pd.concat([df, swings], axis=1)

    except Exception as e:
        print(f"[WARN] detect_swings failed for {symbol}: {e}")
        for col in ["swing_high", "swing_low", "swing_high_val", "swing_low_val"]:
            if col not in df.columns:
                df[col] = np.nan


    # Multi-timeframe analysis for H1 FVG and bias
    df_h1 = df.resample("1H").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    df_h1.dropna(subset=["open"], inplace=True)
    df_h1["ema9"] = df_h1["close"].ewm(span=9, adjust=False).mean()
    df_h1["ema21"] = df_h1["close"].ewm(span=21, adjust=False).mean()
    h1_bias_series = df_h1["ema9"] > df_h1["ema21"]
    h1_bias_for_df = h1_bias_series.reindex(df.index, method="ffill")

    fvg_h1_zones: List[Dict] = []
    prev_h1_high = prev_h1_low = prev_h1_prev_high = prev_h1_prev_low = None
    for t, row_h1 in df_h1.iterrows():
        h_val = float(row_h1["high"])
        l_val = float(row_h1["low"])
        if prev_h1_high is None:
            prev_h1_high, prev_h1_low = h_val, l_val
            continue

        bull_one = l_val > prev_h1_high
        bull_two = (
            prev_h1_prev_high is not None
            and prev_h1_low is not None
            and prev_h1_low > prev_h1_prev_high
            and l_val > prev_h1_prev_high
        )
        bear_one = h_val < prev_h1_low
        bear_two = (
            prev_h1_prev_low is not None
            and prev_h1_high is not None
            and prev_h1_high < prev_h1_prev_low
            and h_val < prev_h1_prev_low
        )

        if bull_one or bull_two:
            if bull_one and not bull_two:
                zone_low = prev_h1_high
                zone_high = l_val
            elif bull_two and not bull_one:
                zone_low = prev_h1_prev_high
                zone_high = min(prev_h1_low, l_val) if prev_h1_low is not None else l_val
            else:
                zone_low = prev_h1_prev_high if prev_h1_prev_high is not None else prev_h1_high
                zone_high = l_val
            fvg_h1_zones.append(
                {"lower": zone_low, "upper": zone_high, "direction": "bull", "created": t, "filled": False}
            )

        if bear_one or bear_two:
            if bear_one and not bear_two:
                zone_low = h_val
                zone_high = prev_h1_low
            elif bear_two and not bear_one:
                zone_low = max(prev_h1_high, h_val) if prev_h1_high is not None else h_val
                zone_high = prev_h1_prev_low
            else:
                zone_low = h_val
                zone_high = prev_h1_prev_low if prev_h1_prev_low is not None else prev_h1_low
            fvg_h1_zones.append(
                {"lower": zone_low, "upper": zone_high, "direction": "bear", "created": t, "filled": False}
            )

        prev_h1_prev_high, prev_h1_prev_low = prev_h1_high, prev_h1_low
        prev_h1_high, prev_h1_low = h_val, l_val

    # Mark H1 FVG zones as filled if later price entered them
    for zone in fvg_h1_zones:
        if zone["direction"] == "bull":
            later_df = df_h1.loc[zone["created"] :]
            if not later_df.empty and float(later_df["low"].min()) <= zone["upper"]:
                zone["filled"] = True
        else:  # bear
            later_df = df_h1.loc[zone["created"] :]
            if not later_df.empty and float(later_df["high"].max()) >= zone["lower"]:
                zone["filled"] = True

    # Peer EMA (SMT)
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
    fvg_zones: List[Dict] = []  # FVG MTF (timeframe actuel) mémorisés pour les retests

    for i, ts in enumerate(df.index):
        row = df.iloc[i]
        price, high, low, open_price = map(
            scalar, (row["close"], row["high"], row["low"], row["open"])
        )
        current_date = ts.date()
        daily_pnl.setdefault(current_date, 0.0)
        daily_trades.setdefault(current_date, 0)
        daily_r.setdefault(current_date, 0.0)

        if prev1_high is None:
            # Initialize previous candle values
            prev1_high, prev1_low = high, low
            continue

        in_kz = is_in_killzone(ts)
        if should_skip_time(
            ts,
            initial_capital,
            in_kz,
            current_date,
            daily_pnl,
            daily_trades,
            risk_cfg,
            killzone_only,
        ):
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                prev1_high, prev1_low, high, low
            )
            continue

        ema9 = scalar(df["ema9"].iloc[i])
        ema21 = scalar(df["ema21"].iloc[i])
        if math.isnan(ema9) or math.isnan(ema21):
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                prev1_high, prev1_low, high, low
            )
            continue

        # Determine bias and structural signals
        bias = "long" if ema9 > ema21 else "short"
        bos_long = price > prev1_high if prev1_high is not None else False
        bos_short = price < prev1_low if prev1_low is not None else False

        # Enhanced Fair Value Gap (FVG) detection
        fvg_bullish_immediate = prev1_high is not None and low > prev1_high
        fvg_bullish_two = (
            prev2_high is not None
            and prev1_low is not None
            and prev1_low > prev2_high
            and low > prev2_high
        )
        fvg_bearish_immediate = prev1_low is not None and high < prev1_low
        fvg_bearish_two = (
            prev2_low is not None
            and prev1_high is not None
            and prev1_high < prev2_low
            and high < prev2_low
        )
        fvg_bullish = fvg_bullish_immediate or fvg_bullish_two
        fvg_bearish = fvg_bearish_immediate or fvg_bearish_two

        # Log and store any new FVG (timeframe actuel)
        if fvg_bullish:
            if fvg_bullish_immediate and not fvg_bullish_two:
                zone_low = prev1_high
                zone_high = low
            elif fvg_bullish_two and not fvg_bullish_immediate:
                zone_low = prev2_high
                zone_high = min(prev1_low, low) if prev1_low is not None else low
            else:
                zone_low = prev2_high if prev2_high is not None else prev1_high
                zone_high = low
            fvg_zones.append(
                {
                    "lower": zone_low,
                    "upper": zone_high,
                    "direction": "bull",
                    "created": ts,
                    "filled": False,
                }
            )
            print(
                f"[FVG DETECTED] Bullish FVG at {ts} from {zone_low:.2f} to {zone_high:.2f}"
            )

        if fvg_bearish:
            if fvg_bearish_immediate and not fvg_bearish_two:
                zone_low = high
                zone_high = prev1_low
            elif fvg_bearish_two and not fvg_bearish_immediate:
                zone_low = max(prev1_high, high) if prev1_high is not None else high
                zone_high = prev2_low
            else:
                zone_low = high
                zone_high = prev2_low if prev2_low is not None else prev1_low
            fvg_zones.append(
                {
                    "lower": zone_low,
                    "upper": zone_high,
                    "direction": "bear",
                    "created": ts,
                    "filled": False,
                }
            )
            print(
                f"[FVG DETECTED] Bearish FVG at {ts} from {zone_low:.2f} to {zone_high:.2f}"
            )

        # Liquidity sweep signals
        sweep_bullish = bool(prev1_high and high > prev1_high and price < prev1_high)
        sweep_bearish = bool(prev1_low and low < prev1_low and price > prev1_low)

        # Determine trade direction or wait for FVG retrace
        direction = None
        retouch_entry = False
        smt_tag = ""

        if bias == "long" and (bos_long or sweep_bullish):
            direction = "long"
        elif bias == "short" and (bos_short or sweep_bearish):
            direction = "short"

        if direction is None:
            # Attempt entry on FVG retest (price returning into a previously detected FVG)
            trade_dir = None
            triggered_zone = None
            for zone in reversed(fvg_zones):
                if not zone["filled"]:
                    if (
                        bias == "long"
                        and zone["direction"] == "bull"
                        and low <= zone["upper"]
                    ):
                        trade_dir = "long"
                        triggered_zone = zone
                        break
                    if (
                        bias == "short"
                        and zone["direction"] == "bear"
                        and high >= zone["lower"]
                    ):
                        trade_dir = "short"
                        triggered_zone = zone
                        break
            if trade_dir:
                direction = trade_dir
                triggered_zone["filled"] = True
                triggered_zone["filled_ts"] = ts
                retouch_entry = True
                print(
                    f"[FVG FILL] {direction.capitalize()} FVG filled at {ts}. Entering {direction} trade."
                )
            else:
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue

        environment_choppy = (
            (prev1_high - prev1_low) / price < 0.001 if price else False
        )

        # SMT divergence check with peer (ES <-> NQ)
        if symbol in ("ES=F", "NQ=F") and peer_ema9 is not None and peer_ema21 is not None:
            peer_e9 = scalar(peer_ema9.get(ts, float("nan")))
            peer_e21 = scalar(peer_ema21.get(ts, float("nan")))
            if not math.isnan(peer_e9) and not math.isnan(peer_e21):
                peer_bias_direction = "long" if peer_e9 > peer_e21 else "short"
                if peer_bias_direction != direction:
                    prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                        prev1_high, prev1_low, high, low
                    )
                    continue
                peer_prefix = "es" if symbol == "NQ=F" else "nq"
                smt_tag = f"{peer_prefix}_{peer_bias_direction}"

        # Count confluence signals for information (BOS, sweep, FVG)
        confluence_count = sum(
            [
                (direction == "long" and sweep_bullish)
                or (direction == "short" and sweep_bearish),
                (direction == "long" and bos_long)
                or (direction == "short" and bos_short),
                (direction == "long" and fvg_bullish)
                or (direction == "short" and fvg_bearish),
            ]
        )
        multi_signal = confluence_count >= 2

        # Symbol-specific gating rules
        if symbol == "ES=F":
            if (not retouch_entry) and (confluence_count < 1 or environment_choppy):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue
            if (not retouch_entry) and (
                (direction == "long" and price < open_price)
                or (direction == "short" and price > open_price)
            ):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue

        if symbol == "NQ=F":
            if (
                not retouch_entry
                and direction == "long"
                and not (sweep_bullish and bos_long)
            ):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue
            if (
                not retouch_entry
                and direction == "short"
                and not (sweep_bearish and bos_short)
            ):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue
            if not retouch_entry and environment_choppy:
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue

        structure_clean = is_structure_clean(
            df.loc[:ts, ["swing_high", "swing_low", "swing_high_val", "swing_low_val"]]
        )

        # Assess higher timeframe bias confluence
        current_h1_bias = bool(h1_bias_for_df.get(ts, True))
        h1_dir = "long" if current_h1_bias else "short"
        has_htf_conf = h1_dir == direction

        # Check for conflicting higher timeframe FVG (opposite imbalance nearby)
        conflicting_signals = False
        if direction == "long":
            for zone in fvg_h1_zones:
                if (
                    not zone["filled"]
                    and zone["direction"] == "bear"
                    and price < zone["lower"]
                ):
                    conflicting_signals = True
                    break
        elif direction == "short":
            for zone in fvg_h1_zones:
                if (
                    not zone["filled"]
                    and zone["direction"] == "bull"
                    and price > zone["upper"]
                ):
                    conflicting_signals = True
                    break

        setup_ctx = SetupContext(
            symbol=symbol,
            direction=direction,
            has_htf_bias_confluence=has_htf_conf,
            has_liquidity_sweep=(
                sweep_bullish if direction == "long" else sweep_bearish
            ),
            has_bos_in_direction=(bos_long if direction == "long" else bos_short),
            uses_fvg=((fvg_bullish if direction == "long" else fvg_bearish) or retouch_entry),
            in_killzone=in_kz,
            environment_choppy=environment_choppy,
            es_bias_direction=smt_tag,
            es_structure_clean=structure_clean,
            multi_signal_confirmed=multi_signal,
            confirmed_after_retouch=retouch_entry,
            conflicting_signals=conflicting_signals,
        )

        score_obj = score_setup_cjr(setup_ctx)
        if use_scorer_filters:
            if symbol == "ES=F" and not should_trade_es(score_obj):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue
            if symbol == "NQ=F" and not should_trade_nq(
                score_obj, daily_r[current_date], -2.0
            ):
                prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                    prev1_high, prev1_low, high, low
                )
                continue

        # Determine stop loss price (previous swing or current FVG retest extreme)
        if retouch_entry:
            stop_price = low if direction == "long" else high
        else:
            stop_price = prev1_low if direction == "long" else prev1_high

        if (direction == "long" and stop_price >= price) or (
            direction == "short" and stop_price <= price
        ):
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                prev1_high, prev1_low, high, low
            )
            continue

        stop_dist = compute_stop_distance(price, stop_price, direction)
        if stop_dist <= 0:
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                prev1_high, prev1_low, high, low
            )
            continue

        atr = scalar(
            df["high"].rolling(14).max().iloc[i]
            - df["low"].rolling(14).min().iloc[i]
        )
        size, risk_amount, risk_pct_eff, leverage = position_size_vol_adjusted(
            capital,
            price,
            stop_dist,
            atr,
            risk_cfg.get("max_risk_per_trade_pct", 1.0),
            risk_cfg.get("max_leverage", 5.0),
        )
        if size <= 0 or risk_amount <= 0:
            prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
                prev1_high, prev1_low, high, low
            )
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

        trades.append(
            Trade(
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
            )
        )

        prev1_high, prev1_low, prev2_high, prev2_low = reset_prev(
            prev1_high, prev1_low, high, low
        )

    equity_series = (
        pd.Series(equity_values, index=pd.to_datetime(equity_times))
        if equity_times
        else pd.Series(dtype=float)
    )
    drawdown_series = (
        pd.Series(drawdowns, index=pd.to_datetime(equity_times))
        if equity_times
        else pd.Series(dtype=float)
    )

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
