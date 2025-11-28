import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

@dataclass
class Trade:
    symbol: str
    direction: Literal["long", "short"]
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    tp_price: float
    size: float
    risk_amount: float
    risk_pct: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    leverage: float
    htf_bias: str = ""
    strategy_mode: str = ""
    smt_tag: str = ""
    context: str = ""
    tp1_price: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    grade: str = ""
    setup_score: int = 0
    market_regime: str = ""
    multi_signal_confirmed: Optional[bool] = None
    setup_tags: List[str] = field(default_factory=list)
    environment_choppy: Optional[bool] = None
    profile: str = ""
    day_type: str = ""  # ✅ NEW FIELD

@dataclass
class BacktestStats:
    symbol: str
    initial_capital: float
    final_capital: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    grade_distribution: Dict[str,int] = field(default_factory=dict)
    r_by_grade: Dict[str,float] = field(default_factory=dict)
    drawdown_curve: Optional[pd.Series] = None
    max_drawdown_pct: Optional[float] = None

    def compute_metrics(self) -> Dict[str, float]:
        if not self.trades:
            return {k: 0.0 for k in [
                "pnl_total", "pnl_pct", "nb_trades", "nb_winners", "nb_losers", "nb_be",
                "winrate", "pnl_mean", "best_trade", "worst_trade", "max_dd", "nb_days",
                "pnl_per_day", "max_risk_pct", "avg_risk_pct", "max_r_multiple",
                "min_r_multiple", "max_leverage", "avg_leverage", "expectancy_r"]}

        pnl_list = [t.pnl for t in self.trades]
        pnl_total = sum(pnl_list)
        nb_trades = len(pnl_list)
        winners = [p for p in pnl_list if p > 0]
        losers = [p for p in pnl_list if p < 0]
        be_count = nb_trades - len(winners) - len(losers)

        equity = pd.Series(pnl_list).cumsum() + self.initial_capital
        drawdowns = equity - equity.cummax()

        r_list = [t.r_multiple for t in self.trades]
        risk_pcts = [t.risk_pct for t in self.trades]
        levs = [abs(t.leverage) for t in self.trades]

        self.grade_distribution = {}
        self.r_by_grade = {}
        for t in self.trades:
            if t.grade:
                self.grade_distribution[t.grade] = self.grade_distribution.get(t.grade, 0) + 1
                self.r_by_grade[t.grade] = self.r_by_grade.get(t.grade, 0.0) + t.r_multiple

        first, last = self.trades[0].exit_time.date(), self.trades[-1].exit_time.date()
        nb_days = (last - first).days + 1 if first and last else 0

        return {
            "pnl_total": pnl_total,
            "pnl_pct": pnl_total / self.initial_capital * 100.0,
            "nb_trades": nb_trades,
            "nb_winners": len(winners),
            "nb_losers": len(losers),
            "nb_be": be_count,
            "winrate": len(winners) / nb_trades * 100.0,
            "pnl_mean": pnl_total / nb_trades,
            "best_trade": max(pnl_list),
            "worst_trade": min(pnl_list),
            "max_dd": drawdowns.min(),
            "nb_days": nb_days,
            "pnl_per_day": pnl_total / nb_days if nb_days > 0 else 0.0,
            "max_risk_pct": max(risk_pcts),
            "avg_risk_pct": sum(risk_pcts)/len(risk_pcts),
            "max_r_multiple": max(r_list),
            "min_r_multiple": min(r_list),
            "max_leverage": max(levs),
            "avg_leverage": sum(levs)/len(levs),
            "expectancy_r": sum(r_list)/nb_trades
        }

def infer_trade_profile(t: Trade) -> str:
    if t.smt_tag and t.tp2_hit and t.multi_signal_confirmed:
        return "momentum"
    elif "sweep" in t.smt_tag.lower() or t.tp1_hit and not t.tp2_hit:
        return "liquidity_sweep"
    elif not t.tp1_hit and not t.tp2_hit:
        return "failed"
    return "neutral"

def get_daily_pnl_summary(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "date": t.entry_time.date(),
        "pnl": t.pnl,
        "is_win": t.pnl > 0,
        "is_loss": t.pnl < 0
    } for t in trades])
    return df.groupby("date").agg(
        trades=("pnl", "count"),
        pnl_total=("pnl", "sum"),
        winrate_pct=("is_win", lambda x: 100 * x.sum() / len(x)),
    ).reset_index()

def write_daily_summary(stats_by_symbol: Dict[str, BacktestStats], run_id: str) -> None:
    for symbol, stats in stats_by_symbol.items():
        daily_df = get_daily_pnl_summary(stats.trades)
        if not daily_df.empty:
            path = os.path.join("journal", f"daily_summary_{symbol.replace('=','_')}_{run_id}.csv")
            daily_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[+] Saved daily summary: {path}")

def export_json(stats_by_symbol: Dict[str, BacktestStats], run_id: str) -> None:
    for symbol, stats in stats_by_symbol.items():
        if not stats.trades:
            continue
        trades_data = [t.__dict__ for t in stats.trades]
        path = os.path.join("journal", f"journal_{symbol.replace('=','_')}_{run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trades_data, f, indent=2, default=str)
        print(f"[+] Saved JSON journal: {path}")

def plot_equity_curve(stats: BacktestStats, run_id: str) -> None:
    if stats.equity_curve.empty:
        return
    plt.figure(figsize=(10, 5))
    stats.equity_curve.plot(label="Equity Curve")
    if stats.drawdown_curve is not None:
        stats.drawdown_curve.plot(label="Drawdown", linestyle="--")
    plt.title(f"Equity Curve - {stats.symbol}")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.legend()
    plt.grid(True)
    path = os.path.join("journal", f"equity_curve_{stats.symbol.replace('=','_')}_{run_id}.png")
    plt.savefig(path)
    plt.close()
    print(f"[+] Saved equity plot: {path}")

def write_trade_journals(stats_by_symbol: Dict[str, BacktestStats], timeframe: str, lookback_days: int, strategy_mode: str, risk_profile: str) -> None:
    os.makedirs("journal", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for symbol, stats in stats_by_symbol.items():
        if not stats.trades:
            continue
        for t in stats.trades:
            t.profile = infer_trade_profile(t)
        df = pd.DataFrame([{
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "duration_min": (t.exit_time - t.entry_time).total_seconds() / 60,
            "day_of_week": t.entry_time.strftime("%A"),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_price": t.stop_price,
            "tp_price": t.tp_price,
            "tp1_price": t.tp1_price,
            "tp1_hit": t.tp1_hit,
            "tp2_hit": t.tp2_hit,
            "size": t.size,
            "risk_amount": t.risk_amount,
            "risk_pct": t.risk_pct,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "r_multiple": t.r_multiple,
            "leverage": t.leverage,
            "htf_bias": t.htf_bias,
            "smt_tag": t.smt_tag,
            "context": t.context,
            "grade": t.grade,
            "setup_score": t.setup_score,
            "market_regime": t.market_regime,
            "multi_signal": t.multi_signal_confirmed,
            "env_choppy": t.environment_choppy,
            "setup_tags": ",".join(t.setup_tags),
            "profile": t.profile,
            "day_type": t.day_type  # ✅ NEW EXPORT
        } for t in stats.trades])
        path = os.path.join("journal", f"journal_{symbol.replace('=','_')}_{run_id}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[+] Saved trade journal: {path}")
        plot_equity_curve(stats, run_id)
    write_daily_summary(stats_by_symbol, run_id)
    export_json(stats_by_symbol, run_id)
    print_summary(stats_by_symbol)

def print_summary(stats_by_symbol: Dict[str, BacktestStats]) -> None:
    print("\n=== Résumé des performances ===")
    for symbol, stats in stats_by_symbol.items():
        m = stats.compute_metrics()
        print(f"\n>> {symbol}")
        print(f"  Trades: {m['nb_trades']} | Winrate: {m['winrate']:.1f}% | PnL: {m['pnl_total']:.2f}")
        print(f"  Expectancy: {m['expectancy_r']:.2f}R | Max DD: {m['max_dd']:.2f} | Leverage max: {m['max_leverage']:.1f}x")

        # ✅ NEW: Synthèse day_type
        dt_counts = Counter(t.day_type for t in stats.trades if t.day_type)
        if dt_counts:
            print(f"  Journée types : " + " | ".join(f"{k}:{v}" for k, v in dt_counts.items()))

def compute_health_flags(
    stats_by_symbol: Dict[str, BacktestStats],
    metrics_by_symbol: Dict[str, dict]
) -> tuple[list[str], list[str]]:
    """
    Analyse les résultats globaux et remonte des signaux d'alerte si nécessaire.
    Retourne deux listes : (console_flags, discord_flags)
    """
    console_flags = []
    discord_flags = []

    for symbol, metrics in metrics_by_symbol.items():
        winrate = metrics.get("winrate", 0)
        expectancy = metrics.get("expectancy_r", 0)
        max_dd = metrics.get("max_dd", 0)
        trades = metrics.get("nb_trades", 0)

        if trades == 0:
            msg = f"{symbol}: Aucun trade exécuté."
            console_flags.append(msg)
            discord_flags.append(msg)
        if winrate < 40:
            msg = f"{symbol}: Winrate faible ({winrate:.1f}%)"
            console_flags.append(msg)
            discord_flags.append(msg)
        if expectancy < 0:
            msg = f"{symbol}: Expectancy négatif ({expectancy:.2f}R)"
            console_flags.append(msg)
            discord_flags.append(msg)
        if max_dd > 200:
            msg = f"{symbol}: Drawdown élevé ({max_dd:.2f}€)"
            console_flags.append(msg)
            discord_flags.append(msg)

    return console_flags, discord_flags
