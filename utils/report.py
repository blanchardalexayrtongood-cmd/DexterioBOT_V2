import os
import matplotlib.pyplot as plt
from core.journal import BacktestStats, Trade

def print_console_report(stats: BacktestStats, metrics: dict):
    print(f"\n📊 Résumé des performances pour {stats.symbol}")
    print("-" * 50)
    print(f"Nombre de trades       : {metrics['nb_trades']}")
    print(f"PnL total              : {metrics['pnl_total']:.2f}€ ({metrics['pnl_pct']:.2f}%)")
    print(f"Winrate                : {metrics['winrate']:.1f}%")
    print(f"Expectancy (R)         : {metrics['expectancy_r']:.2f}")
    print(f"Trade moyen (PnL)      : {metrics['pnl_mean']:.2f}€")
    print(f"Meilleur trade         : {metrics['best_trade']:.2f}€")
    print(f"Pire trade             : {metrics['worst_trade']:.2f}€")
    print(f"Trades gagnants        : {metrics['nb_winners']} | Perdants : {metrics['nb_losers']} | BE : {metrics['nb_be']}")
    print(f"Max Drawdown           : {metrics['max_dd']:.2f}€")
    print(f"Durée backtest (jours) : {metrics['nb_days']}")
    print(f"PnL moyen / jour       : {metrics['pnl_per_day']:.2f}€")
    print(f"Risque max / trade     : {metrics['max_risk_pct']:.2f}%")
    print(f"Levier max             : {metrics['max_leverage']:.2f}x")
    print(f"Levier moyen           : {metrics['avg_leverage']:.2f}x")
    print("-" * 50)

    if stats.trades and hasattr(stats.trades[0], "day_type"):
        print("\n🧭 Breakdown par type de journée")
        daytype_data = {}
        for t in stats.trades:
            dt = getattr(t, "day_type", "unknown")
            if dt not in daytype_data:
                daytype_data[dt] = {"count": 0, "wins": 0, "r": 0.0}
            daytype_data[dt]["count"] += 1
            if t.r_multiple > 0:
                daytype_data[dt]["wins"] += 1
            daytype_data[dt]["r"] += t.r_multiple

        for dt, d in daytype_data.items():
            wr = d["wins"] / d["count"] * 100
            avg_r = d["r"] / d["count"]
            print(f"{dt:>14} : {d['count']} trades | Winrate {wr:.1f}% | Avg R {avg_r:+.2f}")
        print("-" * 50)


def save_equity_curve_png(stats: BacktestStats):
    if stats.equity_curve.empty:
        print(f"[!] Equity curve vide pour {stats.symbol}")
        return

    os.makedirs("plots", exist_ok=True)
    fname = f"plots/equity_{stats.symbol.replace('=','_')}.png"

    plt.figure(figsize=(10, 4))
    stats.equity_curve.plot(label="Equity (€)", linewidth=1.5)
    if stats.drawdown_curve is not None and not stats.drawdown_curve.empty:
        stats.drawdown_curve.plot(label="Drawdown", linestyle="--", alpha=0.7)
    plt.title(f"Equity Curve: {stats.symbol}")
    plt.xlabel("Temps")
    plt.ylabel("Capital (€)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"[✓] Sauvegardé equity plot: {fname}")
