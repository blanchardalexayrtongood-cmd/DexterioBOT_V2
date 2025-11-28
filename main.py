import sys
import os
import traceback
import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from core.data_loader import download_ohlc
from core.engine import run_strategy_on_symbol
from core.journal import (
    write_trade_journals,
    compute_health_flags,
    BacktestStats
)
from core.config import load_config
from utils.report import print_console_report, save_equity_curve_png

def load_discord_webhook_from_env_or_file(default_url: str = "") -> str:
    url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not url and os.path.isfile(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("DISCORD_WEBHOOK_URL"):
                    _, val = line.strip().split("=", 1)
                    return val.strip()
    return url or default_url

def send_discord_message(webhook_url: str, content: str):
    if webhook_url:
        try:
            requests.post(webhook_url, json={"content": content})
        except Exception as e:
            print(f"[!] Discord webhook failed: {e}")

def send_discord_file(webhook_url: str, file_path: str, message: str = ""):
    if webhook_url and os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                data = {"content": message}
                requests.post(webhook_url, data=data, files=files)
        except Exception as e:
            print(f"[!] Failed to upload {file_path} to Discord: {e}")

def summarize_trades(trades, max_count=3) -> str:
    rows = []
    for t in trades[-max_count:]:
        rows.append(
            f"{t.direction.upper()} {t.symbol} | R={t.r_multiple:.2f} | "
            f"{t.entry_time.strftime('%H:%M')}→{t.exit_time.strftime('%H:%M')} | "
            f"PnL={t.pnl:.2f}"
        )
    return "\n".join(rows) if rows else "(aucun trade)"

def generate_r_histogram(all_trades):
    r_values = [t.r_multiple for t in all_trades]
    plt.figure()
    plt.hist(r_values, bins=20, edgecolor='black')
    plt.title("Distribution des R-multiples")
    plt.xlabel("R-multiple")
    plt.ylabel("Nombre de trades")
    plt.grid(True)
    plt.tight_layout()
    output_path = "plots/hist_r_multiples.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_daily_recap(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{"date": t.entry_time.date(), "r": t.r_multiple, "pnl": t.pnl} for t in trades])
    recap = df.groupby("date").agg(trades=("r", "count"), pnl=("pnl", "sum"), total_r=("r", "sum"))
    return recap.reset_index()

def main():
    config = load_config("config.json")
    webhook_url = load_discord_webhook_from_env_or_file(config.discord.get("webhook_url", ""))

    stats_by_symbol = {}
    all_trades = []

    try:
        for symbol in config.symbols:
            print(f"[+] Téléchargement des données: {symbol}")
            df = download_ohlc(symbol, config.timeframe, config.lookback_days)

            stats = run_strategy_on_symbol(
                symbol=symbol,
                df=df,
                strategy_mode=config.strategy_mode,
                risk_cfg=config.risk_params,
                initial_capital=config.initial_capital,
                killzone_only=config.session.get("killzone_only", True)
            )

            metrics = stats.compute_metrics()
            print_console_report(stats, metrics)
            save_equity_curve_png(stats)
            write_trade_journals({symbol: stats}, config.timeframe, config.lookback_days, config.strategy_mode, config.risk_profile)
            stats_by_symbol[symbol] = stats
            all_trades.extend(stats.trades)

            summary = (
                f"📊 `{symbol}` | Trades={metrics['nb_trades']:.0f} | "
                f"Winrate={metrics['winrate']:.1f}% | R={metrics['expectancy_r']:.2f} | "
                f"PnL={metrics['pnl_total']:.2f}"
            )
            trade_details = summarize_trades(stats.trades, 3)
            send_discord_message(webhook_url, summary + "\n```text\n" + trade_details + "\n```")

            send_discord_file(webhook_url, f"plots/equity_{symbol.replace('=','_')}.png", f"📉 Courbe equity {symbol}")
            journal_path = next((f for f in os.listdir("journal") if symbol.replace("=", "_") in f and f.endswith(".csv")), None)
            if journal_path:
                send_discord_file(webhook_url, os.path.join("journal", journal_path), f"📄 Journal trades {symbol}")

            # Résumé par type de journée
            daytype_map = {}
            for t in stats.trades:
                dt = getattr(t, "day_type", None)
                if not dt:
                    continue
                if dt not in daytype_map:
                    daytype_map[dt] = {"count": 0, "wins": 0, "total_r": 0.0}
                daytype_map[dt]["count"] += 1
                if t.r_multiple > 0:
                    daytype_map[dt]["wins"] += 1
                daytype_map[dt]["total_r"] += t.r_multiple

            if daytype_map:
                lines = ["🧭 **Breakdown par type de journée**"]
                for dt, d in daytype_map.items():
                    winrate = 100 * d["wins"] / d["count"]
                    avg_r = d["total_r"] / d["count"]
                    lines.append(f"- `{dt}` : {d['count']} trades | Winrate {winrate:.1f}% | Avg R {avg_r:+.2f}")
                send_discord_message(webhook_url, "\n".join(lines))

        if all_trades:
            best = max(all_trades, key=lambda x: x.r_multiple)
            worst = min(all_trades, key=lambda x: x.r_multiple)
            best_msg = f"🏅 BEST TRADE {best.symbol}: R={best.r_multiple:.2f}, PnL={best.pnl:.2f}, Durée={(best.exit_time - best.entry_time)}"
            worst_msg = f"💀 WORST TRADE {worst.symbol}: R={worst.r_multiple:.2f}, PnL={worst.pnl:.2f}, Durée={(worst.exit_time - worst.entry_time)}"
            send_discord_message(webhook_url, best_msg + "\n" + worst_msg)

            hist_path = generate_r_histogram(all_trades)
            send_discord_file(webhook_url, hist_path, "📊 Histogramme des R-multiples")

            daily_df = generate_daily_recap(all_trades)
            if not daily_df.empty:
                csv_path = "plots/daily_recap.csv"
                daily_df.to_csv(csv_path, index=False)
                send_discord_file(webhook_url, csv_path, "📆 Recap quotidien des performances")

                table = "📅 **Recap quotidien**\n```\nDate       | Trades | PnL    | R\n"
                for _, row in daily_df.iterrows():
                    table += f"{row['date']:<11} | {row['trades']:<6} | {row['pnl']:<6.2f} | {row['total_r']:<5.2f}\n"
                table += "```"
                send_discord_message(webhook_url, table)

        metrics_by_symbol = {sym: st.compute_metrics() for sym, st in stats_by_symbol.items()}
        console_flags, discord_flags = compute_health_flags(stats_by_symbol, metrics_by_symbol)
        for msg in console_flags:
            print("[Flag]", msg)
        if discord_flags:
            alert_msg = "**⚠️ Indicateurs Santé**\n" + "\n".join(f"- {f}" for f in discord_flags)
            send_discord_message(webhook_url, alert_msg)

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        send_discord_message(webhook_url, f"❌ **Erreur critique détectée**\n```py\n{tb}\n```")
        raise

if __name__ == "__main__":
    main()
