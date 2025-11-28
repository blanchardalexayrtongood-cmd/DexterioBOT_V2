import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.engine import run_strategy_on_symbol
from core.journal import write_trade_journals
from core.risk import RISK_PROFILES
from core.config import load_config
from utils.report import print_console_report, save_equity_curve_png
from core.data_loader import download_ohlc
import pandas as pd


def main():
    cfg = load_config("config.json")

    symbols = cfg.symbols
    timeframe = cfg.timeframe
    lookback_days = cfg.lookback_days
    strategy_mode = cfg.strategy_mode
    risk_profile = cfg.risk_profile
    session_cfg = cfg.session
    initial_capital = cfg.initial_capital

    risk_cfg = RISK_PROFILES.get(risk_profile)
    if risk_cfg is None:
        raise ValueError(f"[!] Profil de risque inconnu: {risk_profile}")

    # Téléchargement OHLC
    dfs = {}
    for sym in symbols:
        print(f"[+] Loading data: {sym}")
        dfs[sym] = download_ohlc(sym, timeframe, lookback_days)

    # Execution par symbole
    for symbol in symbols:
        df = dfs[symbol]
        peer_df = None

        # SMT: NQ utilise ES comme référence
        if symbol == "NQ=F" and "ES=F" in dfs:
            peer_df = dfs["ES=F"]

        stats = run_strategy_on_symbol(
            symbol=symbol,
            df=df,
            strategy_mode=strategy_mode,
            risk_cfg=risk_cfg,
            initial_capital=initial_capital,
            peer_df=peer_df,
            killzone_only=True,
            use_scorer_filters=True,
        )

        metrics = stats.compute_metrics()
        print_console_report(stats, metrics)
        save_equity_curve_png(stats)
        write_trade_journals({symbol: stats}, timeframe, lookback_days, strategy_mode, risk_profile)


if __name__ == "__main__":
    main()
