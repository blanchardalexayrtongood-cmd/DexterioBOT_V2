import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.engine import run_strategy_on_symbol
from core.journal import write_trade_journals, BacktestStats
from core.config import load_config
from utils.report import print_console_report, save_equity_curve_png
from core.data_loader import download_batch_ohlc
import pandas as pd


def main():
    """Backtest multi-actifs pour DexterioBOT_V2.

    - Utilise `config.json` pour récupérer : symboles, timeframe, lookback, capital, risque.
    - Lance `run_strategy_on_symbol` pour chaque actif, avec capital partagé.
    - Applique la logique SMT via `peer_df` (ES <-> NQ).
    - Calcule en plus une courbe d'equity globale PORTFOLIO en agrégeant tous les symboles.
    """
    cfg = load_config("config.json")

    symbols = cfg.symbols or []
    timeframe = cfg.timeframe
    lookback_days = cfg.lookback_days
    initial_capital = cfg.initial_capital
    strategy_mode = cfg.strategy_mode
    risk_profile = cfg.risk_profile
    risk_cfg = cfg.risk_params
    killzone_only = cfg.session.get("killzone_only", True)

    if not symbols:
        print("[!] Aucun symbole configuré dans config.json (clé 'symbols').")
        return

    print("\n=== DEXTERIOBOT V2 BACKTEST ===")
    print(f"Mode           : {cfg.mode}")
    print(f"Stratégie      : {strategy_mode}")
    print(f"Profil risque  : {risk_profile}")
    print(f"Symbols        : {', '.join(symbols)}")
    print(f"Timeframe      : {timeframe}")
    print(f"Lookback (j)   : {lookback_days}")
    print(f"Capital init.  : {initial_capital:.2f}")
    print(f"Killzone only  : {killzone_only}")
    print("Risque params  :", risk_cfg)
    print("================================\n")

    # 1) Télécharger toutes les données d'un coup
    print("[+] Téléchargement des données OHLC...")
    ohlc_map = download_batch_ohlc(symbols, timeframe, lookback_days)

    # Filtrer les symboles effectivement disponibles
    available_symbols = [s for s in symbols if s in ohlc_map and not ohlc_map[s].empty]
    if not available_symbols:
        print("[!] Aucune donnée téléchargée pour les symboles demandés.")
        return

    # Capital par sous-stratégie (un sous-compte par symbole)
    n = len(available_symbols)
    capital_per_symbol = initial_capital / n

    stats_by_symbol = {}
    all_trades = []

    # 2) Boucle sur chaque symbole avec support SMT via peer_df
    for symbol in available_symbols:
        df = ohlc_map[symbol]
        if df is None or df.empty:
            print(f"[!] Données vides pour {symbol}, on saute.")
            continue

        # Peer logique pour SMT (ES <-> NQ)
        peer_df = None
        if symbol == "NQ=F" and "ES=F" in ohlc_map:
            peer_df = ohlc_map["ES=F"]
        elif symbol == "ES=F" and "NQ=F" in ohlc_map:
            peer_df = ohlc_map["NQ=F"]

        print(f"\n▶ Backtest {symbol} (capital alloué: {capital_per_symbol:.2f})")

        stats = run_strategy_on_symbol(
            symbol=symbol,
            df=df,
            strategy_mode=strategy_mode,
            risk_cfg=risk_cfg,
            initial_capital=capital_per_symbol,
            peer_df=peer_df,
            killzone_only=killzone_only,
            use_scorer_filters=True,
        )

        metrics = stats.compute_metrics()
        print_console_report(stats, metrics)
        save_equity_curve_png(stats)
        write_trade_journals(
            {symbol: stats},
            timeframe,
            lookback_days,
            strategy_mode,
            risk_profile,
        )

        stats_by_symbol[symbol] = stats
        all_trades.extend(stats.trades)

    # 3) Si plusieurs symboles, construire une courbe d'equity PORTFOLIO globale
    if len(stats_by_symbol) > 1:
        print("\n=== Résumé PORTFOLIO (tous symboles confondus) ===")
        equity_cols = []
        for sym, st in stats_by_symbol.items():
            if st.equity_curve is not None and not st.equity_curve.empty:
                s = st.equity_curve.rename(sym)
                equity_cols.append(s)

        if equity_cols:
            eq_df = pd.concat(equity_cols, axis=1).sort_index().ffill()
            portfolio_equity = eq_df.sum(axis=1)

            # Calcul drawdown %
            peak = portfolio_equity.cummax()
            dd_pct = (peak - portfolio_equity) / peak * 100.0

            portfolio_stats = BacktestStats(
                symbol="PORTFOLIO",
                initial_capital=initial_capital,
                final_capital=float(portfolio_equity.iloc[-1]),
                trades=all_trades,
                equity_curve=portfolio_equity,
            )
            portfolio_stats.drawdown_curve = dd_pct
            portfolio_stats.max_drawdown_pct = float(dd_pct.max())

            port_metrics = portfolio_stats.compute_metrics()
            print_console_report(portfolio_stats, port_metrics)
            save_equity_curve_png(portfolio_stats)
            # Journal optionnel pour le portefeuille global
            write_trade_journals(
                {"PORTFOLIO": portfolio_stats},
                timeframe,
                lookback_days,
                strategy_mode,
                risk_profile,
            )
        else:
            print("[~] Impossible de construire la courbe d'equity globale (courbes individuelles vides).")


if __name__ == "__main__":
    main()
