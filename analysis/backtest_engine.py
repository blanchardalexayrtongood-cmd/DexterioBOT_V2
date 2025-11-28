# analysis/backtest_engine.py
import json

class BacktestEngine:
    """
    Ce module permet de lire l'historique des trades (trade_log.json) et de produire des statistiques de performance:
    - Taux de réussite (win/loss ratio).
    - Profit factor ou expectancy (espérance de gain moyen par trade).
    - Drawdown maximum.
    - Nombre total de trades, etc.
    """
    def __init__(self, log_path="data/logs/trade_log.json"):
        self.log_path = log_path

    def generate_report(self):
        """ 
        Lit le fichier de log de trades et calcule les métriques de performance.
        Retourne un dictionnaire avec les statistiques, et imprime le rapport de manière lisible.
        """
        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("Trade log not found at path:", self.log_path)
            return None
        except json.JSONDecodeError:
            print("Trade log is empty or invalid JSON.")
            return None

        # Récupérer la liste des trades depuis le log.
        if isinstance(data, dict) and "trades" in data:
            trades = data["trades"]
        elif isinstance(data, list):
            trades = data
        else:
            print("No trade data found in log.")
            trades = []
        if not trades:
            print("No trades to analyze.")
            return None

        total_trades = 0
        wins = 0
        losses = 0
        pnl_list = []
        for trade in trades:
            # On s'attend à ce que chaque trade soit un dict contenant 'pnl' ou 'result'.
            if "pnl" in trade and isinstance(trade["pnl"], (int, float)):
                pnl = trade["pnl"]
            else:
                # Si pas de pnl numérique, on essaie de déduire via 'result'
                result = trade.get("result")
                if result == "win":
                    pnl = 1  # +1 unité arbitraire
                elif result == "loss":
                    pnl = -1
                else:
                    continue  # ignorer si pas de résultat exploitable
            pnl_list.append(pnl)
            total_trades += 1
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
        if total_trades == 0:
            print("No valid trades to analyze.")
            return None

        win_rate = (wins / total_trades) * 100.0
        # Calcul de l'espérance (expectancy): moyenne des PnL
        avg_pnl = sum(pnl_list) / total_trades
        # On peut aussi calculer séparément gain moyen et perte moyenne
        avg_win = sum(p for p in pnl_list if p > 0) / wins if wins > 0 else 0
        avg_loss = sum(p for p in pnl_list if p < 0) / losses if losses > 0 else 0
        # Profit factor: somme des gains / somme des pertes (en valeur absolue)
        profit_factor = -1
        if losses == 0:
            profit_factor = float('inf')  # aucun trade perdant
        else:
            total_gain = sum(p for p in pnl_list if p > 0)
            total_loss = -sum(p for p in pnl_list if p < 0)
            profit_factor = total_gain / total_loss if total_loss != 0 else float('inf')

        # Max drawdown: calculer l'équity curve et trouver le plus grand repli depuis un sommet
        equity = 0.0
        peak_equity = 0.0
        max_drawdown = 0.0
        for pnl in pnl_list:
            equity += pnl
            if equity > peak_equity:
                peak_equity = equity
            drawdown = peak_equity - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Préparer le rapport
        report = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_percent": round(win_rate, 2),
            "average_pnl": round(avg_pnl, 2),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "Infinity",
            "max_drawdown": round(max_drawdown, 2)
        }
        # Afficher le rapport de manière lisible
        print("=== Backtest Performance Report ===")
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins} | Losses: {losses} | Win Rate: {report['win_rate_percent']}%")
        print(f"Average PnL per trade: {report['average_pnl']}")
        print(f"Average Win: {report['average_win']} | Average Loss: {report['average_loss']}")
        print(f"Profit Factor: {report['profit_factor']}")
        print(f"Max Drawdown: {report['max_drawdown']}")
        return report
