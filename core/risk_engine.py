# core/risk_engine.py

from datetime import datetime, timedelta
import json
import os


class RiskEngine:
    def __init__(self, settings: dict):
        self.max_risk_per_trade = settings.get("risk_per_trade_max", 0.01)
        self.min_risk_per_trade = settings.get("risk_per_trade_min", 0.005)
        self.max_losses_per_day = settings.get("max_losses_per_day", 2)
        self.pause_duration_hours = settings.get("pause_duration_after_max_loss", 24)
        self.consecutive_wins_for_scale = settings.get("consecutive_wins_for_scaling", 0)

        self.daily_loss_count = 0
        self.last_reset_date = datetime.now().date()
        self.paused_until = None
        self.scaling_unlocked = False

    def reset_daily_counters(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_loss_count = 0
            self.paused_until = None
            self.last_reset_date = today

    def can_take_new_trade(self) -> bool:
        self.reset_daily_counters()
        if self.paused_until and datetime.now() < self.paused_until:
            return False
        if self.daily_loss_count >= self.max_losses_per_day:
            return False
        return True

    def calculate_position_size(self, account_balance, entry_price, stop_price) -> float:
        if None in (account_balance, entry_price, stop_price):
            return 0.0

        price_risk = abs(entry_price - stop_price)
        if price_risk == 0:
            return 0.0

        risk_pct = self.min_risk_per_trade if self.daily_loss_count > 0 else self.max_risk_per_trade
        risk_amount = account_balance * risk_pct
        return round(risk_amount / price_risk, 2)

    def register_trade_result(self, trade_result: dict):
        pnl = trade_result.get("pnl")
        result = trade_result.get("result")

        if pnl is not None:
            if pnl < 0:
                self.daily_loss_count += 1
        elif result == "loss":
            self.daily_loss_count += 1

        if self.daily_loss_count >= self.max_losses_per_day:
            self.paused_until = datetime.now() + timedelta(hours=self.pause_duration_hours)
            self._log_pause_event()

    def _log_pause_event(self):
        log_path = "data/logs/trade_log.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        note = {
            "timestamp": datetime.now().isoformat(),
            "note": f"[RiskEngine] Max daily loss reached. Trading paused for {self.pause_duration_hours}h."
        }

        try:
            if os.path.exists(log_path):
                with open(log_path, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "trades" in data:
                        data["trades"].append(note)
                    elif isinstance(data, list):
                        data.append(note)
                    else:
                        data = [note]
                    f.seek(0)
                    json.dump(data, f, indent=4)
            else:
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump([note], f, indent=4)
        except Exception:
            pass  # Silent fail — logging non bloquant

    def consider_scaling(self, performance_metrics: dict):
        if not self.consecutive_wins_for_scale or not performance_metrics:
            return

        win_days = performance_metrics.get("consecutive_win_days", 0)
        if win_days >= self.consecutive_wins_for_scale and not self.scaling_unlocked:
            self.max_risk_per_trade *= 1.5
            self.scaling_unlocked = True
