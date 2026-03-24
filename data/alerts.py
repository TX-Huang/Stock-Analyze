"""Alert Manager — 價格與技術指標警報。"""
import json, os, sys
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import safe_json_read, safe_json_write
from config.paths import ALERTS_PATH

TW_TZ = timezone(timedelta(hours=8))

ALERT_TYPES = {
    "price_above": "價格突破",
    "price_below": "價格跌破",
    "rsi_above": "RSI 超買",
    "rsi_below": "RSI 超賣",
    "volume_spike": "爆量",
}

class AlertManager:
    def __init__(self):
        self.path = ALERTS_PATH
        self._data = self._load()

    def _load(self):
        data = safe_json_read(self.path)
        return data if data is not None else {"alerts": []}

    def _save(self):
        safe_json_write(self.path, self._data)

    def get_all(self):
        return self._data.get("alerts", [])

    def add_alert(self, ticker, name, alert_type, threshold):
        _ts = datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S")
        _uid = str(id(ticker))[-4:]
        alert = {
            "id": _ts + "_" + _uid,
            "ticker": ticker,
            "name": name,
            "type": alert_type,
            "threshold": threshold,
            "active": True,
            "created_at": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M"),
            "triggered": False,
            "triggered_at": None,
        }
        self._data["alerts"].append(alert)
        self._save()
        return alert["id"]

    def remove_alert(self, alert_id):
        self._data["alerts"] = [a for a in self._data["alerts"] if a.get("id") != alert_id]
        self._save()

    def toggle_alert(self, alert_id):
        for a in self._data["alerts"]:
            if a.get("id") == alert_id:
                a["active"] = not a["active"]
                break
        self._save()

    def check_alerts(self, provider):
        """Check all active alerts against current data. Returns list of triggered alerts."""
        triggered = []
        for a in self._data["alerts"]:
            if not a.get("active") or a.get("triggered"):
                continue
            try:
                ticker = a["ticker"]
                df = provider.get_historical_data(ticker, period="60d", interval="1d")
                if df.empty or len(df) < 1:
                    continue
                current_price = float(df['Close'].iloc[-1])
                vol_avg = float(df['Volume'].tail(20).mean()) if len(df) >= 20 else float(df['Volume'].mean())
                current_vol = float(df['Volume'].iloc[-1])

                alert_type = a["type"]
                threshold = float(a["threshold"])
                fired = False

                if alert_type == "price_above" and current_price >= threshold:
                    fired = True
                elif alert_type == "price_below" and current_price <= threshold:
                    fired = True
                elif alert_type == "rsi_above":
                    from analysis.indicators import calculate_rsi
                    rsi = calculate_rsi(df['Close'])
                    if rsi.iloc[-1] >= threshold:
                        fired = True
                elif alert_type == "rsi_below":
                    from analysis.indicators import calculate_rsi
                    rsi = calculate_rsi(df['Close'])
                    if rsi.iloc[-1] <= threshold:
                        fired = True
                elif alert_type == "volume_spike":
                    if vol_avg > 0 and current_vol >= vol_avg * threshold:
                        fired = True

                if fired:
                    a["triggered"] = True
                    a["triggered_at"] = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
                    triggered.append({**a, "current_price": current_price})
            except Exception:
                continue

        if triggered:
            self._save()
        return triggered

    def reset_alert(self, alert_id):
        for a in self._data["alerts"]:
            if a.get("id") == alert_id:
                a["triggered"] = False
                a["triggered_at"] = None
                break
        self._save()

    def count_active(self):
        return len([a for a in self._data.get("alerts", []) if a.get("active") and not a.get("triggered")])
