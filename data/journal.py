"""Trade Journal — 記錄每筆交易的決策過程與情緒。"""
import json, os, sys
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import safe_json_read, safe_json_write
from config.paths import JOURNAL_PATH

TW_TZ = timezone(timedelta(hours=8))

EMOTION_TAGS = ["按計劃", "FOMO追高", "恐慌賣出", "停損執行", "停利執行", "攤平加碼", "直覺操作", "其他"]

class JournalManager:
    def __init__(self):
        self.path = JOURNAL_PATH
        self._data = self._load()

    def _load(self):
        data = safe_json_read(self.path)
        return data if data is not None else {"entries": []}

    def _save(self):
        safe_json_write(self.path, self._data)

    def get_entries(self):
        return sorted(self._data.get("entries", []), key=lambda x: x.get("date", ""), reverse=True)

    def add_entry(self, date, ticker, name, action, price, shares, reasoning, emotion_tag, outcome_review=""):
        _ts = datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S")
        _uid = str(id(ticker))[-4:]
        entry = {
            "id": _ts + "_" + _uid,
            "date": date,
            "ticker": ticker,
            "name": name,
            "action": action,  # BUY or SELL
            "price": price,
            "shares": shares,
            "reasoning": reasoning,
            "emotion_tag": emotion_tag,
            "outcome_review": outcome_review,
            "created_at": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M"),
        }
        self._data["entries"].append(entry)
        self._save()
        return entry["id"]

    def delete_entry(self, entry_id):
        self._data["entries"] = [e for e in self._data["entries"] if e.get("id") != entry_id]
        self._save()

    def update_review(self, entry_id, review_text):
        for e in self._data["entries"]:
            if e.get("id") == entry_id:
                e["outcome_review"] = review_text
                break
        self._save()

    def get_stats(self):
        entries = self._data.get("entries", [])
        if not entries:
            return {}
        total = len(entries)
        buys = len([e for e in entries if e["action"] == "BUY"])
        sells = len([e for e in entries if e["action"] == "SELL"])
        # Emotion distribution
        emotion_counts = {}
        for e in entries:
            tag = e.get("emotion_tag", "其他")
            emotion_counts[tag] = emotion_counts.get(tag, 0) + 1
        return {"total": total, "buys": buys, "sells": sells, "emotions": emotion_counts}
