"""Watchlist Manager — persistent JSON-based watchlist."""
import json
import os
import sys
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.helpers import safe_json_read, safe_json_write
from config.paths import WATCHLIST_PATH

TW_TZ = timezone(timedelta(hours=8))

class WatchlistManager:
    def __init__(self):
        self.path = WATCHLIST_PATH
        self._data = self._load()

    def _load(self):
        data = safe_json_read(self.path)
        return data if data is not None else {"stocks": [], "groups": ["預設", "AI族", "存股", "觀察中"]}

    def _save(self):
        safe_json_write(self.path, self._data)

    def get_all(self):
        return self._data.get("stocks", [])

    def get_groups(self):
        return self._data.get("groups", ["預設"])

    def add(self, ticker, name="", group="預設", notes=""):
        # Check if already exists
        for s in self._data["stocks"]:
            if s["ticker"] == ticker:
                return False  # already exists
        self._data["stocks"].append({
            "ticker": ticker,
            "name": name,
            "group": group,
            "notes": notes,
            "added_date": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M"),
        })
        self._save()
        return True

    def remove(self, ticker):
        self._data["stocks"] = [s for s in self._data["stocks"] if s["ticker"] != ticker]
        self._save()

    def update_group(self, ticker, new_group):
        for s in self._data["stocks"]:
            if s["ticker"] == ticker:
                s["group"] = new_group
                break
        self._save()

    def add_group(self, group_name):
        if group_name not in self._data["groups"]:
            self._data["groups"].append(group_name)
            self._save()

    def count(self):
        return len(self._data.get("stocks", []))
