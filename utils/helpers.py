import json
import os
import re
import tempfile
import logging
from datetime import timedelta


def safe_json_write(path, data, **kwargs):
    """Atomic JSON write: writes to temp file then renames to prevent corruption.

    Args:
        path: Target file path.
        data: Data to serialize as JSON.
        **kwargs: Extra keyword arguments passed to json.dump (e.g. default=str).
    """
    dir_name = os.path.dirname(path) or '.'
    os.makedirs(dir_name, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def safe_json_read(path, default=None):
    """Read JSON file safely, returning *default* on missing or corrupt files.

    Args:
        path: File path to read.
        default: Value to return when the file is missing or unreadable.
    Returns:
        Parsed JSON data, or *default*.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"safe_json_read: failed to read {path}: {e}")
        return default


def robust_json_extract(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass
    return None


def validate_ticker(ticker, market):
    ticker = str(ticker).strip().upper()
    if "台股" in market:
        # 支援純數字 (2330)、帶字母的 ETF (00631L)、帶後綴的 (.TW/.TWO)
        if re.match(r'^\d{4,6}[A-Z]?$', ticker):
            return True
        if ticker.endswith(".TW") or ticker.endswith(".TWO"):
            return True
        return False
    else:
        if re.match(r'^[A-Z]{1,6}$', ticker):
            return True
        return False


def get_default_sector_map_full(market):
    if "台股" in market:
        return {
            "💾 記憶體": ["2408", "2344", "2337", "8299", "3260"],
            "🤖 AI 伺服器": ["2317", "2382", "3231", "2356", "6669"],
            "❄️ 散熱模組": ["3017", "3324", "2421", "3013"],
            "🚢 航運": ["2603", "2609", "2615", "2606"],
            "💎 權值股": ["2330", "2454", "3035", "3443"]
        }
    else:
        return {"👑 Mag 7": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]}


def get_fallback_supply_chain(keyword, market):
    k = keyword.lower()
    if "台股" in market:
        if "記憶體" in k or "dram" in k:
            return {"IC設計": {"3006": "晶豪科", "8299": "群聯"}, "製造": {"2408": "南亞科", "2344": "華邦電"}, "封測": {"6239": "力成", "8150": "南茂"}}
        if "機器人" in k or "robot" in k:
            return {"關鍵零組件": {"2049": "上銀", "1590": "亞德客"}, "系統整合": {"2317": "鴻海", "2357": "華碩"}}
    return None


def get_date_from_index(idx, df, is_weekly):
    """Helper to project future dates from index"""
    if idx < 0:
        idx = 0
    if idx < len(df):
        return df.index[int(idx)]
    else:
        extra_units = idx - len(df) + 1
        last_date = df.index[-1]
        days_per_unit = 7 if is_weekly else 1.4
        delta = timedelta(days=int(extra_units * days_per_unit))
        return last_date + delta
