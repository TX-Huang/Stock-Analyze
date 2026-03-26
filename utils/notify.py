"""
Telegram 通知工具
用於策略回測結果的自動通知，支援多用戶訂閱推送。

設定方式:
1. 在 Telegram 找 @BotFather，建立新 Bot 取得 BOT_TOKEN
2. 取得自己的 Chat ID（找 @userinfobot 或 @RawDataBot）
3. 將以下內容加入 .streamlit/secrets.toml:
   TELEGRAM_BOT_TOKEN = "你的Bot Token"
   TELEGRAM_CHAT_ID = "你的Chat ID"
"""

import os
import time
import requests
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Subscriber storage path
from config.paths import SUBSCRIBERS_PATH

_SUBSCRIBERS_FILE = SUBSCRIBERS_PATH

TW_TZ = timezone(timedelta(hours=8))


def _load_telegram_config():
    """從 secrets.toml 讀取 Telegram 設定"""
    try:
        import toml
        secrets = toml.load('.streamlit/secrets.toml')
        token = secrets.get('TELEGRAM_BOT_TOKEN', '')
        chat_id = secrets.get('TELEGRAM_CHAT_ID', '')
        if token and chat_id:
            return token, chat_id
    except Exception:
        pass

    # Fallback: 嘗試 streamlit secrets
    try:
        import streamlit as st
        token = st.secrets.get('TELEGRAM_BOT_TOKEN', '')
        chat_id = st.secrets.get('TELEGRAM_CHAT_ID', '')
        if token and chat_id:
            return token, chat_id
    except Exception:
        pass

    return None, None


def _get_bot_token():
    """只取得 bot token（不需要 chat_id）。"""
    try:
        import toml
        secrets = toml.load('.streamlit/secrets.toml')
        token = secrets.get('TELEGRAM_BOT_TOKEN', '')
        if token:
            return token
    except Exception:
        pass
    try:
        import streamlit as st
        token = st.secrets.get('TELEGRAM_BOT_TOKEN', '')
        if token:
            return token
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Subscriber management
# ---------------------------------------------------------------------------

def get_subscribers():
    """
    讀取訂閱者清單。

    Returns:
        list[dict]: 每個 dict 包含 {chat_id, name, subscribed_at}
    """
    from utils.helpers import safe_json_read
    data = safe_json_read(_SUBSCRIBERS_FILE, default=[])
    if not isinstance(data, list):
        return []
    return data


def add_subscriber(chat_id, name=""):
    """
    新增訂閱者。如果 chat_id 已存在則更新 name。

    Args:
        chat_id: Telegram chat ID (str or int)
        name: 訂閱者名稱（選填）

    Returns:
        bool: 是否為新增（True）或更新（False）
    """
    from utils.helpers import safe_json_read, safe_json_write
    subscribers = safe_json_read(_SUBSCRIBERS_FILE, default=[])
    if not isinstance(subscribers, list):
        subscribers = []

    chat_id_str = str(chat_id)

    # Check if already exists
    for sub in subscribers:
        if str(sub.get('chat_id', '')) == chat_id_str:
            if name:
                sub['name'] = name
            safe_json_write(_SUBSCRIBERS_FILE, subscribers)
            logger.info(f"訂閱者已更新: {chat_id_str} ({name})")
            return False

    # New subscriber
    subscribers.append({
        'chat_id': chat_id_str,
        'name': name,
        'subscribed_at': datetime.now(TW_TZ).isoformat(),
    })
    safe_json_write(_SUBSCRIBERS_FILE, subscribers)
    logger.info(f"新增訂閱者: {chat_id_str} ({name})")
    return True


def remove_subscriber(chat_id):
    """
    移除訂閱者。

    Args:
        chat_id: Telegram chat ID (str or int)

    Returns:
        bool: 是否成功移除（找不到回傳 False）
    """
    from utils.helpers import safe_json_read, safe_json_write
    subscribers = safe_json_read(_SUBSCRIBERS_FILE, default=[])
    if not isinstance(subscribers, list):
        return False

    chat_id_str = str(chat_id)
    original_len = len(subscribers)
    subscribers = [s for s in subscribers if str(s.get('chat_id', '')) != chat_id_str]

    if len(subscribers) < original_len:
        safe_json_write(_SUBSCRIBERS_FILE, subscribers)
        logger.info(f"已移除訂閱者: {chat_id_str}")
        return True

    logger.warning(f"訂閱者不存在: {chat_id_str}")
    return False


def _send_to_chat(token, chat_id, message, parse_mode="Markdown"):
    """
    向指定 chat_id 發送訊息（內部函數）。

    Returns:
        bool: 是否成功
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            logger.error(f"Telegram 發送失敗 (chat={chat_id}): {resp.status_code} - {resp.text}")
            # Retry without parse_mode (in case of markdown issues)
            if parse_mode:
                payload.pop("parse_mode")
                resp2 = requests.post(url, json=payload, timeout=10)
                if resp2.status_code == 200:
                    return True
            return False
    except Exception as e:
        logger.error(f"Telegram 發送錯誤 (chat={chat_id}): {e}")
        return False


def send_to_all_subscribers(message, parse_mode="Markdown"):
    """
    發送訊息給所有訂閱者，帶速率限制（每則間隔 50ms）。

    Args:
        message: 要發送的文字訊息
        parse_mode: 解析模式 ("Markdown" or "HTML")

    Returns:
        dict: {sent: int, failed: int, total: int}
    """
    token = _get_bot_token()
    if not token:
        logger.warning("Telegram 未設定 (缺少 TELEGRAM_BOT_TOKEN)，無法群發")
        print("[NOTIFY] Telegram 未設定，訊息僅顯示在本地:")
        print(message)
        return {'sent': 0, 'failed': 0, 'total': 0}

    subscribers = get_subscribers()
    if not subscribers:
        logger.info("無訂閱者，跳過群發")
        return {'sent': 0, 'failed': 0, 'total': 0}

    sent = 0
    failed = 0

    for i, sub in enumerate(subscribers):
        chat_id = sub.get('chat_id')
        if not chat_id:
            failed += 1
            continue

        success = _send_to_chat(token, chat_id, message, parse_mode)
        if success:
            sent += 1
        else:
            failed += 1

        # Rate limiting: 50ms between messages to avoid Telegram API limits
        # (Telegram allows ~30 msgs/sec, 50ms = 20 msgs/sec with margin)
        if i < len(subscribers) - 1:
            time.sleep(0.05)

    logger.info(f"群發完成: {sent}/{len(subscribers)} 成功, {failed} 失敗")
    return {'sent': sent, 'failed': failed, 'total': len(subscribers)}


def send_telegram(message, parse_mode="Markdown"):
    """
    發送 Telegram 訊息（向預設 CHAT_ID）。
    保持向後相容。

    Args:
        message: 要發送的文字訊息
        parse_mode: 解析模式 ("Markdown" or "HTML")

    Returns:
        bool: 是否發送成功
    """
    token, chat_id = _load_telegram_config()
    if not token or not chat_id:
        logger.warning("Telegram 未設定 (缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID)")
        print("[NOTIFY] Telegram 未設定，訊息僅顯示在本地:")
        print(message)
        return False

    return _send_to_chat(token, chat_id, message, parse_mode)


def format_backtest_report(stats, trades, version="V3"):
    """
    格式化回測報告為 Telegram 訊息。

    Args:
        stats: report.get_stats() 的結果
        trades: report.get_trades() 的結果
        version: 策略版本號

    Returns:
        str: 格式化後的訊息文字
    """
    if hasattr(stats, 'to_dict'):
        s = stats.to_dict()
    elif hasattr(stats, '__getitem__'):
        s = stats
    else:
        s = {}

    cagr = s.get('cagr', 0) * 100
    max_dd = s.get('max_drawdown', 0) * 100
    sharpe = s.get('daily_sharpe', 0)
    sortino = s.get('daily_sortino', 0)
    win_ratio = s.get('win_ratio', 0) * 100
    total_return = s.get('total_return', 0) * 100
    calmar = s.get('calmar', 0)
    ytd = s.get('ytd', 0) * 100

    trade_count = len(trades) if trades is not None else 0
    avg_period = 0
    median_period = 0
    if trades is not None and not trades.empty and 'period' in trades.columns:
        avg_period = trades['period'].mean()
        median_period = trades['period'].median()

    # Emoji indicators
    cagr_emoji = "🟢" if cagr > 15 else ("🟡" if cagr > 8 else "🔴")
    dd_emoji = "🟢" if max_dd > -30 else ("🟡" if max_dd > -40 else "🔴")
    sharpe_emoji = "🟢" if sharpe > 0.8 else ("🟡" if sharpe > 0.5 else "🔴")
    win_emoji = "🟢" if win_ratio > 50 else ("🟡" if win_ratio > 40 else "🔴")

    msg = f"""📊 *Isaac {version} 回測報告*

{cagr_emoji} CAGR: `{cagr:.2f}%`
{dd_emoji} Max DD: `{max_dd:.2f}%`
{sharpe_emoji} Sharpe: `{sharpe:.2f}` | Sortino: `{sortino:.2f}`
{win_emoji} Win Ratio: `{win_ratio:.1f}%`

📈 Total Return: `{total_return:.1f}%`
📅 YTD: `{ytd:.1f}%`
⚖️ Calmar: `{calmar:.2f}`

🔢 Trades: `{trade_count}`
⏱️ Avg Hold: `{avg_period:.1f}` days (median: `{median_period:.0f}`)
"""
    return msg.strip()


def format_optimization_diff(before_stats, after_stats, change_desc=""):
    """
    格式化優化前後的對比報告。

    Args:
        before_stats: 修改前的 stats
        after_stats: 修改後的 stats
        change_desc: 修改描述

    Returns:
        str: 格式化後的對比訊息
    """
    def get(s, key):
        if hasattr(s, 'to_dict'):
            s = s.to_dict()
        return s.get(key, 0)

    cagr_b = get(before_stats, 'cagr') * 100
    cagr_a = get(after_stats, 'cagr') * 100
    dd_b = get(before_stats, 'max_drawdown') * 100
    dd_a = get(after_stats, 'max_drawdown') * 100
    sharpe_b = get(before_stats, 'daily_sharpe')
    sharpe_a = get(after_stats, 'daily_sharpe')
    win_b = get(before_stats, 'win_ratio') * 100
    win_a = get(after_stats, 'win_ratio') * 100

    def arrow(before, after, higher_better=True):
        diff = after - before
        if abs(diff) < 0.01:
            return "➡️"
        if higher_better:
            return "⬆️" if diff > 0 else "⬇️"
        else:
            return "⬆️" if diff < 0 else "⬇️"

    msg = f"""🔄 *策略優化對比報告*

*修改內容:* {change_desc}

| 指標 | Before | After | |
|------|--------|-------|---|
| CAGR | `{cagr_b:.2f}%` | `{cagr_a:.2f}%` | {arrow(cagr_b, cagr_a)} |
| Max DD | `{dd_b:.2f}%` | `{dd_a:.2f}%` | {arrow(dd_b, dd_a, False)} |
| Sharpe | `{sharpe_b:.2f}` | `{sharpe_a:.2f}` | {arrow(sharpe_b, sharpe_a)} |
| Win | `{win_b:.1f}%` | `{win_a:.1f}%` | {arrow(win_b, win_a)} |
"""
    return msg.strip()


_REGIME_ZH = {
    'strong_bull': '強多頭 🟢',
    'weak_bull': '弱多頭 🟡',
    'sideways': '盤整 ⚪',
    'weak_bear': '弱空頭 🟠',
    'strong_bear': '強空頭 🔴',
}


def format_v4_daily_signals(result):
    """
    格式化 V4 每日信號為 Telegram 訊息。

    Args:
        result: daily_v4_scan 的輸出 dict

    Returns:
        str: Telegram Markdown 格式訊息
    """
    date = result.get('date', '')
    regime = result.get('regime', 'unknown')
    regime_label = _REGIME_ZH.get(regime, regime)
    allocations = result.get('allocations', {})
    variant_entries = result.get('variant_new_entries', {})
    strategy_signals = result.get('strategy_signals', {})

    lines = [f"📡 *V4 每日信號掃描*  `{date}`"]
    lines.append(f"🌐 市場狀態: *{regime_label}*")
    lines.append("")

    # 各 variant 的配置
    for vk in ['V4.0', 'V4.1', 'V4.2']:
        alloc = allocations.get(vk, {})
        weights = alloc.get('weights', {})
        if not weights:
            continue
        w_str = ' | '.join(f"{k} {int(v*100)}%" for k, v in weights.items() if v > 0.01)
        lines.append(f"*{vk}*: {w_str}")

    lines.append("")

    # 新進場股票（重點）
    has_entries = False
    for vk in ['V4.0', 'V4.1', 'V4.2']:
        entries = variant_entries.get(vk, [])
        if not entries:
            continue
        has_entries = True
        lines.append(f"🆕 *{vk} 新進場:*")
        for e in entries[:10]:
            name = e.get('name', '')
            ticker = e['ticker']
            price = e.get('price', 0)
            strat = e.get('strategy', '')
            weight = e.get('weight', 0)
            score = e.get('score', 0)
            price_str = f"${price:,.1f}" if price else ""
            lines.append(
                f"  `{ticker}` {name} {price_str} "
                f"(Score:{score}, {strat} {weight}%)"
            )
        lines.append("")

    if not has_entries:
        lines.append("✅ 今日無新進場信號")
        lines.append("")

    # 各子策略持倉概覽
    lines.append("📊 *子策略持倉概覽:*")
    for strat_name, sig in strategy_signals.items():
        n_h = sig.get('n_holdings', 0)
        n_in = sig.get('n_entered', 0)
        n_out = sig.get('n_exited', 0)
        lines.append(f"  {strat_name}: {n_h}檔 (+{n_in} -{n_out})")

    return '\n'.join(lines)


if __name__ == "__main__":
    # 測試用
    print("Testing Telegram notification...")
    send_telegram("🤖 AI Invest HQ 通知系統測試成功！")
