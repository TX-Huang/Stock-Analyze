"""
Telegram 通知工具
用於策略回測結果的自動通知。

設定方式:
1. 在 Telegram 找 @BotFather，建立新 Bot 取得 BOT_TOKEN
2. 取得自己的 Chat ID（找 @userinfobot 或 @RawDataBot）
3. 將以下內容加入 .streamlit/secrets.toml:
   TELEGRAM_BOT_TOKEN = "你的Bot Token"
   TELEGRAM_CHAT_ID = "你的Chat ID"
"""

import requests
import logging

logger = logging.getLogger(__name__)


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


def send_telegram(message, parse_mode="Markdown"):
    """
    發送 Telegram 訊息。

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

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram 訊息發送成功")
            return True
        else:
            logger.error(f"Telegram 發送失敗: {resp.status_code} - {resp.text}")
            # Retry without parse_mode (in case of markdown issues)
            if parse_mode:
                payload.pop("parse_mode")
                resp2 = requests.post(url, json=payload, timeout=10)
                if resp2.status_code == 200:
                    return True
            return False
    except Exception as e:
        logger.error(f"Telegram 發送錯誤: {e}")
        return False


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


if __name__ == "__main__":
    # 測試用
    print("Testing Telegram notification...")
    send_telegram("🤖 AI Invest HQ 通知系統測試成功！")
