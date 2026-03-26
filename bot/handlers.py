"""
Telegram Bot 指令處理器
支援: /scan, /portfolio, /add, /remove, /risk, /daily, /cc + AI 對話
"""
import json
import logging
import os
import sys
import asyncio
import subprocess
import numpy as np
from datetime import datetime

from telegram import Update, constants
from telegram.ext import ContextTypes

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_PATH = os.path.join(PROJECT_ROOT, 'data', 'portfolio.json')
SECRETS_PATH = os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml')
CLAUDE_CLI = os.environ.get(
    'CLAUDE_CLI',
    os.path.join(os.path.expanduser('~'), '.local', 'bin', 'claude.exe'),
)

sys.path.insert(0, PROJECT_ROOT)


# ==========================================
# Typing 狀態心跳 (每 4 秒重發，避免 5 秒自動消失)
# ==========================================

async def _run_with_typing(chat_id, bot, executor_func, *args, progress_msg=None):
    """在背景持續發送 typing 狀態，同時執行耗時任務

    progress_msg: 若提供，每 60 秒發送一條進度訊息 (e.g. "Claude Code 執行中")
    """
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, executor_func, *args)
    elapsed = 0
    last_progress = 0
    while True:
        await bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        done, _ = await asyncio.wait({task}, timeout=4.0)
        if done:
            return task.result()
        elapsed += 4
        if progress_msg and elapsed - last_progress >= 60:
            last_progress = elapsed
            minutes = elapsed // 60
            await bot.send_message(
                chat_id=chat_id,
                text=f"⏳ {progress_msg}... ({minutes}分鐘)"
            )


# ==========================================
# Portfolio 持倉管理
# ==========================================

def _load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"holdings": [], "last_updated": None}


def _save_portfolio(data):
    data['last_updated'] = datetime.now().isoformat()
    with open(PORTFOLIO_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==========================================
# /start, /help
# ==========================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "AI Invest HQ Bot (Isaac V3.7)\n"
        "━━━━━━━━━━━━━━━\n"
        "/daily - 每日推薦 (Isaac V3.7 + 即時報價)\n"
        "/paper - 模擬交易持倉狀態\n"
        "/paper_update - 根據推薦更新模擬交易\n"
        "/paper_history - 模擬交易歷史\n"
        "/risk - 即時風控檢查\n"
        "/scan <代碼> - 個股深度分析\n"
        "/portfolio - 查看手動持倉 PnL\n"
        "/add <代碼> <價格> [股數] - 新增持股\n"
        "/remove <代碼> - 移除持股\n"
        "/cc <指令> - Claude Code 遠端操控\n"
        "━━━━━━━━━━━━━━━\n"
        "直接輸入文字 → AI 分析師對話"
    )
    await update.message.reply_text(msg)


# ==========================================
# /scan <ticker>
# ==========================================

def _run_scan(ticker):
    """同步掃描單一股票"""
    try:
        from data.scanner import scan_single_stock_deep
        result = scan_single_stock_deep(
            market="🇹🇼 台股", ticker=ticker,
            strategy="順勢突破", timeframe="1d",
            user_query_name=ticker
        )
        if not result:
            return None

        msg = (
            f"{result.get('代碼', ticker)} {result.get('名稱', '')}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"現價: {result.get('現價', 'N/A')}\n"
            f"漲跌幅: {result.get('漲跌幅%', 'N/A')}%\n"
            f"爆量倍數: {result.get('爆量倍數', 'N/A')}x\n"
            f"RSI: {result.get('RSI', 'N/A')}\n"
            f"PE: {result.get('PE', 'N/A')}\n"
            f"EPS: {result.get('EPS', 'N/A')}\n"
            f"殖利率: {result.get('Yield', 'N/A')}\n"
            f"━━━━━━━━━━━━━━━\n"
        )
        # 信號上下文
        ctx = result.get('signal_context', '')
        if ctx:
            msg += f"信號: {ctx}\n"

        # 趨勢判斷
        verdict = result.get('verdict', {})
        if isinstance(verdict, dict):
            msg += f"趨勢: {verdict.get('trend', 'N/A')}\n"
            msg += f"建議: {verdict.get('signal', 'N/A')}\n"

        return msg
    except Exception as e:
        return f"分析失敗: {e}"


async def scan_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法: /scan <股票代碼>\n例如: /scan 2330")
        return

    ticker = context.args[0].strip()
    await update.message.reply_text(f"正在分析 {ticker}...")

    result = await _run_with_typing(update.effective_chat.id, context.bot, _run_scan, ticker)
    await update.message.reply_text(result or f"無法取得 {ticker} 的資料")


# ==========================================
# /portfolio
# ==========================================

def _format_portfolio():
    """同步取得持倉 PnL"""
    portfolio = _load_portfolio()
    holdings = portfolio.get('holdings', [])

    if not holdings:
        return "投資組合為空。使用 /add <代碼> <進場價> 新增持股。"

    lines = ["投資組合 PnL", "━━━━━━━━━━━━━━━"]
    total_pnl = 0

    try:
        from data.provider import YFinanceProvider
        provider = YFinanceProvider(market_type="TW")
    except Exception:
        provider = None

    for h in holdings:
        try:
            ticker = h['ticker']
            entry = h['entry_price']
            shares = h.get('shares', 1000)

            current_price = None
            # 嘗試 YFinance
            if provider:
                try:
                    df = provider.get_historical_data(ticker, period="5d", interval="1d")
                    if not df.empty:
                        current_price = float(df['Close'].iloc[-1])
                except Exception:
                    pass

            if current_price:
                pnl_pct = ((current_price - entry) / entry) * 100
                pnl_abs = (current_price - entry) * shares
                total_pnl += pnl_abs
                sign = "+" if pnl_pct >= 0 else ""
                lines.append(
                    f"{ticker} {h.get('name', '')} | "
                    f"進:{entry} 現:{current_price:.1f} | "
                    f"{sign}{pnl_pct:.1f}% ({sign}{pnl_abs:,.0f})"
                )
            else:
                lines.append(f"{ticker} {h.get('name', '')} | 進:{entry} | 無法取得報價")
        except Exception as e:
            lines.append(f"{h.get('ticker', '?')} 錯誤: {e}")

    lines.append("━━━━━━━━━━━━━━━")
    lines.append(f"總損益: {total_pnl:+,.0f}")
    lines.append(f"更新時間: {datetime.now().strftime('%H:%M:%S')}")
    return "\n".join(lines)


async def portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
    msg = await _run_with_typing(update.effective_chat.id, context.bot, _format_portfolio)
    await update.message.reply_text(msg)


# ==========================================
# /add <ticker> <price> [shares]
# ==========================================

async def add_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("用法: /add <代碼> <進場價> [股數]\n例如: /add 2330 850 1000")
        return

    ticker = context.args[0].strip()
    try:
        entry_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("進場價格必須為數字")
        return

    shares = int(context.args[2]) if len(context.args) > 2 else 1000

    portfolio = _load_portfolio()
    portfolio['holdings'].append({
        "ticker": ticker,
        "name": ticker,
        "entry_price": entry_price,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "shares": shares,
        "market": "TW"
    })
    _save_portfolio(portfolio)

    await update.message.reply_text(
        f"已新增: {ticker}\n進場價: {entry_price} | 股數: {shares}"
    )


# ==========================================
# /remove <ticker>
# ==========================================

async def remove_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法: /remove <代碼>")
        return

    ticker = context.args[0].strip()
    portfolio = _load_portfolio()
    original = len(portfolio['holdings'])
    portfolio['holdings'] = [h for h in portfolio['holdings'] if h['ticker'] != ticker]

    if len(portfolio['holdings']) < original:
        _save_portfolio(portfolio)
        await update.message.reply_text(f"已移除 {ticker}")
    else:
        await update.message.reply_text(f"找不到 {ticker}")


# ==========================================
# /risk
# ==========================================

def _calculate_risk():
    """同步計算風險指標 (使用即時風控模組)"""
    from data.risk_monitor import RiskMonitor
    monitor = RiskMonitor()
    result = monitor.check_all()
    return monitor.format_risk_text(result)


async def risk_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("正在執行即時風控檢查...")
    msg = await _run_with_typing(update.effective_chat.id, context.bot, _calculate_risk)
    if len(msg) > 4000:
        msg = msg[:4000] + "\n...(截斷)"
    await update.message.reply_text(msg)


# ==========================================
# /daily
# ==========================================

def _run_daily():
    """同步執行每日推薦 (Isaac V3.7 + 永豐金即時報價)"""
    from data.daily_recommender import get_daily_recommendation, format_recommendation_text
    result = get_daily_recommendation()
    return format_recommendation_text(result)


async def daily_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("正在執行 Isaac V3.7 每日推薦...")
    try:
        msg = await _run_with_typing(
            update.effective_chat.id, context.bot, _run_daily,
            progress_msg="Isaac V3.7 策略 + 即時報價取得中"
        )
        if len(msg) > 4000:
            msg = msg[:4000] + "\n...(截斷)"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"執行失敗: {e}")


# ==========================================
# /paper — 模擬交易持倉
# ==========================================

def _paper_status():
    from data.paper_trader import PaperTrader
    trader = PaperTrader()
    return trader.format_status_text()


async def paper_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await _run_with_typing(update.effective_chat.id, context.bot, _paper_status)
    await update.message.reply_text(msg)


# ==========================================
# /paper_update — 更新模擬交易
# ==========================================

def _paper_update():
    from data.paper_trader import PaperTrader
    trader = PaperTrader()
    trader.update()
    return trader.format_status_text()


async def paper_update_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("正在更新模擬交易...")
    try:
        msg = await _run_with_typing(
            update.effective_chat.id, context.bot, _paper_update,
            progress_msg="模擬交易更新中"
        )
        if len(msg) > 4000:
            msg = msg[:4000] + "\n...(截斷)"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"更新失敗: {e}")


# ==========================================
# /paper_history — 模擬交易歷史
# ==========================================

def _paper_history():
    from data.paper_trader import PaperTrader
    trader = PaperTrader()
    return trader.format_history_text()


async def paper_history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await _run_with_typing(update.effective_chat.id, context.bot, _paper_history)
    await update.message.reply_text(msg)


# ==========================================
# /cc — Claude Code CLI 整合
# ==========================================

_cc_logger = logging.getLogger(__name__ + '.cc')

# Command allowlist — only read-only / analysis commands are permitted
_CC_ALLOWED_PREFIXES = ['分析', '查詢', '搜尋', '回測', '報告', 'analyze', 'search', 'report', 'backtest']


def _run_claude_cli(prompt):
    """同步呼叫 Claude Code CLI"""
    if not os.path.exists(CLAUDE_CLI):
        return f"Claude CLI 不存在: {CLAUDE_CLI}"

    try:
        result = subprocess.run(
            [CLAUDE_CLI, '-p', prompt],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=1800,
            cwd=PROJECT_ROOT,
            encoding='utf-8', errors='replace'
        )
        stdout = (result.stdout or '').strip()
        stderr = (result.stderr or '').strip()

        if stdout:
            return stdout
        elif stderr:
            return f"[Claude Code]\n{stderr[:4000]}"
        else:
            return f"(無輸出, exit code: {result.returncode})"
    except subprocess.TimeoutExpired:
        return "Claude Code 執行超時 (>30分鐘)"
    except Exception as e:
        return f"Claude Code 錯誤: {type(e).__name__}: {e}"


async def cc_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """透過 Telegram 呼叫 Claude Code CLI"""
    # 安全檢查: 只允許特定 chat_id 使用
    import toml
    secrets = toml.load(SECRETS_PATH)
    allowed_chat_id = secrets.get('TELEGRAM_CHAT_ID', '')
    if str(update.effective_chat.id) != str(allowed_chat_id):
        await update.message.reply_text("此指令僅限管理員使用")
        return

    if not context.args:
        await update.message.reply_text(
            "用法: /cc <指令>\n"
            "例如:\n"
            "  /cc 回測 Isaac 策略\n"
            "  /cc 分析 strategies/isaac.py 的 Signal A 邏輯\n"
            "  /cc 報告今天策略推薦哪些股票"
        )
        return

    prompt = ' '.join(context.args)

    # Security: command allowlist — only analysis/read-only prompts permitted
    prompt_lower = prompt.strip().lower()
    if not any(prompt_lower.startswith(prefix) for prefix in _CC_ALLOWED_PREFIXES):
        _cc_logger.warning(f"Blocked /cc command from chat_id={update.effective_chat.id}: {prompt[:200]}")
        await update.message.reply_text("此指令未被允許。僅支援分析類指令。")
        return

    _cc_logger.info(f"/cc invoked by chat_id={update.effective_chat.id}: {prompt[:200]}")
    await update.message.reply_text(f"正在執行 Claude Code...\n指令: {prompt[:100]}")

    result = await _run_with_typing(
        update.effective_chat.id, context.bot, _run_claude_cli, prompt,
        progress_msg="Claude Code 執行中"
    )

    # Telegram 訊息長度限制 4096
    if len(result) > 4000:
        # 分段發送
        chunks = [result[i:i+4000] for i in range(0, len(result), 4000)]
        for i, chunk in enumerate(chunks[:5]):  # 最多 5 段
            prefix = f"[{i+1}/{min(len(chunks),5)}] " if len(chunks) > 1 else ""
            await update.message.reply_text(prefix + chunk)
    else:
        await update.message.reply_text(result)


# ==========================================
# AI 對話 (Gemini)
# ==========================================

def _ai_chat(user_message):
    """用 Gemini 回覆自然語言問題"""
    import toml
    secrets = toml.load(SECRETS_PATH)
    gemini_key = secrets.get('GEMINI_API_KEY', '')
    if not gemini_key:
        return "Gemini API 未設定"

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        system_prompt = (
            "你是 AI Invest HQ 的台股分析師機器人。"
            "用繁體中文回答。簡潔有力，直接給出分析和建議。"
            "你熟悉台股、技術分析、基本面分析、籌碼面分析。"
            "回答時使用條列式，重點標粗。"
        )
        response = model.generate_content(
            f"{system_prompt}\n\n用戶問題: {user_message}",
            generation_config=genai.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7,
            )
        )
        return response.text[:4000]
    except Exception as e:
        return f"AI 回覆失敗: {e}"


async def ai_chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理非指令的一般訊息"""
    user_msg = update.message.text
    if not user_msg:
        return

    reply = await _run_with_typing(update.effective_chat.id, context.bot, _ai_chat, user_msg)
    # Telegram 限制 4096 字
    if len(reply) > 4000:
        reply = reply[:4000] + "\n...(截斷)"
    await update.message.reply_text(reply)
