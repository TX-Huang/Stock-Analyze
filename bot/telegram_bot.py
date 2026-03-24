"""
AI Invest HQ Interactive Telegram Bot
======================================
啟動: ./python_embed/python.exe -m bot.telegram_bot

功能:
  /scan <代碼>     — 個股深度分析
  /portfolio       — 持倉 PnL (永豐金/YFinance)
  /add <代碼> <價>  — 新增持股
  /remove <代碼>   — 移除持股
  /risk            — 風險管理面板
  /daily           — 執行每日策略
  一般文字          — Gemini AI 對話
"""
import sys
import os
import logging

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Mock streamlit BEFORE any project imports
from bot._st_mock import install_mock
install_mock()

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'bot_debug.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    import toml
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    from bot.handlers import (
        start_handler, scan_handler, portfolio_handler,
        add_handler, remove_handler, risk_handler,
        daily_handler, cc_handler, ai_chat_handler,
        paper_handler, paper_update_handler, paper_history_handler,
    )

    # Load secrets
    secrets_path = os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml')
    secrets = toml.load(secrets_path)
    bot_token = secrets.get('TELEGRAM_BOT_TOKEN', '')

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in secrets.toml")
        return

    # Build application
    app = ApplicationBuilder().token(bot_token).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", start_handler))
    app.add_handler(CommandHandler("scan", scan_handler))
    app.add_handler(CommandHandler("portfolio", portfolio_handler))
    app.add_handler(CommandHandler("add", add_handler))
    app.add_handler(CommandHandler("remove", remove_handler))
    app.add_handler(CommandHandler("risk", risk_handler))
    app.add_handler(CommandHandler("daily", daily_handler))
    app.add_handler(CommandHandler("paper", paper_handler))
    app.add_handler(CommandHandler("paper_update", paper_update_handler))
    app.add_handler(CommandHandler("paper_history", paper_history_handler))
    app.add_handler(CommandHandler("cc", cc_handler))

    # AI chat for non-command messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai_chat_handler))

    logger.info("AI Invest HQ Bot starting...")
    logger.info(f"Commands: /start /scan /portfolio /add /remove /risk /daily /cc")
    logger.info(f"AI Chat: enabled (Gemini)")

    # Start polling
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
