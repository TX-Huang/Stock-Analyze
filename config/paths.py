"""Centralized path configuration."""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

WATCHLIST_PATH = os.path.join(DATA_DIR, 'watchlist.json')
ALERTS_PATH = os.path.join(DATA_DIR, 'alerts.json')
JOURNAL_PATH = os.path.join(DATA_DIR, 'trade_journal.json')
PAPER_TRADE_PATH = os.path.join(DATA_DIR, 'paper_trade.json')
SCAN_RESULTS_PATH = os.path.join(DATA_DIR, 'scan_results.json')
RECOMMENDATION_PATH = os.path.join(DATA_DIR, 'daily_recommendation.json')
AUTO_TRADE_CONFIG_PATH = os.path.join(DATA_DIR, 'auto_trade_config.json')
ORDER_LOG_PATH = os.path.join(DATA_DIR, 'order_log.json')
RISK_CONFIG_PATH = os.path.join(DATA_DIR, 'risk_config.json')
SUBSCRIBERS_PATH = os.path.join(DATA_DIR, 'telegram_subscribers.json')
V4_SIGNALS_PATH = os.path.join(DATA_DIR, 'daily_v4_signals.json')
