"""
Session state initialization.
"""
import streamlit as st
from config.settings import DEFAULT_CHART_SETTINGS

MAX_CACHE_SIZE = 10


def init_session_state():
    """Initialize all session state keys with defaults."""
    for key, default in {
        'data_cache': {}, 'ai_reports': {}, 'market_mode': "🇹🇼 台股 (TW)",
        'dynamic_name_map': {}, 'view_mode': "list", 'single_stock_data': None,
        'current_source': "🗂️ 預設清單", 'detected_themes': [], 'supply_chain_data': None,
        'leverage_analysis': None, 'leverage_etf_data': None, 'leverage_underlying_data': None,
        'backtest_report': None, 'current_strategy': None, 'backtest_results': {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if 'chart_settings' not in st.session_state:
        st.session_state.chart_settings = DEFAULT_CHART_SETTINGS.copy()


def evict_data_cache():
    """Evict oldest entries from data_cache when it exceeds MAX_CACHE_SIZE."""
    cache = st.session_state.get('data_cache', {})
    if len(cache) > MAX_CACHE_SIZE:
        # dict preserves insertion order in Python 3.7+; remove oldest entries
        keys = list(cache.keys())
        for k in keys[:len(keys) - MAX_CACHE_SIZE]:
            del cache[k]
