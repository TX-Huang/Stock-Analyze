"""
Page: Lab (策略實驗室) — Combines Backtest + Leverage ETF in tabs.
"""
import streamlit as st

from ui.components import cyber_header


def render():
    cyber_header("策略實驗室", "量化回測 | 槓桿ETF | 壓力測試")

    tab_bt, tab_lev = st.tabs(["量化回測", "槓桿ETF評估"])

    with tab_bt:
        try:
            from ui.pages.backtest import render as render_bt
            render_bt(_embedded=True)
        except Exception as e:
            st.warning(f"量化回測模組載入失敗: {e}")

    with tab_lev:
        try:
            from ui.pages.leverage import render as render_lev
            render_lev(_embedded=True)
        except Exception as e:
            st.warning(f"槓桿ETF評估模組載入失敗: {e}")
