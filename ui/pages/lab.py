"""
Page: Lab (策略實驗室) — Combines Backtest + Leverage ETF + Monte Carlo + A/B Comparison in tabs.
"""
import streamlit as st

from ui.components import cyber_header


def render():
    cyber_header("策略實驗室", "量化回測 | 槓桿ETF | 蒙地卡羅 | 策略比較")

    tab_bt, tab_lev, tab_mc, tab_cmp = st.tabs([
        "量化回測", "槓桿ETF評估", "蒙地卡羅模擬", "策略比較",
    ])

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

    with tab_mc:
        try:
            from ui.pages.monte_carlo_panel import render_monte_carlo
            # Pass trades from session state if available (set by backtest page)
            trades_df = st.session_state.get("backtest_trades", None)
            stats = st.session_state.get("backtest_stats", None)
            render_monte_carlo(trades_df=trades_df, stats=stats)
        except Exception as e:
            st.warning(f"蒙地卡羅模擬模組載入失敗: {e}")

    with tab_cmp:
        try:
            from ui.pages.comparison import render_comparison
            render_comparison()
        except Exception as e:
            st.warning(f"策略比較模組載入失敗: {e}")
