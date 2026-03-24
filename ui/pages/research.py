"""
Page: Research (研究分析) — Combines War Room + Watchlist + Heatmap in tabs.
"""
import streamlit as st

from ui.components import cyber_header


def render(client=None, market_mode="", strategy_mode="", tf_code="1d", is_weekly=False):
    cyber_header("研究分析", "個股深度 | 自選股 | 板塊熱力")

    tab_war, tab_watch, tab_heat = st.tabs(["個股分析", "自選股", "板塊熱力圖"])

    with tab_war:
        try:
            from ui.pages.war_room import render as render_war_room
            render_war_room(client=client, market_mode=market_mode, strategy_mode=strategy_mode, tf_code=tf_code, is_weekly=is_weekly, _embedded=True)
        except Exception as e:
            st.warning(f"個股分析模組載入失敗: {e}")

    with tab_watch:
        try:
            from ui.pages.watchlist import render as render_watchlist
            render_watchlist(_embedded=True)
        except ImportError:
            st.info("自選股模組尚未建立 (watchlist.py)")
        except Exception as e:
            st.warning(f"自選股模組載入失敗: {e}")

    with tab_heat:
        try:
            from ui.pages.heatmap import render as render_heatmap
            render_heatmap(_embedded=True)
        except Exception as e:
            st.warning(f"板塊熱力圖模組載入失敗: {e}")
