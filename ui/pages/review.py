"""
Page: Review (覆盤紀錄) — Combines Journal + Calendar in tabs.
"""
import streamlit as st

from ui.components import cyber_header


def render():
    cyber_header("覆盤紀錄", "交易日誌 | 經濟行事曆")

    tab_journal, tab_cal = st.tabs(["交易日誌", "經濟行事曆"])

    with tab_journal:
        try:
            from ui.pages.journal import render as render_journal
            render_journal(_embedded=True)
        except ImportError:
            st.info("交易日誌模組尚未建立 (journal.py)")
        except Exception as e:
            st.warning(f"交易日誌模組載入失敗: {e}")

    with tab_cal:
        try:
            from ui.pages.calendar import render as render_cal
            render_cal(_embedded=True)
        except ImportError:
            st.info("經濟行事曆模組尚未建立 (calendar.py)")
        except Exception as e:
            st.warning(f"經濟行事曆模組載入失敗: {e}")
