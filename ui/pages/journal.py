"""交易日誌 — 記錄決策、情緒追蹤、覆盤學習。"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date

from ui.components import cyber_header, cyber_kpi_strip, cyber_table
from ui.theme import _plotly_dark_layout
from data.journal import JournalManager, EMOTION_TAGS


def render(_embedded=False):
    jm = JournalManager()

    if not _embedded:
        cyber_header("交易日誌", "記錄決策 | 情緒追蹤 | 覆盤學習")

    tab_add, tab_history = st.tabs(["📝 新增紀錄", "📖 歷史紀錄"])

    # ── Tab 1: 新增紀錄 ──
    with tab_add:
        with st.form("journal_form", clear_on_submit=True):
            st.markdown('<p class="sec-header">交易紀錄</p>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            trade_date = c1.date_input("日期", value=date.today())
            ticker = c2.text_input("股票代碼", placeholder="例: 2330")
            name = c3.text_input("股票名稱", placeholder="例: 台積電")

            c4, c5, c6 = st.columns(3)
            action = c4.selectbox("操作方向", ["BUY", "SELL"])
            price = c5.number_input("價格", min_value=0.0, step=0.1, format="%.2f")
            shares = c6.number_input("股數", min_value=0, step=1, value=1000)

            reasoning = st.text_area("交易理由 / 決策邏輯", placeholder="為什麼做這筆交易？技術面、基本面、消息面...")
            emotion_tag = st.selectbox("情緒標籤", EMOTION_TAGS)

            submitted = st.form_submit_button("💾 儲存紀錄", use_container_width=True)

            if submitted:
                if not ticker.strip():
                    st.error("請輸入股票代碼")
                elif price <= 0:
                    st.error("請輸入有效價格")
                else:
                    jm.add_entry(
                        date=trade_date.strftime("%Y-%m-%d"),
                        ticker=ticker.strip(),
                        name=name.strip(),
                        action=action,
                        price=price,
                        shares=int(shares),
                        reasoning=reasoning.strip(),
                        emotion_tag=emotion_tag,
                    )
                    st.success("紀錄已儲存！")
                    st.rerun()

    # ── Tab 2: 歷史紀錄 ──
    with tab_history:
        entries = jm.get_entries()

        if not entries:
            st.markdown(
                '<div class="alert-card alert-info">'
                '<div class="alert-title">尚無紀錄</div>'
                '<div class="alert-body">前往「新增紀錄」開始記錄你的每筆交易。</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        # Stats KPI
        stats = jm.get_stats()
        top_emotion = max(stats.get("emotions", {}), key=stats["emotions"].get, default="—")
        cyber_kpi_strip([
            {"label": "總紀錄", "value": stats.get("total", 0), "accent": "#00f0ff"},
            {"label": "買進", "value": stats.get("buys", 0), "accent": "#ef4444"},
            {"label": "賣出", "value": stats.get("sells", 0), "accent": "#22c55e"},
            {"label": "最常見情緒", "value": top_emotion, "accent": "#f59e0b"},
        ])

        # Emotion distribution chart
        emotions = stats.get("emotions", {})
        if emotions:
            fig = go.Figure(go.Bar(
                x=list(emotions.keys()),
                y=list(emotions.values()),
                marker_color="rgba(0,240,255,0.5)",
                marker_line_color="rgba(0,240,255,0.8)",
                marker_line_width=1,
            ))
            _plotly_dark_layout(fig, height=280, title_text="情緒標籤分布")
            fig.update_layout(xaxis_title="", yaxis_title="次數")
            st.plotly_chart(fig, use_container_width=True)

        # Entry list
        st.markdown('<p class="sec-header">所有紀錄</p>', unsafe_allow_html=True)

        for entry in entries:
            action_tag = (
                '<span class="tag tag-bull">買進</span>'
                if entry["action"] == "BUY"
                else '<span class="tag tag-bear">賣出</span>'
            )
            header_text = f"{entry['date']}　{entry['ticker']} {entry['name']}　{entry['action']}　${entry['price']:,.2f}　x {entry['shares']}"

            with st.expander(header_text, expanded=False):
                st.markdown(f"**操作方向：** {action_tag}", unsafe_allow_html=True)
                st.markdown(f"**價格：** ${entry['price']:,.2f}　**股數：** {entry['shares']:,}")
                st.markdown(f"**情緒標籤：** `{entry.get('emotion_tag', '—')}`")
                st.markdown(f"**交易理由：**\n\n{entry.get('reasoning', '—')}")
                if entry.get("outcome_review"):
                    st.markdown(f"**覆盤筆記：**\n\n{entry['outcome_review']}")
                st.caption(f"建立時間：{entry.get('created_at', '—')}")

                if st.button("🗑️ 刪除此紀錄", key=f"del_{entry['id']}"):
                    jm.delete_entry(entry["id"])
                    st.rerun()
