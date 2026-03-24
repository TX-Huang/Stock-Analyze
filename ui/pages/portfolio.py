"""
投資組合 — Portfolio Dashboard
持倉概覽 | 資產配置 | 績效追蹤
"""
import streamlit as st
import json
import os
import plotly.graph_objects as go

from ui.theme import _tw_color, _tw_color_pct, _plotly_dark_layout
from ui.components import cyber_header, cyber_kpi_strip


try:
    from config.paths import PAPER_TRADE_PATH
except ImportError:
    PAPER_TRADE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'paper_trade.json')


def _load_paper_trade():
    """Load paper_trade.json data."""
    if os.path.exists(PAPER_TRADE_PATH):
        try:
            with open(PAPER_TRADE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def render():
    # --- Header ---
    cyber_header("投資組合", subtitle="持倉概覽 | 資產配置 | 績效追蹤")

    # --- Load data ---
    data = _load_paper_trade()
    if not data:
        st.markdown("""
        <div class="alert-card alert-info" style="text-align:center; padding:40px;">
            <div class="alert-title" style="font-size:1.1rem;">尚無投資組合資料</div>
            <div class="alert-body">請先透過自動交易系統建立模擬交易</div>
        </div>
        """, unsafe_allow_html=True)
        return

    positions = data.get("positions", [])
    cash = float(data.get("cash", 0))
    initial_capital = float(data.get("initial_capital", 1000000))
    closed_trades = data.get("closed_trades", [])

    # --- Calculate portfolio metrics ---
    positions_value = 0
    for p in positions:
        price = float(p.get("current_price", p.get("entry_price", 0)))
        shares = int(p.get("shares", 0))
        positions_value += price * shares

    total_equity = cash + positions_value

    # Unrealized PnL
    unrealized_pnl = 0
    for p in positions:
        entry = float(p.get("entry_price", 0))
        current = float(p.get("current_price", entry))
        shares = int(p.get("shares", 0))
        unrealized_pnl += (current - entry) * shares

    # Realized PnL from closed trades
    realized_pnl = 0
    for t in closed_trades:
        realized_pnl += float(t.get("pnl", 0))

    # Exposure ratio
    exposure_pct = (positions_value / total_equity * 100) if total_equity > 0 else 0

    # PnL colors
    ur_color = "#ef4444" if unrealized_pnl >= 0 else "#22c55e"
    rl_color = "#ef4444" if realized_pnl >= 0 else "#22c55e"

    # --- KPI Strip ---
    cyber_kpi_strip([
        {"label": "總資產", "value": f"${total_equity:,.0f}", "accent": "#00f0ff"},
        {"label": "未實現損益", "value": f"${unrealized_pnl:+,.0f}", "color": ur_color, "accent": ur_color},
        {"label": "已實現損益", "value": f"${realized_pnl:+,.0f}", "color": rl_color, "accent": rl_color},
        {"label": "現金", "value": f"${cash:,.0f}", "accent": "#3b82f6"},
        {"label": "曝險比例", "value": f"{exposure_pct:.1f}%", "accent": "#f59e0b"},
    ])

    # --- Holdings Table ---
    if not positions:
        st.markdown("""
        <div class="alert-card alert-info" style="text-align:center; padding:30px;">
            <div class="alert-title">目前無持倉</div>
            <div class="alert-body">所有部位已平倉</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p class="sec-header">持倉明細</p>', unsafe_allow_html=True)

        headers = ["代碼", "名稱", "股數", "成本價", "現價", "未實現損益", "權重%"]
        th_html = "".join(
            f'<th style="text-align:{"left" if i == 0 else "right"}">{h}</th>'
            for i, h in enumerate(headers)
        )

        tr_html = ""
        for p in positions:
            ticker = p.get("ticker", "")
            name = p.get("name", ticker)
            shares = int(p.get("shares", 0))
            entry_price = float(p.get("entry_price", 0))
            current_price = float(p.get("current_price", entry_price))
            pnl = (current_price - entry_price) * shares
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
            weight = (current_price * shares / total_equity * 100) if total_equity > 0 else 0

            pnl_str = _tw_color(pnl, fmt="+,.0f")
            pnl_pct_tag = f' ({_tw_color_pct(pnl_pct)})'

            tr_html += f"""<tr>
                <td>{ticker}</td>
                <td>{name}</td>
                <td>{shares:,}</td>
                <td>{entry_price:,.1f}</td>
                <td>{current_price:,.1f}</td>
                <td>{pnl_str}{pnl_pct_tag}</td>
                <td>{weight:.1f}%</td>
            </tr>"""

        st.markdown(f"""<table class="pro-table">
        <thead><tr>{th_html}</tr></thead>
        <tbody>{tr_html}</tbody>
        </table>""", unsafe_allow_html=True)

    # --- Allocation Pie Chart ---
    if positions:
        st.markdown('<p class="sec-header">資產配置</p>', unsafe_allow_html=True)

        labels = [p.get('name', '') or p.get('ticker', '') for p in positions]
        values = [float(p.get("current_price", p.get("entry_price", 0))) * int(p.get("shares", 0))
                  for p in positions]

        # Add cash slice
        labels.append("現金")
        values.append(cash)

        colors = [
            "#00f0ff", "#3b82f6", "#8b5cf6", "#ef4444", "#f59e0b",
            "#22c55e", "#ec4899", "#06b6d4", "#84cc16", "#f97316",
        ]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo='label+percent',
            textfont=dict(size=11),
            marker=dict(colors=colors[:len(labels)],
                        line=dict(color='rgba(0,240,255,0.2)', width=1)),
        )])

        _plotly_dark_layout(fig, height=400,
                            title=dict(text="持倉配置", font=dict(size=14, color="#94a3b8")),
                            showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    # --- Equity Curve (if available) ---
    daily_equity = data.get("daily_equity", [])
    if len(daily_equity) > 1:
        st.markdown('<p class="sec-header">權益曲線</p>', unsafe_allow_html=True)
        dates = [e["date"] for e in daily_equity]
        equities = [e["equity"] for e in daily_equity]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=equities,
            mode='lines+markers',
            line=dict(color='#00f0ff', width=2),
            marker=dict(size=5),
            name='總權益',
        ))
        _plotly_dark_layout(fig2, height=300,
                            title=dict(text="每日權益", font=dict(size=14, color="#94a3b8")),
                            yaxis_title="NTD")
        st.plotly_chart(fig2, use_container_width=True)
