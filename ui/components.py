"""
Cyber Component Library — Reusable UI building blocks for the trading terminal.
"""
import streamlit as st
import pandas as pd
from contextlib import contextmanager


# ==========================================
# Cyber Loading Animation
# ==========================================
@contextmanager
def cyber_spinner(message="PROCESSING", sub="", min_display=0.5):
    """
    Cyber-style loading animation for long-running operations.
    Usage: with cyber_spinner("BACKTESTING", "Isaac V3.7 strategy..."):

    Args:
        min_display: Minimum display time in seconds (prevents flash)
    """
    import time as _time
    _inject_spinner_css()
    start = _time.monotonic()

    placeholder = st.empty()
    placeholder.markdown(
        '<div class="cyber-loader-wrapper">'
        '<div class="cyber-loader-container">'
        '<div style="display:flex;justify-content:center"><div class="cyber-loader-ring"></div></div>'
        f'<div class="cyber-loader-title">{message}</div>'
        '<div style="margin-top:10px">'
        '<span class="cyber-loader-dot"></span>'
        '<span class="cyber-loader-dot"></span>'
        '<span class="cyber-loader-dot"></span>'
        '<span class="cyber-loader-dot"></span>'
        '<span class="cyber-loader-dot"></span>'
        '</div>'
        + (f'<div class="cyber-loader-sub">{sub}</div>' if sub else '')
        + '<div class="cyber-loader-bar-track"><div class="cyber-loader-bar-fill"></div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    try:
        yield placeholder
    finally:
        # Ensure minimum display time so animation is visible
        elapsed = _time.monotonic() - start
        if elapsed < min_display:
            _time.sleep(min_display - elapsed)
        placeholder.empty()


def _inject_spinner_css():
    """Inject cyber spinner CSS. Called every time cyber_spinner is used.
    Streamlit clears the DOM on each rerun, so CSS must be re-injected."""
    st.markdown("""<style>
@keyframes cyber-pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1.2); }
}
@keyframes cyber-rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes cyber-bar {
    0% { width: 5%; }
    25% { width: 45%; }
    50% { width: 65%; }
    75% { width: 85%; }
    100% { width: 95%; }
}
@keyframes cyber-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(0,240,255,0.3); }
    50% { box-shadow: 0 0 20px rgba(0,240,255,0.6), 0 0 40px rgba(0,240,255,0.2); }
}
.cyber-loader-wrapper {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 60px 20px; margin: 20px 0;
}
.cyber-loader-ring {
    width: 60px; height: 60px;
    border: 2px solid rgba(0,240,255,0.1);
    border-top: 2px solid #00f0ff;
    border-right: 2px solid rgba(0,240,255,0.4);
    border-radius: 50%;
    animation: cyber-rotate 1.2s linear infinite;
    margin-bottom: 20px;
}
.cyber-loader-dot {
    width: 8px; height: 8px; background: #00f0ff;
    border-radius: 50%; display: inline-block; margin: 0 4px;
}
.cyber-loader-dot:nth-child(1) { animation: cyber-pulse 1.4s 0s infinite; }
.cyber-loader-dot:nth-child(2) { animation: cyber-pulse 1.4s 0.2s infinite; }
.cyber-loader-dot:nth-child(3) { animation: cyber-pulse 1.4s 0.4s infinite; }
.cyber-loader-dot:nth-child(4) { animation: cyber-pulse 1.4s 0.6s infinite; }
.cyber-loader-dot:nth-child(5) { animation: cyber-pulse 1.4s 0.8s infinite; }
.cyber-loader-bar-track {
    width: 200px; height: 3px; background: rgba(0,240,255,0.1);
    border-radius: 2px; overflow: hidden; margin-top: 16px;
}
.cyber-loader-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00f0ff, #8b5cf6, #00f0ff);
    border-radius: 2px; animation: cyber-bar 3s ease-in-out infinite;
}
.cyber-loader-container {
    animation: cyber-glow 2s ease-in-out infinite;
    padding: 30px 40px; border: 1px solid rgba(0,240,255,0.12);
    border-radius: 12px; background: rgba(0,0,0,0.3); text-align: center;
}
.cyber-loader-title {
    font-family: JetBrains Mono, monospace; font-size: 0.85rem;
    font-weight: 700; color: #00f0ff; letter-spacing: 0.15em;
    text-shadow: 0 0 10px rgba(0,240,255,0.3);
}
.cyber-loader-sub {
    font-size: 0.7rem; color: #64748b; margin-top: 10px;
    font-family: JetBrains Mono, monospace;
}
</style>""", unsafe_allow_html=True)


# ==========================================
# Legacy Helpers (backward-compatible)
# ==========================================
def custom_metric(label, value, delta=None):
    delta_str = ""
    if delta:
        delta_str = f" {delta}"
    st.markdown(f"**{label}**: {value} {delta_str}")


def highlight_ret(val):
    if pd.isna(val):
        return ''
    if isinstance(val, (int, float)):
        return 'color: #ef4444' if val > 0 else 'color: #22c55e'
    return ''


# ==========================================
# Cyber Metric Card
# ==========================================
def cyber_metric(label, value, delta=None, accent_color=None):
    """Glassmorphism KPI card with optional accent border and delta."""
    border_style = f"border-left: 3px solid {accent_color};" if accent_color else ""
    delta_html = ""
    if delta is not None:
        try:
            d = float(str(delta).replace('%', '').replace('+', ''))
            d_color = "#ef4444" if d > 0 else "#22c55e" if d < 0 else "#94a3b8"
            d_sign = "+" if d > 0 else ""
            delta_html = f'<div style="font-size:0.7rem; color:{d_color}; font-family:JetBrains Mono,monospace; margin-top:2px;">{d_sign}{delta}</div>'
        except (ValueError, TypeError):
            delta_html = f'<div style="font-size:0.7rem; color:#94a3b8; margin-top:2px;">{delta}</div>'

    st.markdown(f"""
    <div class="kpi-item" style="{border_style}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# Market Ticker Tape (Global Index Bar)
# ==========================================
MARKET_TICKERS = [
    {"symbol": "^TWII",  "label": "加權指數", "flag": "🇹🇼"},
    {"symbol": "0050.TW", "label": "0050",    "flag": ""},
    {"symbol": "^SOX",   "label": "費半",     "flag": "🇺🇸"},
    {"symbol": "^IXIC",  "label": "NASDAQ",   "flag": ""},
    {"symbol": "^GSPC",  "label": "S&P500",   "flag": ""},
    {"symbol": "QQQ",    "label": "QQQ",      "flag": ""},
    {"symbol": "^VIX",   "label": "VIX",      "flag": "⚡"},
]

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_indices():
    """Fetch latest prices for major indices (cached 5 min)."""
    try:
        import yfinance as yf
        import numpy as np
        results = []
        for t in MARKET_TICKERS:
            sym = t["symbol"]
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(period="5d")
                if hist is None or hist.empty or len(hist) < 2:
                    results.append(None)
                    continue
                # Drop NaN rows and get last 2
                close = hist["Close"].dropna()
                if len(close) < 2:
                    results.append(None)
                    continue
                price = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                if np.isnan(price) or np.isnan(prev) or prev == 0:
                    results.append(None)
                    continue
                chg = price - prev
                chg_pct = (chg / prev * 100)
                results.append({
                    "price": price,
                    "change": chg,
                    "change_pct": chg_pct,
                })
            except Exception:
                results.append(None)
        return results
    except Exception:
        return [None] * len(MARKET_TICKERS)


def render_market_tape():
    """Render the global market index ticker tape at the top of every page."""
    indices = _fetch_market_indices()

    cells = []
    for i, t in enumerate(MARKET_TICKERS):
        d = indices[i] if i < len(indices) else None
        if d is None:
            cells.append(
                f'<span style="color:#475569">{t["flag"]}{t["label"]} --</span>'
            )
            continue

        price = d["price"]
        chg = d["change"]
        chg_pct = d["change_pct"]

        # VIX uses inverse color logic (high VIX = bad)
        if t["symbol"] == "^VIX":
            color = "#22c55e" if chg >= 0 else "#ef4444"
        else:
            # Taiwan convention: red = up, green = down
            color = "#ef4444" if chg >= 0 else "#22c55e"

        arrow = "▲" if chg > 0 else "▼" if chg < 0 else "─"

        # Format price
        if price >= 1000:
            p_str = f"{price:,.0f}"
        elif price >= 100:
            p_str = f"{price:,.1f}"
        else:
            p_str = f"{price:,.2f}"

        cells.append(
            f'<span style="white-space:nowrap">'
            f'<span style="color:#64748b;font-size:0.6rem">{t["flag"]}{t["label"]}</span>'
            f'<span style="color:#e2e8f0;font-weight:700"> {p_str}</span>'
            f'<span style="color:{color};font-weight:600"> {arrow}{chg_pct:+.1f}%</span>'
            f'</span>'
        )

    sep = '<span style="color:rgba(0,240,255,0.12);margin:0 2px">│</span>'
    tape_html = (
        '<div style="display:flex;flex-wrap:wrap;gap:4px 10px;padding:5px 8px;'
        'background:linear-gradient(90deg,rgba(0,240,255,0.03),rgba(0,0,0,0),rgba(0,240,255,0.03));'
        'border-bottom:1px solid rgba(0,240,255,0.08);border-top:1px solid rgba(0,240,255,0.08);'
        'font-family:JetBrains Mono,SF Mono,monospace;font-size:0.65rem;'
        'align-items:center;margin-bottom:6px;justify-content:center">'
        + sep.join(cells)
        + '</div>'
    )
    st.markdown(tape_html, unsafe_allow_html=True)


# ==========================================
# Cyber KPI Strip
# ==========================================
def cyber_kpi_strip(items):
    """Render a horizontal strip of KPI items.
    items: list of dicts with keys: label, value, [color], [accent]
    """
    cards = []
    for item in items:
        accent = f"border-left: 3px solid {item['accent']};" if item.get('accent') else ""
        v_color = f"color:{item['color']};" if item.get('color') else ""
        cards.append(f"""
        <div class="kpi-item" style="{accent}">
            <div class="kpi-label">{item['label']}</div>
            <div class="kpi-value" style="{v_color}">{item['value']}</div>
        </div>""")

    st.markdown(f'<div class="kpi-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


# ==========================================
# Cyber Alert Card
# ==========================================
def cyber_alert(title, body, level="info"):
    """Alert card with neon glow. level: info, ok, warn, danger"""
    st.markdown(f"""
    <div class="alert-card alert-{level}">
        <div class="alert-title">{title}</div>
        <div class="alert-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# Cyber Module Header
# ==========================================
def cyber_header(title, subtitle="", tag_text=None, tag_class="tag-new"):
    """Gradient neon module header with optional tag badge."""
    tag_html = f'<span class="tag {tag_class}" style="font-size:0.65rem; vertical-align:middle; margin-left:12px;">{tag_text}</span>' if tag_text else ""
    st.markdown(f"""
    <div style="margin-bottom:4px;">
        <span class="cyber-title">{title}</span>{tag_html}
    </div>
    <div class="cyber-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)


# ==========================================
# Signal Badge
# ==========================================
def signal_badge(signal_type, value=None):
    """Colored signal badge: BUY/SELL/HOLD/WARN"""
    configs = {
        'BUY': ('tag-bull', '買進'),
        'SELL': ('tag-bear', '賣出'),
        'HOLD': ('tag-new', '持有'),
        'WARN': ('tag-warn', '警告'),
        'STRONG_BUY': ('tag-bull', '強力買進'),
        'STRONG_SELL': ('tag-bear', '強力賣出'),
    }
    cls, label = configs.get(signal_type.upper(), ('tag-new', signal_type))
    val_html = f" {value}" if value else ""
    return f'<span class="tag {cls}">{label}{val_html}</span>'


# ==========================================
# Ticker Tape (Market Index Bar)
# ==========================================
def ticker_tape(indices_data):
    """Render horizontal market index ticker tape.
    indices_data: list of dicts with keys: name, price, change, change_pct
    """
    items = []
    for idx in indices_data:
        change = idx.get('change', 0)
        pct = idx.get('change_pct', 0)
        # Taiwan convention: red=up, green=down
        if change > 0:
            clr = "#ef4444"
            sign = "+"
            arrow = "▲"
        elif change < 0:
            clr = "#22c55e"
            sign = ""
            arrow = "▼"
        else:
            clr = "#94a3b8"
            sign = ""
            arrow = "–"

        items.append(f"""
        <div class="ticker-item">
            <span class="ticker-name">{idx['name']}</span>
            <span class="ticker-price">{idx.get('price', 'N/A')}</span>
            <span style="color:{clr}; font-weight:700;">{arrow} {sign}{change:,.2f} ({sign}{pct:.2f}%)</span>
        </div>""")

    st.markdown(f'<div class="ticker-tape">{"".join(items)}</div>', unsafe_allow_html=True)


# ==========================================
# Cyber Table
# ==========================================
def cyber_table(headers, rows, html=True):
    """Render a pro-table with cyber styling.
    headers: list of header strings
    rows: list of lists (each row is a list of cell HTML strings)
    """
    th_html = "".join(f'<th style="text-align:{"left" if i==0 else "right"}">{h}</th>' for i, h in enumerate(headers))
    tr_html = ""
    for row in rows:
        td_html = "".join(f'<td>{cell}</td>' for cell in row)
        tr_html += f"<tr>{td_html}</tr>"

    table = f"""<table class="pro-table">
    <thead><tr>{th_html}</tr></thead>
    <tbody>{tr_html}</tbody>
    </table>"""

    if html:
        st.markdown(table, unsafe_allow_html=True)
    return table
