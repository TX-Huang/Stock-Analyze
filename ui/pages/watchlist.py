"""
自選股清單 — Watchlist Page
即時報價 | 技術指標 | 快速跳轉分析
"""
import streamlit as st
import pandas as pd
import re

from ui.theme import _tw_color, _tw_color_pct, _tw_tag
from ui.components import cyber_header, cyber_table, cyber_spinner
from data.watchlist import WatchlistManager
from data.provider import get_data_provider
from analysis.indicators import calculate_rsi


def _is_tw_ticker(ticker: str) -> bool:
    """4-6 digits (with optional trailing letter) = TW stock."""
    return bool(re.match(r'^\d{4,6}[A-Za-z]?$', str(ticker).strip()))


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price_data(ticker: str):
    """Fetch latest price and compute RSI for a single ticker."""
    try:
        market = "TW" if _is_tw_ticker(ticker) else "US"
        provider = get_data_provider("auto", market_type=market)
        df = provider.get_historical_data(ticker, period="3mo", interval="1d")
        if df is None or df.empty:
            return None

        last_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else last_close
        change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close else 0
        volume = int(df['Volume'].iloc[-1])
        rsi_val = float(calculate_rsi(df['Close'], period=14).iloc[-1]) if len(df) >= 15 else None

        # Simple trend: price vs 20-MA
        ma20 = float(df['Close'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
        if ma20:
            trend = "多" if last_close > ma20 else "空"
        else:
            trend = "—"

        return {
            "price": last_close,
            "change_pct": change_pct,
            "volume": volume,
            "rsi": rsi_val,
            "trend": trend,
        }
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_batch_price_data(tickers_tuple: tuple):
    """Batch fetch price data for all watchlist tickers at once."""
    results = {}
    for ticker in tickers_tuple:
        results[ticker] = _fetch_price_data.__wrapped__(ticker)
    return results


def render(_embedded=False):
    # --- Header ---
    if not _embedded:
        cyber_header("自選股清單", subtitle="即時報價 | 技術指標 | 快速跳轉分析")

    # --- Init watchlist manager ---
    wm = WatchlistManager()

    # --- Add Stock Section ---
    st.markdown('<p class="sec-header">新增自選股</p>', unsafe_allow_html=True)
    col_ticker, col_group, col_btn = st.columns([2, 2, 1])
    with col_ticker:
        new_ticker = st.text_input("股票代碼", placeholder="輸入代碼 (如 2330, AAPL)", label_visibility="collapsed")
    with col_group:
        groups = wm.get_groups()
        selected_group = st.selectbox("群組", groups, label_visibility="collapsed")
    with col_btn:
        add_clicked = st.button("➕ 加入", use_container_width=True, key="wl_add_btn")

    if add_clicked and new_ticker.strip():
        ticker_clean = new_ticker.strip().upper()
        # Try to get name from provider
        try:
            market = "TW" if _is_tw_ticker(ticker_clean) else "US"
            provider = get_data_provider("auto", market_type=market)
            info = provider.get_stock_info(ticker_clean)
            name = info.get("name", ticker_clean)
        except Exception:
            name = ticker_clean

        if wm.add(ticker_clean, name=name, group=selected_group):
            st.success(f"已加入 {ticker_clean} ({name}) 到 [{selected_group}]")
            st.rerun()
        else:
            st.warning(f"{ticker_clean} 已在自選股中")

    st.markdown("---")

    # --- Group Filter ---
    if "watchlist_filter" not in st.session_state:
        st.session_state.watchlist_filter = "全部"

    all_groups = ["全部"] + wm.get_groups()
    filter_cols = st.columns(len(all_groups))
    for i, grp in enumerate(all_groups):
        with filter_cols[i]:
            if st.button(grp, key=f"wl_grp_{grp}", use_container_width=True,
                         type="primary" if st.session_state.watchlist_filter == grp else "secondary"):
                st.session_state.watchlist_filter = grp
                st.rerun()

    # --- Get filtered stocks ---
    stocks = wm.get_all()
    active_filter = st.session_state.watchlist_filter
    if active_filter != "全部":
        stocks = [s for s in stocks if s.get("group") == active_filter]

    if not stocks:
        st.markdown("""
        <div class="alert-card alert-info" style="text-align:center; padding:40px;">
            <div class="alert-title" style="font-size:1.1rem;">尚無自選股</div>
            <div class="alert-body">在上方輸入股票代碼並點擊「加入」開始追蹤</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- Batch fetch all price data with cyber_spinner ---
    all_tickers = tuple(s["ticker"] for s in stocks)

    # Check if any data is uncached (needs fetching)
    _needs_fetch = False
    for t in all_tickers:
        if _fetch_price_data(t) is None:
            # Could be genuinely no data, or first load — either way we show spinner on first batch
            pass
    # Use session key to track initial load
    _wl_cache_key = f"_wl_loaded_{hash(all_tickers)}"
    if _wl_cache_key not in st.session_state:
        with cyber_spinner("LOADING", f"自選股報價載入中... ({len(stocks)} 檔)"):
            price_cache = {}
            for t in all_tickers:
                price_cache[t] = _fetch_price_data(t)
        st.session_state[_wl_cache_key] = True
    else:
        price_cache = {t: _fetch_price_data(t) for t in all_tickers}

    # --- Refresh button ---
    col_title, col_refresh = st.columns([6, 2])
    with col_title:
        st.markdown(f'<p class="sec-header">自選股 ({len(stocks)} 檔)</p>', unsafe_allow_html=True)
    with col_refresh:
        if st.button("🔄 刷新報價", use_container_width=True, key="wl_refresh"):
            _fetch_price_data.clear()
            st.session_state.pop(_wl_cache_key, None)
            st.rerun()

    headers = ["代碼", "名稱", "現價", "漲跌%", "成交量", "RSI", "趨勢", "群組"]
    rows = []
    rsi_alerts = []  # Collect RSI extreme stocks for trader alerts

    for s in stocks:
        ticker = s["ticker"]
        name = s.get("name", ticker)
        group = s.get("group", "預設")

        data = price_cache.get(ticker)
        if data:
            price_str = f"{data['price']:,.2f}"
            chg_str = _tw_color_pct(data['change_pct'])
            vol_str = f"{data['volume']:,.0f}"

            # RSI with color coding: >70 overbought (red), <30 oversold (green)
            rsi = data['rsi']
            if rsi is not None:
                if rsi >= 70:
                    rsi_str = f'<span style="color:#ef4444;font-weight:700">{rsi:.1f}</span>'
                    rsi_alerts.append(f"{ticker} {name} RSI={rsi:.1f} 超買")
                elif rsi <= 30:
                    rsi_str = f'<span style="color:#22c55e;font-weight:700">{rsi:.1f}</span>'
                    rsi_alerts.append(f"{ticker} {name} RSI={rsi:.1f} 超賣")
                else:
                    rsi_str = f"{rsi:.1f}"
            else:
                rsi_str = "—"

            trend = data['trend']
            if trend == "多":
                trend_html = '<span class="tag tag-bull">多</span>'
            elif trend == "空":
                trend_html = '<span class="tag tag-bear">空</span>'
            else:
                trend_html = '<span class="tag" style="background:#1e293b;color:#94a3b8">—</span>'
        else:
            price_str = "N/A"
            chg_str = "—"
            vol_str = "—"
            rsi_str = "—"
            trend_html = "—"

        group_html = f'<span class="tag tag-new">{group}</span>'
        rows.append([ticker, name, price_str, chg_str, vol_str, rsi_str, trend_html, group_html])

    # Show RSI alerts if any
    if rsi_alerts:
        alerts_html = " | ".join(rsi_alerts)
        st.markdown(
            f'<div class="alert-card alert-warn" style="padding:8px 14px;margin-bottom:8px">'
            f'<div class="alert-title" style="font-size:0.8rem">RSI 極端值提醒</div>'
            f'<div class="alert-body" style="font-size:0.75rem">{alerts_html}</div></div>',
            unsafe_allow_html=True,
        )

    # Render HTML table
    th_html = "".join(
        f'<th style="text-align:{"left" if i == 0 else "right"}">{h}</th>'
        for i, h in enumerate(headers)
    )
    tr_html = ""
    for row in rows:
        td_html = "".join(f"<td>{cell}</td>" for cell in row)
        tr_html += f"<tr>{td_html}</tr>"

    st.markdown(f"""<table class="pro-table">
    <thead><tr>{th_html}</tr></thead>
    <tbody>{tr_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # --- Delete buttons (Streamlit buttons below table) ---
    st.markdown('<p class="sec-header">刪除操作</p>', unsafe_allow_html=True)
    del_cols = st.columns(min(len(stocks), 6))
    for i, s in enumerate(stocks):
        col_idx = i % min(len(stocks), 6)
        with del_cols[col_idx]:
            if st.button(f"🗑 {s['ticker']}", key=f"del_{s['ticker']}"):
                wm.remove(s['ticker'])
                st.rerun()
