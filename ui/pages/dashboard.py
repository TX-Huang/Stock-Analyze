"""
Page: Dashboard (交易總覽) — Morning briefing page.
Aggregates holdings, signals, calendar events, and pending orders into one view.
"""
import streamlit as st
import logging
from datetime import datetime, timedelta, timezone

from ui.components import cyber_header, cyber_kpi_strip, cyber_table
from ui.theme import _tw_color, _tw_color_pct, _tw_tag
from utils.helpers import safe_json_read
from config.paths import (
    PAPER_TRADE_PATH, SCAN_RESULTS_PATH, RECOMMENDATION_PATH,
    ORDER_LOG_PATH, AUTO_TRADE_CONFIG_PATH,
)

logger = logging.getLogger(__name__)
TW_TZ = timezone(timedelta(hours=8))

# ── Signal translation ──
_SIG_ZH = {
    'volume_breakout': '帶量突破', 'vcp_breakout': 'VCP突破',
    'breakout': '突破壓力', 'vcp_ready': 'VCP成形',
    'near_resistance': '即將觸壓', 'support_bounce': '支撐反彈',
    'break_support': '跌破支撐', 'volume_break_support': '帶量跌破',
    'None': '', None: '',
}


def _is_market_hours():
    now = datetime.now(TW_TZ)
    if now.weekday() >= 5:
        return False
    from datetime import time as _time
    return _time(8, 45) <= now.time() <= _time(13, 35)


def _get_price_ttl():
    return 10 if _is_market_hours() else 1800


def _fetch_prices_batch(tickers_tuple):
    from data.provider import get_data_provider, SinoPacProvider
    prices = {}
    try:
        sinopac = SinoPacProvider()
        snapshots = sinopac.get_snapshots(list(tickers_tuple))
        if snapshots:
            for snap in snapshots:
                code = getattr(snap, 'code', '') or ''
                close = getattr(snap, 'close', 0) or 0
                if code and close > 0:
                    prices[code] = float(close)
            if len(prices) >= len(tickers_tuple) * 0.5:
                return prices
    except Exception:
        pass
    try:
        provider = get_data_provider("auto", market_type="TW")
        for ticker in tickers_tuple:
            if ticker in prices:
                continue
            try:
                df = provider.get_historical_data(ticker, period="5d", interval="1d")
                if df is not None and not df.empty and 'Close' in df.columns:
                    prices[ticker] = float(df['Close'].iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return prices


def _refresh_live_prices(positions):
    tickers = tuple(p.get('ticker', '') for p in positions if p.get('ticker'))
    if not tickers:
        return
    cache_key = '_dash_price_cache'
    cache_ts_key = '_dash_price_ts'
    ttl = _get_price_ttl()
    now = datetime.now(TW_TZ).timestamp()
    cached = st.session_state.get(cache_key)
    cached_ts = st.session_state.get(cache_ts_key, 0)
    if cached and (now - cached_ts) < ttl and set(tickers).issubset(set(cached.keys())):
        prices = cached
    else:
        prices = _fetch_prices_batch(tickers)
        if prices:
            st.session_state[cache_key] = prices
            st.session_state[cache_ts_key] = now
    if not prices:
        return
    for p in positions:
        ticker = p.get('ticker', '')
        if ticker in prices:
            p['current_price'] = prices[ticker]


def _load_paper_positions():
    """Load paper trading positions with live price refresh."""
    try:
        data = safe_json_read(PAPER_TRADE_PATH, {})
        positions = data.get('positions', [])
        cash = data.get('cash', 0)
        initial = data.get('initial_capital', 1_000_000)

        if positions:
            _refresh_live_prices(positions)

        pos_value = sum(
            p.get('current_price', p.get('entry_price', 0)) * p.get('shares', 0)
            for p in positions
        )
        equity = cash + pos_value
        today_pnl = sum(
            (p.get('current_price', p.get('entry_price', 0)) - p.get('entry_price', 0)) * p.get('shares', 0)
            for p in positions
        )
        return {
            'positions': positions,
            'equity': equity,
            'cash': cash,
            'initial': initial,
            'today_pnl': today_pnl,
            'n_positions': len(positions),
        }
    except Exception:
        return {
            'positions': [], 'equity': 0, 'cash': 0,
            'initial': 1_000_000, 'today_pnl': 0, 'n_positions': 0,
        }


def _load_scan_signals():
    """Load breakout scan results."""
    try:
        data = safe_json_read(SCAN_RESULTS_PATH, {})
        if isinstance(data, list):
            return data
        return data.get('signals', data.get('results', []))
    except Exception:
        return []


def _load_calendar_events():
    """Load upcoming calendar events (next 4 days)."""
    try:
        from ui.pages.calendar import EVENTS_2026
        today = datetime.now().date()
        end = today + timedelta(days=3)
        upcoming = []
        for evt in EVENTS_2026:
            try:
                evt_date = datetime.strptime(evt.get('date', ''), '%Y-%m-%d').date()
                if today <= evt_date <= end:
                    upcoming.append(evt)
            except Exception:
                continue
        return upcoming
    except Exception:
        return []


def _load_pending_orders():
    """Load pending/today's orders from auto_trader order log."""
    try:
        data = safe_json_read(ORDER_LOG_PATH, {})
        orders = data.get('orders', [])
        today_str = datetime.now().strftime('%Y-%m-%d')
        today_orders = [o for o in orders if o.get('date', '') == today_str]
        # Also check for orders with status 'submitted' (pending)
        pending = [o for o in orders if o.get('status') == 'submitted']
        return today_orders, pending
    except Exception:
        return [], []


def render():
    """Render the dashboard page."""
    c_title, c_refresh = st.columns([8, 2])
    with c_title:
        cyber_header("交易總覽", "持倉損益 | 今日信號 | 待辦事項")
    with c_refresh:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        if st.button("🔄 刷新報價", use_container_width=True):
            st.session_state.pop('_dash_price_cache', None)
            st.session_state.pop('_dash_price_ts', None)
            st.rerun()
        ttl = _get_price_ttl()
        cached_ts = st.session_state.get('_dash_price_ts', 0)
        if cached_ts:
            age = int(datetime.now(TW_TZ).timestamp() - cached_ts)
            market_label = "🟢 盤中" if _is_market_hours() else "🔴 盤後"
            st.markdown(
                f'<div style="font-size:0.6rem;color:#64748b;text-align:center;font-family:JetBrains Mono,monospace">'
                f'{market_label} | {ttl}s | {age}s前</div>',
                unsafe_allow_html=True,
            )

    try:
        # ── Load all data ──
        paper = _load_paper_positions()
        signals = _load_scan_signals()
        rec = safe_json_read(RECOMMENDATION_PATH, {})
        today_orders, pending_orders = _load_pending_orders()
        calendar_events = _load_calendar_events()

        # Count today's signals
        n_signals = len(signals)
        if rec:
            n_signals = max(n_signals, len(rec.get('recommendations', [])))

        # ── KPI Strip ──
        equity = paper['equity']
        equity_str = f"${equity:,.0f}" if equity else "$0"
        pnl = paper['today_pnl']
        pnl_color = "#ef4444" if pnl > 0 else "#22c55e" if pnl < 0 else "#94a3b8"
        pnl_str = f"${pnl:+,.0f}" if pnl else "$0"

        cyber_kpi_strip([
            {'label': '總資產', 'value': equity_str, 'accent': '#8b5cf6'},
            {'label': '今日損益', 'value': pnl_str, 'color': pnl_color, 'accent': pnl_color},
            {'label': '未平倉部位', 'value': str(paper['n_positions']), 'accent': '#3b82f6'},
            {'label': '今日信號', 'value': str(n_signals), 'accent': '#f59e0b'},
        ])

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── 3-Column Layout ──
        col_left, col_right = st.columns([6, 4])

        # ── Left: Holdings Snapshot ──
        with col_left:
            st.markdown("#### 持倉快照")
            # P5: Source filter
            source_filter = st.radio(
                "篩選", ["全部", "手動", "策略"],
                horizontal=True, key="dash_source_filter",
                label_visibility="collapsed",
            )
            all_positions = paper['positions']
            if source_filter == "手動":
                positions = [p for p in all_positions if p.get('source', 'strategy:isaac') == 'manual']
            elif source_filter == "策略":
                positions = [p for p in all_positions if p.get('source', 'strategy:isaac').startswith('strategy:')]
            else:
                positions = all_positions
            if positions:
                # Build signal lookup from scan results
                signal_map = {}
                for sig in signals:
                    ticker = sig.get('ticker', sig.get('code', ''))
                    if ticker:
                        raw = sig.get('signal', sig.get('type', ''))
                        signal_map[ticker] = _SIG_ZH.get(raw, raw or '')

                headers = ['代碼', '名稱', '現價', '損益%', '損益$', '信號']
                rows = []
                for p in sorted(positions, key=lambda x: x.get('pnl_pct', _calc_pnl_pct(x)), reverse=True):
                    ticker = p.get('ticker', '')
                    name = p.get('name', '')[:6]
                    cur_price = p.get('current_price', p.get('entry_price', 0))
                    entry_price = p.get('entry_price', 0)
                    shares = p.get('shares', 0)

                    # Calculate PnL
                    if entry_price > 0 and shares > 0:
                        gross = (cur_price - entry_price) * shares
                        est_fees = entry_price * shares * 0.001425 + cur_price * shares * (0.001425 + 0.003)
                        pnl_abs = gross - est_fees
                        pnl_pct = pnl_abs / (entry_price * shares) * 100
                    else:
                        pnl_abs = 0
                        pnl_pct = 0

                    # Signal status
                    sig_status = signal_map.get(ticker, '')
                    if sig_status:
                        sig_html = f'<span class="tag tag-bull" style="font-size:0.65rem">{sig_status}</span>'
                    else:
                        sig_html = '<span style="color:#475569">--</span>'

                    rows.append([
                        f'<span style="font-family:JetBrains Mono,monospace;font-weight:700">{ticker}</span>',
                        name,
                        f'{cur_price:,.1f}',
                        _tw_color_pct(pnl_pct),
                        _tw_color(pnl_abs, fmt="+,.0f"),
                        sig_html,
                    ])

                cyber_table(headers, rows)
            else:
                st.info("目前無持倉資料。執行 Paper Trader 後即可顯示。")

        # ── Right Column ──
        with col_right:
            # ── Calendar Events ──
            st.markdown("#### 今日行事")
            if calendar_events:
                for evt in calendar_events[:5]:
                    evt_date = evt.get('date', '')
                    evt_title = evt.get('title', evt.get('event', ''))
                    evt_tag = evt.get('tag', '')
                    tag_html = f' <span class="tag tag-warn" style="font-size:0.6rem">{evt_tag}</span>' if evt_tag else ''
                    st.markdown(
                        f'<div style="padding:4px 0;border-bottom:1px solid #1e293b;">'
                        f'<span style="color:#64748b;font-size:0.75rem">{evt_date}</span> '
                        f'<span style="color:#e2e8f0">{evt_title}</span>{tag_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("近期無行事曆事件")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # ── Breakout Signal Summary ──
            st.markdown("#### 突破信號摘要")
            if signals:
                headers_sig = ['代碼', '名稱', '類型', '價格']
                rows_sig = []
                for sig in signals[:5]:
                    ticker = sig.get('ticker', sig.get('code', ''))
                    sig_name = sig.get('name', '')
                    raw_type = sig.get('signal', sig.get('type', ''))
                    sig_type = _SIG_ZH.get(raw_type, raw_type or '—')
                    price = sig.get('close', sig.get('price', 0))
                    rows_sig.append([
                        f'<span style="font-family:JetBrains Mono,monospace">{ticker}</span>',
                        sig_name,
                        f'<span class="tag tag-bull" style="font-size:0.65rem">{sig_type}</span>',
                        f'{price:,.1f}' if isinstance(price, (int, float)) else str(price),
                    ])
                cyber_table(headers_sig, rows_sig)
            elif rec and rec.get('recommendations'):
                # Fallback: show recommendations as signals
                headers_sig = ['代碼', '名稱', 'Score']
                rows_sig = []
                for r in rec['recommendations'][:5]:
                    rows_sig.append([
                        f'<span style="font-family:JetBrains Mono,monospace">{r.get("ticker", "")}</span>',
                        r.get('name', ''),
                        f'{r.get("score", 0):.0f}',
                    ])
                cyber_table(headers_sig, rows_sig)
            else:
                st.caption("目前無突破信號")

        # ── Bottom: Pending Orders ──
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        all_pending = today_orders + [o for o in pending_orders if o not in today_orders]
        if all_pending:
            st.markdown("#### 待執行委託")
            headers_ord = ['動作', '代碼', '名稱', '價格', '股數', '狀態']
            rows_ord = []
            for o in all_pending[:10]:
                action = o.get('action', '')
                action_cls = 'tag-bull' if action == 'BUY' else 'tag-bear'
                status = o.get('status', '')
                status_color = '#22c55e' if status == 'filled' else '#f59e0b' if status == 'submitted' else '#ef4444'
                rows_ord.append([
                    f'<span class="tag {action_cls}" style="font-size:0.65rem">{action}</span>',
                    f'<span style="font-family:JetBrains Mono,monospace">{o.get("ticker", "")}</span>',
                    o.get('name', ''),
                    f'{o.get("price", 0):,.1f}',
                    f'{o.get("shares", 0):,}',
                    f'<span style="color:{status_color}">{status}</span>',
                ])
            cyber_table(headers_ord, rows_ord)

    except Exception as e:
        st.error(f"Dashboard 載入錯誤: {e}")
        st.caption("請確認資料檔案是否存在，或先執行一次策略推薦。")


def _calc_pnl_pct(p):
    """Helper to calculate PnL% for sorting."""
    try:
        cur = p.get('current_price', p.get('entry_price', 0))
        entry = p.get('entry_price', 0)
        if entry > 0:
            return (cur - entry) / entry * 100
    except Exception:
        pass
    return 0
