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
    ORDER_LOG_PATH, AUTO_TRADE_CONFIG_PATH, V4_SIGNALS_PATH,
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
    """即時報價：SinoPac 快照優先 → YFinance 歷史備援。"""
    from data.provider import get_data_provider, SinoPacProvider
    prices = {}
    source_label = "N/A"

    # Phase 1: SinoPac 即時快照（速度最快、資料最即時）
    try:
        sinopac = SinoPacProvider()
        snapshots = sinopac.get_snapshots(list(tickers_tuple))
        if snapshots:
            for snap in snapshots:
                code = getattr(snap, 'code', '') or ''
                close = getattr(snap, 'close', 0) or 0
                if code and close > 0:
                    prices[code] = float(close)
            if prices:
                source_label = "SinoPac"
    except (ConnectionError, TimeoutError, AttributeError) as e:
        logger.debug(f"SinoPac 即時報價失敗: {e}")

    # Phase 2: 補齊缺失的 ticker（YFinance fallback）
    missing = [t for t in tickers_tuple if t not in prices]
    if missing:
        try:
            provider = get_data_provider("auto", market_type="TW")
            for ticker in missing:
                try:
                    df = provider.get_historical_data(ticker, period="5d", interval="1d")
                    if df is not None and not df.empty and 'Close' in df.columns:
                        prices[ticker] = float(df['Close'].iloc[-1])
                except (ConnectionError, TimeoutError, KeyError, IndexError) as e:
                    logger.debug(f"YFinance 報價失敗 ({ticker}): {e}")
            if source_label == "N/A" and prices:
                source_label = "YFinance"
            elif missing and any(t in prices for t in missing):
                source_label = f"{source_label}+YFinance" if source_label != "N/A" else "YFinance"
        except (ImportError, ConnectionError) as e:
            logger.warning(f"YFinance fallback 報價失敗: {e}")

    return prices, source_label


def _refresh_live_prices(positions):
    tickers = tuple(p.get('ticker', '') for p in positions if p.get('ticker'))
    if not tickers:
        return
    cache_key = '_dash_price_cache'
    cache_ts_key = '_dash_price_ts'
    cache_src_key = '_dash_price_source'
    ttl = _get_price_ttl()
    now = datetime.now(TW_TZ).timestamp()
    cached = st.session_state.get(cache_key)
    cached_ts = st.session_state.get(cache_ts_key, 0)
    if cached and (now - cached_ts) < ttl and set(tickers).issubset(set(cached.keys())):
        prices = cached
    else:
        prices, source_label = _fetch_prices_batch(tickers)
        if prices:
            st.session_state[cache_key] = prices
            st.session_state[cache_ts_key] = now
            st.session_state[cache_src_key] = source_label
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
    except Exception as e:
        logger.warning(f"載入持倉資料失敗: {e}")
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
    except (TypeError, KeyError, AttributeError) as e:
        logger.warning(f"載入掃描結果失敗: {e}")
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
            except (ValueError, TypeError):
                continue
        return upcoming
    except (ImportError, AttributeError) as e:
        logger.debug(f"載入行事曆失敗: {e}")
        return []


_REGIME_ZH = {
    'strong_bull': '強多頭', 'weak_bull': '弱多頭', 'sideways': '盤整',
    'weak_bear': '弱空頭', 'strong_bear': '強空頭',
}
_REGIME_COLOR = {
    'strong_bull': '#22c55e', 'weak_bull': '#84cc16', 'sideways': '#94a3b8',
    'weak_bear': '#f59e0b', 'strong_bear': '#ef4444',
}


def _load_v4_signals():
    """Load V4 daily scan signals."""
    try:
        data = safe_json_read(V4_SIGNALS_PATH, {})
        if not data or not data.get('date'):
            return None
        return data
    except Exception as e:
        logger.warning(f"載入 V4 信號失敗: {e}")
        return None


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
    except (TypeError, KeyError, AttributeError) as e:
        logger.warning(f"載入委託記錄失敗: {e}")
        return [], []


def _render_v4_signals(v4_data):
    """Render V4 daily signal section."""
    date = v4_data.get('date', '')
    regime = v4_data.get('regime', 'unknown')
    regime_label = _REGIME_ZH.get(regime, regime)
    regime_color = _REGIME_COLOR.get(regime, '#94a3b8')
    allocations = v4_data.get('allocations', {})
    variant_entries = v4_data.get('variant_new_entries', {})
    strategy_signals = v4_data.get('strategy_signals', {})

    # Header with regime badge
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
        f'<span style="font-size:1.1rem;font-weight:700;color:#e2e8f0">V4 每日信號</span>'
        f'<span style="font-size:0.7rem;color:#64748b">{date}</span>'
        f'<span style="background:{regime_color};color:#0f172a;padding:2px 10px;'
        f'border-radius:4px;font-size:0.7rem;font-weight:700">{regime_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Allocation overview (3 columns)
    cols = st.columns(3)
    for i, vk in enumerate(['V4.0', 'V4.1', 'V4.2']):
        alloc = allocations.get(vk, {})
        weights = alloc.get('weights', {})
        with cols[i]:
            st.markdown(
                f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:10px;'
                f'border-left:3px solid #8b5cf6">'
                f'<div style="font-size:0.75rem;color:#a78bfa;font-weight:700;margin-bottom:6px">{vk}</div>',
                unsafe_allow_html=True,
            )
            if weights:
                for sname, w in weights.items():
                    if w > 0.01:
                        bar_w = int(w * 100)
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
                            f'<span style="font-size:0.65rem;color:#cbd5e1;min-width:90px">{sname}</span>'
                            f'<div style="flex:1;height:8px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden">'
                            f'<div style="width:{bar_w}%;height:100%;background:#8b5cf6;border-radius:2px"></div></div>'
                            f'<span style="font-size:0.65rem;color:#a78bfa;font-family:JetBrains Mono,monospace">{bar_w}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    '<span style="font-size:0.7rem;color:#475569">N/A</span>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # New entries table
    all_entries = []
    for vk in ['V4.0', 'V4.1', 'V4.2']:
        for e in variant_entries.get(vk, []):
            e['variant'] = vk
            all_entries.append(e)

    if all_entries:
        # Deduplicate by ticker, keep all variant info
        seen = {}
        for e in all_entries:
            t = e['ticker']
            if t not in seen:
                seen[t] = {**e, 'variants': [e['variant']]}
            else:
                if e['variant'] not in seen[t]['variants']:
                    seen[t]['variants'].append(e['variant'])
        unique_entries = sorted(seen.values(), key=lambda x: -x.get('score', 0))

        st.markdown(
            '<div style="margin-top:10px;font-size:0.85rem;font-weight:700;color:#f59e0b">'
            '🆕 今日新進場信號</div>',
            unsafe_allow_html=True,
        )
        headers = ['代碼', '名稱', '現價', 'Score', '策略', '出現版本']
        rows = []
        for e in unique_entries[:15]:
            variants_str = ' '.join(
                f'<span class="tag tag-bull" style="font-size:0.55rem">{v}</span>'
                for v in e['variants']
            )
            price = e.get('price', 0)
            rows.append([
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700">{e["ticker"]}</span>',
                e.get('name', '')[:6],
                f'{price:,.1f}' if price else '—',
                f'{e.get("score", 0):.1f}',
                e.get('strategy', ''),
                variants_str,
            ])
        cyber_table(headers, rows)
    else:
        st.markdown(
            '<div style="margin-top:10px;padding:12px;background:rgba(0,0,0,0.15);'
            'border-radius:8px;text-align:center;color:#64748b;font-size:0.8rem">'
            '✅ 今日無新進場信號</div>',
            unsafe_allow_html=True,
        )

    # Sub-strategy summary (compact)
    if strategy_signals:
        st.markdown(
            '<div style="margin-top:8px;display:flex;gap:12px;flex-wrap:wrap">',
            unsafe_allow_html=True,
        )
        for sname, sig in strategy_signals.items():
            n_h = sig.get('n_holdings', 0)
            n_in = sig.get('n_entered', 0)
            n_out = sig.get('n_exited', 0)
            in_color = '#22c55e' if n_in > 0 else '#475569'
            out_color = '#ef4444' if n_out > 0 else '#475569'
            st.markdown(
                f'<span style="font-size:0.65rem;color:#94a3b8;background:rgba(0,0,0,0.2);'
                f'padding:3px 8px;border-radius:4px">'
                f'{sname}: <b>{n_h}</b>檔 '
                f'<span style="color:{in_color}">+{n_in}</span> '
                f'<span style="color:{out_color}">-{n_out}</span>'
                f'</span>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


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

        source_label = st.session_state.get('_dash_price_source', 'N/A')
        src_color = '#22c55e' if source_label == 'SinoPac' else '#f59e0b' if source_label == 'YFinance' else '#64748b'
        st.markdown(
            f'<div style="font-size:0.6rem;color:#64748b;text-align:right;font-family:JetBrains Mono,monospace">'
            f'報價來源: <span style="color:{src_color}">{source_label}</span></div>',
            unsafe_allow_html=True,
        )

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

        # ── V4 Daily Signals ──
        v4_data = _load_v4_signals()
        if v4_data:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            _render_v4_signals(v4_data)

        # ── Risk Dashboard: Daily VaR + Performance Attribution ──
        if paper['positions']:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            col_var, col_attr = st.columns(2)

            with col_var:
                st.markdown("#### 每日 VaR 風險值")
                try:
                    from analysis.risk_calc import calculate_var
                    from data.provider import get_data_provider
                    provider = get_data_provider("auto", market_type="TW")
                    var_result = calculate_var(paper['positions'], provider, confidence=0.95, period='6mo')
                    if var_result:
                        var_color = "#ef4444" if var_result['var_amount'] < 0 else "#22c55e"
                        cvar_color = "#ef4444" if var_result['cvar_amount'] < 0 else "#22c55e"
                        st.markdown(
                            f'<div style="padding:12px;background:rgba(0,0,0,0.2);border-radius:8px;'
                            f'border-left:3px solid {var_color}">'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
                            f'<div><span style="color:#64748b;font-size:0.7rem">95% VaR</span><br>'
                            f'<span style="color:{var_color};font-size:1.1rem;font-weight:700;font-family:JetBrains Mono,monospace">'
                            f'${abs(var_result["var_amount"]):,.0f}</span></div>'
                            f'<div style="text-align:right"><span style="color:#64748b;font-size:0.7rem">CVaR (ES)</span><br>'
                            f'<span style="color:{cvar_color};font-size:1.1rem;font-weight:700;font-family:JetBrains Mono,monospace">'
                            f'${abs(var_result["cvar_amount"]):,.0f}</span></div></div>'
                            f'<div style="font-size:0.7rem;color:#94a3b8">{var_result["interpretation"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("VaR 資料不足（需至少 30 天歷史資料）")
                except Exception as e:
                    st.caption(f"VaR 計算暫時無法使用: {type(e).__name__}")

            with col_attr:
                st.markdown("#### 損益歸因")
                try:
                    from analysis.attribution import calculate_attribution
                    from data.provider import get_data_provider
                    provider = get_data_provider("auto", market_type="TW")
                    attr_results, total_pnl = calculate_attribution(paper['positions'], provider)
                    if attr_results:
                        # 簡易歸因表
                        for r in attr_results[:5]:
                            pnl = r['daily_pnl']
                            pnl_color = "#ef4444" if pnl >= 0 else "#22c55e"
                            contrib = r['contribution_pct']
                            bar_width = min(abs(contrib), 100)
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;'
                                f'border-bottom:1px solid rgba(255,255,255,0.05)">'
                                f'<span style="font-size:0.75rem;color:#e2e8f0;min-width:80px">{r["name"][:6]}</span>'
                                f'<div style="flex:1;height:12px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden">'
                                f'<div style="width:{bar_width}%;height:100%;background:{pnl_color};border-radius:2px"></div></div>'
                                f'<span style="font-size:0.7rem;color:{pnl_color};font-family:JetBrains Mono,monospace;min-width:70px;text-align:right">'
                                f'${pnl:+,.0f}</span></div>',
                                unsafe_allow_html=True,
                            )
                        total_color = "#ef4444" if total_pnl >= 0 else "#22c55e"
                        st.markdown(
                            f'<div style="text-align:right;margin-top:6px;font-size:0.75rem;color:{total_color};'
                            f'font-family:JetBrains Mono,monospace;font-weight:700">合計: ${total_pnl:+,.0f}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("歸因分析需要持倉資料")
                except Exception as e:
                    st.caption(f"歸因分析暫時無法使用: {type(e).__name__}")

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
    except (TypeError, ZeroDivisionError):
        pass
    return 0
