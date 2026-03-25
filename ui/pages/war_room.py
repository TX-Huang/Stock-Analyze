"""
Page: AI 戰情室 (War Room) — Default landing page for AI Invest HQ.

Layers:
1. Market overview KPI strip (index, volume, advance/decline, sentiment)
2. AI scan results — opportunity cards sorted by confidence
3. Portfolio risk summary — exposure, unrealized P&L, warnings
4. Quick diagnosis search bar — inline stock report
"""
import streamlit as st
import logging
import os
import re
import pandas as pd
from datetime import datetime, timedelta, timezone

from ui.theme import inject_cyber_theme
from ui.components import custom_metric, cyber_spinner
from utils.helpers import safe_json_read
from config.paths import PAPER_TRADE_PATH, SCAN_RESULTS_PATH


def _detect_market_type(ticker_input: str) -> str:
    """Auto-detect market type from ticker input.

    Returns 'TW', 'US', or 'unknown'.
    - Pure digits (e.g. '2330') -> TW
    - Pure letters (e.g. 'AAPL', 'TSLA') -> US
    - Mixed / Chinese characters -> 'unknown' (needs AI resolution)
    """
    text = str(ticker_input).strip()
    if re.match(r'^\d{4,6}[A-Za-z]?$', text):
        return 'TW'
    if re.match(r'^[A-Za-z]{1,5}$', text):
        return 'US'
    return 'unknown'

logger = logging.getLogger(__name__)
TW_TZ = timezone(timedelta(hours=8))

# ── Signal label translation ──
_SIG_ZH = {
    'volume_breakout': '帶量突破', 'vcp_breakout': 'VCP突破',
    'breakout': '突破壓力', 'vcp_ready': 'VCP成形',
    'near_resistance': '即將觸壓', 'support_bounce': '支撐反彈',
    'break_support': '跌破支撐', 'volume_break_support': '帶量跌破',
    'None': '', None: '',
}

# ── Confidence badge colors ──
_CONFIDENCE_COLORS = {
    'high': ('#ef4444', 'rgba(239,68,68,0.15)', '#fca5a5'),
    'medium': ('#f59e0b', 'rgba(245,158,11,0.12)', '#fcd34d'),
    'low': ('#3b82f6', 'rgba(59,130,246,0.12)', '#93c5fd'),
}


def _confidence_level(score):
    """Map numeric confidence to level string."""
    if score is None:
        return 'low'
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 'low'
    if s >= 7:
        return 'high'
    if s >= 4:
        return 'medium'
    return 'low'


def _load_scan_results():
    """Load scan results from JSON, returning (results_list, scan_time, error)."""
    try:
        data = safe_json_read(SCAN_RESULTS_PATH, None)
        if data is None:
            return [], None, None

        if isinstance(data, list):
            results = data
            scan_time = None
        else:
            results = data.get('signals', data.get('results', []))
            scan_time = data.get('scan_time', data.get('timestamp'))

        return results, scan_time, None
    except Exception as e:
        logger.warning(f"載入掃描結果失敗: {e}")
        return [], None, str(e)


def _load_portfolio():
    """Load portfolio data for risk summary."""
    try:
        data = safe_json_read(PAPER_TRADE_PATH, {})
        positions = data.get('positions', [])
        cash = data.get('cash', 0)
        initial = data.get('initial_capital', 1_000_000)

        total_value = sum(
            p.get('current_price', p.get('entry_price', 0)) * p.get('shares', 0)
            for p in positions
        )
        equity = cash + total_value

        unrealized_pnl = sum(
            (p.get('current_price', p.get('entry_price', 0)) - p.get('entry_price', 0))
            * p.get('shares', 0)
            for p in positions
        )

        return {
            'positions': positions,
            'equity': equity,
            'cash': cash,
            'initial': initial,
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'n_positions': len(positions),
        }
    except Exception as e:
        logger.warning(f"載入持倉資料失敗: {e}")
        return {
            'positions': [], 'equity': 0, 'cash': 0,
            'initial': 1_000_000, 'total_value': 0,
            'unrealized_pnl': 0, 'n_positions': 0,
        }


def _scan_freshness_indicator(scan_time):
    """Render scan freshness badge. Returns None if no scan_time available."""
    if not scan_time:
        return
    try:
        if isinstance(scan_time, str):
            dt = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
        else:
            return
        now = datetime.now(tz=TW_TZ)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TW_TZ)
        age = now - dt
        hours = age.total_seconds() / 3600

        if hours < 1:
            color = '#22c55e'
            label = '最新掃描'
            bg = 'rgba(34,197,94,0.12)'
        elif hours < 24:
            color = '#94a3b8'
            label = f'{int(hours)} 小時前掃描'
            bg = 'rgba(148,163,184,0.08)'
        else:
            color = '#f59e0b'
            label = '掃描結果已超過 24 小時，建議重新掃描'
            bg = 'rgba(245,158,11,0.12)'

        st.markdown(
            f'<div style="display:inline-block;padding:3px 10px;border-radius:4px;'
            f'font-size:0.72rem;font-family:JetBrains Mono,monospace;'
            f'color:{color};background:{bg};border:1px solid {color}30;margin-bottom:8px">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )
    except (ValueError, TypeError):
        pass


def _render_market_kpi():
    """Layer 1: Market overview KPI strip.

    Auto-detects market context: if the user's last search was a US stock,
    show S&P 500 / NASDAQ data instead of TAIEX.
    """
    current_market = st.session_state.get('_war_room_market', 'TW')

    if current_market == 'US':
        _render_us_market_kpi()
    else:
        _render_tw_market_kpi()


def _render_tw_market_kpi():
    """Render Taiwan market KPI strip (TAIEX)."""
    st.markdown(
        '<div class="sec-header">\U0001f1f9\U0001f1fc 今日市場概覽</div>',
        unsafe_allow_html=True,
    )

    from config.paths import RECOMMENDATION_PATH
    rec = safe_json_read(RECOMMENDATION_PATH, {})
    market = rec.get('market_overview', {})

    index_val = market.get('taiex', market.get('index', '--'))
    index_chg = market.get('taiex_change', market.get('change', 0))
    volume_b = market.get('volume', '--')
    adv_dec = market.get('advance_decline', market.get('breadth', '--'))
    sentiment = market.get('sentiment_score', market.get('sentiment', '--'))

    if isinstance(index_val, (int, float)):
        idx_str = f"{index_val:,.0f}"
    else:
        idx_str = str(index_val)

    if isinstance(index_chg, (int, float)):
        idx_delta = f"{index_chg:+.0f}" if abs(index_chg) > 1 else f"{index_chg:+.2f}%"
    else:
        idx_delta = None

    if isinstance(volume_b, (int, float)):
        vol_str = f"{volume_b/1e8:,.0f} 億" if volume_b > 1e7 else str(volume_b)
    else:
        vol_str = str(volume_b)

    if isinstance(sentiment, (int, float)):
        sent_str = f"{sentiment:.0f}/100"
    else:
        sent_str = str(sentiment)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("加權指數", idx_str, delta=idx_delta)
    with c2:
        st.metric("成交量", vol_str)
    with c3:
        st.metric("漲跌比", str(adv_dec))
    with c4:
        st.metric("大盤情緒", sent_str)


def _render_us_market_kpi():
    """Render US market KPI strip (S&P 500 / NASDAQ) via YFinance."""
    st.markdown(
        '<div class="sec-header">\U0001f1fa\U0001f1f8 US Market Overview</div>',
        unsafe_allow_html=True,
    )

    sp500 = '--'
    sp500_delta = None
    nasdaq = '--'
    nasdaq_delta = None

    try:
        import yfinance as yf

        # S&P 500
        spy = yf.download('^GSPC', period='5d', interval='1d', progress=False)
        if spy is not None and not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            close_vals = spy['Close'].dropna()
            if len(close_vals) >= 2:
                sp_price = float(close_vals.iloc[-1])
                sp_prev = float(close_vals.iloc[-2])
                sp_chg_pct = (sp_price - sp_prev) / sp_prev * 100 if sp_prev else 0
                sp500 = f"{sp_price:,.0f}"
                sp500_delta = f"{sp_chg_pct:+.2f}%"

        # NASDAQ Composite
        ndx = yf.download('^IXIC', period='5d', interval='1d', progress=False)
        if ndx is not None and not ndx.empty:
            if isinstance(ndx.columns, pd.MultiIndex):
                ndx.columns = ndx.columns.get_level_values(0)
            close_vals = ndx['Close'].dropna()
            if len(close_vals) >= 2:
                nd_price = float(close_vals.iloc[-1])
                nd_prev = float(close_vals.iloc[-2])
                nd_chg_pct = (nd_price - nd_prev) / nd_prev * 100 if nd_prev else 0
                nasdaq = f"{nd_price:,.0f}"
                nasdaq_delta = f"{nd_chg_pct:+.2f}%"
    except Exception as e:
        logger.warning("[WarRoom] Failed to fetch US market data: %s", e)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("S&P 500", sp500, delta=sp500_delta)
    with c2:
        st.metric("NASDAQ", nasdaq, delta=nasdaq_delta)
    with c3:
        # VIX (fear index)
        vix_str = '--'
        try:
            import yfinance as yf
            vix = yf.download('^VIX', period='5d', interval='1d', progress=False)
            if vix is not None and not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_val = float(vix['Close'].dropna().iloc[-1])
                vix_str = f"{vix_val:.1f}"
        except Exception:
            pass
        st.metric("VIX 恐慌指數", vix_str)
    with c4:
        # Market context indicator
        st.metric("市場模式", "US")


def _render_opportunity_cards(results, scan_time):
    """Layer 2: AI scan result cards sorted by confidence."""
    st.markdown('<div class="sec-header">今日機會 — AI 掃描結果</div>', unsafe_allow_html=True)

    # Freshness indicator
    _scan_freshness_indicator(scan_time)

    n = len(results)
    time_str = ""
    if scan_time:
        try:
            if isinstance(scan_time, str):
                dt = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            else:
                time_str = str(scan_time)
        except (ValueError, TypeError):
            time_str = str(scan_time)

    info_parts = [f"共 {n} 檔機會"]
    if time_str:
        info_parts.append(f"掃描時間: {time_str}")

    st.markdown(
        f'<div style="font-size:0.72rem;color:#64748b;font-family:JetBrains Mono,monospace;'
        f'margin-bottom:12px">{"  |  ".join(info_parts)}</div>',
        unsafe_allow_html=True,
    )

    if not results:
        return

    def _sort_key(r):
        score = r.get('confidence', r.get('score', r.get('composite_score', 0)))
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0

    sorted_results = sorted(results, key=_sort_key, reverse=True)

    for i, sig in enumerate(sorted_results):
        ticker = sig.get('ticker', sig.get('code', ''))
        name = sig.get('name', sig.get('stock_name', ''))
        signal_type = sig.get('signal', sig.get('type', sig.get('signal_type', '')))
        signal_label = _SIG_ZH.get(signal_type, signal_type or '--')
        confidence = sig.get('confidence', sig.get('score', sig.get('composite_score', 0)))
        price = sig.get('close', sig.get('price', sig.get('current_price', 0)))
        change_pct = sig.get('change_pct', sig.get('漲跌幅%', 0))

        level = _confidence_level(confidence)
        badge_border, badge_bg, badge_text = _CONFIDENCE_COLORS[level]

        try:
            conf_display = f"{float(confidence):.0f}" if confidence else "--"
        except (TypeError, ValueError):
            conf_display = str(confidence) if confidence else "--"

        if isinstance(price, (int, float)) and price > 0:
            price_str = f"{price:,.1f}"
        else:
            price_str = "--"

        try:
            chg_val = float(change_pct)
            chg_color = "#ef4444" if chg_val > 0 else "#22c55e" if chg_val < 0 else "#94a3b8"
            chg_str = f"{chg_val:+.1f}%"
        except (TypeError, ValueError):
            chg_color = "#94a3b8"
            chg_str = ""

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;padding:10px 14px;'
            f'background:var(--bg-card);border:1px solid var(--glass-border);border-radius:8px;'
            f'margin-bottom:6px;backdrop-filter:blur(8px);'
            f'transition:border-color 0.2s,box-shadow 0.2s;cursor:default" '
            f'onmouseover="this.style.borderColor=\'rgba(0,240,255,0.25)\';this.style.boxShadow=\'0 0 15px rgba(0,240,255,0.08)\'" '
            f'onmouseout="this.style.borderColor=\'\';this.style.boxShadow=\'\'">'
            f'<div style="min-width:110px">'
            f'<span style="font-family:JetBrains Mono,monospace;font-weight:800;font-size:0.95rem;'
            f'color:#e2e8f0">{ticker}</span>'
            f'<br><span style="font-size:0.72rem;color:#94a3b8">{name}</span>'
            f'</div>'
            f'<div style="min-width:80px">'
            f'<span class="tag tag-bull" style="font-size:0.68rem">{signal_label}</span>'
            f'</div>'
            f'<div style="min-width:100px;text-align:right">'
            f'<span style="font-family:JetBrains Mono,monospace;font-weight:700;color:#e2e8f0">{price_str}</span>'
            f' <span style="font-size:0.72rem;color:{chg_color}">{chg_str}</span>'
            f'</div>'
            f'<div style="min-width:70px;text-align:center">'
            f'<span style="display:inline-block;padding:2px 10px;border-radius:4px;font-size:0.72rem;'
            f'font-weight:700;font-family:JetBrains Mono,monospace;'
            f'background:{badge_bg};color:{badge_text};border:1px solid {badge_border}40">'
            f'{conf_display}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        btn_key = f"war_diag_{ticker}_{i}"
        if st.button("一鍵診斷", key=btn_key, use_container_width=False):
            st.session_state['_war_room_diag_ticker'] = ticker


def _render_portfolio_risk(portfolio):
    """Layer 3: Portfolio risk summary."""
    st.markdown('<div class="sec-header">持倉風險摘要</div>', unsafe_allow_html=True)

    positions = portfolio['positions']
    equity = portfolio['equity']
    initial = portfolio['initial']
    unrealized = portfolio['unrealized_pnl']
    n_pos = portfolio['n_positions']
    total_value = portfolio['total_value']

    # First-time user experience: no paper_trade.json or no positions
    if not positions:
        has_paper_trade = os.path.exists(PAPER_TRADE_PATH)
        if not has_paper_trade:
            # No paper_trade.json at all — first-time user
            st.markdown(
                '<div style="padding:30px 20px;text-align:center;background:var(--bg-card);'
                'border:1px solid var(--glass-border);border-radius:12px">'
                '<div style="font-size:1.5rem;margin-bottom:8px">&#x1F4BC;</div>'
                '<div style="font-size:0.95rem;color:#e2e8f0;margin-bottom:6px;font-weight:600">'
                '尚無持倉資料</div>'
                '<div style="font-size:0.78rem;color:#94a3b8">'
                '試試模擬交易或手動新增持倉</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding:16px;text-align:center;color:#64748b;font-size:0.82rem;'
                'background:var(--bg-card);border:1px solid var(--glass-border);border-radius:8px">'
                '目前無持倉。執行策略後即可查看風險摘要。'
                '</div>',
                unsafe_allow_html=True,
            )
        return

    exposure_pct = (total_value / equity * 100) if equity > 0 else 0
    dd_from_initial = ((equity - initial) / initial * 100) if initial > 0 else 0

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("持倉檔數", str(n_pos))
    with col_b:
        st.metric("總曝險", f"{exposure_pct:.0f}%")
    with col_c:
        st.metric("未實現損益", f"${unrealized:+,.0f}")
    with col_d:
        st.metric("累計報酬", f"{dd_from_initial:+.1f}%")


def _render_quick_diagnosis():
    """Layer 4: Quick diagnosis search bar with loading animation."""
    st.markdown('<div class="sec-header">快速診斷</div>', unsafe_allow_html=True)

    ticker_input = st.text_input(
        "輸入股票代碼，立即產出 AI 診斷報告",
        placeholder="例如: 2330、NVDA",
        key="_war_room_quick_ticker",
        label_visibility="collapsed",
    )

    card_ticker = st.session_state.pop('_war_room_diag_ticker', None)
    target_ticker = card_ticker or ticker_input

    if target_ticker:
        target_ticker = str(target_ticker).strip()
        if target_ticker:
            # --- Smart market detection ---
            detected = _detect_market_type(target_ticker)
            if detected == 'TW':
                market_type = 'TW'
            elif detected == 'US':
                market_type = 'US'
                target_ticker = target_ticker.upper()
            else:
                # Mixed / Chinese — try AI resolution
                try:
                    from analysis.ai_core import resolve_ticker_and_market
                    from config.settings import GEMINI_MODEL
                    client = st.session_state.get('gemini_client')
                    resolved_ticker, resolved_market, _ = resolve_ticker_and_market(
                        target_ticker, client=client,
                        gemini_model=GEMINI_MODEL if client else None,
                    )
                    if resolved_ticker and resolved_market:
                        target_ticker = resolved_ticker
                        market_type = 'US' if '美股' in resolved_market else 'TW'
                    else:
                        market_type = 'TW'  # default fallback
                except Exception:
                    market_type = 'TW'

            # Store current market context for KPI strip
            st.session_state['_war_room_market'] = market_type

            # Pulsing neon loading animation
            market_flag = '\U0001f1fa\U0001f1f8' if market_type == 'US' else '\U0001f1f9\U0001f1fc'
            st.markdown(
                f'<style>'
                f'@keyframes war-pulse {{ 0%,100% {{ opacity:0.4 }} 50% {{ opacity:1 }} }}'
                f'.war-loading {{ animation:war-pulse 1.2s ease-in-out infinite;'
                f'color:#00f0ff;font-family:JetBrains Mono,monospace;font-size:0.85rem;'
                f'text-align:center;padding:12px;'
                f'text-shadow:0 0 8px rgba(0,240,255,0.5) }}'
                f'</style>'
                f'<div class="war-loading">{market_flag} AI 正在分析 {target_ticker}...</div>',
                unsafe_allow_html=True,
            )
            _render_inline_report(target_ticker, market_type=market_type)


def _render_inline_report(ticker, market_type="TW"):
    """Render an inline stock report for the given ticker."""
    market_flag = '\U0001f1fa\U0001f1f8' if market_type == 'US' else '\U0001f1f9\U0001f1fc'
    st.markdown(
        f'<div style="margin:8px 0 4px;padding:6px 12px;background:rgba(0,240,255,0.04);'
        f'border:1px solid rgba(0,240,255,0.12);border-radius:6px">'
        f'<span style="font-size:0.72rem;color:var(--neon-cyan);font-family:JetBrains Mono,monospace">'
        f'{market_flag} DIAGNOSING: {ticker}</span></div>',
        unsafe_allow_html=True,
    )

    try:
        from analysis.stock_report import generate_stock_report, format_report_summary
        report = generate_stock_report(ticker, market_type=market_type)
        summary = format_report_summary(report)
        st.info(summary)
        # Render full profile with market-aware provider
        from ui.stock_profile import render_stock_profile
        render_stock_profile(ticker, show_actions=True, market_type=market_type)
    except ImportError:
        try:
            from ui.stock_profile import render_stock_profile
            render_stock_profile(ticker, show_actions=True, market_type=market_type)
        except TypeError:
            # Fallback if render_stock_profile doesn't accept market_type yet
            from ui.stock_profile import render_stock_profile
            render_stock_profile(ticker, show_actions=True)
        except Exception as e:
            logger.warning(f"個股報告載入失敗 ({ticker}): {e}")
            st.error(f"無法載入 {ticker} 的診斷報告: {e}")


# =============================================
# Main Entry Point — Landing Page
# =============================================

def render_war_room():
    """Render the AI War Room — the default landing page."""
    inject_cyber_theme()

    st.markdown(
        '<div style="margin-bottom:4px">'
        '<span class="cyber-title">AI 戰情室</span>'
        '</div>'
        '<div class="cyber-subtitle">市場概覽 | 機會掃描 | 風險監控 | 快速診斷</div>',
        unsafe_allow_html=True,
    )

    # Layer 1: Market Overview KPI
    _render_market_kpi()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Layer 2: AI Scan Results
    results, scan_time, error = _load_scan_results()

    if error:
        st.markdown(
            f'<div class="alert-card alert-danger">'
            f'<div class="alert-title">掃描失敗</div>'
            f'<div class="alert-body">{error}'
            + (f'<br>上次成功: {scan_time}' if scan_time else '')
            + f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("重新掃描", key="war_retry_scan"):
            st.rerun()
    elif not results:
        # EMPTY state — warm welcome with neon-styled button
        st.markdown(
            '<div style="padding:40px 20px;text-align:center;background:var(--bg-card);'
            'border:1px solid var(--glass-border);border-radius:12px;margin-bottom:12px">'
            '<div style="font-size:2rem;margin-bottom:12px">&#x1F6F0;</div>'
            '<div style="font-size:1.05rem;color:#e2e8f0;margin-bottom:8px;font-weight:600">'
            '歡迎來到 AI 戰情室！</div>'
            '<div style="font-size:0.82rem;color:#94a3b8;margin-bottom:20px">'
            '點擊下方按鈕開始第一次掃描</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if scan_time:
            _scan_freshness_indicator(scan_time)
        st.markdown(
            '<style>.war-scan-btn button {background:linear-gradient(135deg,#00f0ff,#0ea5e9) !important;'
            'color:#0f172a !important;font-weight:700 !important;font-size:1rem !important;'
            'padding:0.6rem 2rem !important;border:none !important;'
            'box-shadow:0 0 20px rgba(0,240,255,0.3) !important}</style>',
            unsafe_allow_html=True,
        )
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown('<div class="war-scan-btn">', unsafe_allow_html=True)
            if st.button("開始掃描", key="war_trigger_scan", type="primary", use_container_width=True):
                st.info("請前往「研究分析」頁面執行完整市場掃描")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        _render_opportunity_cards(results, scan_time)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Layer 3: Portfolio Risk Summary
    portfolio = _load_portfolio()
    _render_portfolio_risk(portfolio)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Layer 4: Quick Diagnosis
    _render_quick_diagnosis()


# =============================================
# Legacy: Stock War Room (股市戰情室) scanner
# Kept for backward compatibility — imported by research.py
# =============================================

def render(client, market_mode, strategy_mode, tf_code, is_weekly, _embedded=False):
    from config.settings import GEMINI_MODEL
    from analysis.ai_core import (
        resolve_ticker_and_market, detect_hot_themes,
        generate_supply_chain_structure, generate_ai_analysis,
    )
    from data.scanner import scan_single_stock_deep, scan_tickers_from_map
    from ui.charts import render_trend_chart, render_supply_chain_graph
    from utils.helpers import get_default_sector_map_full, get_fallback_supply_chain

    api_key = client is not None

    st.markdown("### 🔎 1. 全市場狙擊")
    single_input = st.text_input("輸入代碼/名稱 (如 凡甲, NVDA)", placeholder="輸入股票代碼...")
    if st.button("🚀 分析個股"):
        if not single_input:
            st.error("請輸入代碼")
        else:
            st.session_state.supply_chain_data = None
            st.session_state.detected_themes = []
            with cyber_spinner("TARGETING", f"{single_input} 深度分析中..."):
                target_ticker, detected_market, target_name = resolve_ticker_and_market(
                    single_input, client=client, gemini_model=GEMINI_MODEL)
                if not target_ticker and not api_key:
                    if re.match(r'^\d{4,6}$', single_input):
                        target_ticker = single_input; detected_market = "🇹🇼 台股 (TW)"; target_name = single_input
                    elif re.match(r'^[A-Z]{1,5}$', single_input.upper()):
                        target_ticker = single_input.upper(); detected_market = "🗽 美股 (US)"; target_name = single_input.upper()
                    else:
                        st.error("無法識別股票名稱。建議輸入代碼 (如 2330) 或輸入 Gemini API Key。")
                if target_ticker and detected_market:
                    st.success(f"已識別: {target_name} ({target_ticker})")
                    st.session_state.dynamic_name_map[target_ticker] = target_name
                    st.session_state.market_mode = detected_market
                    data = scan_single_stock_deep(detected_market, target_ticker, strategy_mode, timeframe=tf_code, user_query_name=target_name)
                    if data:
                        st.session_state.single_stock_data = data
                        st.session_state.view_mode = "single"
                        verdict = data.get('verdict', {})
                        trend_msg = f"趨勢：{verdict.get('trend')}。{verdict.get('signal')}。"
                        ai_report = generate_ai_analysis(
                            detected_market, target_ticker, data['名稱'], data['現價'], data['漲跌幅%'],
                            "個股", data['extra_info'], strategy_mode, trend_msg,
                            timeframe=tf_code, signal_context=data['signal_context'],
                            client=client, gemini_model=GEMINI_MODEL)
                        st.session_state.ai_reports[f"SINGLE_{target_ticker}"] = ai_report
                        st.rerun()
                    else:
                        st.error("無法取得數據")
                elif api_key:
                    st.error("無法識別股票")

    st.markdown("### 📡 2. 族群熱點掃描")
    if st.button("🔥 掃描今日熱門話題"):
        if not api_key:
            st.error("無 API Key")
        else:
            st.session_state.supply_chain_data = None
            st.session_state.single_stock_data = None
            with cyber_spinner("AI SCAN", "新聞熱點掃描中..."):
                themes = detect_hot_themes(st.session_state.market_mode, client=client, gemini_model=GEMINI_MODEL)
                if themes:
                    st.session_state.detected_themes = themes
                    st.session_state.view_mode = "list"
                    st.success("偵測完成！")
                else:
                    st.error("偵測失敗")

    st.markdown("### ⛓️ 3. 產業鏈搜尋")
    custom_input = st.text_input("輸入族群關鍵字:", placeholder="例: 記憶體, 機器人")
    if st.button("✨ 繪製供應鏈圖"):
        if custom_input:
            st.session_state.single_stock_data = None
            st.session_state.detected_themes = []
            with cyber_spinner("SUPPLY CHAIN", f"「{custom_input}」供應鏈分析中..."):
                fallback = get_fallback_supply_chain(custom_input, st.session_state.market_mode)
                structure = fallback if fallback else generate_supply_chain_structure(
                    st.session_state.market_mode, custom_input, client=client, gemini_model=GEMINI_MODEL)
                if structure:
                    st.session_state.supply_chain_data = {"keyword": custom_input, "structure": structure}
                    df = scan_tickers_from_map(st.session_state.market_mode, structure, strategy_mode, timeframe=tf_code)
                    st.session_state.data_cache[st.session_state.market_mode] = df
                    st.session_state.current_source = f"⛓️ {custom_input} 供應鏈"
                    st.session_state.view_mode = "list"
                else:
                    st.error("供應鏈拆解失敗")

    # === Main Display ===
    if st.session_state.view_mode == "single" and st.session_state.single_stock_data:
        data = st.session_state.single_stock_data
        verdict = data.get('verdict', {})
        patterns = data.get('patterns', [])
        st.button("🔙 返回列表模式", on_click=lambda: st.session_state.update({"view_mode": "list"}))
        st.markdown(f"## 🎯 {data['代碼']} {data['名稱']} 個股戰情室 ({tf_code})")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: custom_metric("現價", data['現價'], f"{data['漲跌幅%']}%")
        with c2: custom_metric("爆量", f"{data['爆量倍數']}x", None)
        with c3: custom_metric("本益比", f"{data['PE']}", None)
        with c4: custom_metric("EPS", f"{data['EPS']}", None)
        with c5: custom_metric("殖利率", f"{data.get('Yield', 'N/A')}", None)
        trend_val = verdict.get('trend', '')
        if '多' in trend_val or '上升' in trend_val:
            border_color = '#ef4444'
        elif '空' in trend_val or '下降' in trend_val:
            border_color = '#22c55e'
        else:
            border_color = '#f59e0b'
        # --- Verdict HTML (防止特殊字元破壞 HTML 結構) ---
        import html as _html
        details_html = ""
        if verdict.get('details'):
            details_items = "".join(
                f'<div style="font-size:0.75rem;color:#94a3b8">&bull; {_html.escape(str(d))}</div>'
                for d in verdict.get('details')
            )
            details_html = f'<hr style="border-color:#334155;margin:8px 0">{details_items}'
        signal_ctx = _html.escape(str(data.get('signal_context', '無')))
        signal_val = _html.escape(str(verdict.get('signal', '')))
        verdict_html = (
            f'<div class="alert-card" style="border-color:{border_color};padding:14px 18px">'
            f'<div class="alert-title" style="color:{border_color}">'
            f'&#9878; \u7a0b\u5f0f\u5224\u6c7a\uff1a{_html.escape(str(trend_val))}</div>'
            f'<div class="alert-body" style="margin-top:6px">'
            f'<div style="margin-bottom:6px"><b>\u8a0a\u865f</b>\uff1a{signal_val}</div>'
            f'{details_html}'
            f'<hr style="border-color:#334155;margin:8px 0">'
            f'<div><b>\u6df1\u5ea6\u638d\u63cf</b>\uff1a{signal_ctx}</div>'
            f'</div></div>'
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
        render_trend_chart(data['df'], patterns, st.session_state.market_mode,
                          is_box=verdict.get('is_box', False), height=900, is_weekly=is_weekly,
                          candle_patterns=data.get('candle_patterns', []))
        report_key = f"SINGLE_{data['代碼']}"
        if report_key in st.session_state.ai_reports:
            st.markdown("### AI 深度評論")
            st.markdown(st.session_state.ai_reports[report_key])
    else:
        if st.session_state.detected_themes:
            st.markdown("### 🔥 請點擊感興趣的主題：")
            cols = st.columns(len(st.session_state.detected_themes))
            for i, theme in enumerate(st.session_state.detected_themes):
                safe_theme_label = str(theme)
                if cols[i].button(safe_theme_label, use_container_width=True):
                    st.session_state.single_stock_data = None
                    with cyber_spinner("MINING", f"「{safe_theme_label}」供應鏈挖掘中..."):
                        structure = generate_supply_chain_structure(
                            st.session_state.market_mode, safe_theme_label, client=client, gemini_model=GEMINI_MODEL)
                        if structure and isinstance(structure, dict):
                            st.session_state.supply_chain_data = {"keyword": safe_theme_label, "structure": structure}
                            df = scan_tickers_from_map(st.session_state.market_mode, structure, strategy_mode, timeframe=tf_code)
                            st.session_state.data_cache[st.session_state.market_mode] = df
                            st.session_state.current_source = f"🔥 {safe_theme_label}"
                        else:
                            st.error("AI 正在思考中，請再試一次")
        st.divider()
        if st.button("🔙 回到預設清單 (全市場掃描)"):
            st.session_state.supply_chain_data = None; st.session_state.single_stock_data = None
            st.session_state.detected_themes = []; st.session_state.view_mode = "list"
            with cyber_spinner("DATABASE", "完整資料庫載入中..."):
                default_map = get_default_sector_map_full(st.session_state.market_mode)
                df = scan_tickers_from_map(st.session_state.market_mode, default_map, strategy_mode, timeframe=tf_code)
                st.session_state.data_cache[st.session_state.market_mode] = df
                st.session_state.current_source = "🗂️ 預設清單"
        if st.session_state.supply_chain_data:
            st.markdown(f"## 🗺️ {st.session_state.supply_chain_data['keyword']} 產業供應鏈地圖")
            render_supply_chain_graph(st.session_state.supply_chain_data['keyword'],
                                     st.session_state.supply_chain_data['structure'], st.session_state.market_mode)
            st.divider()
        current_df = st.session_state.data_cache.get(st.session_state.market_mode)
        if current_df is not None and not current_df.empty:
            st.subheader(f"{st.session_state.current_source} 數據掃描 ({tf_code})")
            for idx, row in current_df.iterrows():
                ticker = row['代碼']; name = row['名稱']
                with st.expander(f"{ticker} {name} | {row['族群']} | {row['趨勢']}", expanded=(idx == 0)):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: custom_metric("現價", row['現價'], f"{row['漲跌幅%']}%")
                    with c2: custom_metric("爆量", f"{row['爆量倍數']}x", None)
                    with c3: custom_metric("短均", row['短均'], None)
                    with c4: custom_metric("長均", row['長均'], None)
                    with c5: custom_metric("RSI", row['RSI'], None)
                    trend_val = row['趨勢']
                    if '多' in trend_val or '上升' in trend_val: color = "red"
                    elif '空' in trend_val or '下降' in trend_val: color = "green"
                    else: color = "gray"
                    st.markdown(f":{color}-background[**⚖️ {trend_val}**] | {row.get('signal_context', '')}")
                    render_trend_chart(row['df'], row['patterns'], st.session_state.market_mode,
                                      is_box=row.get('verdict', {}).get('is_box', False), height=600,
                                      is_weekly=is_weekly, candle_patterns=row.get('candle_patterns', []))
                    cache_key = f"{st.session_state.market_mode}_{ticker}_{strategy_mode}"
                    if cache_key in st.session_state.ai_reports:
                        st.markdown(st.session_state.ai_reports[cache_key])
                    else:
                        if st.button(f"AI 分析 {name}", key=f"btn_{ticker}"):
                            with cyber_spinner("ANALYZING", "個股深度分析中..."):
                                tech_str = f"短均{row['短均']}, 長均{row['長均']}, RSI{row['RSI']}"
                                ai_report = generate_ai_analysis(
                                    st.session_state.market_mode, ticker, name, row['現價'], row['漲跌幅%'],
                                    row['族群'], tech_str, strategy_mode, f"趨勢：{row['趨勢']}",
                                    timeframe=tf_code, signal_context=row.get('signal_context', ''),
                                    client=client, gemini_model=GEMINI_MODEL)
                                st.session_state.ai_reports[cache_key] = ai_report
                                st.rerun()
        else:
            if current_df is not None:
                st.warning("無符合資料。")
            else:
                st.info("請選擇側邊欄的搜尋方式開始。")
