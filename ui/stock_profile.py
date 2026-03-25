"""
個股全貌面板 — Stock Profile Panel
任何頁面呼叫 render_stock_profile(ticker) 即可展開完整個股資訊。
整合：K線圖 + 基本面 + 籌碼面 + 突破信號 + 持倉狀態 + 操作按鈕
"""
import streamlit as st
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from ui.theme import _tw_color, _tw_color_pct, _plotly_dark_layout

logger = logging.getLogger(__name__)


def render_stock_profile(ticker, show_actions=True, market_type=None):
    """
    渲染個股全貌面板。

    Args:
        ticker: 股票代碼 (e.g. "2330")
        show_actions: 是否顯示操作按鈕列
        market_type: 'TW' or 'US'. If None, auto-detect from ticker format.
    """
    if not ticker:
        return

    ticker = str(ticker).strip()

    # Auto-detect market_type if not provided
    if market_type is None:
        import re as _re
        if _re.match(r'^\d{4,6}[A-Za-z]?$', ticker):
            market_type = 'TW'
        elif _re.match(r'^[A-Za-z]{1,5}$', ticker):
            market_type = 'US'
        else:
            market_type = 'TW'

    # ── Fetch Data ──
    df, info, name = _fetch_stock_data(ticker, market_type=market_type)
    if df is None or df.empty:
        st.warning(f"無法取得 {ticker} 的資料")
        return

    price = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2]) if len(df) >= 2 else price
    chg = price - prev
    chg_pct = (chg / prev * 100) if prev != 0 else 0

    # Color
    color = "#ef4444" if chg >= 0 else "#22c55e"
    arrow = "▲" if chg > 0 else "▼" if chg < 0 else "─"

    # ── Header ──
    # Check watchlist status
    wl_status = _check_watchlist(ticker)
    wl_badge = '⭐ 已加自選' if wl_status else ''

    # Check position status
    pos_info = _check_position(ticker)

    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 12px;background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.12);'
        f'border-radius:8px;margin-bottom:12px">'
        f'<div>'
        f'<span style="font-size:1.3rem;font-weight:800;color:#e2e8f0;font-family:JetBrains Mono,monospace">{ticker}</span>'
        f' <span style="color:#94a3b8;font-size:0.9rem">{name}</span>'
        f' <span style="font-size:0.7rem;color:#f59e0b">{wl_badge}</span>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<span style="font-size:1.2rem;font-weight:800;color:#e2e8f0">{price:,.1f}</span>'
        f' <span style="color:{color};font-weight:700;font-size:0.9rem">{arrow}{chg_pct:+.1f}%</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chart ──
    try:
        from ui.charts import render_trend_chart
        from analysis.trend import calculate_trend_logic
        from analysis.patterns import detect_complex_patterns

        verdict = calculate_trend_logic(df)
        peaks = df.get('peaks', pd.Series(dtype=float)).dropna()
        troughs = df.get('troughs', pd.Series(dtype=float)).dropna()
        patterns = detect_complex_patterns(df, peaks, troughs)

        entry_price = pos_info.get('entry_price') if pos_info else None

        render_trend_chart(
            df, patterns, "TW",
            entry_price=entry_price,
            is_weekly=False,
            candle_patterns=patterns,
        )
    except Exception as e:
        st.caption(f"圖表載入失敗: {e}")

    # ── 3-Column Info Cards ──
    col_fund, col_chip, col_signal = st.columns(3)

    # ── Fundamentals ──
    with col_fund:
        pe = info.get('pe', info.get('trailingPE', '--'))
        eps = info.get('eps', info.get('trailingEps', '--'))
        div_yield = info.get('dividend_yield', info.get('dividendYield', 0))
        if isinstance(div_yield, (int, float)) and div_yield > 0:
            div_str = f"{div_yield*100:.1f}%" if div_yield < 1 else f"{div_yield:.1f}%"
        else:
            div_str = '--'

        _info_card("基本面", [
            ("P/E", f"{pe:.1f}" if isinstance(pe, (int, float)) else str(pe)),
            ("EPS", f"{eps:.1f}" if isinstance(eps, (int, float)) else str(eps)),
            ("殖利率", div_str),
        ], accent="#8b5cf6")

    # ── Chip Data ──
    with col_chip:
        chip = _fetch_chip_data(ticker)
        if chip:
            streak = chip.get('foreign_streak', 0)
            streak_str = f"+{streak}天連買" if streak > 0 else f"{streak}天連賣" if streak < 0 else "中性"
            streak_color = "#ef4444" if streak > 0 else "#22c55e" if streak < 0 else "#94a3b8"

            _info_card("籌碼面", [
                ("外資", f'<span style="color:{streak_color}">{streak_str}</span>'),
                ("投信", f"{chip.get('trust_streak', 0):+d}天"),
                ("籌碼分", f"{chip.get('chip_score', 0)}/6"),
            ], accent="#3b82f6")
        else:
            _info_card("籌碼面", [
                ("外資", "--"),
                ("投信", "--"),
                ("籌碼分", "--"),
            ], accent="#3b82f6")

    # ── Signal Status ──
    with col_signal:
        sig = _fetch_signal_data(ticker, df)
        if sig and sig.get('signal'):
            sig_info = sig.get('signal_info', {}) or {}
            label = sig_info.get('label', sig['signal'])
            level = sig_info.get('level', 'info')
            level_color = {'critical': '#ef4444', 'warning': '#f59e0b', 'info': '#3b82f6'}.get(level, '#94a3b8')

            dist_r = sig.get('distance_to_resistance_pct')
            dist_str = f"{dist_r*100:.1f}%" if dist_r is not None else '--'

            vcp_score = sig.get('vcp', {}).get('vcp_score', 0)

            _info_card("信號", [
                ("狀態", f'<span style="color:{level_color};font-weight:700">{label}</span>'),
                ("距壓力", dist_str),
                ("VCP", f"{vcp_score}/4"),
            ], accent=level_color)
        else:
            _info_card("信號", [
                ("狀態", "無信號"),
                ("距壓力", "--"),
                ("VCP", "--"),
            ], accent="#475569")

    # ── Position Status ──
    if pos_info:
        entry = pos_info.get('entry_price', 0)
        shares = pos_info.get('shares', 0)
        pnl_pct = ((price - entry) / entry * 100) if entry > 0 else 0
        pnl_abs = (price - entry) * shares
        days_held = pos_info.get('days_held', '--')

        pnl_color = "#ef4444" if pnl_pct > 0 else "#22c55e" if pnl_pct < 0 else "#94a3b8"

        st.markdown(
            f'<div style="padding:8px 12px;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.15);'
            f'border-radius:6px;margin:8px 0;display:flex;gap:24px;align-items:center;font-size:0.78rem">'
            f'<span style="color:#8b5cf6;font-weight:700">📦 持倉中</span>'
            f'<span>持有 <strong>{shares:,}</strong> 股</span>'
            f'<span>進場 <strong>{entry:,.1f}</strong></span>'
            f'<span style="color:{pnl_color};font-weight:700">損益 {pnl_pct:+.1f}% (${pnl_abs:+,.0f})</span>'
            f'<span style="color:#64748b">持有 {days_held} 天</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Trade Thesis ──
    try:
        from analysis.thesis import generate_thesis, generate_ai_narrative
        from analysis.breakout import detect_levels, detect_signal

        levels = detect_levels(df) if len(df) >= 30 else {}
        sig = _fetch_signal_data(ticker, df)
        chip = _fetch_chip_data(ticker)

        thesis = generate_thesis(
            ticker, df,
            chip_data=chip,
            signal_data=sig,
            levels=levels,
            position=pos_info,
        )

        if thesis and thesis.get('composite_score', 0) > 0:
            cs = thesis['composite_score']
            verdict = thesis['verdict']
            tech = thesis['technical']
            chip_s = thesis['chip']
            risk_s = thesis['risk']
            action = thesis['action']

            # Score bar color
            if cs >= 7:
                bar_color = "#ef4444"
            elif cs >= 5:
                bar_color = "#f59e0b"
            else:
                bar_color = "#22c55e"

            bar_width = cs * 10

            # Score bar
            st.markdown(
                f'<div style="padding:10px 12px;background:rgba(0,0,0,0.3);border:1px solid {bar_color}33;'
                f'border-radius:8px;margin:10px 0">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                f'<span style="font-size:0.8rem;font-weight:800;color:#e2e8f0">交易論述</span>'
                f'<span style="font-size:0.85rem;font-weight:800;color:{bar_color}">{cs}/10 {verdict}</span>'
                f'</div>'
                f'<div style="background:#1e293b;border-radius:4px;height:8px;overflow:hidden">'
                f'<div style="width:{bar_width}%;height:100%;background:{bar_color};border-radius:4px;'
                f'transition:width 0.3s"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 3-column detail scores
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                _thesis_card("📊 技術面", tech['score'], tech['details'], "#8b5cf6")
            with tc2:
                _thesis_card("💰 籌碼面", chip_s['score'], chip_s['details'], "#3b82f6")
            with tc3:
                _thesis_card("⚠️ 風險", risk_s['score'], risk_s['details'], "#f59e0b")

            # Action plan
            if action.get('entry'):
                sl = action.get('stop_loss', 0)
                tp = action.get('take_profit', 0)
                rr = action.get('risk_reward')
                shares = action.get('suggested_shares', 0)
                reason = action.get('reason', '')

                sl_pct = ((sl - price) / price * 100) if sl and price > 0 else 0
                tp_pct = ((tp - price) / price * 100) if tp and price > 0 else 0

                st.markdown(
                    f'<div style="display:flex;gap:12px;padding:6px 10px;background:rgba(0,240,255,0.03);'
                    f'border:1px solid rgba(0,240,255,0.1);border-radius:6px;margin:6px 0;'
                    f'font-size:0.72rem;font-family:JetBrains Mono,monospace;flex-wrap:wrap;align-items:center">'
                    f'<span style="color:var(--neon-cyan);font-weight:700">💡 {reason}</span>'
                    f'<span>進場 <strong>{action["entry"]:,.0f}</strong></span>'
                    f'<span style="color:#22c55e">停損 <strong>{sl:,.0f}</strong> ({sl_pct:+.1f}%)</span>'
                    f'<span style="color:#ef4444">停利 <strong>{tp:,.0f}</strong> ({tp_pct:+.1f}%)</span>'
                    f'{"<span>風報比 <strong>" + str(rr) + ":1</strong></span>" if rr else ""}'
                    f'{"<span>建議 <strong>" + f"{shares:,}" + " 股</strong></span>" if shares > 0 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # AI Narrative (optional, only if Gemini client exists)
            try:
                from google import genai
                client = st.session_state.get('_gemini_client')
                if client is None:
                    api_key = ''
                    try: api_key = st.secrets.get("GEMINI_API_KEY", "")
                    except Exception: pass
                    if api_key:
                        client = genai.Client(api_key=api_key)
                        st.session_state['_gemini_client'] = client

                if client:
                    with st.expander("🤖 AI 解讀", expanded=False):
                        narrative = generate_ai_narrative(ticker, thesis, _client=client)
                        if narrative:
                            st.markdown(
                                f'<div style="font-size:0.8rem;color:#cbd5e1;line-height:1.6;'
                                f'padding:8px;background:rgba(0,0,0,0.2);border-radius:6px;'
                                f'border-left:3px solid var(--neon-cyan)">{narrative}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("AI 解讀生成失敗")
            except Exception as e:
                logger.debug(f"AI 解讀跳過: {e}")

    except Exception as e:
        logger.warning(f"交易論述生成失敗 ({ticker}): {e}")
        st.caption(f"交易論述生成失敗: {e}")

    # ── Action Buttons ──
    if show_actions:
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            if not wl_status:
                if st.button("⭐ 加自選", key=f"sp_wl_{ticker}", use_container_width=True):
                    try:
                        from data.watchlist import WatchlistManager
                        wm = WatchlistManager()
                        wm.add(ticker, name=name)
                        st.success(f"已加入自選股: {ticker}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"加入失敗: {e}")
            else:
                st.button("⭐ 已加自選", key=f"sp_wl_{ticker}", use_container_width=True, disabled=True)

        with ac2:
            if st.button("🔔 設警報", key=f"sp_alert_{ticker}", use_container_width=True):
                try:
                    from data.alerts import AlertManager
                    am = AlertManager()
                    am.add_alert(ticker=ticker, name=name, alert_type="price_above",
                                 threshold=price * 1.05, message=f"{ticker} 突破 {price*1.05:.0f}")
                    st.success(f"已設定 {ticker} 漲 5% 警報")
                except Exception as e:
                    st.error(f"設定失敗: {e}")

        with ac3:
            if st.button("📊 深入分析", key=f"sp_deep_{ticker}", use_container_width=True):
                st.session_state['target_ticker'] = ticker
                st.session_state['view_mode'] = 'single'
                st.info("請切換到「🔍 研究分析」頁面查看完整分析")

        with ac4:
            if pos_info:
                if st.button("🔴 平倉", key=f"sp_sell_{ticker}", use_container_width=True, type="primary"):
                    st.warning("請至「⚡ 交易執行」手動平倉或等待自動交易系統執行")
            else:
                if st.button("💰 買入", key=f"sp_buy_{ticker}", use_container_width=True, type="primary"):
                    st.info("請至「⚡ 交易執行」執行買入或等待自動交易系統")


# ==========================================
# Helper Functions
# ==========================================

def _thesis_card(title, score, details, accent="#00f0ff"):
    """Render a thesis scoring card."""
    details_html = "".join(
        f'<div style="font-size:0.68rem;color:#94a3b8;padding:1px 0">• {d}</div>'
        for d in details[:4]
    )
    st.markdown(
        f'<div style="padding:6px 8px;background:rgba(0,0,0,0.2);border:1px solid {accent}22;'
        f'border-radius:6px;border-left:3px solid {accent}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">'
        f'<span style="font-size:0.68rem;font-weight:700;color:{accent}">{title}</span>'
        f'<span style="font-size:0.75rem;font-weight:800;color:#e2e8f0">{score}/10</span>'
        f'</div>'
        f'{details_html}</div>',
        unsafe_allow_html=True,
    )


def _info_card(title, items, accent="#00f0ff"):
    """Render a compact info card."""
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04)">'
        f'<span style="color:#64748b;font-size:0.72rem">{k}</span>'
        f'<span style="color:#e2e8f0;font-size:0.72rem;font-weight:600">{v}</span>'
        f'</div>'
        for k, v in items
    )
    st.markdown(
        f'<div style="padding:8px 10px;background:rgba(0,0,0,0.2);border:1px solid {accent}22;'
        f'border-radius:6px;border-left:3px solid {accent}">'
        f'<div style="font-size:0.68rem;font-weight:700;color:{accent};margin-bottom:4px">{title}</div>'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_stock_data(ticker, market_type="TW"):
    """Fetch historical data + basic info for a ticker."""
    try:
        from data.provider import get_data_provider
        provider = get_data_provider("auto", market_type=market_type)
        df = provider.get_historical_data(ticker, period="6mo", interval="1d")
        if df is None or df.empty:
            return None, {}, ticker

        info = {}
        try:
            stock_info = provider.get_stock_info(ticker)
            if stock_info:
                info = stock_info
        except (ConnectionError, TimeoutError, AttributeError) as e:
            logger.debug(f"取得 {ticker} 基本面資訊失敗: {e}")

        name = info.get('shortName', info.get('longName', ticker))
        # Clean up Yahoo Finance name
        if name and len(name) > 20:
            name = name[:20]

        return df, info, name
    except Exception as e:
        logger.warning(f"取得 {ticker} 股票資料失敗: {e}")
        return None, {}, ticker


def _check_watchlist(ticker):
    """Check if ticker is in watchlist."""
    try:
        from data.watchlist import WatchlistManager
        wm = WatchlistManager()
        return any(s['ticker'] == ticker for s in wm.get_all())
    except Exception as e:
        logger.debug(f"自選股查詢失敗 ({ticker}): {e}")
        return False


def _check_position(ticker):
    """Check if ticker is in current positions."""
    try:
        from utils.helpers import safe_json_read
        from config.paths import PAPER_TRADE_PATH
        data = safe_json_read(PAPER_TRADE_PATH, {})
        for p in data.get('positions', []):
            if str(p.get('ticker', '')) == str(ticker):
                # Calculate days held
                days_held = '--'
                try:
                    entry_date = datetime.fromisoformat(p.get('entry_date', ''))
                    days_held = (datetime.now() - entry_date).days
                except (ValueError, TypeError):
                    pass
                return {
                    'entry_price': p.get('entry_price', 0),
                    'shares': p.get('shares', 0),
                    'days_held': days_held,
                }
        return None
    except Exception as e:
        logger.debug(f"持倉查詢失敗 ({ticker}): {e}")
        return None


def _fetch_chip_data(ticker):
    """Fetch institutional chip data if FinLab token available."""
    try:
        finlab_token = st.session_state.get('finlab_token', '')
        if not finlab_token:
            return None
        from analysis.chip import get_institutional_data, analyze_chip_for_ticker
        chip_data = get_institutional_data(finlab_token)
        if chip_data:
            return analyze_chip_for_ticker(ticker, chip_data)
        return None
    except Exception as e:
        logger.warning(f"籌碼資料取得失敗 ({ticker}): {e}")
        return None


def _fetch_signal_data(ticker, df):
    """Run breakout detection on this ticker."""
    try:
        if df is None or df.empty or len(df) < 30:
            return None
        from analysis.breakout import detect_levels, detect_signal
        levels = detect_levels(df)
        return detect_signal(df, levels)
    except Exception as e:
        logger.warning(f"信號偵測失敗 ({ticker}): {e}")
        return None
