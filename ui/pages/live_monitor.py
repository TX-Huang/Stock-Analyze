"""
Module: Live Monitor (實盤監控)
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

from ui.components import cyber_spinner
from ui.theme import _plotly_dark_layout, _load_recommendation


def render(_embedded=False):
    if not _embedded:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
            <div style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">實盤監控</div>
            <span><span class="status-dot status-live"></span><span style="color:#22c55e; font-size:0.8rem;">運作中</span></span>
        </div>
        """, unsafe_allow_html=True)

    tab_rec, tab_paper, tab_risk = st.tabs(["📋 每日精選", "💰 模擬交易", "🛡️ 風控中心"])

    with tab_rec:
        render_daily_picks()

    with tab_paper:
        render_paper_trading()

    with tab_risk:
        _render_risk_control()


def _render_risk_control():
    """風控中心 tab content — 完整風險監控。"""
    from data.risk_monitor import RiskMonitor
    from config.paths import PAPER_TRADE_PATH
    monitor = RiskMonitor()
    risk_path = PAPER_TRADE_PATH
    if os.path.exists(risk_path):
        result = monitor.check_all()
        alerts = result.get('alerts', [])
        pr = result.get('portfolio_risk', {})
        mr = result.get('market_risk', {}).get('etf_0050', {})
        positions_risk = result.get('positions_risk', [])
        n_danger = result.get('n_danger', 0); n_warning = result.get('n_warning', 0)
        if n_danger > 0:
            st.markdown(f'<div class="alert-card alert-danger"><div class="alert-title">危險：{n_danger} 個重大警報</div></div>', unsafe_allow_html=True)
        elif n_warning > 0:
            st.markdown(f'<div class="alert-card alert-warn"><div class="alert-title">警告：{n_warning} 個警報</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-card alert-ok"><div class="alert-title">狀態正常</div><div class="alert-body">未偵測到風險警報。</div></div>', unsafe_allow_html=True)
        for a in alerts:
            cls = 'alert-danger' if a['level'] == 'DANGER' else 'alert-warn'
            st.markdown(f'<div class="alert-card {cls}"><div class="alert-title">[{a["level"]}] {a["type"].upper()}</div><div class="alert-body">{a["message"]}</div></div>', unsafe_allow_html=True)
        if st.button("重新檢查風險", use_container_width=True, type="primary"):
            st.rerun()
    else:
        st.markdown('<div class="alert-card alert-info"><div class="alert-title">無投資組合資料</div><div class="alert-body">請先在模擬交易頁面初始化帳戶。</div></div>', unsafe_allow_html=True)


def render_daily_picks():
    """每日精選 — 獨立函式，可從 trading.py 直接呼叫。"""
    rec_data = _load_recommendation()

    if rec_data:
        etf = rec_data.get('etf_0050', {})
        etf_close = etf.get('close', 0); etf_chg = etf.get('change_rate', 0); ma60 = etf.get('ma60', 0); ma120 = etf.get('ma120', 0)
        regime = rec_data.get('market_regime', ''); exposure = rec_data.get('exposure', 1.0)
        hedge = rec_data.get('hedge_status', ''); n_short = rec_data.get('n_short_signals', 0)

        # Market Environment Strip
        regime_clr = '#22c55e' if regime == '多頭' else '#f59e0b' if '盤整' in regime else '#ef4444'
        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-item" style="border-left:3px solid {regime_clr}"><div class="kpi-label">大盤環境</div><div class="kpi-value" style="color:{regime_clr}">{regime}</div></div>
            <div class="kpi-item"><div class="kpi-label">0050</div><div class="kpi-value">{etf_close:.2f} <span style="font-size:0.75rem" class="{'c-up' if etf_chg > 0 else 'c-down' if etf_chg < 0 else 'c-flat'}">{etf_chg:+.2f}%</span></div></div>
            <div class="kpi-item"><div class="kpi-label">MA60 / MA120</div><div class="kpi-value">{ma60:.1f} / {ma120:.1f}</div></div>
            <div class="kpi-item"><div class="kpi-label">曝險度</div><div class="kpi-value">{exposure*100:.0f}%</div></div>
            <div class="kpi-item"><div class="kpi-label">避險</div><div class="kpi-value">{hedge}</div></div>
            <div class="kpi-item"><div class="kpi-label">日期</div><div class="kpi-value">{rec_data.get('date', '')}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations Table
        recs = rec_data.get('recommendations', [])
        if not recs:
            st.markdown('<div class="alert-card alert-info"><div class="alert-body">今日無符合條件的推薦標的。</div></div>', unsafe_allow_html=True)
        if recs:
            # Fill missing names from session_state or FinLab
            _name_map = st.session_state.get('dynamic_name_map', {})
            for r in recs:
                if not r.get('name'):
                    r['name'] = _name_map.get(r['ticker'], '')
            # Batch lookup remaining empty names
            missing = [r['ticker'] for r in recs if not r.get('name')]
            if missing:
                try:
                    from finlab import data as _fdata
                    cats = _fdata.get('security_categories')
                    if cats is not None and 'stock_id' in cats.columns:
                        cats_idx = cats.set_index('stock_id')
                        for r in recs:
                            if not r.get('name') and r['ticker'] in cats_idx.index:
                                r['name'] = str(cats_idx.loc[r['ticker'], 'name']) if 'name' in cats_idx.columns else ''
                                if r['name'] == 'nan':
                                    r['name'] = ''
                except Exception:
                    pass

            rows_html = ""
            ticker_list = []
            for r in recs:
                score = r.get('score', 0); pct = min(score / 10.0, 1.0)
                score_clr = f"hsl({120 * pct}, 70%, 50%)"
                new_tag = '<span class="tag tag-new">新</span>' if r.get('is_new') else ''

                # Volume ratio: volume / volume_ma20
                vol = r.get('volume', 0)
                vol_ma20 = r.get('volume_ma20', 0)
                if vol_ma20 and vol_ma20 > 0:
                    vol_ratio = vol / vol_ma20
                    vol_display = f"{vol_ratio:.1f}x"
                    vol_cls = 'c-up' if vol_ratio > 1.5 else 'c-flat' if vol_ratio >= 0.8 else 'c-down'
                else:
                    vol_display = f"{vol:,.0f}"
                    vol_cls = 'c-flat'

                # Signal reason
                signal = r.get('signal_type', r.get('reason', ''))
                if not signal:
                    # Fallback to Isaac signal description from score components
                    parts = []
                    if r.get('is_new'):
                        parts.append('新進場')
                    if score >= 8:
                        parts.append('強勢')
                    elif score >= 6:
                        parts.append('觀望')
                    signal = '+'.join(parts) if parts else '--'

                ticker_list.append(f"{r['ticker']} {r.get('name', '')}")
                rows_html += f"""<tr>
                    <td style="text-align:left"><span style="color:#3b82f6; font-weight:700;">#{r['rank']}</span></td>
                    <td style="text-align:left"><strong>{r['ticker']}</strong> <span style="color:#64748b">{r.get('name', '')}</span> {new_tag}</td>
                    <td><div class="score-bar"><div class="score-fill" style="width:{pct*100}%;background:{score_clr}"></div></div><span style="font-size:0.75rem">{score:.0f}</span></td>
                    <td>{r.get('close', 0):,.1f}</td>
                    <td><span class="{vol_cls}">{vol_display}</span></td>
                    <td><span style="color:#a78bfa;font-size:0.75rem">{signal}</span></td>
                    <td style="white-space:nowrap"><span style="cursor:pointer" title="加入自選股">&#11088;</span> <span style="cursor:pointer" title="加入手動持倉">&#128203;</span></td>
                </tr>"""

            st.markdown(f"""
            <table class="pro-table">
                <thead><tr><th style="text-align:left">排名</th><th style="text-align:left">股票</th><th>評分</th><th>現價</th><th>量比</th><th>信號</th><th>操作</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            # Action controls below table
            sel_ticker = st.selectbox("選擇操作標的", ticker_list, key="rec_action_ticker")
            act_c1, act_c2 = st.columns(2)
            with act_c1:
                if st.button("⭐ 加入自選股", use_container_width=True, key="btn_watchlist"):
                    try:
                        from data.watchlist import WatchlistManager
                        ticker_code = sel_ticker.split()[0]
                        rec_match = next((r for r in recs if r['ticker'] == ticker_code), {})
                        wm = WatchlistManager()
                        if wm.add(ticker_code, name=rec_match.get('name', ''), group="預設",
                                  notes=f"每日精選 score={rec_match.get('score', 0):.0f}"):
                            st.success(f"已加入自選股: {ticker_code}（至「研究分析 → 自選股」查看）")
                        else:
                            st.info(f"{ticker_code} 已在自選股中（至「研究分析 → 自選股」查看）")
                    except Exception as e:
                        st.error(f"錯誤: {e}")
            with act_c2:
                if st.button("📋 加入手動持倉", use_container_width=True, key="btn_manual_pos"):
                    try:
                        from data.paper_trader import PaperTrader
                        ticker_code = sel_ticker.split()[0]
                        rec_match = next((r for r in recs if r['ticker'] == ticker_code), {})
                        pt = PaperTrader()
                        pt.add_manual_position(
                            ticker=ticker_code,
                            name=rec_match.get('name', ''),
                            entry_price=rec_match.get('close', 0),
                            shares=1000,
                            note=f"從每日精選加入 score={rec_match.get('score', 0):.0f}",
                        )
                        st.success(f"已加入手動持倉: {ticker_code}")
                    except Exception as e:
                        st.error(f"錯誤: {e}")

            # Score Distribution Chart — score-based gradient colors
            fig_scores = go.Figure()
            tickers = [f"{r['ticker']}" for r in recs]
            scores = [r.get('score', 0) for r in recs]
            colors = [f"hsl({min(s / 10.0, 1.0) * 120}, 70%, 50%)" for s in scores]
            fig_scores.add_trace(go.Bar(x=tickers, y=scores, marker_color=colors, text=[f"{s:.0f}" for s in scores], textposition='outside', textfont=dict(size=10, color='#94a3b8')))
            _plotly_dark_layout(fig_scores, height=280, title_text="評分分佈")
            fig_scores.update_layout(yaxis_title="評分", xaxis_tickangle=-45)
            st.plotly_chart(fig_scores, use_container_width=True)

        # Entries & Exits
        new_entries = rec_data.get('new_entries', []); exits = rec_data.get('exits', [])
        if new_entries or exits:
            e1, e2 = st.columns(2)
            with e1:
                if new_entries:
                    st.markdown('<p class="sec-header">新進場</p>', unsafe_allow_html=True)
                    for t in new_entries:
                        rec = next((r for r in recs if r['ticker'] == t), {})
                        st.markdown(f'<div class="ob-row"><span><span class="tag tag-new">買入</span> <strong>{t}</strong> {rec.get("name", "")}</span><span>評分: {rec.get("score", 0):.0f} | {rec.get("close", 0):,.1f}</span></div>', unsafe_allow_html=True)
            with e2:
                if exits:
                    st.markdown('<p class="sec-header">出場信號</p>', unsafe_allow_html=True)
                    for ex in exits:
                        st.markdown(f'<div class="ob-row"><span><span class="tag tag-exit">賣出</span> <strong>{ex["ticker"]}</strong> {ex.get("name", "")}</span><span>{ex.get("close", "N/A")}</span></div>', unsafe_allow_html=True)

        # Refresh button
        if st.button("重新整理", use_container_width=True, type="primary"):
            try:
                from data.daily_recommender import get_daily_recommendation
                with cyber_spinner("ISAAC V3.7", "策略運算中..."):
                    get_daily_recommendation()
                st.rerun()
            except Exception as e: st.error(f"錯誤: {e}")
    else:
        st.markdown('<div class="alert-card alert-info"><div class="alert-title">無資料</div><div class="alert-body">請先執行 python data/daily_recommender.py 或點擊下方按鈕。</div></div>', unsafe_allow_html=True)
        if st.button("產生每日推薦", type="primary"):
            try:
                from data.daily_recommender import get_daily_recommendation
                with cyber_spinner("ISAAC V3.7", "策略運算中..."):
                    get_daily_recommendation()
                st.rerun()
            except Exception as e: st.error(f"錯誤: {e}")

def render_paper_trading():
    """模擬交易 — 獨立函式，可從 trading.py 直接呼叫。"""
    from data.paper_trader import PaperTrader
    from data.signal_format import AVAILABLE_STRATEGIES
    from datetime import date as _date

    trader = PaperTrader()
    status = trader.get_status()

    # Portfolio KPIs
    equity = status.get('equity', 0); initial = status.get('initial_capital', 1_000_000)
    ret_pct = status.get('return_pct', 0); unrealized = status.get('total_unrealized', 0)
    realized = status.get('total_realized', 0); cash = status.get('cash', 0)
    n_pos = status.get('n_positions', 0); win_rate = status.get('win_rate', 0)

    ret_cls = 'c-up' if ret_pct > 0 else 'c-down' if ret_pct < 0 else 'c-flat'
    ur_cls = 'c-up' if unrealized > 0 else 'c-down' if unrealized < 0 else 'c-flat'
    rl_cls = 'c-up' if realized > 0 else 'c-down' if realized < 0 else 'c-flat'

    st.markdown(f"""
    <div class="kpi-strip">
    <div class="kpi-item" style="border-left:3px solid {'#22c55e' if ret_pct >= 0 else '#ef4444'}">
        <div class="kpi-label">總權益</div>
        <div class="kpi-value">${equity:,.0f} <span class="{ret_cls}" style="font-size:0.8rem">{ret_pct:+.2f}%</span></div>
    </div>
    <div class="kpi-item"><div class="kpi-label">初始資金</div><div class="kpi-value">${initial:,.0f}</div></div>
    <div class="kpi-item"><div class="kpi-label">未實現損益</div><div class="kpi-value"><span class="{ur_cls}">${unrealized:+,.0f}</span></div></div>
    <div class="kpi-item"><div class="kpi-label">已實現損益</div><div class="kpi-value"><span class="{rl_cls}">${realized:+,.0f}</span></div></div>
    <div class="kpi-item"><div class="kpi-label">現金</div><div class="kpi-value">${cash:,.0f}</div></div>
    <div class="kpi-item"><div class="kpi-label">持倉數</div><div class="kpi-value">{n_pos}</div></div>
    <div class="kpi-item"><div class="kpi-label">勝率</div><div class="kpi-value">{win_rate:.1f}%</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    # Manual Portfolio Section
    # ================================================================
    st.markdown("#### 📋 手動持倉")

    manual_positions = trader.get_positions(source='manual')

    # Add manual position form
    with st.form("add_manual_pos", clear_on_submit=True):
        fm1, fm2, fm3 = st.columns(3)
        mp_ticker = fm1.text_input("代碼", placeholder="例: 2330")
        mp_price = fm2.number_input("進場價", min_value=0.0, value=0.0, step=0.1)
        mp_shares = fm3.number_input("股數", min_value=1, value=1000, step=100)
        fm4, fm5 = st.columns(2)
        mp_date = fm4.date_input("進場日", value=_date.today())
        mp_note = fm5.text_input("備註 (選填)", placeholder="自行研究")
        mp_submit = st.form_submit_button("新增持倉", use_container_width=True, type="primary")

    if mp_submit and mp_ticker and mp_price > 0:
        trader.add_manual_position(
            ticker=mp_ticker.strip(),
            name="",
            entry_price=mp_price,
            shares=mp_shares,
            entry_date=mp_date,
            note=mp_note,
        )
        st.success(f"已新增手動持倉: {mp_ticker.strip()}")
        st.rerun()

    if manual_positions:
        mp_rows = ""
        for idx, p in enumerate(manual_positions):
            pnl_pct = p.get('pnl_pct', 0)
            pnl_cls = 'c-up' if pnl_pct > 0 else 'c-down' if pnl_pct < 0 else 'c-flat'
            cur_price = p.get('current_price', p['entry_price'])
            note_tag = f' <span style="color:#64748b;font-size:0.7rem">({p.get("note", "")})</span>' if p.get('note') else ''
            mp_rows += f"""<tr>
                <td style="text-align:left"><strong>{p['ticker']}</strong> <span style="color:#64748b">{p.get('name','')}</span>{note_tag}</td>
                <td>{p['entry_price']:,.1f}</td>
                <td>{cur_price:,.1f}</td>
                <td>{p['shares']:,}</td>
                <td><span class="{pnl_cls}">{pnl_pct:+.2f}%</span></td>
                <td>{p.get('entry_date', '')}</td>
            </tr>"""
        st.markdown(f"""
        <table class="pro-table">
            <thead><tr><th style="text-align:left">股票</th><th>進場價</th><th>現價</th><th>股數</th><th>損益%</th><th>進場日</th></tr></thead>
            <tbody>{mp_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        manual_tickers = [f"{p['ticker']} {p.get('name','')}" for p in manual_positions]
        close_sel = st.selectbox("選擇平倉標的", manual_tickers, key="manual_close_sel")
        if st.button("平倉", key="btn_close_manual", use_container_width=True):
            close_ticker = close_sel.split()[0]
            if trader.remove_position(close_ticker, source='manual'):
                st.success(f"已平倉: {close_ticker}")
                st.rerun()
    else:
        st.caption("目前無手動持倉。使用上方表單新增。")

    st.markdown("---")

    # ================================================================
    # Strategy Simulation Section
    # ================================================================
    st.markdown("#### 🤖 策略模擬")

    strat_labels = [s['label'] for s in AVAILABLE_STRATEGIES]
    selected_strat_idx = st.selectbox(
        "模擬策略", range(len(strat_labels)),
        format_func=lambda i: strat_labels[i],
        key="sim_strategy_selector",
    )
    selected_strat = AVAILABLE_STRATEGIES[selected_strat_idx]
    selected_source = selected_strat['source_tag']

    strategy_positions = trader.get_positions(source='strategy:')
    filtered_positions = [
        p for p in strategy_positions
        if p.get('source', 'strategy:isaac') == selected_source
    ]
    if selected_source == 'strategy:isaac':
        for p in trader.get_positions():
            src = p.get('source', 'strategy:isaac')
            if src == 'strategy:isaac' and p not in filtered_positions:
                filtered_positions.append(p)

    positions = filtered_positions  # used below for charts
    all_positions = status.get('positions', [])  # all for charts

    if positions:
        st.markdown(f'<p class="sec-header">持有部位 — {selected_strat["label"]}</p>', unsafe_allow_html=True)
        pos_rows = ""
        for p in sorted(positions, key=lambda x: x.get('pnl_pct', 0), reverse=True):
            pnl_pct = p.get('pnl_pct', 0); pnl_abs = p.get('pnl_abs', 0)
            pnl_cls = 'c-up' if pnl_pct > 0 else 'c-down' if pnl_pct < 0 else 'c-flat'
            weight = p.get('current_price', p['entry_price']) * p['shares'] / equity * 100 if equity > 0 else 0
            pos_rows += f"""<tr>
                <td style="text-align:left"><strong>{p['ticker']}</strong> <span style="color:#64748b">{p.get('name','')}</span></td>
                <td>{p['entry_price']:,.1f}</td><td>{p.get('current_price', p['entry_price']):,.1f}</td>
                <td>{p['shares']:,}</td><td><span class="{pnl_cls}">{pnl_pct:+.2f}%</span></td>
                <td><span class="{pnl_cls}">${pnl_abs:+,.0f}</span></td><td>{weight:.1f}%</td>
                <td>{p.get('entry_date', '')}</td></tr>"""
        st.markdown(f'<table class="pro-table"><thead><tr><th style="text-align:left">股票</th><th>進場價</th><th>現價</th><th>股數</th><th>損益%</th><th>損益$</th><th>權重</th><th>進場日</th></tr></thead><tbody>{pos_rows}</tbody></table>', unsafe_allow_html=True)
    else:
        st.caption(f"目前 {selected_strat['label']} 無持倉。")

    # ── 持倉籌碼分析 ──
    all_pos_for_chip = positions if positions else all_positions
    if all_pos_for_chip:
        st.markdown('<p class="sec-header">持倉籌碼面</p>', unsafe_allow_html=True)
        try:
            from analysis.chip import get_institutional_data, analyze_chip_for_ticker, chip_score_color
            finlab_token = st.session_state.get('finlab_token', '')
            chip_data = get_institutional_data(finlab_token)
            if chip_data:
                for pos in all_pos_for_chip:
                    ticker = pos.get('ticker', '')
                    chip = analyze_chip_for_ticker(ticker, chip_data)
                    score = chip['chip_score']
                    color = chip_score_color(score)
                    signals_text = ' | '.join(chip['chip_signals']) if chip['chip_signals'] else '無特殊信號'
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:6px 12px;margin-bottom:4px;background:rgba(0,0,0,0.2);border-radius:6px;'
                        f'border-left:3px solid {color}">'
                        f'<span style="font-weight:700;font-size:0.8rem">{pos.get("name", ticker)} ({ticker})</span>'
                        f'<span style="font-size:0.72rem;color:{color}">籌碼 [{score:+d}] {signals_text}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("需要 Finlab Token 才能取得法人籌碼資料")
        except Exception as e:
            st.caption(f"籌碼分析暫無法使用: {e}")

    # Charts (use filtered positions, not all)
    chart_positions = positions if positions else all_pos_for_chip
    if chart_positions:
        ch1, ch2 = st.columns(2)
        with ch1:
            labels = [p.get('name', '') or p['ticker'] for p in chart_positions]
            values = [p.get('current_price', p['entry_price']) * p['shares'] for p in chart_positions]
            fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=px.colors.qualitative.Set3),
                                      textinfo='label+percent', textfont=dict(size=10)))
            _plotly_dark_layout(fig_pie, height=300, title_text="持倉權重分佈")
            st.plotly_chart(fig_pie, use_container_width=True)
        with ch2:
            sorted_pos = sorted(chart_positions, key=lambda x: x.get('pnl_pct', 0))
            names = [p.get('name', '') or p['ticker'] for p in sorted_pos]
            pnls = [p.get('pnl_pct', 0) for p in sorted_pos]
            colors = ['#ef4444' if v > 0 else '#22c55e' for v in pnls]
            fig_pnl = go.Figure(go.Bar(x=pnls, y=names, orientation='h', marker_color=colors,
                                      text=[f"{v:+.1f}%" for v in pnls], textposition='outside', textfont=dict(size=10, color='#94a3b8')))
            _plotly_dark_layout(fig_pnl, height=300, title_text="個股損益")
            fig_pnl.update_layout(xaxis_title="報酬率 %")
            st.plotly_chart(fig_pnl, use_container_width=True)

    # ── 持股相關性分析 ──
    corr_positions = chart_positions if chart_positions else all_pos_for_chip
    if corr_positions and len(corr_positions) >= 2:
        st.markdown('<p class="sec-header">持股相關性</p>', unsafe_allow_html=True)
        try:
            from analysis.correlation import calculate_correlation_matrix, render_correlation_heatmap, get_concentration_risk
            from data.provider import get_data_provider
            held_tickers = [p['ticker'] for p in corr_positions if p.get('ticker')]
            name_map = {p['ticker']: p.get('name', '') or p['ticker'] for p in corr_positions}
            if len(held_tickers) >= 2:
                provider = get_data_provider("auto", market_type="TW")
                corr = calculate_correlation_matrix(held_tickers, provider, period="6mo")
                if corr is not None:
                    # Replace ticker codes with names in correlation matrix
                    display_labels = [name_map.get(t, t) for t in corr.columns]
                    corr_display = corr.copy()
                    corr_display.columns = display_labels
                    corr_display.index = display_labels
                    fig = render_correlation_heatmap(corr_display)
                    st.plotly_chart(fig, use_container_width=True)
                    risk = get_concentration_risk(corr_display)
                    risk_color = {'low': '#22c55e', 'medium': '#f59e0b', 'high': '#ef4444'}.get(risk['risk_level'], '#64748b')
                    st.markdown(
                        f'<div class="alert-card" style="border-color:{risk_color}">'
                        f'<div class="alert-title">集中度風險: {risk["risk_level"].upper()}</div>'
                        f'<div class="alert-body">{risk["suggestion"]}'
                        f'<br><span style="font-size:0.72rem;color:#64748b">'
                        f'平均相關係數: {risk["avg_correlation"]:.2f} | 最高: {risk["max_correlation"]:.2f}</span>'
                        f'</div></div>', unsafe_allow_html=True)
                    if risk['high_corr_pairs']:
                        st.markdown('<p style="font-size:0.75rem;color:#f59e0b;margin-top:8px">高相關配對:</p>', unsafe_allow_html=True)
                        for t1, t2, c in risk['high_corr_pairs'][:5]:
                            st.markdown(f'<span style="font-size:0.72rem;color:#94a3b8">{t1} <-> {t2}: {c:.2f}</span>', unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"相關性分析暫無法使用: {e}")

    # Equity Curve
    equity_hist = trader.account.get('daily_equity', [])
    if equity_hist:
        eq_df = pd.DataFrame(equity_hist)
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        fig_eq.add_trace(go.Scatter(x=eq_df['date'], y=eq_df['equity'], mode='lines', name='權益', line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'), row=1, col=1)
        fig_eq.add_hline(y=initial, line_dash="dash", line_color="#64748b", row=1, col=1)
        fig_eq.add_trace(go.Bar(x=eq_df['date'], y=eq_df.get('n_positions', [0]*len(eq_df)), name='持倉數', marker_color='#6366f1'), row=2, col=1)
        _plotly_dark_layout(fig_eq, height=400, title_text="權益曲線")
        st.plotly_chart(fig_eq, use_container_width=True)

    # Closed Trades
    closed = trader.account.get('closed_trades', [])
    if closed:
        with st.expander(f"已平倉交易 ({len(closed)})", expanded=False):
            ct_rows = ""
            for t in reversed(closed[-30:]):
                t_cls = 'c-up' if t.get('pnl_pct', 0) > 0 else 'c-down'
                ct_rows += f"""<tr><td style="text-align:left"><strong>{t['ticker']}</strong> {t.get('name','')}</td>
                    <td>{t.get('entry_price',0):,.1f}</td><td>{t.get('exit_price',0):,.1f}</td>
                    <td><span class="{t_cls}">{t.get('pnl_pct',0):+.2f}%</span></td>
                    <td>{t.get('hold_days',0)}d</td><td>{t.get('entry_date','')}</td><td>{t.get('exit_date','')}</td></tr>"""
            st.markdown(f'<table class="pro-table"><thead><tr><th style="text-align:left">股票</th><th>進場價</th><th>出場價</th><th>損益</th><th>天數</th><th>進場</th><th>出場</th></tr></thead><tbody>{ct_rows}</tbody></table>', unsafe_allow_html=True)

    # Catch-up indicator
    last_updated = trader.account.get('last_updated', '')
    if last_updated:
        try:
            from datetime import date as _d2, datetime as _dt
            last_dt = _dt.strptime(last_updated[:10], '%Y-%m-%d').date()
            gap = (_d2.today() - last_dt).days
            if gap > 1:
                st.markdown(
                    f'<div class="alert-card alert-warn" style="padding:8px 14px">'
                    f'<div class="alert-title">⚠️ 偵測到 {gap} 天未更新</div>'
                    f'<div class="alert-body">上次更新: {last_updated[:10]}。'
                    f'建議使用「補跟遺漏」自動模擬這段期間的策略操作。</div></div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    # Action buttons
    bc1, bc2, bc3, bc4 = st.columns(4)
    def _post_update_rerun():
        """Clear ALL caches and rerun after update."""
        st.session_state.pop('_dash_price_cache', None)
        st.session_state.pop('_dash_price_ts', None)
        # Clear scanner cache so position charts refresh
        for k in list(st.session_state.keys()):
            if k.startswith('_scan_cache_'):
                st.session_state.pop(k, None)
        st.rerun()

    if bc1.button("📈 更新報價", use_container_width=True, type="primary", help="只更新現有持倉的最新報價，不會增減股票"):
        with cyber_spinner("UPDATING", "持倉報價更新中..."):
            trader.update()
        _post_update_rerun()
    if bc2.button("🔄 重跑策略", use_container_width=True, help="重新執行 Isaac 策略，產生新推薦，並更新持倉（可能增減股票）"):
        try:
            from data.daily_recommender import get_daily_recommendation
            with cyber_spinner("ISAAC V3.7", "策略運算 + 持倉更新中..."):
                get_daily_recommendation()
                trader.update()
            _post_update_rerun()
        except Exception as e:
            st.error(f"錯誤: {e}")
    if bc3.button("補跟遺漏", use_container_width=True):
        with cyber_spinner("CATCH UP", "補跟遺漏交易日中..."):
            caught = trader.catch_up()
        if caught > 0:
            st.success(f"已補跟 {caught} 個交易日")
        else:
            st.info("無需補跟，已是最新狀態")
        _post_update_rerun()
    if bc4.button("重置帳戶", use_container_width=True):
        trader.reset()
        _post_update_rerun()
