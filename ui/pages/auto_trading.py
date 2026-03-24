"""
Module: Auto Trading System (自動交易系統)
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from analysis.indicators import calculate_bbands
from analysis.patterns import detect_candlestick_patterns
from data.provider import get_data_provider
from ui.charts import render_position_chart
from ui.components import cyber_spinner
from ui.theme import _plotly_dark_layout, _load_recommendation


def _get_trading_context():
    """初始化交易上下文（單次建立，供所有 tab 共用）"""
    from data.auto_trader import AutoTrader
    from data.paper_trader import PaperTrader
    trader = AutoTrader()
    config = trader.config
    status = trader.get_status()
    pt = PaperTrader()
    pt_status = pt.get_status()
    positions = pt_status.get('positions', []) if isinstance(pt_status, dict) else []
    return trader, config, status, pt, pt_status, positions


def render(_embedded=False):
    trader, config, status, pt, pt_status, positions = _get_trading_context()

    mode_dot = 'status-live' if config['mode'] == 'live' else 'status-sim' if config['enabled'] else 'status-off'
    mode_label = '實盤' if config['mode'] == 'live' else '模擬'
    mode_clr = '#ef4444' if config['mode'] == 'live' else '#f59e0b'

    if not _embedded:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
            <div style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">自動交易引擎</div>
            <span><span class="status-dot {mode_dot}"></span><span style="color:{mode_clr}; font-size:0.8rem; font-weight:700;">{mode_label}</span></span>
            <span class="tag {'tag-ok' if config['enabled'] else 'tag-warn'}">{'已啟用' if config['enabled'] else '已停用'}</span>
        </div>
        """, unsafe_allow_html=True)

    tab_ctrl, tab_charts, tab_orders, tab_config = st.tabs(["🎮 控制面板", "📊 持倉線圖", "📋 委託紀錄", "⚙️ 系統設定"])

    with tab_ctrl:
        _render_control_panel(trader, config, status, pt, positions)
    with tab_charts:
        _render_position_charts(positions, pt_status)
    with tab_orders:
        _render_order_book(trader)
    with tab_config:
        _render_config(trader, config)


def _render_control_panel(trader, config, status, pt, positions):
    # ------ TAB 1: Control Panel ------
        mode_label = '實盤' if config['mode'] == 'live' else '模擬'
        mode_clr = '#ef4444' if config['mode'] == 'live' else '#f59e0b'
        today_count = status.get('today_count', 0)
        total_orders = status.get('total_orders', 0)
        recent = status.get('recent_30d', {})

        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-item" style="border-left:3px solid {mode_clr}">
                <div class="kpi-label">模式</div><div class="kpi-value" style="color:{mode_clr}">{mode_label}</div>
            </div>
            <div class="kpi-item"><div class="kpi-label">今日委託</div><div class="kpi-value">{today_count}</div></div>
            <div class="kpi-item"><div class="kpi-label">累計委託</div><div class="kpi-value">{total_orders}</div></div>
            <div class="kpi-item"><div class="kpi-label">30日買入</div><div class="kpi-value">${recent.get('buy_amount',0):,.0f}</div></div>
            <div class="kpi-item"><div class="kpi-label">30日賣出</div><div class="kpi-value">${recent.get('sell_amount',0):,.0f}</div></div>
            <div class="kpi-item"><div class="kpi-label">委託類型</div><div class="kpi-value">{config.get('order_type','market').upper()}</div></div>
            <div class="kpi-item"><div class="kpi-label">排程時間</div><div class="kpi-value">{config.get('schedule_time','09:05')}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Execution Pipeline
        st.markdown('<p class="sec-header">執行管線</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; gap:8px; align-items:center; padding:16px 0; color:#94a3b8; font-size:0.85rem;">
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem;">📊</div>
                <div style="font-weight:600;">1. 策略信號</div>
                <div style="font-size:0.7rem;">Isaac V3.7</div>
            </div>
            <div style="color:#334155;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem;">🛡️</div>
                <div style="font-weight:600;">2. 風控預檢</div>
                <div style="font-size:0.7rem;">下單前檢查</div>
            </div>
            <div style="color:#334155;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem;">📝</div>
                <div style="font-weight:600;">3. 產生委託</div>
                <div style="font-size:0.7rem;">委託簿</div>
            </div>
            <div style="color:#334155;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem;">⚡</div>
                <div style="font-weight:600;">4. 執行下單</div>
                <div style="font-size:0.7rem;">Shioaji API</div>
            </div>
            <div style="color:#334155;">→</div>
            <div style="text-align:center; flex:1;">
                <div style="font-size:1.5rem;">📨</div>
                <div style="font-weight:600;">5. 推播通知</div>
                <div style="font-size:0.7rem;">Telegram</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action Buttons
        st.markdown('<p class="sec-header">操作</p>', unsafe_allow_html=True)
        ac1, ac2, ac3, ac4 = st.columns(4)

        # Preview orders (dry run)
        if ac1.button("預覽委託單", use_container_width=True):
            rec = _load_recommendation()
            if rec:
                check = trader.pre_trade_check(rec)
                if not check['passed']:
                    for issue in check['issues']:
                        st.markdown(f'<div class="alert-card alert-warn"><div class="alert-body">{issue}</div></div>', unsafe_allow_html=True)
                orders = trader.generate_orders(rec)
                if orders:
                    ord_rows = ""
                    for o in orders:
                        act_cls = 'tag-new' if o['action'] == 'BUY' else 'tag-exit'
                        ord_rows += f"""<tr>
                            <td style="text-align:left"><span class="tag {act_cls}">{o['action']}</span></td>
                            <td style="text-align:left"><strong>{o['ticker']}</strong> {o.get('name','')}</td>
                            <td>{o.get('price',0):,.1f}</td>
                            <td>{o.get('shares','—'):,}</td>
                            <td>${o.get('cost',0):,.0f}</td>
                            <td>{o.get('score','—')}</td>
                            <td>{o.get('order_type','market').upper()}</td>
                        </tr>"""
                    st.markdown(f"""<table class="pro-table"><thead><tr><th style="text-align:left">操作</th><th style="text-align:left">股票</th><th>價格</th><th>股數</th><th>金額</th><th>評分</th><th>類型</th></tr></thead><tbody>{ord_rows}</tbody></table>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-card alert-info"><div class="alert-body">今日無委託可執行。</div></div>', unsafe_allow_html=True)
            else:
                st.error("找不到推薦資料。")

        # Execute
        if ac2.button("立即執行", use_container_width=True, type="primary"):
            if not config['enabled']:
                st.markdown('<div class="alert-card alert-warn"><div class="alert-title">未啟用</div><div class="alert-body">請先在系統設定中啟用自動交易。</div></div>', unsafe_allow_html=True)
            else:
                with cyber_spinner("EXECUTING", "每日交易管線執行中..."):
                    result = trader.execute_daily()
                    if result['status'] == 'completed':
                        s = result['summary']
                        st.markdown(f'<div class="alert-card alert-ok"><div class="alert-title">執行完成</div><div class="alert-body">成交: {s["filled"]}/{s["total_orders"]} | 買入: {s["buy_orders"]} | 賣出: {s["sell_orders"]} | 金額: ${s["total_cost"]:,.0f}</div></div>', unsafe_allow_html=True)
                    elif result['status'] == 'blocked':
                        for issue in result.get('risk_check', {}).get('issues', []):
                            st.markdown(f'<div class="alert-card alert-danger"><div class="alert-body">{issue}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-card alert-info"><div class="alert-body">{result.get("message","")}</div></div>', unsafe_allow_html=True)

        # Enable/Disable
        if config['enabled']:
            if ac3.button("🔴 停用", use_container_width=True):
                trader.update_config(enabled=False); st.rerun()
        else:
            if ac3.button("🟢 啟用", use_container_width=True):
                if config['mode'] == 'live':
                    st.markdown('<div class="alert-card alert-danger"><div class="alert-title">⚠️ LIVE MODE</div><div class="alert-body">正在以 LIVE 模式啟用，將使用真實資金下單！請確認 Shioaji API 設定正確。</div></div>', unsafe_allow_html=True)
                trader.update_config(enabled=True); st.rerun()

        # Risk Check
        if ac4.button("風控檢查", use_container_width=True):
            rec = _load_recommendation()
            if rec:
                check = trader.pre_trade_check(rec)
                if check['passed']:
                    st.markdown('<div class="alert-card alert-ok"><div class="alert-title">所有檢查通過</div></div>', unsafe_allow_html=True)
                else:
                    for issue in check['issues']:
                        st.markdown(f'<div class="alert-card alert-warn"><div class="alert-body">{issue}</div></div>', unsafe_allow_html=True)

        # Today's executed orders
        today_orders = status.get('today_orders', [])
        st.markdown('<p class="sec-header">今日執行紀錄</p>', unsafe_allow_html=True)
        if not today_orders:
            st.markdown('<div style="text-align:center; padding:20px; color:#334155; font-size:0.8rem;">今日尚無已執行委託</div>', unsafe_allow_html=True)
        if today_orders:
            for o in today_orders:
                status_tag = 'tag-ok' if o.get('status') == 'filled' else 'tag-danger' if o.get('status') == 'failed' else 'tag-warn'
                act_tag = 'tag-new' if o.get('action') == 'BUY' else 'tag-exit'
                st.markdown(f"""
                <div class="ob-row">
                    <span><span class="tag {act_tag}">{o.get('action','')}</span> <strong>{o.get('ticker','')}</strong> {o.get('name','')}</span>
                    <span><span class="tag {status_tag}">{o.get('status','').upper()}</span> {o.get('filled_price', o.get('price', 0)):,.1f} x {o.get('shares',0):,}</span>
                </div>
                """, unsafe_allow_html=True)


def _render_position_charts(positions, pt_status=None):
    if pt_status is None:
        pt_status = {}
    # ------ TAB 2: Position Charts (Multi-Chart Dashboard) ------
    # Compact control bar
    ctrl_cols = st.columns([1, 1, 1, 1, 1, 1.2])
    st.session_state.chart_settings['trendline'] = ctrl_cols[0].checkbox("壓力/支撐", value=st.session_state.chart_settings.get('trendline', True), key="at_trendline")
    st.session_state.chart_settings['gaps'] = ctrl_cols[1].checkbox("跳空", value=st.session_state.chart_settings.get('gaps', True), key="at_gaps")
    st.session_state.chart_settings['ma'] = ctrl_cols[2].checkbox("MA", value=st.session_state.chart_settings.get('ma', True), key="at_ma")
    st.session_state.chart_settings['bbands'] = ctrl_cols[3].checkbox("BB", value=st.session_state.chart_settings.get('bbands', False), key="at_bbands")
    st.session_state.chart_settings['candle_patterns'] = ctrl_cols[4].checkbox("K線型態", value=st.session_state.chart_settings.get('candle_patterns', True), key="at_candle")
    if ctrl_cols[5].button("🔄 重新整理", use_container_width=True):
        st.rerun()

    # Master legend
    st.markdown("""
    <div style="display:flex; gap:16px; padding:4px 8px; font-size:0.7rem; color:#64748b; border-bottom:1px solid #1e293b; margin-bottom:8px;">
        <span><span style="color:#ef4444;">■</span> 漲 (紅K)</span>
        <span><span style="color:#22c55e;">■</span> 跌 (綠K)</span>
        <span><span style="color:orange;">─</span> MA短</span>
        <span><span style="color:#3b82f6;">─</span> MA長</span>
        <span><span style="color:#22c55e;">╌</span> 壓力線</span>
        <span><span style="color:#ef4444;">╌</span> 支撐線</span>
        <span><span style="color:#f59e0b;">╌</span> 進場價</span>
        <span style="color:#94a3b8;">🖱️ 滾輪縮放 | 拖曳平移</span>
    </div>
    """, unsafe_allow_html=True)

    if not positions:
        st.markdown("""
        <div class="alert-card alert-info">
            <div class="alert-title">無持倉</div>
            <div class="alert-body">目前沒有持倉。請先在模擬交易建立部位，或執行自動交易產生持倉後，此處會自動顯示所有持股的即時線圖。</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        n_pos = len(positions)

        # Compact KPI bar
        unr = pt_status.get('total_unrealized', 0)
        ret = pt_status.get('return_pct', 0)
        st.markdown(f"""
        <div style="display:flex; gap:24px; padding:6px 12px; background:#0f172a; border-radius:6px; margin-bottom:10px; font-size:0.8rem;">
            <span style="color:#94a3b8;">持倉 <strong style="color:#e2e8f0;">{n_pos}</strong></span>
            <span style="color:#94a3b8;">權益 <strong style="color:#e2e8f0;">${pt_status.get('equity', 0):,.0f}</strong></span>
            <span style="color:#94a3b8;">未實現 <strong style="color:{'#ef4444' if unr >= 0 else '#22c55e'}">{unr:+,.0f}</strong></span>
            <span style="color:#94a3b8;">報酬 <strong style="color:{'#ef4444' if ret >= 0 else '#22c55e'}">{ret:+.2f}%</strong></span>
        </div>
        """, unsafe_allow_html=True)

        # 2-column compact chart grid
        market_mode = st.session_state.get('market_mode', '🇹🇼 台股 (TW)')
        market_type = "TW" if "台股" in market_mode else "US"

        for i in range(0, n_pos, 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= n_pos:
                    break
                pos = positions[idx]
                ticker = pos.get('ticker', '')
                name = pos.get('name', ticker)
                pnl_pct = pos.get('pnl_pct', 0)
                entry_price = pos.get('entry_price', 0)
                cur_price = pos.get('current_price', entry_price)
                shares = pos.get('shares', 0)

                with col:
                    pnl_clr = '#ef4444' if pnl_pct >= 0 else '#22c55e'
                    pnl_abs = pos.get('pnl_abs', (cur_price - entry_price) * shares)
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 10px; background:#1e293b; border-radius:4px 4px 0 0; border-left:3px solid {pnl_clr};">
                        <div>
                            <span style="font-weight:800; color:#e2e8f0; font-size:0.95rem;">{ticker}</span>
                            <span style="color:#64748b; font-size:0.7rem; margin-left:6px;">{name}</span>
                        </div>
                        <div style="text-align:right; font-size:0.8rem;">
                            <span style="color:{pnl_clr}; font-weight:700;">{pnl_pct:+.2f}%</span>
                            <span style="color:#475569; margin-left:4px;">{pnl_abs:+,.0f}</span>
                            <span style="color:#334155; margin-left:4px;">|</span>
                            <span style="color:#94a3b8; margin-left:4px;">{cur_price:,.0f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    try:
                        provider = get_data_provider("auto", market_type=market_type)
                        df = provider.get_historical_data(ticker, period="6mo", interval="1d")
                        if df is None or df.empty or len(df) <= 30:
                            st.markdown(f'<div style="height:60px; display:flex; align-items:center; justify-content:center; background:#0f172a; border-radius:0 0 4px 4px; color:#475569; font-size:0.75rem;">⚠ {ticker} 無法取得資料（Yahoo Finance 可能不支援此代碼）</div>', unsafe_allow_html=True)
                            continue

                        df['BB_Upper'], _, df['BB_Lower'] = calculate_bbands(df)
                        candle_pats = detect_candlestick_patterns(df)
                        patterns = []

                        render_position_chart(
                            df, patterns, market_mode,
                            entry_price=entry_price,
                            height=260,
                            candle_patterns=candle_pats,
                        )
                    except Exception as e:
                        st.error(f"{ticker}: {e}")


def _render_order_book(trader):
    # ------ TAB 3: Order Book ------
    history = trader.get_order_history(days=30)
    if history:
        st.markdown(f'<p class="sec-header">委託歷史（近 30 天，共 {len(history)} 筆）</p>', unsafe_allow_html=True)
        oh_rows = "".join(
            f"""<tr>
                <td style="text-align:left">{o.get('timestamp','')[:16]}</td>
                <td style="text-align:left"><span class="tag {'tag-new' if o.get('action')=='BUY' else 'tag-exit'}">{o.get('action','')}</span></td>
                <td style="text-align:left"><strong>{o.get('ticker','')}</strong></td>
                <td>{o.get('price',0):,.1f}</td>
                <td>{o.get('shares',0):,}</td>
                <td><span class="tag {'tag-ok' if o.get('status')=='filled' else 'tag-danger' if o.get('status')=='failed' else 'tag-warn'}">{o.get('status','').upper()}</span></td>
                <td style="font-size:0.75rem;color:#64748b">{o.get('message','')}</td>
            </tr>""" for o in history[:100]
        )
        st.markdown(f"""<table class="pro-table"><thead><tr><th style="text-align:left">時間</th><th style="text-align:left">操作</th><th style="text-align:left">股票</th><th>價格</th><th>股數</th><th>狀態</th><th>訊息</th></tr></thead><tbody>{oh_rows}</tbody></table>""", unsafe_allow_html=True)

        # Daily summary chart
        summaries = status.get('daily_summaries', [])
        if summaries:
            sum_df = pd.DataFrame(summaries)
            fig_sum = go.Figure()
            fig_sum.add_trace(go.Bar(x=sum_df['date'], y=sum_df.get('buy_orders', [0]*len(sum_df)), name='買入', marker_color='#3b82f6'))
            fig_sum.add_trace(go.Bar(x=sum_df['date'], y=sum_df.get('sell_orders', [0]*len(sum_df)), name='賣出', marker_color='#f97316'))
            _plotly_dark_layout(fig_sum, height=300, title_text="每日委託數量", barmode='stack')
            st.plotly_chart(fig_sum, use_container_width=True)
    else:
        st.markdown('<div class="alert-card alert-info"><div class="alert-body">尚無委託紀錄。請執行管線以產生委託。</div></div>', unsafe_allow_html=True)


def _render_config(trader, config):
    # ------ TAB 4: Configuration ------
    st.markdown('<p class="sec-header">交易參數設定</p>', unsafe_allow_html=True)

    cfg_c1, cfg_c2 = st.columns(2)
    with cfg_c1:
        new_mode = st.selectbox("交易模式", ['simulation', 'live'], index=0 if config['mode'] == 'simulation' else 1, format_func=lambda x: '模擬' if x == 'simulation' else '實盤')
        new_order_type = st.selectbox("委託類型", ['market', 'limit'], index=0 if config['order_type'] == 'market' else 1, format_func=lambda x: '市價' if x == 'market' else '限價')
        new_max_order = st.number_input("單筆最大金額 ($)", value=config.get('max_order_value', 200000), step=10000)
        new_capital = st.number_input("初始資金 ($)", value=config.get('initial_capital', 1000000), step=100000)
    with cfg_c2:
        new_limit_offset = st.number_input("限價偏移 (%)", value=config.get('limit_offset_pct', 0.5), step=0.1)
        new_daily_limit = st.number_input("當日虧損上限 (%)", value=config.get('daily_loss_limit', 3.0), step=0.5)
        new_schedule = st.text_input("排程時間", value=config.get('schedule_time', '09:05'))
        new_pre_check = st.checkbox("下單前風控檢查", value=config.get('pre_trade_checks', True))

    if st.button("儲存設定", use_container_width=True, type="primary"):
        trader.update_config(
            mode=new_mode, order_type=new_order_type, max_order_value=new_max_order,
            initial_capital=new_capital, limit_offset_pct=new_limit_offset,
            daily_loss_limit=new_daily_limit, schedule_time=new_schedule,
            pre_trade_checks=new_pre_check,
        )
        st.markdown('<div class="alert-card alert-ok"><div class="alert-title">設定已儲存</div></div>', unsafe_allow_html=True)
        st.rerun()

    if new_mode == 'live':
        st.markdown("""
        <div class="alert-card alert-danger">
            <div class="alert-title">⚠️ 實盤交易警告</div>
            <div class="alert-body">
                實盤模式將透過永豐金 Shioaji API 執行真實下單，使用真實資金。
                請確認 API 金鑰設定正確，並充分了解交易風險。
                系統將使用 ROD（當日有效）委託單。
            </div>
        </div>
        """, unsafe_allow_html=True)
