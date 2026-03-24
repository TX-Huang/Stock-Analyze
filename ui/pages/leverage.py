"""
Module: Leveraged ETF Analysis (槓桿ETF評估)
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ui.components import cyber_spinner

from analysis.indicators import calculate_rsi, calculate_macd, calculate_bbands
from analysis.patterns import detect_candlestick_patterns
from analysis.trend import calculate_trend_logic
from data.provider import get_data_provider
from ui.charts import render_trend_chart
from ui.theme import _plotly_dark_layout


def render(_embedded=False):
    from analysis.leverage import (
        LEVERAGED_ETF_PRESETS, compute_hv, calculate_volatility_decay,
        classify_hv_regime, calculate_decay_cost_per_day, calculate_breakeven_move,
        compare_actual_vs_theoretical, generate_decay_heatmap_data,
        calculate_entry_signal_score, calculate_optimal_leverage,
    )
    compute_hv_standalone = compute_hv

    # Header
    if not _embedded:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
            <div style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">槓桿 ETF 分析系統</div>
            <span class="tag tag-new">PRO</span>
        </div>
        <div style="color:#64748b; font-size:0.85rem; margin-bottom:24px;">
            波動衰減分析 | HV 波動率分類 | Kelly 最佳槓桿 | 進場信號評分
        </div>
        """, unsafe_allow_html=True)

    # Input Panel
    col_sel, col_etf, col_und, col_lev = st.columns([2.5, 1.5, 1.5, 1])
    with col_sel:
        preset_options = ["自訂"] + [f"{k} - {v['name']}" for k, v in LEVERAGED_ETF_PRESETS.items()]
        preset_choice = st.selectbox("快速選擇", preset_options, label_visibility="collapsed")
    if preset_choice != "自訂":
        pkey = preset_choice.split(" - ")[0]; preset = LEVERAGED_ETF_PRESETS[pkey]
        d_etf, d_und, d_lev, d_mkt = pkey, preset['underlying'], preset['leverage'], preset['market']
    else:
        d_etf, d_und, d_lev, d_mkt = "00631L", "0050", 2.0, "TW"
    with col_etf: etf_ticker = st.text_input("ETF 代碼", value=d_etf, placeholder="00631L")
    with col_und: underlying_ticker = st.text_input("標的代碼", value=d_und, placeholder="0050")
    with col_lev: leverage_ratio = st.number_input("槓桿倍數", value=d_lev, min_value=-3.0, max_value=3.0, step=0.5)

    ds_c1, ds_c2 = st.columns([3, 1])
    with ds_c1:
        data_source = st.radio("資料來源", ["永豐金 (預設)", "Yahoo Finance"], horizontal=True, label_visibility="collapsed")

    lev_analyze_btn = st.button("開始分析", use_container_width=True, type="primary")

    if lev_analyze_btn:
        if not etf_ticker or not underlying_ticker:
            st.error("請輸入 ETF 和標的代碼")
        else:
            st.session_state.leverage_etf_data = None; st.session_state.leverage_underlying_data = None; st.session_state.leverage_analysis = None
            with cyber_spinner("FETCHING", f"{etf_ticker} & {underlying_ticker} 資料載入中..."):
                try:
                    if data_source == "Yahoo Finance":
                        provider = get_data_provider("yfinance", market_type=d_mkt)
                    else:
                        provider = get_data_provider("auto", market_type=d_mkt)
                    etf_df = provider.get_historical_data(etf_ticker, period="2y", interval="1d")
                    und_df = provider.get_historical_data(underlying_ticker, period="2y", interval="1d")
                    if etf_df.empty:
                        st.error(f"無法取得 {etf_ticker} 的資料")
                    elif und_df.empty:
                        st.error(f"無法取得 {underlying_ticker} 的資料")
                    else:
                        etf_df['RSI'] = calculate_rsi(etf_df['Close'])
                        _, _, etf_df['MACD_Hist'] = calculate_macd(etf_df)
                        und_returns = und_df['Close'].pct_change()
                        hv_series = compute_hv_standalone(und_df['Close'], 20)
                        hv_regime_df = classify_hv_regime(hv_series, lookback=120)
                        decay_df = calculate_volatility_decay(und_returns, leverage_ratio)
                        heatmap_df = generate_decay_heatmap_data(und_returns, leverage_ratio)
                        comparison_df = compare_actual_vs_theoretical(etf_df, und_df, leverage_ratio)
                        candle_patterns = detect_candlestick_patterns(etf_df)
                        trend_verdict = calculate_trend_logic(etf_df)
                        current_hv_regime = hv_regime_df['hv_regime'].iloc[-1] if not hv_regime_df.empty else 'Normal'
                        signal_result = calculate_entry_signal_score(etf_df, candle_patterns, current_hv_regime, trend_verdict)
                        st.session_state.leverage_etf_data = etf_df
                        st.session_state.leverage_underlying_data = und_df
                        st.session_state.leverage_analysis = {
                            'etf_ticker': etf_ticker, 'underlying_ticker': underlying_ticker, 'leverage': leverage_ratio,
                            'hv_series': hv_series, 'hv_regime_df': hv_regime_df, 'decay_df': decay_df,
                            'heatmap_df': heatmap_df, 'comparison_df': comparison_df, 'candle_patterns': candle_patterns,
                            'trend_verdict': trend_verdict, 'signal_result': signal_result, 'und_returns': und_returns, 'market': d_mkt,
                        }
                except Exception as e:
                    st.error(f"分析失敗: {e}"); import traceback; st.code(traceback.format_exc())

    # === Results ===
    if st.session_state.leverage_analysis is None and not lev_analyze_btn:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#475569;">
            <div style="font-size:2rem; margin-bottom:12px;">📊</div>
            <div style="font-size:1rem; color:#64748b;">選擇 ETF 後點擊「開始分析」</div>
            <div style="font-size:0.8rem; color:#334155; margin-top:8px;">支援台股 (00631L, 00632R) 及美股 (TQQQ, SOXL, UPRO) 槓桿 ETF</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.leverage_analysis is not None:
        ana = st.session_state.leverage_analysis
        etf_df = st.session_state.leverage_etf_data
        und_df = st.session_state.leverage_underlying_data

        hv_regime_df = ana['hv_regime_df']
        current_hv = hv_regime_df['hv'].iloc[-1] if not hv_regime_df.empty else 0
        current_regime = hv_regime_df['hv_regime'].iloc[-1] if not hv_regime_df.empty else 'N/A'
        current_pctl = hv_regime_df['hv_percentile'].iloc[-1] if not hv_regime_df.empty else 0
        daily_decay_bps = calculate_decay_cost_per_day(current_hv, ana['leverage'])
        decay_30d = ana['decay_df']['decay_30d'].iloc[-1] * 100 if 'decay_30d' in ana['decay_df'].columns else 0
        breakeven = calculate_breakeven_move(abs(decay_30d), ana['leverage'])
        signal = ana['signal_result']
        trend_str = ana['trend_verdict'].get('trend', 'N/A') if ana['trend_verdict'] else 'N/A'

        # KPI Strip
        regime_colors = {'Low': '#22c55e', 'Normal': '#f59e0b', 'High': '#ef4444'}
        regime_clr = regime_colors.get(current_regime, '#94a3b8')
        signal_clr = '#22c55e' if signal['score'] >= 50 else '#f59e0b' if signal['score'] >= 30 else '#ef4444'

        st.markdown(f"""
        <div class="kpi-strip">
            <div class="kpi-item" style="border-left: 3px solid {regime_clr};">
                <div class="kpi-label">波動率區間</div>
                <div class="kpi-value" style="color:{regime_clr}">{current_regime} <span style="font-size:0.7rem;color:#64748b">P{current_pctl:.0f}</span></div>
            </div>
            <div class="kpi-item"><div class="kpi-label">年化波動率</div><div class="kpi-value">{current_hv:.4f}</div></div>
            <div class="kpi-item"><div class="kpi-label">日衰減</div><div class="kpi-value">{daily_decay_bps:.1f} <span style="font-size:0.7rem">bps</span></div></div>
            <div class="kpi-item"><div class="kpi-label">30日累計衰減</div><div class="kpi-value">{decay_30d:.2f}%</div></div>
            <div class="kpi-item"><div class="kpi-label">損益平衡漲幅</div><div class="kpi-value">{breakeven:.2f}%</div></div>
            <div class="kpi-item"><div class="kpi-label">趨勢</div><div class="kpi-value">{trend_str}</div></div>
            <div class="kpi-item" style="border-left: 3px solid {signal_clr};">
                <div class="kpi-label">進場信號</div>
                <div class="kpi-value" style="color:{signal_clr}">{signal['score']}/100 <span style="font-size:0.7rem">{signal['recommendation_zh']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if current_regime == 'High' and not any(kw in trend_str for kw in ['多頭', '上升']):
            st.markdown('<div class="alert-card alert-danger"><div class="alert-title">高衰減區域警告</div><div class="alert-body">HV 偏高且無上升趨勢。建議避免左側攤平，等待趨勢確認後右側進場。</div></div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📉 波動衰減", "🕯️ 進場信號", "🔬 情境模擬", "⚖️ 實際vs理論"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig_hv = go.Figure()
                hv_clean = hv_regime_df.dropna(subset=['hv'])
                fig_hv.add_trace(go.Scatter(x=hv_clean.index, y=hv_clean['hv'], mode='lines', name='HV (20D)', line=dict(color='#6366f1', width=2)))
                if 'hv_p20' in hv_clean.columns:
                    fig_hv.add_trace(go.Scatter(x=hv_clean.index, y=hv_clean['hv_p20'], mode='lines', name='P20', line=dict(color='#22c55e', width=1, dash='dot')))
                if 'hv_p80' in hv_clean.columns:
                    fig_hv.add_trace(go.Scatter(x=hv_clean.index, y=hv_clean['hv_p80'], mode='lines', name='P80', line=dict(color='#ef4444', width=1, dash='dot')))
                _plotly_dark_layout(fig_hv, height=350, title_text="HV 波動率區間分類")
                st.plotly_chart(fig_hv, use_container_width=True)
            with c2:
                decay_plot = ana['decay_df'].dropna().tail(250) * 100
                fig_decay = go.Figure()
                dcolors = ['#22c55e', '#eab308', '#f97316', '#ef4444']
                for i, col in enumerate(decay_plot.columns):
                    fig_decay.add_trace(go.Scatter(x=decay_plot.index, y=decay_plot[col], mode='lines', name=col.replace('decay_', ''), line=dict(color=dcolors[i % len(dcolors)], width=2)))
                fig_decay.add_hline(y=0, line_dash="dash", line_color="#334155")
                _plotly_dark_layout(fig_decay, height=350, title_text="波動衰減曲線")
                st.plotly_chart(fig_decay, use_container_width=True)

            # Heatmap + Kelly side by side
            c3, c4 = st.columns(2)
            with c3:
                heatmap_data = ana['heatmap_df'].dropna()
                if not heatmap_data.empty:
                    fig_heat = px.imshow(heatmap_data.T, labels=dict(x="日期", y="持有天數", color="衰減 (%)"), color_continuous_scale='RdYlGn_r', aspect='auto')
                    _plotly_dark_layout(fig_heat, height=280, title_text="衰減熱力圖")
                    st.plotly_chart(fig_heat, use_container_width=True)
            with c4:
                exp_daily_ret = und_df['Close'].pct_change().mean() if und_df is not None else 0
                opt_result = calculate_optimal_leverage(current_hv, exp_daily_ret)
                if opt_result['leverage_curve']:
                    curve_df = pd.DataFrame(opt_result['leverage_curve'])
                    fig_kelly = go.Figure()
                    fig_kelly.add_trace(go.Scatter(x=curve_df['leverage'], y=curve_df['expected_daily_return'], mode='lines+markers', name='Expected Daily (bps)', line=dict(color='#6366f1', width=2), marker=dict(size=4)))
                    fig_kelly.add_vline(x=ana['leverage'], line_dash="dash", annotation_text=f"目前 {ana['leverage']}x", line_color="#ef4444")
                    fig_kelly.add_vline(x=opt_result['optimal_leverage'], line_dash="dash", annotation_text=f"最佳 {opt_result['optimal_leverage']}x", line_color="#22c55e")
                    _plotly_dark_layout(fig_kelly, height=280, title_text=f"Kelly 最佳槓桿 (最佳: {opt_result['optimal_leverage']}x)")
                    st.plotly_chart(fig_kelly, use_container_width=True)

        with tab2:
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                score_val = signal['score']
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=score_val, domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': signal['recommendation_zh'], 'font': {'size': 16, 'color': '#94a3b8'}},
                    gauge={'axis': {'range': [0, 100], 'tickcolor': '#334155'},
                           'bar': {'color': signal_clr},
                           'bgcolor': '#1e293b',
                           'steps': [{'range': [0, 30], 'color': 'rgba(239,68,68,0.2)'}, {'range': [30, 50], 'color': 'rgba(245,158,11,0.2)'}, {'range': [50, 100], 'color': 'rgba(34,197,94,0.2)'}]}))
                _plotly_dark_layout(fig_gauge, height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with sc2:
                for factor in signal['factors']:
                    icon = "+" if any(k in factor for k in ['多頭', '反轉', '強力', '爆增', '低檔', '上升']) else "-" if any(k in factor for k in ['空頭', '注意', '高檔', '下降']) else "~"
                    cls = 'c-up' if icon == '+' else 'c-down' if icon == '-' else 'c-flat'
                    st.markdown(f'<span class="{cls}">[{icon}]</span> {factor}', unsafe_allow_html=True)

            market_for_chart = "🇹🇼 台股 (TW)" if ana.get('market', 'TW') == "TW" else "🗽 美股 (US)"
            render_trend_chart(etf_df, [], market_for_chart, is_box=False, height=600, is_weekly=False, candle_patterns=ana['candle_patterns'])

        with tab3:
            sim_c1, sim_c2 = st.columns(2)
            with sim_c1:
                sim_days = st.slider("持有天數", 1, 120, 30)
                sim_move = st.slider("預期漲跌幅 (%)", -30.0, 30.0, 5.0, 0.5)
            with sim_c2:
                sim_hv = st.slider("波動率假設 (HV)", 0.05, 0.80, float(max(0.05, current_hv)), 0.01)
                sim_lev = st.number_input("槓桿倍數", value=float(ana['leverage']), min_value=-3.0, max_value=3.0, step=0.5, key="sim_lev")

            daily_var = (sim_hv / np.sqrt(252)) ** 2
            projected_decay = -0.5 * sim_lev * (sim_lev - 1) * daily_var * sim_days
            projected_lev_return = sim_lev * sim_move / 100 + projected_decay
            sim_breakeven = calculate_breakeven_move(abs(projected_decay) * 100, sim_lev)
            sim_daily_bps = calculate_decay_cost_per_day(sim_hv, sim_lev)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("預估衰減", f"{projected_decay*100:.2f}%")
            r2.metric("槓桿淨報酬", f"{projected_lev_return*100:.2f}%")
            r3.metric("損益平衡漲幅", f"{sim_breakeven:.2f}%")
            r4.metric("每日衰減", f"{sim_daily_bps:.1f} bps")

            hv_range = np.linspace(0.05, 0.80, 50)
            decay_by_hv = [-0.5 * sim_lev * (sim_lev - 1) * (h / np.sqrt(252))**2 * sim_days * 100 for h in hv_range]
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=hv_range * 100, y=decay_by_hv, mode='lines', name='累計衰減 (%)', line=dict(color='#ef4444', width=2), fill='tozeroy', fillcolor='rgba(239,68,68,0.1)'))
            fig_sim.add_vline(x=sim_hv * 100, line_dash="dash", annotation_text=f"HV={sim_hv*100:.0f}%", line_color="#3b82f6")
            _plotly_dark_layout(fig_sim, height=350, title_text=f"衰減 vs 波動率 ({sim_days}天, {sim_lev}x)")
            st.plotly_chart(fig_sim, use_container_width=True)

        with tab4:
            comp = ana['comparison_df']
            if comp is not None and not comp.empty:
                fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
                fig_comp.add_trace(go.Scatter(x=comp.index, y=comp['actual_etf'], mode='lines', name='實際ETF', line=dict(color='#3b82f6', width=2)), row=1, col=1)
                fig_comp.add_trace(go.Scatter(x=comp.index, y=comp['theoretical_leveraged'], mode='lines', name='理論值', line=dict(color='#f97316', width=1, dash='dot')), row=1, col=1)
                fig_comp.add_trace(go.Scatter(x=comp.index, y=comp['underlying_linear'], mode='lines', name='線性', line=dict(color='#64748b', width=1, dash='dash')), row=1, col=1)
                fig_comp.add_trace(go.Scatter(x=comp.index, y=comp['tracking_error'], mode='lines', name='追蹤誤差', line=dict(color='#ef4444', width=1), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)'), row=2, col=1)
                _plotly_dark_layout(fig_comp, height=500, title_text=f"{ana['etf_ticker']} 實際 vs 理論")
                st.plotly_chart(fig_comp, use_container_width=True)

                perf_c1, perf_c2, perf_c3, perf_c4 = st.columns(4)
                actual_ret = (comp['actual_etf'].iloc[-1] - 1) * 100; theo_ret = (comp['theoretical_leveraged'].iloc[-1] - 1) * 100
                linear_ret = (comp['underlying_linear'].iloc[-1] - 1) * 100; track_err = actual_ret - theo_ret
                perf_c1.metric("實際報酬", f"{actual_ret:.2f}%"); perf_c2.metric("理論報酬", f"{theo_ret:.2f}%")
                perf_c3.metric("線性報酬", f"{linear_ret:.2f}%"); perf_c4.metric("追蹤誤差", f"{track_err:.2f}%")
            else:
                st.warning("無法計算，數據日期可能不重疊。")
