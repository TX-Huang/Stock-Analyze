"""
Module: Stock War Room (股市戰情室)
"""
import streamlit as st
import re

from config.settings import GEMINI_MODEL
from analysis.ai_core import (
    resolve_ticker_and_market, detect_hot_themes,
    generate_supply_chain_structure, generate_ai_analysis
)
from data.scanner import scan_single_stock_deep, scan_tickers_from_map
from ui.charts import render_trend_chart, render_supply_chain_graph
from ui.components import custom_metric, cyber_spinner
from utils.helpers import get_default_sector_map_full, get_fallback_supply_chain


def render(client, market_mode, strategy_mode, tf_code, is_weekly, _embedded=False):
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
        details_html = ""
        if verdict.get('details'):
            details_html = '<hr style="border-color:#334155;margin:8px 0">'
            details_html += "".join(f'<div style="font-size:0.75rem;color:#94a3b8">• {d}</div>' for d in verdict.get('details'))
        signal_ctx = data.get('signal_context', '無')
        signal_val = verdict.get('signal', '')
        verdict_html = (
            f'<div class="alert-card" style="border-color:{border_color};padding:14px 18px">'
            f'<div class="alert-title" style="color:{border_color}">⚖️ 程式判決：{trend_val}</div>'
            f'<div class="alert-body" style="margin-top:6px">'
            f'<strong>訊號</strong>：{signal_val}'
            f'{details_html}'
            f'<hr style="border-color:#334155;margin:8px 0">'
            f'<strong>深度掃描</strong>：{signal_ctx}'
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
