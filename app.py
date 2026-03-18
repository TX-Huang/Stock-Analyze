# ==========================================
# Alpha Global v93.0 (Final UI Fix)
# ==========================================
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from google import genai
from google.genai import types
import json
import time
import os
import re
from datetime import datetime, timedelta, timezone
import numpy as np

from config.settings import GEMINI_MODEL, DEFAULT_CHART_SETTINGS
from analysis.indicators import (
    calculate_rsi, calculate_macd, calculate_bbands, calculate_stoch, calculate_obv
)
from analysis.patterns import detect_candlestick_patterns, detect_complex_patterns
from analysis.trend import calculate_trend_logic
from analysis.ai_core import (
    resolve_ticker_and_market, analyze_signals, detect_hot_themes,
    generate_supply_chain_structure, generate_ai_analysis
)
from data.provider import get_data_provider
from data.scanner import scan_single_stock_deep, scan_tickers_from_map
from ui.charts import render_trend_chart, render_supply_chain_graph
from ui.backtest_dashboard import render_backtest_dashboard
from ui.components import custom_metric
from utils.helpers import (
    robust_json_extract, validate_ticker, get_default_sector_map_full,
    get_fallback_supply_chain
)

# --- Config ---
st.set_page_config(page_title="Alpha Global v93.0", layout="wide", page_icon="📈")

# Fix Pandas Styler Limit
pd.set_option("styler.render.max_elements", 1_000_000)

# --- Session State ---
if 'data_cache' not in st.session_state: st.session_state.data_cache = {}
if 'ai_reports' not in st.session_state: st.session_state.ai_reports = {}
if 'market_mode' not in st.session_state: st.session_state.market_mode = "🇹🇼 台股 (TW)"
if 'dynamic_name_map' not in st.session_state: st.session_state.dynamic_name_map = {}
if 'view_mode' not in st.session_state: st.session_state.view_mode = "list"
if 'single_stock_data' not in st.session_state: st.session_state.single_stock_data = None
if 'current_source' not in st.session_state: st.session_state.current_source = "🗂️ 預設清單"
if 'detected_themes' not in st.session_state: st.session_state.detected_themes = []
if 'supply_chain_data' not in st.session_state: st.session_state.supply_chain_data = None

if 'chart_settings' not in st.session_state:
    st.session_state.chart_settings = DEFAULT_CHART_SETTINGS.copy()

# --- API Key ---
st.sidebar.header("🔑 啟動金鑰")
# Sidebar Navigation
app_mode = st.sidebar.radio("功能模組", ["📈 股市戰情室", "🧬 量化回測系統", "📂 自訂策略實驗室"])

st.sidebar.info("中文搜尋需 API Key。代碼搜尋 (如 2330, NVDA) 可免填。")

try:
    default_gemini = st.secrets.get("GEMINI_API_KEY", "")
except FileNotFoundError:
    default_gemini = ""
except Exception:
    default_gemini = ""

api_key = st.sidebar.text_input("輸入 Gemini API Key", type="password", value=default_gemini)

client = None
if api_key:
    try: client = genai.Client(api_key=api_key);
    except Exception as e: st.sidebar.error(f"連線錯誤: {e}")

# ==========================================
# Main UI
# ==========================================
st.title(f"💎 Alpha Global v93.0 (UI Perfection)")

with st.sidebar:
    if app_mode == "📈 股市戰情室":
        market_mode = st.radio("🌐 戰場狀態 (Auto)", ["🇹🇼 台股 (TW)", "🗽 美股 (US)"], index=0 if "台股" in st.session_state.market_mode else 1)
        st.session_state.market_mode = market_mode

        st.divider()
        timeframe = st.radio("🕒 K線週期", ["1d (日線)", "1wk (週線)"], index=0)
        tf_code = "1wk" if "週線" in timeframe else "1d"
        is_weekly = (tf_code == "1wk")

        strategy_mode = st.radio("⚔️ 交易風格", ["🔥 順勢突破 (Momentum)", "🛡️ 拉回抄底 (Dip Buy)"])

        st.divider()
        st.markdown("### ⚖️ 判官工具箱")
        c1, c2 = st.columns(2)
        st.session_state.chart_settings['trendline'] = c1.checkbox("自動支撐壓力線", value=True)
        st.session_state.chart_settings['gaps'] = c2.checkbox("標示跳空缺口", value=True)
        c3, c4 = st.columns(2)
        st.session_state.chart_settings['ma'] = c3.checkbox("移動平均線 (MA)", value=True)
        st.session_state.chart_settings['bbands'] = c4.checkbox("布林通道 (BBands)", value=False)
        st.session_state.chart_settings['candle_patterns'] = st.checkbox("K線型態辨識", value=True)
    else:
        st.info("目前位於「量化回測系統」模式。")

# === Main Content Area Switching ===

if app_mode == "🧬 量化回測系統":
    st.header("🧬 量化策略回測實驗室")

    with st.expander("🛠️ 策略設定 (Strategy Settings)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                default_finlab = st.secrets.get("FINLAB_API_TOKEN", "")
            except:
                default_finlab = ""
            finlab_token = st.text_input("Finlab API Token", type="password", value=default_finlab, help="請輸入您的 Finlab API 金鑰")
        with col2:
            strategy_type = st.selectbox("選擇回測策略", ["純做多策略 (Long Only)", "多空策略 (Long + Short)", "VCP 波動收縮策略 (Minervini)", "Isaac 頂級多因子策略 (Growth + Reversion)"])

        run_btn = st.button("🔬 執行回測 (Run Backtest)", use_container_width=True, type="primary")

    if 'backtest_report' not in st.session_state:
        st.session_state.backtest_report = None
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = None

    if run_btn:
        if not finlab_token:
            st.error("請輸入 Finlab API Token")
        else:
            st.session_state.backtest_report = None
            st.session_state.current_strategy = None

            with st.spinner(f"正在執行 {strategy_type} 回測... (這可能需要幾分鐘)"):
                try:
                    import importlib

                    if "純做多" in strategy_type:
                        import strategies.long_only as strategy_long
                        importlib.reload(strategy_long)
                        report = strategy_long.run_long_strategy(finlab_token)
                    elif "VCP" in strategy_type:
                        import strategies.vcp as strategy_vcp
                        importlib.reload(strategy_vcp)
                        report = strategy_vcp.run_vcp_strategy(finlab_token)
                    elif "Isaac" in strategy_type:
                        import strategies.isaac as strategy_isaac
                        importlib.reload(strategy_isaac)
                        report = strategy_isaac.run_isaac_strategy(finlab_token)
                    else:
                        import strategies.long_short as strategy_long_short
                        importlib.reload(strategy_long_short)
                        report = strategy_long_short.run_long_short_strategy(finlab_token)

                    st.success("回測完成！")
                    st.session_state.backtest_report = report
                    st.session_state.current_strategy = strategy_type

                except Exception as e:
                    st.error(f"回測執行發生錯誤: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    if st.session_state.backtest_report is not None:
        report = st.session_state.backtest_report

        equity = getattr(report, 'creturn', None)
        benchmark = getattr(report, 'benchmark', None)
        drawdown = equity / equity.cummax() - 1 if equity is not None else None
        trades = report.get_trades()
        stats = report.get_stats()

        with st.expander("🔍 Debug: stats 物件診斷 (開發者用)", expanded=False):
            st.write(f"stats type: `{type(stats).__name__}`")
            if hasattr(stats, 'keys'):
                st.write(f"stats keys: `{list(stats.keys())}`")
            st.write(f"trades count: `{len(trades)}`")
            for key in ['cagr', 'max_drawdown', 'win_ratio', 'win_rate', 'daily_mean', 'daily_sharpe']:
                try:
                    val = stats.get(key, 'KEY_NOT_FOUND') if hasattr(stats, 'get') else 'NO_GET'
                    st.write(f"stats.get('{key}'): `{val}`")
                except Exception as ex:
                    st.write(f"stats.get('{key}'): ERROR - `{ex}`")

        cagr = stats.get('cagr', 0)
        mdd = stats.get('max_drawdown', 0)
        win_rate = stats.get('win_ratio', 0)

        avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty else 0
        avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty else 0
        risk_reward = avg_win / avg_loss if avg_loss != 0 else 0

        avg_hold_win = trades[trades['return'] > 0]['period'].mean() if not trades.empty else 0
        avg_hold_loss = trades[trades['return'] <= 0]['period'].mean() if not trades.empty else 0

        exposure = (equity != equity.shift(1)).mean() if equity is not None else 0

        tab1, tab2, tab3 = st.tabs(["📊 實戰戰情室 (Core Metrics)", "🛡️ 參數強健性 (Stress Test)", "📋 交易明細 (Trades)"])

        with tab1:
            st.markdown("### 🏆 核心五大戰略指標")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🛡️ 心理極限 (MDD)", f"{mdd*100:.1f}%", help="最大資金回撤：你能承受的痛")
            with col2:
                st.metric("⚖️ 獲利引擎 (勝率/風報)", f"{win_rate*100:.0f}% | {risk_reward:.1f}", help="賠1塊賺幾塊？")
            with col3:
                st.metric("📈 真實複利 (CAGR)", f"{cagr*100:.1f}%", help="年化報酬率：資產翻倍速度")
            with col4:
                st.metric("⏳ 資金效率 (贏/輸天數)", f"{avg_hold_win:.0f} / {avg_hold_loss:.0f} 天", help="贏家抱多久 vs 輸家跑多快")
            with col5:
                st.metric("🛡️ 避險能力 (曝險)", f"{exposure*100:.0f}%", help="資金留在市場的時間比例")

            st.markdown("---")

            if equity is not None:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("資產權益曲線 (Equity Curve)", "資金回撤 (Drawdown)"),
                                  vertical_spacing=0.1, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                       mode='lines', name='策略報酬',
                                       line=dict(color='#22c55e', width=2)), row=1, col=1)
                if benchmark is not None:
                     fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values,
                                       mode='lines', name='大盤基準',
                                       line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                if drawdown is not None:
                    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                           mode='lines', name='回撤幅度',
                                           line=dict(color='#ef4444', width=1), fill='tozeroy'), row=2, col=1)
                fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("🛡️ 參數強健性掃描 (Stress Test)")
            st.info("此功能將執行多次回測，以檢驗策略在不同停損/停利參數下的穩定性。請耐心等待。")
            c_p1, c_p2 = st.columns(2)
            stop_loss_range = c_p1.slider("停損範圍 (%)", 5, 15, (8, 12))
            take_profit_range = c_p2.slider("停利範圍 (%)", 15, 40, (20, 30))

            if st.button("🔥 開始壓力測試 (Run Grid Search)"):
                if "Isaac" not in strategy_type:
                    st.warning("目前僅支援 Isaac 策略進行參數掃描。")
                else:
                    import importlib
                    import strategies.isaac as strategy_isaac
                    importlib.reload(strategy_isaac)
                    results = []
                    sl_steps = range(stop_loss_range[0], stop_loss_range[1]+1, 2)
                    tp_steps = range(take_profit_range[0], take_profit_range[1]+1, 5)
                    progress_bar = st.progress(0)
                    total_steps = len(sl_steps) * len(tp_steps)
                    step_count = 0
                    for sl in sl_steps:
                        for tp in tp_steps:
                            try:
                                rep = strategy_isaac.run_isaac_strategy(finlab_token, stop_loss=sl/100, take_profit=tp/100)
                                stats_grid = rep.get_stats()
                                results.append({
                                    'Stop Loss': sl, 'Take Profit': tp,
                                    'CAGR': stats_grid.get('cagr', 0),
                                    'Sharpe': stats_grid.get('sharpe', 0)
                                })
                            except Exception:
                                pass
                            step_count += 1
                            if step_count <= total_steps:
                                progress_bar.progress(step_count / total_steps)
                    df_grid = pd.DataFrame(results)
                    if not df_grid.empty:
                        pivot_table = df_grid.pivot(index='Stop Loss', columns='Take Profit', values='CAGR')
                        fig_heat = px.imshow(pivot_table,
                                           labels=dict(x="Take Profit (%)", y="Stop Loss (%)", color="CAGR"),
                                           color_continuous_scale='RdYlGn', text_auto='.1%')
                        st.plotly_chart(fig_heat, use_container_width=True)
                        st.success("壓力測試完成！請尋找紅色的高原區 (Plateau)。")

        with tab3:
            st.subheader("📋 詳細交易紀錄 (Trade Log)")
            if not trades.empty:
                rename_map = {
                    "stock_id": "股票代碼", "entry_date": "進場日期", "exit_date": "出場日期",
                    "entry_price": "進場價", "exit_price": "出場價", "return": "報酬率",
                    "mae": "最大不利(MAE)", "mfe": "最大有利(MFE)", "period": "持有天數"
                }
                trades_display = trades.copy()
                trades_display.rename(columns=rename_map, inplace=True)
                if '進場日期' in trades_display.columns:
                    trades_display['進場日期'] = pd.to_datetime(trades_display['進場日期'])
                if '出場日期' in trades_display.columns:
                    trades_display['出場日期'] = pd.to_datetime(trades_display['出場日期'], errors='coerce')
                try:
                    today = report.position.index[-1]
                except Exception:
                    today = datetime.now()

                def calculate_holding_main(row):
                    if pd.notna(row.get('出場日期')):
                        return (row['出場日期'] - row['進場日期']).days + 1
                    elif pd.notna(row.get('進場日期')):
                        return (today - row['進場日期']).days + 1
                    return row.get('持有天數', 0)

                trades_display['持有天數'] = trades_display.apply(calculate_holding_main, axis=1)
                min_date = trades_display['進場日期'].min().date()
                max_date = trades_display['進場日期'].max().date()
                c_filter1, c_filter2 = st.columns(2)
                start_date = c_filter1.date_input("開始日期", value=max_date - timedelta(days=365), min_value=min_date, max_value=max_date)
                end_date = c_filter2.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)
                trades_filtered = trades_display[
                    (trades_display['進場日期'].dt.date >= start_date) &
                    (trades_display['進場日期'].dt.date <= end_date)
                ]
                csv = trades_filtered.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 下載完整交易明細 (.csv)", data=csv,
                    file_name=f'trade_log_{datetime.now().strftime("%Y%m%d")}.csv', mime='text/csv')
                items_per_page = 1000
                total_items = len(trades_filtered)
                total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
                page = st.number_input("頁數 (Page)", min_value=1, max_value=total_pages, value=1)
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                st.info(f"顯示第 {start_idx + 1} 至 {end_idx} 筆交易 (共 {total_items} 筆)")
                available_cols = ['股票代碼', '進場日期', '出場日期', '進場價', '出場價', '報酬率', '持有天數', '最大不利(MAE)', '最大有利(MFE)']
                cols_to_show = [c for c in available_cols if c in trades_filtered.columns]
                trades_final = trades_filtered[cols_to_show].sort_values("進場日期", ascending=False).iloc[start_idx:end_idx]
                csv = trades_final.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 下載交易明細 (.csv)", data=csv,
                    file_name=f'trade_log_{strategy_type}_{datetime.now().strftime("%Y%m%d")}.csv', mime='text/csv')

                def highlight_ret(val):
                    if pd.isna(val): return ''
                    if isinstance(val, (int, float)):
                        return 'color: #ef4444' if val > 0 else 'color: #22c55e'
                    return ''

                st.dataframe(
                    trades_final.style.format({
                        '報酬率': '{:.2%}', '最大不利(MAE)': '{:.2%}', '最大有利(MFE)': '{:.2%}',
                        '進場價': '{:.2f}', '出場價': '{:.2f}'
                    }, na_rep="N/A").map(highlight_ret, subset=['報酬率']),
                    use_container_width=True, height=600)
            else:
                st.info("無交易紀錄")

elif app_mode == "📂 自訂策略實驗室":
    st.header("📂 自訂策略實驗室 (Lab)")
    st.info("⚠️ 注意：請確保上傳的策略程式碼來源可信。此功能將直接執行 Python 腳本。")
    col1, col2 = st.columns([1, 1])
    with col1:
        try:
            default_finlab = st.secrets.get("FINLAB_API_TOKEN", "")
        except:
            default_finlab = ""
        finlab_token = st.text_input("Finlab API Token", type="password", value=default_finlab)
    with col2:
        with open("strategies/template.py", "rb") as f:
            st.download_button("📥 下載策略範本 (Template)", f, file_name="template_strategy.py", mime="text/x-python")

    uploaded_file = st.file_uploader("上傳您的策略 (.py)", type=["py"])
    if uploaded_file is not None:
        if st.button("🚀 執行回測"):
            if not finlab_token:
                st.error("請輸入 Finlab API Token")
            else:
                with st.spinner("正在編譯並執行您的策略..."):
                    try:
                        import importlib.util
                        import sys
                        temp_filename = f"temp_strategy_{int(time.time())}.py"
                        with open(temp_filename, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        spec = importlib.util.spec_from_file_location("custom_strategy", temp_filename)
                        module = importlib.util.module_from_spec(spec)
                        sys.modules["custom_strategy"] = module
                        spec.loader.exec_module(module)
                        if hasattr(module, 'run_strategy'):
                            report = module.run_strategy(finlab_token)
                            st.success("執行成功！")
                            render_backtest_dashboard(report)
                        else:
                            st.error("錯誤：您的策略檔案中找不到 `run_strategy(api_token)` 函式。請參考範本。")
                        os.remove(temp_filename)
                    except Exception as e:
                        st.error(f"執行失敗: {e}")

elif app_mode == "📈 股市戰情室":
    st.markdown("### 🔎 1. 全市場狙擊")
    single_input = st.text_input("輸入代碼/名稱 (如 凡甲, NVDA)", placeholder="Sniper Input...")
    if st.button("🚀 分析個股"):
        if not single_input:
            st.error("請輸入代碼")
        else:
            st.session_state.supply_chain_data = None
            st.session_state.detected_themes = []
            with st.spinner(f"正在鎖定目標 {single_input}..."):
                target_ticker, detected_market, target_name = resolve_ticker_and_market(
                    single_input, client=client, gemini_model=GEMINI_MODEL)
                if not target_ticker and not api_key:
                    if re.match(r'^\d{4,6}$', single_input):
                        target_ticker = single_input; detected_market = "🇹🇼 台股 (TW)"; target_name = single_input
                    elif re.match(r'^[A-Z]{1,5}$', single_input.upper()):
                        target_ticker = single_input.upper(); detected_market = "🗽 美股 (US)"; target_name = single_input.upper()
                    else:
                        st.error("⚠️ 無法識別股票名稱。建議輸入代碼 (如 2330) 或輸入 Gemini API Key 以啟用完整 AI 搜尋功能。")
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
                        st.error("無法取得數據，請檢查代碼或網路")
                else:
                    if api_key:
                        st.error("無法識別股票，請嘗試輸入代碼 (例如 2330)")

    st.markdown("### 📡 2. 族群熱點掃描")
    if st.button("🔥 掃描今日熱門話題"):
        if not api_key:
            st.error("無 API Key")
        else:
            st.session_state.supply_chain_data = None
            st.session_state.single_stock_data = None
            with st.spinner("AI 正在閱讀新聞..."):
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
            with st.spinner(f"AI 正在拆解「{custom_input}」供應鏈結構..."):
                fallback = get_fallback_supply_chain(custom_input, st.session_state.market_mode)
                if fallback:
                    structure = fallback
                    st.success(f"已啟用內建資料庫：{custom_input}")
                else:
                    structure = generate_supply_chain_structure(
                        st.session_state.market_mode, custom_input,
                        client=client, gemini_model=GEMINI_MODEL)
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
        with c5: custom_metric("殖利率 (Yield)", f"{data.get('Yield', 'N/A')}", None)

        trend_val = verdict.get('trend', '')
        if '多' in trend_val or '上升' in trend_val:
            status_box = st.error
        elif '空' in trend_val or '下降' in trend_val:
            status_box = st.success
        else:
            status_box = st.warning

        with status_box(icon="⚖️", body=f"**程式判決：{trend_val}**"):
            st.write(f"**訊號**：{verdict.get('signal')}")
            if verdict.get('details'):
                st.markdown("---")
                for d in verdict.get('details'):
                    st.caption(f"• {d}")
            st.markdown("---")
            st.markdown(f"**⚡ 深度掃描**：{data.get('signal_context', '無')}")

        render_trend_chart(data['df'], patterns, st.session_state.market_mode,
                          is_box=verdict.get('is_box', False), height=900, is_weekly=is_weekly,
                          candle_patterns=data.get('candle_patterns', []))

        report_key = f"SINGLE_{data['代碼']}"
        if report_key in st.session_state.ai_reports:
            st.markdown("### 🦄 AI 深度評論")
            st.markdown(f"<div class='ai-box'>{st.session_state.ai_reports[report_key]}</div>", unsafe_allow_html=True)

    else:
        if st.session_state.detected_themes:
            st.markdown("### 🔥 請點擊感興趣的主題：")
            cols = st.columns(len(st.session_state.detected_themes))
            for i, theme in enumerate(st.session_state.detected_themes):
                safe_theme_label = str(theme)
                if cols[i].button(safe_theme_label, use_container_width=True):
                    st.session_state.single_stock_data = None
                    with st.spinner(f"正在挖掘「{safe_theme_label}」供應鏈..."):
                        structure = generate_supply_chain_structure(
                            st.session_state.market_mode, safe_theme_label,
                            client=client, gemini_model=GEMINI_MODEL)
                        if structure and isinstance(structure, dict):
                            st.session_state.supply_chain_data = {"keyword": safe_theme_label, "structure": structure}
                            df = scan_tickers_from_map(st.session_state.market_mode, structure, strategy_mode, timeframe=tf_code)
                            st.session_state.data_cache[st.session_state.market_mode] = df
                            st.session_state.current_source = f"🔥 {safe_theme_label}"
                        else:
                            st.error("AI 正在思考中，請再試一次或換個主題")
        st.divider()

        if st.button("🔙 回到預設清單 (全市場掃描)"):
            st.session_state.supply_chain_data = None
            st.session_state.single_stock_data = None
            st.session_state.detected_themes = []
            st.session_state.view_mode = "list"
            with st.spinner("載入完整資料庫..."):
                default_map = get_default_sector_map_full(st.session_state.market_mode)
                df = scan_tickers_from_map(st.session_state.market_mode, default_map, strategy_mode, timeframe=tf_code)
                st.session_state.data_cache[st.session_state.market_mode] = df
                st.session_state.current_source = "🗂️ 預設清單"

        if st.session_state.supply_chain_data:
            st.markdown(f"## 🗺️ {st.session_state.supply_chain_data['keyword']} 產業供應鏈地圖")
            render_supply_chain_graph(
                st.session_state.supply_chain_data['keyword'],
                st.session_state.supply_chain_data['structure'],
                st.session_state.market_mode)
            st.divider()

        current_df = st.session_state.data_cache.get(st.session_state.market_mode)
        if current_df is not None and not current_df.empty:
            st.subheader(f"{st.session_state.current_source} 數據掃描 ({tf_code})")
            for idx, row in current_df.iterrows():
                ticker = row['代碼']; name = row['名稱']
                with st.expander(f"{ticker} {name} | {row['族群']} | {row['趨勢']}", expanded=(idx==0)):
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
                        st.markdown(f"<div class='ai-box'><strong>🦄 AI 分析：</strong><br>{st.session_state.ai_reports[cache_key]}</div>", unsafe_allow_html=True)
                    else:
                        if st.button(f"🧠 AI 分析 {name}", key=f"btn_{ticker}"):
                            with st.spinner("分析中..."):
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
                st.info("👈 請選擇側邊欄的搜尋方式開始。")
