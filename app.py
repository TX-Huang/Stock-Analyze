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
import yfinance as yf
import requests
import graphviz
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
from data_provider import get_data_provider

# --- Config ---
st.set_page_config(page_title="Alpha Global v93.0", layout="wide", page_icon="📈")
GEMINI_MODEL = 'gemini-3-pro-preview' # 鎖定最穩定模型

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
    st.session_state.chart_settings = {
        "trendline": True, "support": True, "gaps": True, "log_scale": False,
        "zigzag": False, "obv": True, "rectangle": True,
        "patterns": True, "ghost_lines": True,
        "volume_strict": True,
        "rounding": True, "fan": True, "wedge": True, "broadening": True,
        "diamond": True, "bbands": True, "macd": True, "kd": True,
        "structure": True # New Setting
    }

# --- API Key ---
st.sidebar.header("🔑 啟動金鑰")
# Sidebar Navigation
app_mode = st.sidebar.radio("功能模組", ["📈 股市戰情室", "🧬 量化回測系統", "📂 自訂策略實驗室"])

st.sidebar.info("中文搜尋需 API Key。代碼搜尋 (如 2330, NVDA) 可免填。")

# [Security Note]: Defaults are hardcoded for user convenience in local dev.
# In production, use .streamlit/secrets.toml or env vars.
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
# 1. Pure Pandas Indicator Functions
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_bbands(df, length=20, std=2):
    mavg = df['Close'].rolling(window=length).mean()
    mstd = df['Close'].rolling(window=length).std()
    upper = mavg + (mstd * std)
    lower = mavg - (mstd * std)
    return upper, lower

def calculate_stoch(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    return k, d

def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

# ==========================================
# 2. Helper Functions
# ==========================================

def calculate_structural_lines(df, lookback=100):
    """
    計算長期結構線：
    1. 線性回歸通道 (Linear Regression Channel)
    2. 主要支撐/壓力位 (Major S/R Levels)
    """
    structure = {"channel": None, "levels": []}
    if len(df) < 30: return structure

    # 限制回看週期
    data = df.tail(lookback).copy()
    if data.empty: return structure

    # 1. 通道計算 (High+Low)/2
    data['Mid'] = (data['High'] + data['Low']) / 2
    x = np.arange(len(data))

    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, data['Mid'])

        # 計算殘差與標準差
        line = slope * x + intercept
        residuals = data['Mid'] - line
        std_resid = np.std(residuals)

        # 定義通道 (2倍標準差)
        upper_line = line + 2 * std_resid
        lower_line = line - 2 * std_resid

        structure["channel"] = {
            "slope": slope,
            "intercept": intercept,
            "std": std_resid,
            "x_start": data.index[0],
            "x_end": data.index[-1],
            "y_start_mid": line[0],
            "y_end_mid": line[-1],
            "y_start_upper": upper_line[0],
            "y_end_upper": upper_line[-1],
            "y_start_lower": lower_line[0],
            "y_end_lower": lower_line[-1]
        }
    except: pass

    # 2. 主要支撐壓力 (Pivot Clustering)
    n = 20 # 較大的週期找大底/大頂
    data['peaks'] = data.iloc[argrelextrema(data.High.values, np.greater_equal, order=n)[0]]['High']
    data['troughs'] = data.iloc[argrelextrema(data.Low.values, np.less_equal, order=n)[0]]['Low']

    pivots = pd.concat([data['peaks'].dropna(), data['troughs'].dropna()])
    if not pivots.empty:
        pivots = pivots.sort_values()
        # 分群 (2% 誤差內視為同一層級)
        clusters = []
        if len(pivots) > 0:
            current_cluster = [pivots.iloc[0]]
            for p in pivots.iloc[1:]:
                if (p - current_cluster[-1]) / current_cluster[-1] < 0.02:
                    current_cluster.append(p)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [p]
            clusters.append(current_cluster)

        # 過濾有效層級 (至少觸碰 2 次)
        for c in clusters:
            if len(c) >= 2:
                avg_price = np.mean(c)
                strength = len(c)
                structure["levels"].append({"price": avg_price, "strength": strength})

    return structure

def calculate_pattern_convergence(df, peaks, troughs):
    """
    計算型態收斂與結構轉折區間 (2/3 ~ 3/4)
    Apex Convergence & Reversal Zone
    """
    # 確保取得有效的索引
    p_idx = peaks.dropna().index
    t_idx = troughs.dropna().index

    # 至少需要兩個高點與兩個低點來定義趨勢
    if len(p_idx) < 2 or len(t_idx) < 2: return None

    # 取最後兩個主要轉折點
    p1_idx, p2_idx = p_idx[-2], p_idx[-1]
    t1_idx, t2_idx = t_idx[-2], t_idx[-1]

    p1_val, p2_val = peaks[p1_idx], peaks[p2_idx]
    t1_val, t2_val = troughs[t1_idx], troughs[t2_idx]

    # 轉換為數值索引 (0, 1, 2...)
    x_p1, x_p2 = df.index.get_loc(p1_idx), df.index.get_loc(p2_idx)
    x_t1, x_t2 = df.index.get_loc(t1_idx), df.index.get_loc(t2_idx)

    # 避免垂直線 (x相同)
    if x_p2 == x_p1 or x_t2 == x_t1: return None

    # 計算斜率 (m) 與 截距 (c)
    # Highs: Resistance Line
    m_p = (p2_val - p1_val) / (x_p2 - x_p1)
    c_p = p1_val - m_p * x_p1

    # Lows: Support Line
    m_t = (t2_val - t1_val) / (x_t2 - x_t1)
    c_t = t1_val - m_t * x_t1

    # 檢查是否收斂
    # 如果斜率非常接近 (平行)，視為通道而非收斂三角形
    if abs(m_p - m_t) < 1e-4: return None

    # 計算交點 x (Apex)
    # m_p * x + c_p = m_t * x + c_t
    # x * (m_p - m_t) = c_t - c_p
    x_int = (c_t - c_p) / (m_p - m_t)

    # 定義型態起始點 (取四點中最早的)
    x_start = min(x_p1, x_t1)

    # 計算型態總長度
    length = x_int - x_start

    # 交點必須在未來相對於起始點 (且最好不要是反向發散)
    # 若 length < 0 代表交點在過去 (發散型態)，此處主要抓收斂突破
    # 若 user 想要擴散型態，需另外處理。這裡針對 "收斂末端"
    if length <= 10: return None # 太短或發散

    # 計算關鍵區間 (2/3 ~ 3/4)
    x_zone_start = x_start + length * 0.66
    x_zone_end = x_start + length * 0.75

    return {
        "x_int": x_int,
        "y_int": m_p * x_int + c_p,
        "x_start": x_start,
        "x_zone_start": x_zone_start,
        "x_zone_end": x_zone_end,
        "m_p": m_p, "c_p": c_p, # 壓力線參數
        "m_t": m_t, "c_t": c_t  # 支撐線參數
    }

def get_date_from_index(idx, df, is_weekly):
    """Helper to project future dates from index"""
    if idx < 0: idx = 0
    if idx < len(df):
        return df.index[int(idx)]
    else:
        # 推算未來日期
        extra_units = idx - len(df) + 1
        last_date = df.index[-1]
        # 簡單推算：日線+1天(不含週末略估)，週線+7天
        # 為了準確繪圖，使用 pd.Timedelta
        days_per_unit = 7 if is_weekly else 1.4 # 1.4 for business days approx
        delta = timedelta(days=int(extra_units * days_per_unit))
        return last_date + delta

def render_backtest_dashboard(report):
    """
    通用型回測儀表板渲染函數
    """
    equity = getattr(report, 'creturn', None)
    benchmark = getattr(report, 'benchmark', None)
    drawdown = equity / equity.cummax() - 1 if equity is not None else None
    trades = report.get_trades()
    stats = report.get_stats()

    # Core Metrics Calculation
    cagr = stats.get('cagr', 0)
    mdd = stats.get('max_drawdown', 0)
    win_rate = stats.get('win_rate', 0)

    # Risk/Reward Ratio
    avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty else 0
    avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty else 0
    risk_reward = avg_win / avg_loss if avg_loss != 0 else 0

    # Holding Period
    avg_hold_win = trades[trades['return'] > 0]['period'].mean() if not trades.empty else 0
    avg_hold_loss = trades[trades['return'] <= 0]['period'].mean() if not trades.empty else 0

    # Exposure Time
    exposure = (equity != equity.shift(1)).mean() if equity is not None else 0

    tab1, tab2, tab3 = st.tabs(["📊 實戰戰情室 (Metrics)", "📈 資金曲線 (Chart)", "📋 交易明細 (Log)"])

    with tab1:
        st.markdown("### 🏆 核心五大戰略指標")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛡️ MDD", f"{mdd*100:.1f}%", "心理極限")
        c2.metric("⚖️ 勝率/風報", f"{win_rate*100:.0f}% | {risk_reward:.1f}", "獲利引擎")
        c3.metric("📈 CAGR", f"{cagr*100:.1f}%", "複利速度")
        c4.metric("⏳ 持倉 (贏/輸)", f"{avg_hold_win:.0f}/{avg_hold_loss:.0f}天", "資金效率")
        c5.metric("🛡️ 曝險", f"{exposure*100:.0f}%", "避險能力")

    with tab2:
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

    with tab3:
        st.subheader("📋 詳細交易紀錄")
        if not trades.empty:
            trades_display = trades.copy()
            if 'entry_date' in trades_display.columns:
                trades_display['entry_date'] = pd.to_datetime(trades_display['entry_date'])

            # Use all trades as filtered trades since no filter is applied here yet
            trades_filtered = trades_display

            # CSV Download
            # Use the filtered but UN-PAGINATED data for download
            # [FIX]: Indentation corrected to be inside `if not trades.empty:`
            csv = trades_filtered.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載完整交易明細 (.csv)",
                data=csv,
                file_name=f'trade_log_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

            # === Pagination ===
            items_per_page = 1000
            total_items = len(trades_filtered)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            page = st.number_input("頁數 (Page)", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)

            st.info(f"顯示第 {start_idx + 1} 至 {end_idx} 筆交易 (共 {total_items} 筆)")

            # Select relevant columns
            available_cols = ['股票代碼', '進場日期', '出場日期', '進場價', '出場價', '報酬率', '持有天數', '最大不利(MAE)', '最大有利(MFE)']
            cols_to_show = [c for c in available_cols if c in trades_filtered.columns]

            trades_final = trades_filtered[cols_to_show].sort_values("進場日期", ascending=False).iloc[start_idx:end_idx]

            # CSV Download (Page)
            csv = trades_final.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載交易明細 (.csv)",
                data=csv,
                file_name=f'trade_log_{strategy_type}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

            # Formatting style function
            def highlight_ret(val):
                color = ''
                if pd.isna(val): return ''
                if isinstance(val, (int, float)):
                    color = 'color: #22c55e' if val > 0 else 'color: #ef4444'
                return color

            st.dataframe(
                trades_final.style.format({
                    '報酬率': '{:.2%}',
                    '最大不利(MAE)': '{:.2%}',
                    '最大有利(MFE)': '{:.2%}',
                    '進場價': '{:.2f}',
                    '出場價': '{:.2f}'
                }, na_rep="N/A").map(highlight_ret, subset=['報酬率']),
                use_container_width=True,
                height=600
            )
        else:
            st.info("無交易紀錄")

def custom_metric(label, value, delta=None):
    delta_str = ""
    if delta:
        delta_str = f" {delta}"
    st.markdown(f"**{label}**: {value} {delta_str}")

def robust_json_extract(text):
    try: return json.loads(text)
    except: pass
    try:
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match: return json.loads(match.group(1))
    except: pass
    return None

def validate_ticker(ticker, market):
    ticker = str(ticker).strip().upper()
    if "台股" in market:
        if re.match(r'^\d{4,6}$', ticker): return True
        if ticker.endswith(".TW") or ticker.endswith(".TWO"): return True
        return False
    else:
        if re.match(r'^[A-Z]{1,6}$', ticker): return True
        return False

def get_default_sector_map_full(market):
    if "台股" in market:
        return {
            "💾 記憶體": ["2408", "2344", "2337", "8299", "3260"],
            "🤖 AI 伺服器": ["2317", "2382", "3231", "2356", "6669"],
            "❄️ 散熱模組": ["3017", "3324", "2421", "3013"],
            "🚢 航運": ["2603", "2609", "2615", "2606"],
            "💎 權值股": ["2330", "2454", "3035", "3443"]
        }
    else:
        return {"👑 Mag 7": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]}

def get_fallback_supply_chain(keyword, market):
    k = keyword.lower()
    if "台股" in market:
        if "記憶體" in k or "dram" in k:
            return {"IC設計": {"3006":"晶豪科", "8299":"群聯"}, "製造": {"2408":"南亞科", "2344":"華邦電"}, "封測": {"6239":"力成", "8150":"南茂"}}
        if "機器人" in k or "robot" in k:
            return {"關鍵零組件": {"2049":"上銀", "1590":"亞德客"}, "系統整合": {"2317":"鴻海", "2357":"華碩"}}
    return None

# ==========================================
# 3. AI Core
# ==========================================
def resolve_ticker_and_market(query):
    query = str(query).strip()
    if re.match(r'^\d{4,6}$', query): return query, "🇹🇼 台股 (TW)", query
    if re.match(r'^[A-Z]{1,5}$', query.upper()): return query.upper(), "🗽 美股 (US)", query.upper()

    if not client: return None, None, None
    prompt = f"將'{query}'轉為股票代碼。回傳JSON:{{'market':'TW'或'US', 'ticker':'代碼', 'name':'中文名'}}。台股代碼僅數字。"
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        data = robust_json_extract(res.text)
        if data and 'market' in data: return data['ticker'], "🇹🇼 台股 (TW)" if data['market'] == "TW" else "🗽 美股 (US)", data.get('name', query)
        return None, None, None
    except Exception as e:
        st.sidebar.error(f"AI 翻譯失敗: {e}")
        return None, None, None

def analyze_signals(df):
    if df.empty or len(df) < 30: return "資料不足"
    signals = []

    price_slope = (df['Close'].iloc[-1] - df['Close'].iloc[-10])
    rsi_slope = (df['RSI'].iloc[-1] - df['RSI'].iloc[-10])
    if price_slope > 0 and rsi_slope < 0: signals.append("⚠️ 頂背離")
    elif price_slope < 0 and rsi_slope > 0: signals.append("✨ 底背離")

    k, d = df['K'].iloc[-1], df['D'].iloc[-1]
    prev_k = df['K'].iloc[-2]
    if prev_k < df['D'].iloc[-2] and k > d and k < 80: signals.append("⚡ KD 金叉")
    elif prev_k > df['D'].iloc[-2] and k < d and k > 20: signals.append("💀 KD 死叉")

    macd = df['MACD'].iloc[-1]
    if df['MACD'].iloc[-2] < 0 and macd > 0: signals.append("🔥 MACD 翻紅")
    elif df['MACD'].iloc[-2] > 0 and macd < 0: signals.append("❄️ MACD 翻綠")

    return " | ".join(signals) if signals else "無明顯訊號"

def detect_hot_themes(market):
    if not client: return []
    q = "今日台股熱門族群" if "台股" in market else "Top US sectors today"
    prompt = f"搜'{q}'，歸納3~5個主題，回傳List JSON (純文字列表)。"
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return robust_json_extract(res.text) or []
    except: return []

def generate_supply_chain_structure(market, keyword):
    if not client: return None
    prompt = f"拆解'{keyword}'產業鏈，回傳JSON: {{'部位': {{'代碼': '中文名'}}}}"
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return robust_json_extract(res.text)
    except: return None

def generate_ai_analysis(market, ticker, name, price, change, sector, technicals, strategy, extra_data="", timeframe="1d", signal_context=""):
    if not client: return "請先輸入 API Key。"
    desc = "週線(Weekly)" if timeframe == "1wk" else "日線(Daily)"
    prompt = f"""
    角色：全方位技術分析大師。標的：{market} {ticker} {name}。
    分析週期：{desc}。數據：{price} ({change}%) | {technicals}
    **重點訊號：{signal_context}**
    {extra_data}

    請進行分析 (Markdown)：
    1. 🔍 訊號判讀
    2. 📐 形態與趨勢
    3. 🛡️ 實戰指令 ({strategy})
    """
    try:
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return res.text
    except Exception as e: return f"分析失敗: {e}"

# ==========================================
# 4. Logic Engines
# ==========================================
def detect_candlestick_patterns(df):
    """
    User requests:
    - 晨星 (Morning Star) / 暮星 (Evening Star)
    - 紅三兵 (Three White Soldiers) / 黑三鴉 (Three Black Crows)
    - 吞噬 (Engulfing) / 烏雲罩頂 (Dark Cloud Cover) / 貫穿線 (Piercing Pattern)
    - 孕線 (Harami)
    - 實體K線 (Marubozu) / 十字星 (Doji)
    - 錘子/吊人 (Hammer / Hanging Man) / 倒錘/流星 (Inverted Hammer / Shooting Star)

    14-day average baselines, 10-day local extremes, priority scanning.
    """
    patterns = []
    if len(df) < 15:
        return patterns

    # Define baseline metrics
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(14).mean()

    # 10-day local extremes for trend detection
    df['LocalHigh'] = df['High'].rolling(10).max()
    df['LocalLow'] = df['Low'].rolling(10).min()

    # Helper functions
    def is_bullish(row): return row['Close'] > row['Open']
    def is_bearish(row): return row['Close'] < row['Open']
    def body_size(row): return abs(row['Close'] - row['Open'])
    def is_doji(row): return body_size(row) <= row['AvgBody'] * 0.1
    def is_marubozu(row): return body_size(row) >= row['AvgBody'] * 1.5 and (row['High'] - max(row['Open'], row['Close'])) < row['AvgBody'] * 0.1 and (min(row['Open'], row['Close']) - row['Low']) < row['AvgBody'] * 0.1
    def upper_shadow(row): return row['High'] - max(row['Open'], row['Close'])
    def lower_shadow(row): return min(row['Open'], row['Close']) - row['Low']
    def at_support(row): return row['Low'] <= row['LocalLow'] * 1.02
    def at_resistance(row): return row['High'] >= row['LocalHigh'] * 0.98

    # Scan the entire history (or last 60 days to avoid clutter)
    scan_df = df.tail(60)
    if len(scan_df) < 3: return patterns

    for i in range(2, len(scan_df)):
        idx0 = scan_df.index[i-2]
        idx1 = scan_df.index[i-1]
        idx2 = scan_df.index[i]

        row0 = scan_df.loc[idx0]
        row1 = scan_df.loc[idx1]
        row2 = scan_df.loc[idx2]

        found = False

        # Priority 1: Multi-Candle Patterns
        if is_bearish(row0) and body_size(row0) > row0['AvgBody'] and \
           is_bullish(row2) and body_size(row2) > row2['AvgBody'] and \
           row2['Close'] > (row0['Open'] + row0['Close']) / 2 and \
           body_size(row1) <= row1['AvgBody'] * 0.5 and \
           max(row1['Open'], row1['Close']) < row0['Close'] and \
           at_support(row1):
            patterns.append({"name": "🌅 晨星 (Morning Star)", "date": idx2, "type": "Bullish", "points": [idx0, idx1, idx2]})
            found = True

        elif is_bullish(row0) and body_size(row0) > row0['AvgBody'] and \
             is_bearish(row2) and body_size(row2) > row2['AvgBody'] and \
             row2['Close'] < (row0['Open'] + row0['Close']) / 2 and \
             body_size(row1) <= row1['AvgBody'] * 0.5 and \
             min(row1['Open'], row1['Close']) > row0['Close'] and \
             at_resistance(row1):
            patterns.append({"name": "🌃 暮星 (Evening Star)", "date": idx2, "type": "Bearish", "points": [idx0, idx1, idx2]})
            found = True

        elif is_bullish(row0) and is_bullish(row1) and is_bullish(row2) and \
             row1['Close'] > row0['Close'] and row2['Close'] > row1['Close'] and \
             row1['Open'] > row0['Open'] and row2['Open'] > row1['Open'] and \
             body_size(row0) > row0['AvgBody'] and body_size(row1) > row1['AvgBody'] and body_size(row2) > row2['AvgBody']:
             patterns.append({"name": "🚀 紅三兵", "date": idx2, "type": "Bullish", "points": [idx0, idx1, idx2]})
             found = True

        elif is_bearish(row0) and is_bearish(row1) and is_bearish(row2) and \
             row1['Close'] < row0['Close'] and row2['Close'] < row1['Close'] and \
             row1['Open'] < row0['Open'] and row2['Open'] < row1['Open'] and \
             body_size(row0) > row0['AvgBody'] and body_size(row1) > row1['AvgBody'] and body_size(row2) > row2['AvgBody']:
             patterns.append({"name": "🦅 黑三鴉", "date": idx2, "type": "Bearish", "points": [idx0, idx1, idx2]})
             found = True

        # Priority 2: Two-Candle Patterns
        if not found:
            if is_bearish(row1) and is_bullish(row2) and \
               row2['Open'] < row1['Close'] and row2['Close'] > row1['Open'] and \
               at_support(row1):
                patterns.append({"name": "🟢 多頭吞噬", "date": idx2, "type": "Bullish", "points": [idx1, idx2]})
                found = True

            elif is_bullish(row1) and is_bearish(row2) and \
                 row2['Open'] > row1['Close'] and row2['Close'] < row1['Open'] and \
                 at_resistance(row1):
                patterns.append({"name": "🔴 空頭吞噬", "date": idx2, "type": "Bearish", "points": [idx1, idx2]})
                found = True

            elif is_bullish(row1) and body_size(row1) > row1['AvgBody'] and \
                 is_bearish(row2) and body_size(row2) > row2['AvgBody'] and \
                 row2['Open'] > row1['High'] and \
                 row2['Close'] < (row1['Open'] + row1['Close']) / 2 and \
                 row2['Close'] > row1['Open'] and \
                 at_resistance(row1):
                patterns.append({"name": "🌧️ 烏雲罩頂", "date": idx2, "type": "Bearish", "points": [idx1, idx2]})
                found = True

            elif is_bearish(row1) and body_size(row1) > row1['AvgBody'] and \
                 is_bullish(row2) and body_size(row2) > row2['AvgBody'] and \
                 row2['Open'] < row1['Low'] and \
                 row2['Close'] > (row1['Open'] + row1['Close']) / 2 and \
                 row2['Close'] < row1['Open'] and \
                 at_support(row1):
                patterns.append({"name": "🗡️ 貫穿線", "date": idx2, "type": "Bullish", "points": [idx1, idx2]})
                found = True

            elif is_bearish(row1) and body_size(row1) > row1['AvgBody'] and \
                 is_bullish(row2) and body_size(row2) < row2['AvgBody'] and \
                 row2['Open'] > row1['Close'] and row2['Close'] < row1['Open'] and \
                 at_support(row1):
                 patterns.append({"name": "🤰 多頭孕線", "date": idx2, "type": "Bullish", "points": [idx1, idx2]})
                 found = True

            elif is_bullish(row1) and body_size(row1) > row1['AvgBody'] and \
                 is_bearish(row2) and body_size(row2) < row2['AvgBody'] and \
                 row2['Open'] < row1['Close'] and row2['Close'] > row1['Open'] and \
                 at_resistance(row1):
                 patterns.append({"name": "🤰 空頭孕線", "date": idx2, "type": "Bearish", "points": [idx1, idx2]})
                 found = True

        # Priority 3: Single-Candle Patterns (Only mark if significant)
        if not found:
            if is_marubozu(row2) and (at_support(row2) or at_resistance(row2)):
                if is_bullish(row2):
                    patterns.append({"name": "🟩 大長紅", "date": idx2, "type": "Bullish", "points": [idx2]})
                else:
                    patterns.append({"name": "🟥 大長黑", "date": idx2, "type": "Bearish", "points": [idx2]})
            elif is_doji(row2) and (at_support(row2) or at_resistance(row2)):
                patterns.append({"name": "➕ 十字星", "date": idx2, "type": "Neutral", "points": [idx2]})
            else:
                if lower_shadow(row2) >= 2 * body_size(row2) and upper_shadow(row2) <= body_size(row2) * 0.2:
                    if at_support(row2):
                        patterns.append({"name": "🔨 錘子", "date": idx2, "type": "Bullish", "points": [idx2]})
                    elif at_resistance(row2):
                        patterns.append({"name": "🪝 吊人", "date": idx2, "type": "Bearish", "points": [idx2]})
                elif upper_shadow(row2) >= 2 * body_size(row2) and lower_shadow(row2) <= body_size(row2) * 0.2:
                    if at_support(row2):
                        patterns.append({"name": "🏹 倒錘", "date": idx2, "type": "Bullish", "points": [idx2]})
                    elif at_resistance(row2):
                        patterns.append({"name": "🌠 流星", "date": idx2, "type": "Bearish", "points": [idx2]})

    return patterns

def detect_complex_patterns(df, peaks, troughs):
    patterns = []
    if df.empty or len(peaks) < 2: return patterns

    if len(peaks) >= 3 and len(troughs) >= 3:
        p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
        t1, t2, t3 = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
        if (p2 > p1 and p2 > p3) and (t2 < t1 and t2 < t3):
            patterns.append({"name": "💎 鑽石頂", "points": [peaks.index[-3], peaks.index[-1]], "type": "Bearish", "is_broadening": False})

    if len(peaks) >= 2 and len(troughs) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]
        t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]
        if p2 > p1 and t2 < t1:
            patterns.append({
                "name": "🎺 擴散", "points": [p1_idx, p2_idx], "type": "Volatility", "is_broadening": True,
                "p_coords": [(p1_idx, p1), (p2_idx, p2)], "t_coords": [(t1_idx, t1), (t2_idx, t2)]
            })

    if len(peaks) >= 3:
        p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
        p1_idx, p2_idx, p3_idx = peaks.index[-3], peaks.index[-2], peaks.index[-1]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.15:
            patterns.append({"name": "頭肩頂", "points": [p1_idx, p2_idx, p3_idx], "type": "Bearish", "is_broadening": False})

    if len(peaks) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]
        if abs(p1 - p2) / p1 < 0.03 and (p2_idx - p1_idx).days > 10:
            patterns.append({"name": "M頭", "points": [p1_idx, p2_idx], "type": "Bearish", "is_broadening": False})

    if len(troughs) >= 2:
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]
        if abs(t1 - t2) / t1 < 0.03 and (t2_idx - t1_idx).days > 10:
            patterns.append({"name": "W底", "points": [t1_idx, t2_idx], "type": "Bullish", "is_broadening": False})

    return patterns

def calculate_trend_logic(df, n=10, is_weekly=False):
    verdict = {"trend": "盤整/不明", "signal": "觀望", "color": "gray", "details": [], "is_box": False}
    if df.empty: return verdict

    df['peaks'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
    df['troughs'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']
    peaks, troughs = df['peaks'].dropna(), df['troughs'].dropna()

    volatility = df['Close'].rolling(5).std() / df['Close']
    if volatility.iloc[-1] < 0.005:
        verdict["trend"] = "🌀 線圈狀態"; verdict["color"] = "orange"; verdict["details"].append("波動率極度壓縮")

    recent = df.tail(40)
    r_max, r_min = recent['High'].max(), recent['Low'].min()
    if (r_max - r_min) / r_min < 0.10:
        verdict["trend"] = "📦 矩形整理"; verdict["color"] = "blue"; verdict["is_box"] = True
        if df['Close'].iloc[-1] > r_max * 1.01: verdict["signal"] = "🚀 箱型突破"; verdict["color"] = "green"
        return verdict

    if len(peaks) >= 2 and len(troughs) >= 2:
        p_last, p_prev = peaks.iloc[-1], peaks.iloc[-2]
        t_last, t_prev = troughs.iloc[-1], troughs.iloc[-2]

        x_p1, x_p2 = df.index.get_loc(peaks.index[-2]), df.index.get_loc(peaks.index[-1])
        x_t1, x_t2 = df.index.get_loc(troughs.index[-2]), df.index.get_loc(troughs.index[-1])

        if x_p2 != x_p1 and x_t2 != x_t1:
            m_peak = (p_last - p_prev) / (x_p2 - x_p1)
            m_trough = (t_last - t_prev) / (x_t2 - x_t1)

            if p_last > p_prev and t_last > t_prev:
                if m_trough > m_peak * 1.2: verdict["trend"] = "⚠️ 上升楔形"; verdict["color"] = "green"
                else: verdict["trend"] = "🔴 多頭趨勢"; verdict["color"] = "red"
            elif p_last < p_prev and t_last < t_prev:
                if m_peak < m_trough * 1.2: verdict["trend"] = "✨ 下降楔形"; verdict["color"] = "red"
                else: verdict["trend"] = "🟢 空頭趨勢"; verdict["color"] = "green"
            elif p_last < p_prev and t_last > t_prev: verdict["trend"] = "📐 收斂整理"; verdict["color"] = "orange"
            elif p_last > p_prev and t_last < t_prev: verdict["trend"] = "🎺 擴散型態"; verdict["color"] = "orange"

    return verdict

# ==========================================
# 5. Data Scanners (Robust + Fixed Columns)
# ==========================================
def scan_single_stock_deep(market, ticker, strategy, timeframe="1d", user_query_name=""):
    if timeframe == "1wk": interval = "1wk"; period = "5y"; is_weekly = True
    else: interval = "1d"; period = "2y"; is_weekly = False

    market_type = "TW" if "台股" in market else "US"
    if market_type == "TW": ma_short, ma_long = 5, 20
    else: ma_short, ma_long = 20, 50

    # Use Data Provider layer instead of raw yf
    provider = get_data_provider("yfinance", market_type=market_type)
    df = provider.get_historical_data(ticker, period=period, interval=interval)

    if df.empty or len(df) <= 30: return None
    try:
        info_data = provider.get_stock_info(ticker)
        final_full_t = info_data.get('raw_ticker', ticker)

        close = df['Close'].iloc[-1]; chg = ((close - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        vol_curr = df['Volume'].iloc[-1]; vol_avg = df['Volume'].iloc[:-5].mean()
        r_vol = vol_curr / vol_avg if vol_avg > 0 else 0
        ma_s = df['Close'].rolling(ma_short).mean().iloc[-1]
        ma_l = df['Close'].rolling(ma_long).mean().iloc[-1]

        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df)
        df['BB_Upper'], df['BB_Lower'] = calculate_bbands(df)
        df['K'], df['D'] = calculate_stoch(df)
        df['OBV'] = calculate_obv(df)

        signal_str = analyze_signals(df)

        name = st.session_state.dynamic_name_map.get(ticker, ticker)
        if info_data['name'] != ticker:
            name = info_data['name']
        pe = info_data.get('pe', 'N/A')
        eps = info_data.get('eps', 'N/A')
        div_yield = info_data.get('yield', 'N/A')

        if name == ticker and user_query_name: name = user_query_name

        verdict = calculate_trend_logic(df, is_weekly=is_weekly)
        patterns = detect_complex_patterns(df, df['peaks'].dropna(), df['troughs'].dropna())
        candle_patterns = detect_candlestick_patterns(df)

        # [Fix]: Define extra safely
        extra = f"趨勢: {verdict.get('trend')}。{verdict.get('signal')}。"
        if patterns: extra += f" 型態: {', '.join([p['name'] for p in patterns])}。"
        if candle_patterns: extra += f" K線: {', '.join([p['name'] for p in candle_patterns])}。"

        tech_summary = f"RSI:{round(df['RSI'].iloc[-1],1)} MACD:{'多' if df['MACD'].iloc[-1]>0 else '空'}"

        return {
            "代碼": ticker, "名稱": name, "全名": final_full_t,
            "現價": round(close, 2), "漲跌幅%": round(chg, 2), "爆量倍數": round(r_vol, 1),
            "短均": round(ma_s, 1), "長均": round(ma_l, 1), "RSI": round(df['RSI'].iloc[-1], 1),
            "PE": pe, "EPS": eps, "Yield": div_yield,
            "df": df, "verdict": verdict, "extra_info": extra + " " + tech_summary,
            "patterns": patterns, "candle_patterns": candle_patterns, "signal_context": signal_str
        }
    except Exception:
        return None

def scan_tickers_from_map(market, sector_map, strategy, timeframe="1d"):
    if timeframe == "1wk": interval = "1wk"; period = "5y"; is_weekly = True
    else: interval = "1d"; period = "2y"; is_weekly = False

    market_type = "TW" if "台股" in market else "US"
    if market_type == "TW": ma_short, ma_long = 5, 20
    else: ma_short, ma_long = 20, 50

    provider = get_data_provider("yfinance", market_type=market_type)

    all_data = []
    unique_tickers = []
    ticker_to_sector = {}

    for sec, tickers in sector_map.items():
        if isinstance(tickers, dict): st.session_state.dynamic_name_map.update(tickers); ticker_list = list(tickers.keys())
        else: ticker_list = tickers
        for t in ticker_list:
            if validate_ticker(t, market) and t not in unique_tickers: unique_tickers.append(t); ticker_to_sector[t] = sec

    progress = st.progress(0); st_text = st.empty()
    for i, t in enumerate(unique_tickers):
        st_text.text(f"掃描中: {t}...");
        if (i+1) <= len(unique_tickers): progress.progress((i+1)/len(unique_tickers))

        df = provider.get_historical_data(t, period=period, interval=interval)
        if df.empty or len(df) <= 30: continue

        info_data = provider.get_stock_info(t)
        final_t = info_data.get('raw_ticker', t)

        try:
            close = df['Close'].iloc[-1]; chg = ((close - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
            ma_s = df['Close'].rolling(ma_short).mean().iloc[-1]; ma_l = df['Close'].rolling(ma_long).mean().iloc[-1]

            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], _, _ = calculate_macd(df)
            df['K'], df['D'] = calculate_stoch(df)
            df['BB_Upper'], df['BB_Lower'] = calculate_bbands(df)

            verdict = calculate_trend_logic(df, is_weekly=is_weekly)
            clean = t.replace(".TW","").replace(".TWO","")
            patterns = detect_complex_patterns(df, df['peaks'].dropna(), df['troughs'].dropna())
            candle_patterns = detect_candlestick_patterns(df)
            signal_str = analyze_signals(df)

            all_data.append({
                "代碼": clean, "名稱": st.session_state.dynamic_name_map.get(clean, clean), "族群": ticker_to_sector.get(t, "其他"),
                "現價": round(close, 2), "漲跌幅%": round(chg, 2), "爆量倍數": 0, "趨勢": verdict['trend'].split(" ")[0],
                "短均": round(ma_s, 1), "長均": round(ma_l, 1), "RSI": round(df['RSI'].iloc[-1], 1), "t_color": "#f3f4f6", "t_border": "#9ca3af", "raw_ticker": final_t, "df": df,
                "patterns": patterns, "candle_patterns": candle_patterns, "verdict": verdict, "signal_context": signal_str
            })
        except: continue
    progress.empty(); st_text.empty()
    return pd.DataFrame(all_data)

# ==========================================
# 6. Visualization & Main UI
# ==========================================
def render_supply_chain_graph(keyword, structure, market):
    if not structure: return
    try:
        dot = graphviz.Digraph(comment=keyword)
        dot.attr(rankdir='LR')
        dot.attr('node', fontname='Noto Sans CJK TC')
        dot.node('ROOT', keyword, shape='doubleoctagon', style='filled', fillcolor='#f3f4f6', fontcolor='#111827', fontsize='16')
        for part, tickers in structure.items():
            part_id = f"PART_{part}"
            dot.node(part_id, part, shape='box', style='filled', fillcolor='#dbeafe', fontcolor='#1e40af')
            dot.edge('ROOT', part_id)
            ticker_iter = tickers.items() if isinstance(tickers, dict) else [(t, t) for t in tickers]
            for t, t_name in ticker_iter:
                if not validate_ticker(t, market): continue
                t_clean = t.replace(".TW","").replace(".TWO","")
                name = t_name if t_name != t else st.session_state.dynamic_name_map.get(t_clean, t_clean)
                stock_label = f"{name}\n({t_clean})"
                stock_id = f"STOCK_{t_clean}"
                dot.node(stock_id, stock_label, shape='ellipse', style='filled', fillcolor='#f9fafb', fontcolor='#374151')
                dot.edge(part_id, stock_id)
        st.graphviz_chart(dot)
    except Exception as e:
        st.warning("⚠️ 無法繪製供應鏈圖 (可能是電腦未安裝 Graphviz 軟體)，改為顯示文字清單：")
        st.write(structure)

def render_trend_chart(df, patterns, market, is_box=False, height=600, is_weekly=False, candle_patterns=None):
    try:
        rows = 2

        if "台股" in market: ma_s='MA5'; ma_l='MA20'; s_win=5; l_win=20
        else: ma_s='MA20'; ma_l='MA50'; s_win=20; l_win=50
        df[ma_s] = df['Close'].rolling(s_win).mean()
        df[ma_l] = df['Close'].rolling(l_win).mean()

        n = 10
        df['peaks'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
        df['troughs'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2], subplot_titles=("價格與壓力/支撐", "成交量"))

        # Taiwan standard: Red for Up (increasing), Green for Down (decreasing)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'), row=1, col=1)

        if st.session_state.chart_settings.get('ma', True):
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_s], line=dict(color='orange', width=1), name=f'{ma_s}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_l], line=dict(color='blue', width=1), name=f'{ma_l}'), row=1, col=1)

        if st.session_state.chart_settings.get('bbands', False) and 'BB_Upper' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB上軌'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)', name='BB下軌'), row=1, col=1)

        y_min = df['Low'].min() * 0.95; y_max = df['High'].max() * 1.05
        fig.update_yaxes(range=[y_min, y_max], fixedrange=False, row=1, col=1)

        # Draw Support and Resistance lines (based on peaks/troughs logic)
        if st.session_state.chart_settings.get('trendline', True):
            peaks, troughs = df['peaks'].dropna(), df['troughs'].dropna()

            # Draw Convergence & Reversal Zone
            zone_data = calculate_pattern_convergence(df, peaks, troughs)
            if zone_data:
                z_start_date = get_date_from_index(zone_data['x_zone_start'], df, is_weekly)
                z_end_date = get_date_from_index(zone_data['x_zone_end'], df, is_weekly)
                apex_date = get_date_from_index(zone_data['x_int'], df, is_weekly)

                # Render Reversal Zone Rectangle
                fig.add_vrect(
                    x0=z_start_date, x1=z_end_date,
                    fillcolor="rgba(255, 165, 0, 0.2)", layer="below", line_width=0,
                    annotation_text="轉折熱區", annotation_position="top left"
                )

                # Render Apex Point Marker
                if zone_data['x_int'] > len(df) * 0.5: # 確保交點合理
                    fig.add_trace(go.Scatter(
                        x=[apex_date], y=[zone_data['y_int']],
                        mode='markers', marker=dict(color="purple", size=8, symbol="star"),
                        name='預期收斂點 (Apex)'
                    ), row=1, col=1)

            if len(peaks) >= 2:
                p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]; p1_val, p2_val = peaks.iloc[-2], peaks.iloc[-1]
                fig.add_trace(go.Scatter(x=[p1_idx, p2_idx], y=[p1_val, p2_val], mode='lines', line=dict(color="Green", width=1.5, dash="dash"), name='壓力線'), row=1, col=1)
                # Extension to current date
                x1, x2 = df.index.get_loc(p1_idx), df.index.get_loc(p2_idx)
                if x2 != x1:
                    slope = (p2_val - p1_val) / (x2 - x1)
                    # Use get_date_from_index if extending into future based on zone_data, else just current edge
                    end_idx_for_line = max(int(zone_data['x_int']) + 2 if zone_data else len(df)-1, len(df)-1)
                    end_date = get_date_from_index(end_idx_for_line, df, is_weekly)
                    proj = p2_val + slope * (end_idx_for_line - x2)
                    fig.add_trace(go.Scatter(x=[p2_idx, end_date], y=[p2_val, proj], mode='lines', line=dict(color="Green", width=1, dash="dot"), name='壓力線延伸'), row=1, col=1)

            if len(troughs) >= 2:
                t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]; t1_val, t2_val = troughs.iloc[-2], troughs.iloc[-1]
                fig.add_trace(go.Scatter(x=[t1_idx, t2_idx], y=[t1_val, t2_val], mode='lines', line=dict(color="Red", width=1.5, dash="dash"), name='支撐線'), row=1, col=1)
                # Extension to current date
                x1, x2 = df.index.get_loc(t1_idx), df.index.get_loc(t2_idx)
                if x2 != x1:
                    slope = (t2_val - t1_val) / (x2 - x1)
                    end_idx_for_line = max(int(zone_data['x_int']) + 2 if zone_data else len(df)-1, len(df)-1)
                    end_date = get_date_from_index(end_idx_for_line, df, is_weekly)
                    proj = t2_val + slope * (end_idx_for_line - x2)
                    fig.add_trace(go.Scatter(x=[t2_idx, end_date], y=[t2_val, proj], mode='lines', line=dict(color="Red", width=1, dash="dot"), name='支撐線延伸'), row=1, col=1)

        # [New] Implement Gap Visualization (跳空)
        if st.session_state.chart_settings.get('gaps', True):
            # Gap up: Today's Low > Yesterday's High
            # Gap down: Today's High < Yesterday's Low
            df['Prev_High'] = df['High'].shift(1)
            df['Prev_Low'] = df['Low'].shift(1)

            # Find gaps (using slightly more than 0% to filter noise, e.g. 0.5% gap)
            for i in range(1, len(df)):
                curr_low = df['Low'].iloc[i]
                prev_high = df['Prev_High'].iloc[i]
                curr_high = df['High'].iloc[i]
                prev_low = df['Prev_Low'].iloc[i]

                # Draw Up Gap (Red for Bullish in TW)
                if curr_low > prev_high * 1.005:
                    fig.add_shape(type="rect", x0=df.index[i-1], x1=df.index[i], y0=prev_high, y1=curr_low,
                                  line=dict(width=0), fillcolor="rgba(239, 68, 68, 0.3)", row=1, col=1)
                    # Extend Support Line
                    fig.add_trace(go.Scatter(x=[df.index[i], df.index[-1]], y=[prev_high, prev_high], mode='lines', line=dict(color="rgba(239, 68, 68, 0.5)", width=1, dash="dot"), showlegend=False), row=1, col=1)

                # Draw Down Gap (Green for Bearish in TW)
                if curr_high < prev_low * 0.995:
                    fig.add_shape(type="rect", x0=df.index[i-1], x1=df.index[i], y0=curr_high, y1=prev_low,
                                  line=dict(width=0), fillcolor="rgba(34, 197, 94, 0.3)", row=1, col=1)
                    # Extend Resistance Line
                    fig.add_trace(go.Scatter(x=[df.index[i], df.index[-1]], y=[prev_low, prev_low], mode='lines', line=dict(color="rgba(34, 197, 94, 0.5)", width=1, dash="dot"), showlegend=False), row=1, col=1)

        # Candle Patterns Annotations
        if st.session_state.chart_settings.get('candle_patterns', True) and candle_patterns:
            for p in candle_patterns:
                date = p['date']
                y_val = df.loc[date, 'High'] * 1.02 if p['type'] == 'Bearish' else df.loc[date, 'Low'] * 0.98
                # Font color matches TW standard (Red for Bullish, Green for Bearish)
                fig.add_annotation(
                    x=date, y=y_val,
                    text=p['name'],
                    showarrow=False,
                    font=dict(color="green" if p['type'] == 'Bearish' else "red", size=10),
                    row=1, col=1
                )

        # Volume
        colors = ['#ef4444' if row['Close'] >= row['Open'] else '#22c55e' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"繪圖錯誤: {e}")

# ==========================================
# 7. Main UI (Fixed Verdict Box)
# ==========================================
st.title(f"💎 Alpha Global v93.0 (UI Perfection)")

with st.sidebar:
    # Only show stock analysis filters if in Dashboard mode
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

    # Input Area
    with st.expander("🛠️ 策略設定 (Strategy Settings)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            # [Security Note]: Defaults are hardcoded for user convenience in local dev.
            try:
                default_finlab = st.secrets.get("FINLAB_API_TOKEN", "")
            except:
                default_finlab = ""

            finlab_token = st.text_input("Finlab API Token", type="password", value=default_finlab, help="請輸入您的 Finlab API 金鑰")
        with col2:
            strategy_type = st.selectbox("選擇回測策略", ["純做多策略 (Long Only)", "多空策略 (Long + Short)", "VCP 波動收縮策略 (Minervini)", "Isaac 頂級多因子策略 (Growth + Reversion)"])

        run_btn = st.button("🔬 執行回測 (Run Backtest)", use_container_width=True, type="primary")

    # Session State Logic for Report Persistence
    if 'backtest_report' not in st.session_state:
        st.session_state.backtest_report = None
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = None

    if run_btn:
        if not finlab_token:
            st.error("請輸入 Finlab API Token")
        else:
            with st.spinner(f"正在執行 {strategy_type} 回測... (這可能需要幾分鐘)"):
                try:
                    # Import dynamically to avoid top-level dependency if not used
                    if "純做多" in strategy_type:
                        import strategy_long
                        report = strategy_long.run_long_strategy(finlab_token)
                    elif "VCP" in strategy_type:
                        import strategy_vcp
                        report = strategy_vcp.run_vcp_strategy(finlab_token)
                    elif "Isaac" in strategy_type:
                        import strategy_isaac
                        report = strategy_isaac.run_isaac_strategy(finlab_token)
                    else:
                        import strategy_long_short
                        report = strategy_long_short.run_long_short_strategy(finlab_token)

                    st.success("回測完成！")

                    # Store in Session State
                    st.session_state.backtest_report = report
                    st.session_state.current_strategy = strategy_type

                except Exception as e:
                    st.error(f"回測執行發生錯誤: {e}")

    # Render Report from Session State
    if st.session_state.backtest_report is not None:
        report = st.session_state.backtest_report

        # Clear previous report if strategy changed (Optional, but safer to keep current)
        # We rely on user clicking Run again to update.

        # === Data Preparation ===
        equity = getattr(report, 'creturn', None)
        benchmark = getattr(report, 'benchmark', None)
        drawdown = equity / equity.cummax() - 1 if equity is not None else None
        trades = report.get_trades()
        stats = report.get_stats()

        # Core Metrics Calculation
        cagr = stats.get('cagr', 0)
        mdd = stats.get('max_drawdown', 0)
        win_rate = stats.get('win_rate', 0)

        # Risk/Reward Ratio
        avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty else 0
        avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty else 0
        risk_reward = avg_win / avg_loss if avg_loss != 0 else 0

        # Holding Period
        avg_hold_win = trades[trades['return'] > 0]['period'].mean() if not trades.empty else 0
        avg_hold_loss = trades[trades['return'] <= 0]['period'].mean() if not trades.empty else 0

        # Exposure Time
        exposure = (equity != equity.shift(1)).mean() if equity is not None else 0

        # === Tab Layout ===
        tab1, tab2, tab3 = st.tabs(["📊 實戰戰情室 (Core Metrics)", "🛡️ 參數強健性 (Stress Test)", "📋 交易明細 (Trades)"])

        with tab1:
            # Big 5 Core Metrics Display
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

            # Charts
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
                    import strategy_isaac
                    results = []
                    sl_steps = range(stop_loss_range[0], stop_loss_range[1]+1, 2)
                    tp_steps = range(take_profit_range[0], take_profit_range[1]+1, 5)

                    progress_bar = st.progress(0)
                    total_steps = len(sl_steps) * len(tp_steps)
                    step_count = 0

                    for sl in sl_steps:
                        for tp in tp_steps:
                            # Run strategy with overrides (We need to modify strategy to accept args)
                            # For now, we simulate or pass args if implemented.
                            # Let's assume strategy_isaac.run_isaac_strategy accepts kwargs
                            try:
                                # Note: We need to update strategy_isaac.py to accept sl/tp
                                rep = strategy_isaac.run_isaac_strategy(finlab_token, stop_loss=sl/100, take_profit=tp/100)
                                stats_grid = rep.get_stats()
                                results.append({
                                    'Stop Loss': sl,
                                    'Take Profit': tp,
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
                                           color_continuous_scale='RdYlGn',
                                           text_auto='.1%')
                        st.plotly_chart(fig_heat, use_container_width=True)
                        st.success("壓力測試完成！請尋找紅色的高原區 (Plateau)。")

        with tab3:
            st.subheader("📋 詳細交易紀錄 (Trade Log)")
            if not trades.empty:
                rename_map = {
                    "stock_id": "股票代碼",
                    "entry_date": "進場日期",
                    "exit_date": "出場日期",
                    "entry_price": "進場價",
                    "exit_price": "出場價",
                    "return": "報酬率",
                    "mae": "最大不利(MAE)",
                    "mfe": "最大有利(MFE)",
                    "period": "持有天數"
                }

                trades_display = trades.copy()
                trades_display.rename(columns=rename_map, inplace=True)

                # Ensure entry_date is datetime
                if '進場日期' in trades_display.columns:
                    trades_display['進場日期'] = pd.to_datetime(trades_display['進場日期'])

                # === Date Filter ===
                min_date = trades_display['進場日期'].min().date()
                max_date = trades_display['進場日期'].max().date()

                c_filter1, c_filter2 = st.columns(2)
                start_date = c_filter1.date_input("開始日期", value=max_date - timedelta(days=365), min_value=min_date, max_value=max_date)
                end_date = c_filter2.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)

                # Filter Data
                trades_filtered = trades_display[
                    (trades_display['進場日期'].dt.date >= start_date) &
                    (trades_display['進場日期'].dt.date <= end_date)
                ]

                # CSV Download (Full Data)
                # Use the filtered but UN-PAGINATED data for download
                csv = trades_filtered.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下載完整交易明細 (.csv)",
                    data=csv,
                    file_name=f'trade_log_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                )

                # === Pagination ===
                items_per_page = 1000
                total_items = len(trades_filtered)
                total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

                page = st.number_input("頁數 (Page)", min_value=1, max_value=total_pages, value=1)
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)

                st.info(f"顯示第 {start_idx + 1} 至 {end_idx} 筆交易 (共 {total_items} 筆)")

                # Select relevant columns
                available_cols = ['股票代碼', '進場日期', '出場日期', '進場價', '出場價', '報酬率', '持有天數', '最大不利(MAE)', '最大有利(MFE)']
                cols_to_show = [c for c in available_cols if c in trades_filtered.columns]

                trades_final = trades_filtered[cols_to_show].sort_values("進場日期", ascending=False).iloc[start_idx:end_idx]

                # CSV Download (Page)
                csv = trades_final.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下載交易明細 (.csv)",
                    data=csv,
                    file_name=f'trade_log_{strategy_type}_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                )

                # Formatting style function
                def highlight_ret(val):
                    color = ''
                    if pd.isna(val): return ''
                    if isinstance(val, (int, float)):
                        color = 'color: #22c55e' if val > 0 else 'color: #ef4444'
                    return color

                st.dataframe(
                    trades_final.style.format({
                        '報酬率': '{:.2%}',
                        '最大不利(MAE)': '{:.2%}',
                        '最大有利(MFE)': '{:.2%}',
                        '進場價': '{:.2f}',
                        '出場價': '{:.2f}'
                    }, na_rep="N/A").map(highlight_ret, subset=['報酬率']),
                    use_container_width=True,
                    height=600
                )
            else:
                st.info("無交易紀錄")

elif app_mode == "📂 自訂策略實驗室":
    st.header("📂 自訂策略實驗室 (Lab)")
    st.info("⚠️ 注意：請確保上傳的策略程式碼來源可信。此功能將直接執行 Python 腳本。")

    col1, col2 = st.columns([1, 1])
    with col1:
        # [Security Note]: Defaults are hardcoded for user convenience in local dev.
        try:
            default_finlab = st.secrets.get("FINLAB_API_TOKEN", "")
        except:
            default_finlab = ""
        finlab_token = st.text_input("Finlab API Token", type="password", value=default_finlab)

    with col2:
        with open("template_strategy.py", "rb") as f:
            st.download_button("📥 下載策略範本 (Template)", f, file_name="template_strategy.py", mime="text/x-python")

    uploaded_file = st.file_uploader("上傳您的策略 (.py)", type=["py"])

    if uploaded_file is not None:
        if st.button("🚀 執行回測"):
            if not finlab_token:
                st.error("請輸入 Finlab API Token")
            else:
                with st.spinner("正在編譯並執行您的策略..."):
                    try:
                        # Save uploaded file to temp
                        import importlib.util
                        import sys

                        temp_filename = f"temp_strategy_{int(time.time())}.py"
                        with open(temp_filename, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Dynamic Import
                        spec = importlib.util.spec_from_file_location("custom_strategy", temp_filename)
                        module = importlib.util.module_from_spec(spec)
                        sys.modules["custom_strategy"] = module
                        spec.loader.exec_module(module)

                        # Run Strategy
                        if hasattr(module, 'run_strategy'):
                            report = module.run_strategy(finlab_token)
                            st.success("執行成功！")
                            render_backtest_dashboard(report)
                        else:
                            st.error("錯誤：您的策略檔案中找不到 `run_strategy(api_token)` 函式。請參考範本。")

                        # Cleanup
                        os.remove(temp_filename)

                    except Exception as e:
                        st.error(f"執行失敗: {e}")
                        # import traceback
                        # st.code(traceback.format_exc())

# Only show original dashboard if in Dashboard Mode
elif app_mode == "📈 股市戰情室":
    # Reuse original input logic but moved here
    st.markdown("### 🔎 1. 全市場狙擊")
    single_input = st.text_input("輸入代碼/名稱 (如 凡甲, NVDA)", placeholder="Sniper Input...")
    if st.button("🚀 分析個股"):
        if not single_input: st.error("請輸入代碼")
        else:
            st.session_state.supply_chain_data = None
            st.session_state.detected_themes = []
            with st.spinner(f"正在鎖定目標 {single_input}..."):
                if not api_key:
                    if re.match(r'^\d{4,6}$', single_input):
                        target_ticker = single_input; detected_market = "🇹🇼 台股 (TW)"; target_name = single_input
                    elif re.match(r'^[A-Z]{1,5}$', single_input.upper()):
                        target_ticker = single_input.upper(); detected_market = "🗽 美股 (US)"; target_name = single_input.upper()
                    else:
                        st.error("⚠️ 請輸入 Gemini API Key 以啟用中文搜尋。或直接輸入代碼 (如 2330)。"); target_ticker=None
                else:
                    target_ticker, detected_market, target_name = resolve_ticker_and_market(single_input)

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
                        report = generate_ai_analysis(detected_market, target_ticker, data['名稱'], data['現價'], data['漲跌幅%'], "個股", data['extra_info'], strategy_mode, trend_msg, timeframe=tf_code, signal_context=data['signal_context'])
                        st.session_state.ai_reports[f"SINGLE_{target_ticker}"] = report
                        st.rerun()
                    else: st.error("無法取得數據，請檢查代碼或網路")
                else:
                    if api_key: st.error("無法識別股票，請嘗試輸入代碼 (例如 2330)")

    st.markdown("### 📡 2. 族群熱點掃描")
    if st.button("🔥 掃描今日熱門話題"):
        if not api_key: st.error("無 API Key")
        else:
            st.session_state.supply_chain_data = None
            st.session_state.single_stock_data = None
            with st.spinner("AI 正在閱讀新聞..."):
                themes = detect_hot_themes(st.session_state.market_mode)
                if themes:
                    st.session_state.detected_themes = themes
                    st.session_state.view_mode = "list"
                    st.success("偵測完成！")
                else: st.error("偵測失敗")

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
                    structure = generate_supply_chain_structure(st.session_state.market_mode, custom_input)

                if structure:
                    st.session_state.supply_chain_data = {"keyword": custom_input, "structure": structure}
                    df = scan_tickers_from_map(st.session_state.market_mode, structure, strategy_mode, timeframe=tf_code)
                    st.session_state.data_cache[st.session_state.market_mode] = df
                    st.session_state.current_source = f"⛓️ {custom_input} 供應鏈"
                    st.session_state.view_mode = "list"
                else: st.error("供應鏈拆解失敗")

    # === Main Display for Dashboard Mode ===

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

        # [Native UI Fix]: Use Streamlit's native info/success/error box
        trend_val = verdict.get('trend', '')
        if '多' in trend_val or '上升' in trend_val:
            status_box = st.success
        elif '空' in trend_val or '下降' in trend_val:
            status_box = st.error
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

        render_trend_chart(data['df'], patterns, st.session_state.market_mode, is_box=verdict.get('is_box', False), height=900, is_weekly=is_weekly, candle_patterns=data.get('candle_patterns', []))

        # AI Report Display
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
                        structure = generate_supply_chain_structure(st.session_state.market_mode, safe_theme_label)
                        if structure and isinstance(structure, dict):
                            st.session_state.supply_chain_data = {"keyword": safe_theme_label, "structure": structure}
                            df = scan_tickers_from_map(st.session_state.market_mode, structure, strategy_mode, timeframe=tf_code)
                            st.session_state.data_cache[st.session_state.market_mode] = df
                            st.session_state.current_source = f"🔥 {safe_theme_label}"
                        else: st.error("AI 正在思考中，請再試一次或換個主題")
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
                st.session_state.market_mode
            )
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

                    verdict = row.get('verdict', {})
                    # Native UI for List View
                    trend_val = row['趨勢']
                    if '多' in trend_val or '上升' in trend_val: color = "green"
                    elif '空' in trend_val or '下降' in trend_val: color = "red"
                    else: color = "gray"

                    st.markdown(f":{color}-background[**⚖️ {trend_val}**] | {row.get('signal_context', '')}")

                    render_trend_chart(row['df'], row['patterns'], st.session_state.market_mode, is_box=row.get('verdict', {}).get('is_box', False), height=600, is_weekly=is_weekly, candle_patterns=row.get('candle_patterns', []))

                    cache_key = f"{st.session_state.market_mode}_{ticker}_{strategy_mode}"
                    if cache_key in st.session_state.ai_reports:
                        st.markdown(f"<div class='ai-box'><strong>🦄 AI 分析：</strong><br>{st.session_state.ai_reports[cache_key]}</div>", unsafe_allow_html=True)
                    else:
                        if st.button(f"🧠 AI 分析 {name}", key=f"btn_{ticker}"):
                            with st.spinner("分析中..."):
                                tech_str = f"短均{row['短均']}, 長均{row['長均']}, RSI{row['RSI']}"
                                report = generate_ai_analysis(st.session_state.market_mode, ticker, name, row['現價'], row['漲跌幅%'], row['族群'], tech_str, strategy_mode, f"趨勢：{row['趨勢']}", timeframe=tf_code, signal_context=row.get('signal_context', ''))
                                st.session_state.ai_reports[cache_key] = report
                                st.rerun()
        else:
            if current_df is not None: st.warning("無符合資料。")
            else: st.info("👈 請選擇側邊欄的搜尋方式開始。")
