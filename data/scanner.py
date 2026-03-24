import pandas as pd
import streamlit as st
import hashlib

from data.provider import get_data_provider
from analysis.indicators import (
    calculate_rsi, calculate_macd, calculate_bbands, calculate_stoch, calculate_obv
)
from analysis.trend import calculate_trend_logic
from analysis.patterns import detect_complex_patterns, detect_candlestick_patterns
from analysis.ai_core import analyze_signals
from utils.helpers import validate_ticker


def _get_provider_for_market(market_type, data_source="auto", **provider_kwargs):
    """
    根據市場類型和資料來源選擇合適的 Provider。
    - data_source="auto": 台股使用永豐金優先 + YFinance 備援，美股使用 YFinance
    - data_source="sinopac": 強制使用永豐金
    - data_source="yfinance": 強制使用 YFinance
    """
    return get_data_provider(data_source, market_type=market_type, **provider_kwargs)


def scan_single_stock_deep(market, ticker, strategy, timeframe="1d", user_query_name="",
                           data_source="auto", **provider_kwargs):
    if timeframe == "1wk":
        interval = "1wk"; period = "5y"; is_weekly = True
    else:
        interval = "1d"; period = "2y"; is_weekly = False

    market_type = "TW" if "台股" in market else "US"
    if market_type == "TW":
        ma_short, ma_long = 5, 20
    else:
        ma_short, ma_long = 20, 50

    provider = _get_provider_for_market(market_type, data_source=data_source, **provider_kwargs)
    source_label = "永豐金" if (data_source == "sinopac" or (data_source == "auto" and market_type == "TW" and provider_kwargs.get('api_key'))) else "yfinance"

    df = provider.get_historical_data(ticker, period=period, interval=interval)

    if df is None or df.empty or len(df) <= 30:
        row_count = len(df) if df is not None else 0
        st.warning(f"⚠️ {source_label} 無法取得 {ticker} 的歷史資料 (rows={row_count}). 請檢查代碼是否正確或網路連線。")
        return None
    try:
        df = df.copy()  # Don't mutate cached data
        info_data = provider.get_stock_info(ticker)
        final_full_t = info_data.get('raw_ticker', ticker)

        close = df['Close'].iloc[-1]
        chg = ((close - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        vol_curr = df['Volume'].iloc[-1]
        vol_avg = df['Volume'].iloc[-21:-1].mean() if len(df) >= 22 else df['Volume'].iloc[:-1].mean()
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

        if name == ticker and user_query_name:
            name = user_query_name

        verdict = calculate_trend_logic(df, is_weekly=is_weekly)
        patterns = detect_complex_patterns(df, df['peaks'].dropna(), df['troughs'].dropna())
        candle_patterns = detect_candlestick_patterns(df)

        extra = f"趨勢: {verdict.get('trend')}。{verdict.get('signal')}。"
        if patterns:
            extra += f" 型態: {', '.join([p['name'] for p in patterns])}。"
        if candle_patterns:
            extra += f" K線: {', '.join([p['name'] for p in candle_patterns])}。"

        tech_summary = f"RSI:{round(df['RSI'].iloc[-1],1)} MACD:{'多' if df['MACD'].iloc[-1]>0 else '空'}"

        return {
            "代碼": ticker, "名稱": name, "全名": final_full_t,
            "現價": round(close, 2), "漲跌幅%": round(chg, 2), "爆量倍數": round(r_vol, 1),
            "短均": round(ma_s, 1), "長均": round(ma_l, 1), "RSI": round(df['RSI'].iloc[-1], 1),
            "PE": pe, "EPS": eps, "Yield": div_yield,
            "df": df, "verdict": verdict, "extra_info": extra + " " + tech_summary,
            "patterns": patterns, "candle_patterns": candle_patterns, "signal_context": signal_str
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.warning(f"⚠️ 分析過程出錯: {type(e).__name__}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_get_historical(ticker, period, interval, data_source, market_type):
    """Cached wrapper around provider.get_historical_data() to avoid redundant fetches."""
    provider = _get_provider_for_market(market_type, data_source=data_source)
    return provider.get_historical_data(ticker, period=period, interval=interval)


def scan_tickers_from_map(market, sector_map, strategy, timeframe="1d",
                          data_source="auto", use_cache=True, **provider_kwargs):
    """掃描指定 sector_map 中的所有股票。use_cache=True 時，同一 market+timeframe 組合在同一 session 內只掃一次。"""
    cache_key = f"_scan_cache_{market}_{timeframe}"
    if use_cache and cache_key in st.session_state:
        return st.session_state[cache_key]

    if timeframe == "1wk":
        interval = "1wk"; period = "5y"; is_weekly = True
    else:
        interval = "1d"; period = "2y"; is_weekly = False

    market_type = "TW" if "台股" in market else "US"
    if market_type == "TW":
        ma_short, ma_long = 5, 20
    else:
        ma_short, ma_long = 20, 50

    provider = _get_provider_for_market(market_type, data_source=data_source, **provider_kwargs)

    all_data = []
    unique_tickers = []
    ticker_to_sector = {}

    for sec, tickers in sector_map.items():
        if isinstance(tickers, dict):
            st.session_state.dynamic_name_map.update(tickers)
            ticker_list = list(tickers.keys())
        else:
            ticker_list = tickers
        for t in ticker_list:
            if validate_ticker(t, market) and t not in unique_tickers:
                unique_tickers.append(t)
                ticker_to_sector[t] = sec

    progress = st.progress(0)
    st_text = st.empty()
    for i, t in enumerate(unique_tickers):
        st_text.text(f"掃描中: {t}...")
        if (i+1) <= len(unique_tickers):
            progress.progress((i+1)/len(unique_tickers))

        df = _cached_get_historical(t, period=period, interval=interval,
                                     data_source=data_source, market_type=market_type)
        if df.empty or len(df) <= 30:
            continue

        info_data = provider.get_stock_info(t)
        final_t = info_data.get('raw_ticker', t)

        try:
            close = df['Close'].iloc[-1]
            chg = ((close - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100
            ma_s = df['Close'].rolling(ma_short).mean().iloc[-1]
            ma_l = df['Close'].rolling(ma_long).mean().iloc[-1]

            df['RSI'] = calculate_rsi(df['Close'])
            df['MACD'], _, _ = calculate_macd(df)
            df['K'], df['D'] = calculate_stoch(df)
            df['BB_Upper'], df['BB_Lower'] = calculate_bbands(df)

            verdict = calculate_trend_logic(df, is_weekly=is_weekly)
            clean = t.replace(".TW", "").replace(".TWO", "")
            patterns = detect_complex_patterns(df, df['peaks'].dropna(), df['troughs'].dropna())
            candle_patterns = detect_candlestick_patterns(df)
            signal_str = analyze_signals(df)

            all_data.append({
                "代碼": clean,
                "名稱": st.session_state.dynamic_name_map.get(clean, clean),
                "族群": ticker_to_sector.get(t, "其他"),
                "現價": round(close, 2), "漲跌幅%": round(chg, 2),
                "爆量倍數": 0,
                "趨勢": verdict['trend'].split(" ")[0],
                "短均": round(ma_s, 1), "長均": round(ma_l, 1),
                "RSI": round(df['RSI'].iloc[-1], 1),
                "t_color": "#f3f4f6", "t_border": "#9ca3af",
                "raw_ticker": final_t, "df": df,
                "patterns": patterns, "candle_patterns": candle_patterns,
                "verdict": verdict, "signal_context": signal_str
            })
        except Exception:
            continue
    progress.empty()
    st_text.empty()
    result_df = pd.DataFrame(all_data)
    if use_cache:
        st.session_state[cache_key] = result_df
    return result_df
