import pandas as pd
import streamlit as st

from data.provider import get_data_provider
from analysis.indicators import (
    calculate_rsi, calculate_macd, calculate_bbands, calculate_stoch, calculate_obv
)
from analysis.trend import calculate_trend_logic
from analysis.patterns import detect_complex_patterns, detect_candlestick_patterns
from analysis.ai_core import analyze_signals
from utils.helpers import validate_ticker


def scan_single_stock_deep(market, ticker, strategy, timeframe="1d", user_query_name=""):
    if timeframe == "1wk":
        interval = "1wk"; period = "5y"; is_weekly = True
    else:
        interval = "1d"; period = "2y"; is_weekly = False

    market_type = "TW" if "台股" in market else "US"
    if market_type == "TW":
        ma_short, ma_long = 5, 20
    else:
        ma_short, ma_long = 20, 50

    provider = get_data_provider("yfinance", market_type=market_type)
    df = provider.get_historical_data(ticker, period=period, interval=interval)

    if df.empty or len(df) <= 30:
        st.warning(f"⚠️ yfinance 無法取得 {ticker} 的歷史資料 (rows={len(df)}). 請檢查代碼是否正確或網路連線。")
        return None
    try:
        info_data = provider.get_stock_info(ticker)
        final_full_t = info_data.get('raw_ticker', ticker)

        close = df['Close'].iloc[-1]
        chg = ((close - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        vol_curr = df['Volume'].iloc[-1]
        vol_avg = df['Volume'].iloc[:-5].mean()
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


def scan_tickers_from_map(market, sector_map, strategy, timeframe="1d"):
    if timeframe == "1wk":
        interval = "1wk"; period = "5y"; is_weekly = True
    else:
        interval = "1d"; period = "2y"; is_weekly = False

    market_type = "TW" if "台股" in market else "US"
    if market_type == "TW":
        ma_short, ma_long = 5, 20
    else:
        ma_short, ma_long = 20, 50

    provider = get_data_provider("yfinance", market_type=market_type)

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

        df = provider.get_historical_data(t, period=period, interval=interval)
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
    return pd.DataFrame(all_data)
