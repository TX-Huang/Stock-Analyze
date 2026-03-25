"""
Stock Report Service — AI 戰情室核心聚合服務
============================================
提供 generate_stock_report() 作為所有分析模組的統一入口。
ui/stock_profile.py 和 AI 診斷報告頁面都使用此服務。
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 30-minute cache
# ---------------------------------------------------------------------------
_report_cache: dict = {}
_CACHE_TTL = 30 * 60  # 30 minutes in seconds


def _cache_key(ticker: str) -> tuple:
    """Generate cache key rounded to 30-min window."""
    ts_rounded = int(time.time() // _CACHE_TTL)
    return (ticker, ts_rounded)


def _clean_expired_cache():
    """Remove expired entries from the cache."""
    now = time.time()
    expired = [
        k for k, v in _report_cache.items()
        if now - v.get('_cached_at', 0) > _CACHE_TTL
    ]
    for k in expired:
        del _report_cache[k]


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_stock_report(
    ticker: str,
    market_type: str = "TW",
    gemini_client=None,
    gemini_model=None,
) -> dict:
    """
    Aggregate all analysis modules into a single unified report dict.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. '2330' for TW, 'AAPL' for US).
    market_type : str
        'TW' or 'US'.
    gemini_client : optional
        google.genai.Client instance for AI analysis.
    gemini_model : optional
        Gemini model name string.

    Returns
    -------
    dict
        Unified report with sections: price_info, technical, fundamental,
        ai_signals, ai_analysis, thesis, risk_warnings, strategy_signals.
    """
    # --- Cache check ---
    _clean_expired_cache()
    key = _cache_key(ticker)
    if key in _report_cache:
        logger.info("[StockReport] Cache hit for %s", ticker)
        return _report_cache[key]

    # --- Data fetch (single fetch, shared across all sections) ---
    from data.provider import get_data_provider

    provider = get_data_provider(source_name="auto", market_type=market_type)
    df = pd.DataFrame()
    stock_info = {}

    try:
        df = provider.get_historical_data(ticker, period="2y", interval="1d")
    except Exception as e:
        logger.warning("[StockReport] Failed to fetch historical data for %s: %s", ticker, e)

    try:
        stock_info = provider.get_stock_info(ticker)
    except Exception as e:
        logger.warning("[StockReport] Failed to fetch stock info for %s: %s", ticker, e)

    name = stock_info.get('name', ticker)

    report = {
        'ticker': ticker,
        'name': name,
        'market': market_type,
        'timestamp': datetime.now().isoformat(),
        'price_info': None,
        'technical': None,
        'fundamental': None,
        'ai_signals': None,
        'ai_analysis': None,
        'thesis': None,
        'risk_warnings': None,
        'strategy_signals': None,
    }

    if df is None or df.empty:
        logger.warning("[StockReport] No data for %s, returning empty report", ticker)
        report['_cached_at'] = time.time()
        _report_cache[key] = report
        return report

    # ── Section: price_info ──
    try:
        price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0.0
        volume = float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0

        report['price_info'] = {
            'price': round(price, 2),
            'prev_close': round(prev_close, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'volume': volume,
        }
    except Exception as e:
        logger.warning("[StockReport] price_info failed for %s: %s", ticker, e)

    # ── Section: technical ──
    try:
        report['technical'] = _build_technical(df)
    except Exception as e:
        logger.warning("[StockReport] technical section failed for %s: %s", ticker, e)

    # ── Section: fundamental ──
    try:
        if market_type == "US":
            # US stocks: YFinance uses trailingPE, trailingEps, dividendYield
            report['fundamental'] = {
                'pe': stock_info.get('pe', stock_info.get('trailingPE', 'N/A')),
                'eps': stock_info.get('eps', stock_info.get('trailingEps', 'N/A')),
                'dividend_yield': stock_info.get('yield', stock_info.get('dividendYield', 'N/A')),
                'market_cap': stock_info.get('marketCap', 'N/A'),
                'revenue_mom': 'N/A',
                'revenue_yoy': 'N/A',
            }
        else:
            report['fundamental'] = {
                'pe': stock_info.get('pe', 'N/A'),
                'eps': stock_info.get('eps', 'N/A'),
                'dividend_yield': stock_info.get('yield', 'N/A'),
                'revenue_mom': 'N/A',
                'revenue_yoy': 'N/A',
            }
    except Exception as e:
        logger.warning("[StockReport] fundamental section failed for %s: %s", ticker, e)

    # ── Section: ai_signals (simple rule-based detection) ──
    try:
        df_with_indicators = _ensure_indicator_columns(df)
        from analysis.ai_core import analyze_signals
        report['ai_signals'] = analyze_signals(df_with_indicators)
    except Exception as e:
        logger.warning("[StockReport] ai_signals failed for %s: %s", ticker, e)

    # ── Section: ai_analysis (Gemini, optional) ──
    try:
        if gemini_client is not None:
            from analysis.ai_core import generate_ai_analysis

            tech = report.get('technical') or {}
            price_info = report.get('price_info') or {}
            technicals_str = (
                f"RSI={tech.get('rsi', {}).get('value', 'N/A')}, "
                f"MACD={tech.get('macd', {}).get('value', 'N/A')}, "
                f"ADX={tech.get('adx', {}).get('value', 'N/A')}"
            )
            signal_context = report.get('ai_signals', '') or ''

            report['ai_analysis'] = generate_ai_analysis(
                market=market_type,
                ticker=ticker,
                name=name,
                price=price_info.get('price', 0),
                change=price_info.get('change_pct', 0),
                sector='',
                technicals=technicals_str,
                strategy='',
                signal_context=str(signal_context),
                client=gemini_client,
                gemini_model=gemini_model,
            )
        else:
            report['ai_analysis'] = None
    except Exception as e:
        logger.warning("[StockReport] ai_analysis failed for %s: %s", ticker, e)

    # ── Section: thesis ──
    try:
        from analysis.breakout import detect_levels, detect_signal
        from analysis.thesis import generate_thesis

        levels = detect_levels(df)
        signal_data = detect_signal(df, levels)

        report['thesis'] = generate_thesis(
            ticker=ticker,
            df=df,
            signal_data=signal_data,
            levels=levels,
        )
    except Exception as e:
        logger.warning("[StockReport] thesis failed for %s: %s", ticker, e)

    # ── Section: risk_warnings ──
    try:
        from ui.widgets.risk_warnings import generate_stock_warnings

        # generate_stock_warnings expects RSI column in df if available
        df_for_warnings = _ensure_indicator_columns(df)
        report['risk_warnings'] = generate_stock_warnings(df_for_warnings, ticker)
    except Exception as e:
        logger.warning("[StockReport] risk_warnings failed for %s: %s", ticker, e)

    # ── Section: strategy_signals ──
    try:
        report['strategy_signals'] = _check_strategy_signals(ticker, df)
    except Exception as e:
        logger.warning("[StockReport] strategy_signals failed for %s: %s", ticker, e)

    # --- Store in cache ---
    report['_cached_at'] = time.time()
    _report_cache[key] = report
    return report


# ---------------------------------------------------------------------------
# Technical section builder
# ---------------------------------------------------------------------------

def _build_technical(df: pd.DataFrame) -> dict:
    """Build the full technical analysis section from OHLCV data."""
    from analysis.indicators import (
        calculate_rsi,
        calculate_macd,
        calculate_bbands,
        calculate_stoch,
        calculate_atr,
        calculate_adx,
    )
    from analysis.trend import calculate_trend_logic

    result = {}

    # Trend verdict
    try:
        df_copy = df.copy()
        trend = calculate_trend_logic(df_copy)
        result['trend_verdict'] = trend
    except Exception as e:
        logger.warning("[StockReport] trend_verdict failed: %s", e)
        result['trend_verdict'] = None

    # MA alignment
    try:
        close = df['Close']
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma120 = float(close.rolling(120).mean().iloc[-1]) if len(df) >= 120 else None
        price = float(close.iloc[-1])

        positions = []
        for label, val in [('Price', price), ('MA20', ma20), ('MA50', ma50), ('MA120', ma120)]:
            if val is not None and not np.isnan(val):
                positions.append((label, val))
        positions.sort(key=lambda x: x[1], reverse=True)

        result['ma_alignment'] = {
            'ma20': round(ma20, 2) if not np.isnan(ma20) else None,
            'ma50': round(ma50, 2) if not np.isnan(ma50) else None,
            'ma120': round(ma120, 2) if ma120 is not None and not np.isnan(ma120) else None,
            'order': [p[0] for p in positions],
        }
    except Exception as e:
        logger.warning("[StockReport] ma_alignment failed: %s", e)
        result['ma_alignment'] = None

    # RSI
    try:
        rsi_series = calculate_rsi(df['Close'], period=14)
        rsi_val = float(rsi_series.iloc[-1])
        if rsi_val >= 70:
            rsi_label = '超買'
        elif rsi_val <= 30:
            rsi_label = '超賣'
        else:
            rsi_label = '中性'
        result['rsi'] = {'value': round(rsi_val, 1), 'label': rsi_label}
    except Exception as e:
        logger.warning("[StockReport] rsi failed: %s", e)
        result['rsi'] = None

    # MACD
    try:
        macd_line, signal_line, histogram = calculate_macd(df)
        macd_val = float(macd_line.iloc[-1])
        hist_val = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2]) if len(histogram) >= 2 else 0

        if prev_hist <= 0 < hist_val:
            macd_dir = '翻紅'
        elif prev_hist >= 0 > hist_val:
            macd_dir = '翻綠'
        elif hist_val > 0:
            macd_dir = '多方'
        else:
            macd_dir = '空方'

        result['macd'] = {
            'value': round(macd_val, 2),
            'histogram': round(hist_val, 2),
            'direction': macd_dir,
        }
    except Exception as e:
        logger.warning("[StockReport] macd failed: %s", e)
        result['macd'] = None

    # Bollinger Bands
    try:
        upper, mid, lower = calculate_bbands(df)
        price = float(df['Close'].iloc[-1])
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        bb_mid = float(mid.iloc[-1])
        bb_range = bb_upper - bb_lower
        position_pct = ((price - bb_lower) / bb_range * 100) if bb_range > 0 else 50.0

        result['bbands'] = {
            'upper': round(bb_upper, 2),
            'mid': round(bb_mid, 2),
            'lower': round(bb_lower, 2),
            'position_pct': round(position_pct, 1),
        }
    except Exception as e:
        logger.warning("[StockReport] bbands failed: %s", e)
        result['bbands'] = None

    # Stochastic
    try:
        k, d = calculate_stoch(df)
        k_val = float(k.iloc[-1])
        d_val = float(d.iloc[-1])
        prev_k = float(k.iloc[-2]) if len(k) >= 2 else k_val
        prev_d = float(d.iloc[-2]) if len(d) >= 2 else d_val

        if prev_k < prev_d and k_val > d_val:
            stoch_cross = 'golden_cross'
        elif prev_k > prev_d and k_val < d_val:
            stoch_cross = 'death_cross'
        else:
            stoch_cross = None

        result['stochastic'] = {
            'k': round(k_val, 1),
            'd': round(d_val, 1),
            'cross': stoch_cross,
        }
    except Exception as e:
        logger.warning("[StockReport] stochastic failed: %s", e)
        result['stochastic'] = None

    # ADX
    try:
        adx, plus_di, minus_di = calculate_adx(
            df['High'], df['Low'], df['Close'],
        )
        adx_val = float(adx.iloc[-1])
        if adx_val >= 40:
            strength = '極強趨勢'
        elif adx_val >= 25:
            strength = '強趨勢'
        elif adx_val >= 20:
            strength = '弱趨勢'
        else:
            strength = '無趨勢'

        result['adx'] = {
            'value': round(adx_val, 1),
            'plus_di': round(float(plus_di.iloc[-1]), 1),
            'minus_di': round(float(minus_di.iloc[-1]), 1),
            'trend_strength': strength,
        }
    except Exception as e:
        logger.warning("[StockReport] adx failed: %s", e)
        result['adx'] = None

    # ATR
    try:
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        atr_val = float(atr.iloc[-1])
        result['atr'] = {'value': round(atr_val, 2)}
    except Exception as e:
        logger.warning("[StockReport] atr failed: %s", e)
        result['atr'] = None

    return result


# ---------------------------------------------------------------------------
# Helper: ensure indicator columns exist for ai_core.analyze_signals
# ---------------------------------------------------------------------------

def _ensure_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with RSI, MACD, K, D columns required by
    ``analyze_signals``.  If columns already exist they are left as-is.
    """
    from analysis.indicators import (
        calculate_rsi,
        calculate_macd,
        calculate_stoch,
    )

    out = df.copy()

    if 'RSI' not in out.columns:
        try:
            out['RSI'] = calculate_rsi(out['Close'], period=14)
        except Exception:
            pass

    if 'MACD' not in out.columns:
        try:
            macd_line, _, histogram = calculate_macd(out)
            out['MACD'] = histogram
        except Exception:
            pass

    if 'K' not in out.columns or 'D' not in out.columns:
        try:
            k, d = calculate_stoch(out)
            out['K'] = k
            out['D'] = d
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# Helper: check built-in strategy signals for this stock
# ---------------------------------------------------------------------------

def _check_strategy_signals(ticker: str, df: pd.DataFrame) -> list:
    """
    Check if the stock currently has buy/sell signals from built-in strategies.

    Returns a list of dicts:
        [{'strategy_name': str, 'signal': str|None, 'win_rate': float|None,
          'sample_size': int|None}]
    """
    signals = []

    # VCP strategy check via breakout module
    try:
        from analysis.breakout import detect_vcp
        vcp = detect_vcp(df)
        signal = None
        if vcp.get('is_vcp'):
            signal = 'VCP 成形'
        elif vcp.get('vcp_score', 0) >= 2:
            signal = 'VCP 部分成形'

        signals.append({
            'strategy_name': 'VCP (Minervini)',
            'signal': signal,
            'win_rate': None,
            'sample_size': None,
        })
    except Exception as e:
        logger.warning("[StockReport] VCP strategy check failed: %s", e)

    # Breakout signal check
    try:
        from analysis.breakout import detect_levels, detect_signal, SIGNAL_TYPES
        levels = detect_levels(df)
        sig_result = detect_signal(df, levels)
        if sig_result and sig_result.get('signal'):
            sig_type = sig_result['signal']
            sig_info = SIGNAL_TYPES.get(sig_type, {})
            signals.append({
                'strategy_name': '突破偵測',
                'signal': sig_info.get('label', sig_type),
                'win_rate': None,
                'sample_size': None,
            })
    except Exception as e:
        logger.warning("[StockReport] breakout strategy check failed: %s", e)

    # Candlestick pattern check
    try:
        from analysis.patterns import detect_candlestick_patterns
        patterns = detect_candlestick_patterns(df.copy())
        if patterns:
            # Report only the most recent pattern
            latest = patterns[-1]
            signals.append({
                'strategy_name': 'K 線型態',
                'signal': f"{latest['name']} ({latest['type']})",
                'win_rate': None,
                'sample_size': None,
            })
    except Exception as e:
        logger.warning("[StockReport] candlestick strategy check failed: %s", e)

    return signals


# ---------------------------------------------------------------------------
# Summary formatter
# ---------------------------------------------------------------------------

def format_report_summary(report: dict) -> str:
    """
    Format a one-line text summary from a stock report.

    Example output:
        "\U0001f1f9\U0001f1fc 2330 台積電 | 多頭 | RSI 65 | MACD 翻紅 | 綜合分 7.2 | 近壓力位"
        "\U0001f1fa\U0001f1f8 AAPL Apple | Bullish | RSI 65 | MACD 翻紅 | 綜合分 7.2"

    Color convention note:
        TW: 紅漲綠跌 (red=up, green=down)
        US: 綠漲紅跌 (green=up, red=down)
    """
    ticker = report.get('ticker', '?')
    name = report.get('name', '')
    market = report.get('market', 'TW')

    # Market flag
    flag = '\U0001f1fa\U0001f1f8' if market == 'US' else '\U0001f1f9\U0001f1fc'

    # Price change with market-aware color indicator
    price_info = report.get('price_info') or {}
    change_pct = price_info.get('change_pct')
    if change_pct is not None:
        if market == 'US':
            # US: green up, red down
            arrow = '\u25b2' if change_pct > 0 else '\u25bc' if change_pct < 0 else '\u2500'
        else:
            # TW: red up, green down (arrow direction same, color handled by UI)
            arrow = '\u25b2' if change_pct > 0 else '\u25bc' if change_pct < 0 else '\u2500'
        chg_str = f"{arrow}{change_pct:+.1f}%"
    else:
        chg_str = ''

    # Trend
    tech = report.get('technical') or {}
    trend_info = tech.get('trend_verdict') or {}
    trend = trend_info.get('trend', '---')

    # RSI
    rsi_info = tech.get('rsi') or {}
    rsi_val = rsi_info.get('value')
    rsi_str = f"RSI {rsi_val:.0f}" if rsi_val is not None else "RSI --"

    # MACD
    macd_info = tech.get('macd') or {}
    macd_dir = macd_info.get('direction', '--')
    macd_str = f"MACD {macd_dir}"

    # Composite score from thesis
    thesis = report.get('thesis') or {}
    score = thesis.get('composite_score')
    score_str = f"綜合分 {score}" if score is not None else "綜合分 --"

    # Risk warnings (pick first if any)
    warnings = report.get('risk_warnings') or []
    warn_str = ''
    if warnings:
        first_warn = warnings[0]
        title = first_warn.get('title', '')
        warn_str = f" | {title}"

    parts = [f"{flag} {ticker} {name}"]
    if chg_str:
        parts.append(chg_str)
    parts.extend([trend, rsi_str, macd_str, score_str])
    result = " | ".join(parts)
    if warn_str:
        result += warn_str
    return result


def get_change_color(change_pct: float, market_type: str = "TW") -> str:
    """Return the appropriate color for a price change value.

    TW convention: 紅漲綠跌 (red=up, green=down)
    US convention: 綠漲紅跌 (green=up, red=down)
    """
    if change_pct > 0:
        return "#ef4444" if market_type == "TW" else "#22c55e"
    elif change_pct < 0:
        return "#22c55e" if market_type == "TW" else "#ef4444"
    return "#94a3b8"
