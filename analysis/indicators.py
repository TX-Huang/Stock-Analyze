"""
Technical Indicators Library
============================
Pure pandas/numpy implementations of common technical analysis indicators.
No ta-lib dependency required.

All functions accept pd.Series or pd.DataFrame inputs and return NaN
for periods with insufficient data rather than raising errors.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_series(series, name="series"):
    """Ensure input is a pd.Series. Convert if possible, raise TypeError otherwise."""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            return series.iloc[:, 0]
        raise TypeError(f"{name} must be a pd.Series, got DataFrame with {series.shape[1]} columns")
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(series).__name__}")
    return series


def _validate_df_columns(df, required_cols, func_name="function"):
    """Ensure *df* is a DataFrame containing all *required_cols*."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{func_name} requires a pd.DataFrame, got {type(df).__name__}")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{func_name} requires columns {missing} in the DataFrame")


# ===================================================================
# Trend / Moving Average indicators
# ===================================================================

def calculate_ema(series, period=20):
    """
    Exponential Moving Average (EMA).

    Parameters
    ----------
    series : pd.Series
        Price or value series.
    period : int
        Look-back window (default 20).

    Returns
    -------
    pd.Series
        EMA values; first *period-1* values will be NaN.
    """
    series = _validate_series(series, "series")
    return series.ewm(span=period, adjust=False).mean()


def calculate_tema(series, period=20):
    """
    Triple Exponential Moving Average (TEMA).

    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where EMA1 = EMA(series), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2).

    Parameters
    ----------
    series : pd.Series
        Price or value series.
    period : int
        Look-back window (default 20).

    Returns
    -------
    pd.Series
        TEMA values.
    """
    series = _validate_series(series, "series")
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


# ===================================================================
# Momentum / Oscillator indicators
# ===================================================================

def calculate_rsi(series, period=14):
    """
    Relative Strength Index using Wilder's smoothing method (EWM with alpha=1/period).

    Parameters
    ----------
    series : pd.Series
        Typically the close price series.
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100]. When loss==0 and gain>0 -> 100;
        when both==0 -> 50.
    """
    series = _validate_series(series, "series")
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Fill NaN where loss was zero
    fill_values = pd.Series(np.where(gain > 0, 100.0, 50.0), index=rsi.index)
    rsi = rsi.fillna(fill_values)
    return rsi


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line EMA period (default 9).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (macd_line, signal_line, histogram)
    """
    _validate_df_columns(df, ['Close'], 'calculate_macd')
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bbands(df, length=20, std=2):
    """
    Bollinger Bands.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    length : int
        Rolling window length (default 20).
    std : float
        Standard deviation multiplier (default 2).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper_band, middle_band, lower_band)
    """
    _validate_df_columns(df, ['Close'], 'calculate_bbands')
    mid = df['Close'].rolling(window=length).mean()
    mstd = df['Close'].rolling(window=length).std()
    upper = mid + mstd * std
    lower = mid - mstd * std
    return upper, mid, lower


def calculate_stoch(df, k_window=14, d_window=3):
    """
    Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High', 'Low', 'Close'.
    k_window : int
        Look-back period for %K (default 14).
    d_window : int
        Smoothing period for %D (default 3).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (%K, %D) both in [0, 100].
    """
    _validate_df_columns(df, ['High', 'Low', 'Close'], 'calculate_stoch')
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((df['Close'] - low_min) / denom)
    k = k.fillna(50)
    d = k.rolling(window=d_window).mean()
    return k, d


def calculate_obv(df):
    """
    On-Balance Volume (OBV) -- cumulative volume flow indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' and 'Volume'.

    Returns
    -------
    pd.Series
        Cumulative OBV values.
    """
    _validate_df_columns(df, ['Close', 'Volume'], 'calculate_obv')
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv


def calculate_cci(high, low, close, period=20):
    """
    Commodity Channel Index (CCI).

    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    where TP = (High + Low + Close) / 3.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    period : int
        Look-back window (default 20).

    Returns
    -------
    pd.Series
        CCI values. Values above +100 suggest overbought; below -100 oversold.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period).mean()
    # Mean deviation (average absolute deviation from SMA)
    mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return cci


def calculate_atr(high, low, close, period=14):
    """
    Average True Range (ATR) using Wilder's smoothing.

    True Range = max(H-L, |H-Prev_C|, |L-Prev_C|).
    ATR is the rolling mean of True Range.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        ATR values. First *period* values will be NaN.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's smoothing (equivalent to EWM with alpha=1/period)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    # Mask the first row (no prev_close)
    atr.iloc[0] = np.nan
    return atr


def calculate_williams_r(high, low, close, period=14):
    """
    Williams %R oscillator.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        Williams %R values in [-100, 0]. Values below -80 suggest oversold;
        above -20 suggest overbought.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    denom = (hh - ll).replace(0, np.nan)
    wr = -100 * (hh - close) / denom
    return wr


def calculate_mfi(high, low, close, volume, period=14):
    """
    Money Flow Index (MFI) -- volume-weighted RSI.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    volume : pd.Series
    period : int
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        MFI values in [0, 100].
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    tp = (high + low + close) / 3.0
    raw_mf = tp * volume

    # Positive / negative money flow
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, raw_mf, 0.0), index=raw_mf.index)
    neg_mf = pd.Series(np.where(tp_diff < 0, raw_mf, 0.0), index=raw_mf.index)

    pos_sum = pos_mf.rolling(window=period).sum()
    neg_sum = neg_mf.rolling(window=period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    # When neg_sum == 0 and pos_sum > 0 -> MFI = 100
    mfi = mfi.fillna(pd.Series(np.where(pos_sum > 0, 100.0, 50.0), index=mfi.index))
    return mfi


def calculate_vwap(high, low, close, volume):
    """
    Volume-Weighted Average Price (VWAP).

    Running cumulative VWAP = cumsum(TP * Volume) / cumsum(Volume).
    For true intraday VWAP, pass only one day's data at a time.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    volume : pd.Series

    Returns
    -------
    pd.Series
        Cumulative VWAP values.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    tp = (high + low + close) / 3.0
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


# ===================================================================
# Trend Strength indicators
# ===================================================================

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) with +DI and -DI.

    Uses Wilder's smoothing throughout.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    period : int
        Look-back window (default 14).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (adx, plus_di, minus_di). ADX above 25 indicates a strong trend.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    # True Range for ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smooth_minus_dm / atr.replace(0, np.nan)

    # Directional index
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_ichimoku(high, low, close,
                       tenkan_period=9, kijun_period=26,
                       senkou_b_period=52, displacement=26):
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo).

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    tenkan_period : int
        Tenkan-sen (conversion line) period (default 9).
    kijun_period : int
        Kijun-sen (base line) period (default 26).
    senkou_b_period : int
        Senkou Span B period (default 52).
    displacement : int
        Forward shift for Senkou spans / backward shift for Chikou (default 26).

    Returns
    -------
    dict[str, pd.Series]
        Keys: 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'.
        Senkou spans are shifted forward by *displacement* periods.
        Chikou is shifted backward by *displacement* periods.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    def _midpoint(h, l, p):
        return (h.rolling(window=p).max() + l.rolling(window=p).min()) / 2.0

    tenkan = _midpoint(high, low, tenkan_period)
    kijun = _midpoint(high, low, kijun_period)

    senkou_a = ((tenkan + kijun) / 2.0).shift(displacement)
    senkou_b = _midpoint(high, low, senkou_b_period).shift(displacement)
    chikou = close.shift(-displacement)

    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou,
    }


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    SuperTrend indicator.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    period : int
        ATR look-back period (default 10).
    multiplier : float
        ATR multiplier for band width (default 3.0).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (supertrend, direction).
        *direction*: 1 = bullish (price above SuperTrend),
                    -1 = bearish (price below SuperTrend).
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")

    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    # Basic bands
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(close)
    st = np.full(n, np.nan)
    dirn = np.ones(n, dtype=int)
    f_upper = basic_upper.values.copy()
    f_lower = basic_lower.values.copy()
    cl = close.values

    # Initialise: start bullish, SuperTrend = lower band
    st[0] = f_lower[0]

    for i in range(1, n):
        # Skip if ATR-derived bands are NaN
        if np.isnan(f_upper[i]) or np.isnan(f_lower[i]):
            st[i] = np.nan
            dirn[i] = dirn[i - 1]
            continue

        # Final upper band: only tighten, never widen
        if not np.isnan(f_upper[i - 1]):
            if f_upper[i] > f_upper[i - 1] and cl[i - 1] <= f_upper[i - 1]:
                f_upper[i] = f_upper[i - 1]

        # Final lower band: only raise, never lower
        if not np.isnan(f_lower[i - 1]):
            if f_lower[i] < f_lower[i - 1] and cl[i - 1] >= f_lower[i - 1]:
                f_lower[i] = f_lower[i - 1]

        # Direction and SuperTrend value
        if dirn[i - 1] == 1:  # was bullish
            if cl[i] < f_lower[i]:
                dirn[i] = -1
                st[i] = f_upper[i]
            else:
                dirn[i] = 1
                st[i] = f_lower[i]
        else:  # was bearish
            if cl[i] > f_upper[i]:
                dirn[i] = 1
                st[i] = f_lower[i]
            else:
                dirn[i] = -1
                st[i] = f_upper[i]

    supertrend = pd.Series(st, index=close.index)
    direction = pd.Series(dirn, index=close.index)
    return supertrend, direction


# ===================================================================
# Channel / Band indicators
# ===================================================================

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel (price breakout channel).

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    period : int
        Look-back window (default 20).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper, lower, mid). Upper = highest high; Lower = lowest low;
        Mid = (Upper + Lower) / 2.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")

    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def calculate_keltner(high, low, close, period=20, atr_mult=2.0):
    """
    Keltner Channel (EMA +/- ATR-based bands).

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    period : int
        EMA and ATR look-back period (default 20).
    atr_mult : float
        ATR multiplier for band width (default 2.0).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper, lower, mid). Mid = EMA(close); Upper/Lower = Mid +/- atr_mult * ATR.
    """
    close = _validate_series(close, "close")

    mid = close.ewm(span=period, adjust=False).mean()
    atr = calculate_atr(high, low, close, period)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return upper, lower, mid
