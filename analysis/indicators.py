import numpy as np


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    # Handle division by zero: loss=0 → RSI=100, gain=0 → RSI=0
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When loss=0 and gain>0 → RSI=100; when both=0 → RSI=50
    import pandas as pd
    fill_values = pd.Series(np.where(gain > 0, 100, 50), index=rsi.index)
    rsi = rsi.fillna(fill_values)
    return rsi


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
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((df['Close'] - low_min) / denom)
    k = k.fillna(50)
    d = k.rolling(window=d_window).mean()
    return k, d


def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv
