from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_long_short_strategy(api_token):
    if api_token:
        finlab.login(api_token)

    # 1. Fetch Data
    close = data.get('price:收盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # 2. Indicators
    ma20 = close.average(20)
    ma50 = close.average(50)
    ma150 = close.average(150)
    ma200 = close.average(200)

    high250 = close.rolling(250).max().shift(1)
    low250 = close.rolling(250).min().shift(1)
    vol_avg = vol.average(20)

    daily_range = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / daily_range).fillna(0)

    # Liquidity
    is_liquid = vol_avg > 500000

    # --- LONG LOGIC ---
    l_cond1 = (ma50 > ma150) & (ma150 > ma200)
    l_cond2 = (close > ma50) & (close > ma150) & (close > ma200)
    l_cond3 = close > (low250 * 1.30)
    # Refined price range for Long
    l_cond4 = (close >= high250 * 0.85) & (close <= high250 * 1.05)
    l_cond5 = ma200 > ma200.shift(20)
    l_cond6 = vol > (vol_avg * 1.5)
    l_cond7 = close_pos >= 0.75

    buy_signal = l_cond1 & l_cond2 & l_cond3 & l_cond4 & l_cond5 & l_cond6 & l_cond7 & is_liquid
    # Relaxed Long Exit: Close < 50MA
    long_exit = close < ma50

    # --- SHORT LOGIC (Inverse) ---
    s_cond1 = (ma50 < ma150) & (ma150 < ma200)
    s_cond2 = (close < ma50) & (close < ma150) & (close < ma200)
    s_cond3 = close < (high250 * 0.70)
    # Short near low
    s_cond4 = (close <= low250 * 1.15)
    s_cond5 = ma200 < ma200.shift(20)
    s_cond6 = vol > (vol_avg * 1.5)
    s_cond7 = close_pos <= 0.25

    short_signal = s_cond1 & s_cond2 & s_cond3 & s_cond4 & s_cond5 & s_cond6 & s_cond7 & is_liquid
    # Relaxed Short Exit: Close > 50MA
    short_exit = close > ma50

    # --- POSITION CONSTRUCTION ---
    long_pos = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    long_pos[buy_signal] = 1
    long_pos[long_exit] = 0
    long_pos = long_pos.ffill().fillna(0)

    short_pos = pd.DataFrame(np.nan, index=short_signal.index, columns=short_signal.columns)
    short_pos[short_signal] = -1
    short_pos[short_exit] = 0
    short_pos = short_pos.ffill().fillna(0)

    final_pos = long_pos + short_pos

    report = backtest.sim(final_pos, resample='D', name='Long/Short Breakout Strategy', upload=False)
    return report
