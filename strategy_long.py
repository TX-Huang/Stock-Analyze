from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_long_strategy(api_token):
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

    # 3. Long Entry Conditions

    # Cond 1: 50MA > 150MA > 200MA
    cond1 = (ma50 > ma150) & (ma150 > ma200)

    # Cond 2: Price > 50MA & 150MA & 200MA
    cond2 = (close > ma50) & (close > ma150) & (close > ma200)

    # Cond 3: Price at least 30% above 250-day low
    cond3 = close > (low250 * 1.30)

    # Cond 4: Price within 15% of 250-day high (Breakout point)
    cond4 = (close >= high250 * 0.85) & (close <= high250 * 1.05)

    # Cond 5: 200MA trending up for 1 month
    cond5 = ma200 > ma200.shift(20)

    # Cond 6: Volume Spike > 1.5x Avg
    cond6 = vol > (vol_avg * 1.5)

    # Cond 7: Strong Close (Top 25% of daily range)
    daily_range = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / daily_range).fillna(0)
    cond7 = close_pos >= 0.75

    # Liquidity Filter (Optimized)
    is_liquid = vol_avg > 500000 # > 500 lots

    buy_signal = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & is_liquid

    # 4. Long Exit Condition (Optimized)
    # Previous: Close < 20MA (Caused 1-day whipsaws)
    # New: Close < 50MA (Classic Trend Following) OR Close < 0.9 * Highest High since entry (10% Trailing Stop)
    # For vectorized backtest, simple Trailing Stop is hard.
    # Let's switch to Close < 50MA to allow the trend to develop, avoiding 1-day exits.
    exit_signal = close < ma50

    # 5. Position Construction
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    # Run Backtest
    report = backtest.sim(position, resample='D', name='Long Strategy (Breakout)', upload=False)
    return report
