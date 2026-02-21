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

    high250 = close.rolling(250).max()
    low250 = close.rolling(250).min()
    vol_avg = vol.average(20)

    # 3. Long Entry Conditions
    cond1 = (ma50 > ma150) & (ma150 > ma200)
    cond2 = (close > ma50) & (close > ma150) & (close > ma200)
    cond3 = close > (low250 * 1.30)
    cond4 = close >= (high250 * 0.85)
    cond5 = ma200 > ma200.shift(20)
    cond6 = vol > (vol_avg * 1.5)

    daily_range = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / daily_range).fillna(0)
    cond7 = close_pos >= 0.75

    buy_signal = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7

    # 4. Long Exit Condition:
    # Option A: Close below 20MA (Tighter stop for momentum trades)
    # Option B: Trailing Stop of 10% from Highest High while in trade (Hard to code without iterative loop or special function)
    # Option C: Close below 50MA (Standard trend following)

    # Let's use a combination: Close below 20MA to lock in profits faster since we enter on a breakout.
    # Breakout strategies often fail quickly if momentum is lost.
    exit_signal = close < ma20

    # 5. Position Construction
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    # Run Backtest
    report = backtest.sim(position, resample='D', name='Long Strategy (Breakout)', upload=False)
    return report
