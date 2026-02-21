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

    high250 = close.rolling(250).max()
    low250 = close.rolling(250).min()
    vol_avg = vol.average(20)

    daily_range = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / daily_range).fillna(0)

    # --- LONG LOGIC ---
    l_cond1 = (ma50 > ma150) & (ma150 > ma200)
    l_cond2 = (close > ma50) & (close > ma150) & (close > ma200)
    l_cond3 = close > (low250 * 1.30)
    l_cond4 = close >= (high250 * 0.85)
    l_cond5 = ma200 > ma200.shift(20)
    l_cond6 = vol > (vol_avg * 1.5)
    l_cond7 = close_pos >= 0.75

    buy_signal = l_cond1 & l_cond2 & l_cond3 & l_cond4 & l_cond5 & l_cond6 & l_cond7
    long_exit = close < ma20

    # --- SHORT LOGIC (Inverse of Long) ---
    # 1. Bearish Alignment: 50 < 150 < 200
    s_cond1 = (ma50 < ma150) & (ma150 < ma200)
    # 2. Price below all MAs
    s_cond2 = (close < ma50) & (close < ma150) & (close < ma200)
    # 3. Price dropped 30% from High (Breakdown context)
    s_cond3 = close < (high250 * 0.70)
    # 4. Price near Low (within 15% of Low)
    s_cond4 = close <= (low250 * 1.15)
    # 5. 200MA Trending Down
    s_cond5 = ma200 < ma200.shift(20)
    # 6. Volume Spike (Panic selling often has huge volume too)
    s_cond6 = vol > (vol_avg * 1.5)
    # 7. Weak Close (Bottom 25%)
    s_cond7 = close_pos <= 0.25

    short_signal = s_cond1 & s_cond2 & s_cond3 & s_cond4 & s_cond5 & s_cond6 & s_cond7
    # Short Exit: Price rallies above 20MA
    short_exit = close > ma20

    # --- POSITION CONSTRUCTION ---
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)

    # Assign signals (Latest signal takes precedence if conflict, though rare)
    # 1 = Long, -1 = Short, 0 = Neutral/Exit

    # Logic:
    # If Long Signal -> 1
    # If Short Signal -> -1
    # If currently Long (1) and Long Exit -> 0 (unless Short Signal present)
    # If currently Short (-1) and Short Exit -> 0 (unless Long Signal present)

    # Simplified approach using iterative forward fill concept or separate series
    # Let's build two separate position series and combine them?
    # Or just one state machine:
    # State = 0 (Neutral)
    # If State == 0: if Buy -> 1, elif Sell -> -1
    # If State == 1: if BuyExit -> 0 (or -1 if Sell trigger)
    # If State == -1: if SellExit -> 0 (or 1 if Buy trigger)

    # Since vectorization is hard for state machine with multiple triggers:
    # We can create a "signal_value" series:
    # +1 for Buy, -1 for Short
    # +0.1 for Long Exit (soft exit), -0.1 for Short Exit (soft exit)

    # Let's try this:
    # Construct a raw signal series
    # 2 = Buy, 1 = Long Exit
    # -2 = Short, -1 = Short Exit
    # 0 = No Change

    # This is complex. Let's simplify.
    # Long Position:
    long_pos = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    long_pos[buy_signal] = 1
    long_pos[long_exit] = 0
    long_pos = long_pos.ffill().fillna(0)

    # Short Position:
    short_pos = pd.DataFrame(np.nan, index=short_signal.index, columns=short_signal.columns)
    short_pos[short_signal] = -1
    short_pos[short_exit] = 0
    short_pos = short_pos.ffill().fillna(0)

    # Combine:
    # If Long is active (1), we are Long.
    # If Short is active (-1), we are Short.
    # If both active? (Conflict). Rare with trend filters.
    # If both 1 and -1, usually one trend dominates.
    # MAs (50>150 vs 50<150) make them mutually exclusive mostly.

    final_pos = long_pos + short_pos

    # Run Backtest
    report = backtest.sim(final_pos, resample='D', name='Long/Short Breakout Strategy', upload=False)
    return report
