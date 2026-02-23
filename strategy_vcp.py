from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_vcp_strategy(api_token):
    if api_token:
        finlab.login(api_token)

    # 1. Fetch Data
    close = data.get('price:收盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')
    benchmark = data.get('price:收盤價')['0050']

    # 2. Indicators
    ma50 = close.average(50)
    ma150 = close.average(150)
    ma200 = close.average(200)

    high250 = close.rolling(250).max().shift(1)
    low250 = close.rolling(250).min().shift(1)
    vol_avg = vol.average(50)

    # 3. Mark Minervini's Trend Template
    cond1 = close > ma150
    cond2 = ma150 > ma200
    cond3 = ma200 > ma200.shift(20)
    cond4 = ma50 > ma150
    cond5 = close > ma50

    # 4. Price Structure
    cond6 = close > (low250 * 1.30)
    cond7 = close >= (high250 * 0.75)

    # 5. Volatility Contraction
    std = close.rolling(20).std()
    upper = close.average(20) + 2 * std
    lower = close.average(20) - 2 * std
    bandwidth = (upper - lower) / close.average(20)
    is_contracting = bandwidth < bandwidth.rolling(60).quantile(0.25)

    # 6. Relative Strength
    stock_ret = close.pct_change(120)
    market_ret = benchmark.pct_change(120)
    rs_rating = stock_ret.sub(market_ret, axis=0) > 0

    # 7. Liquidity Filter (Updated to match others)
    is_liquid = vol_avg > 500000

    # 8. Trigger (Breakout)
    breakout = (close > close.rolling(20).max().shift(1)) & (vol > vol.average(20) * 1.5)

    buy_signal = (cond1 & cond2 & cond3 & cond4 & cond5 &
                  cond6 & cond7 &
                  is_contracting & rs_rating & is_liquid & breakout)

    # 9. Exit Strategy (Optimized)
    # Relaxed from 20MA to 50MA to allow trend to ride
    exit_signal = close < ma50

    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    report = backtest.sim(position, resample='D', name='VCP Strategy (Minervini)', upload=False)
    return report
