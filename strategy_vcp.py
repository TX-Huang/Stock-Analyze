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

    # --- New Conditions (Fundamental, Technical, Chip) ---
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        cond_rev = (rev_growth > 30).reindex(close.index, method='ffill').fillna(False)
    except: cond_rev = pd.DataFrame(False, index=close.index, columns=close.columns)

    body = (close - data.get('price:開盤價')).abs()
    lower_shadow = (data.get('price:開盤價').combine(close, min) - low)
    cond_shadow = (close > data.get('price:開盤價')) & (lower_shadow > body * 2)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數')
        trust_buy = data.get('institutional_investors:投信買賣超股數')
        cond_chip = (foreign_buy.fillna(0) + trust_buy.fillna(0)) > 0
    except: cond_chip = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 8. Trigger (Breakout)
    breakout_std = (close > close.rolling(20).max().shift(1)) & (vol > vol.average(20) * 1.5)

    # VCP is strict on pattern (cond1-cond7 + contracting + RS), so we add new factors as enhancers to the TRIGGER
    base_setup = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & is_contracting & rs_rating & is_liquid

    # Trigger: Standard Breakout OR (Special Signal + Breakout of Resistance)
    trigger = breakout_std | ((cond_rev | cond_shadow | cond_chip) & (close > close.rolling(10).max().shift(1)))

    buy_signal = base_setup & trigger

    # 9. Exit Strategy (Optimized)
    # Relaxed from 20MA to 50MA to allow trend to ride
    exit_signal = close < ma50

    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    report = backtest.sim(position, resample='D', name='VCP Strategy (Minervini)', upload=False)
    return report
