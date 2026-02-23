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
    # Benchmark for RS Rating (Use 0050 ETF as a proxy for the market)
    benchmark = data.get('price:收盤價')['0050']

    # 2. Indicators
    ma50 = close.average(50)
    ma150 = close.average(150)
    ma200 = close.average(200)

    high250 = close.rolling(250).max()
    low250 = close.rolling(250).min()
    vol_avg = vol.average(50) # Use 50-day average volume for liquidity

    # 3. Mark Minervini's Trend Template (Stage 2 Uptrend)
    cond1 = close > ma150
    cond2 = ma150 > ma200
    cond3 = ma200 > ma200.shift(20) # 200MA trending up (at least over 1 month)
    cond4 = ma50 > ma150 # 50MA above 150MA
    cond5 = close > ma50 # Price above 50MA

    # 4. Price Structure
    cond6 = close > (low250 * 1.30) # At least 30% above 52-week low
    cond7 = close >= (high250 * 0.75) # Within 25% of 52-week high (Consolidating near highs)

    # 5. Volatility Contraction (VCP Characteristic)
    # Bandwidth or ATR contraction. Let's use Bollinger Bandwidth as a proxy for contraction.
    # Bandwidth = (Upper - Lower) / Middle
    std = close.rolling(20).std()
    upper = close.average(20) + 2 * std
    lower = close.average(20) - 2 * std
    bandwidth = (upper - lower) / close.average(20)
    # Check if bandwidth is in the lowest 25% of the last 60 days
    is_contracting = bandwidth < bandwidth.rolling(60).quantile(0.25)

    # 6. Relative Strength (RS Rating)
    # Simple RS: Stock Return vs Market Return over 6 months (120 days)
    stock_ret = close.pct_change(120)
    market_ret = benchmark.pct_change(120)
    # Broadcast market return to match stock dataframe shape
    rs_rating = stock_ret.sub(market_ret, axis=0) > 0 # Outperforming the market

    # 7. Liquidity Filter
    is_liquid = vol_avg > 1000 * 1000 # Average volume > 1000 lots (assuming data is in shares? Finlab usually shares)
    # Actually Finlab volume is usually shares. 1000 lots = 1,000,000 shares.
    # Let's assume reasonable liquidity: > 500,000 shares (500 lots) daily avg.
    is_liquid = vol_avg > 500000

    # 8. Trigger (Breakout)
    # Price breaks above 20-day high with Volume > 1.5x Avg
    breakout = (close > close.rolling(20).max().shift(1)) & (vol > vol.average(20) * 1.5)

    # Combine All Entry Conditions
    buy_signal = (cond1 & cond2 & cond3 & cond4 & cond5 &
                  cond6 & cond7 &
                  is_contracting & rs_rating & is_liquid & breakout)

    # 9. Exit Strategy (Chandelier Exit / Trailing Stop)
    # Exit if Close < Highest High since Entry - 3 * ATR
    # For backtest.sim, we usually provide a position matrix.
    # Implementing complex trailing stop in vectorized form is hard.
    # We will use a simplified trailing stop: Close < 20MA (Standard for trend following)
    # OR if we want to be tighter: Close < 10MA

    # Let's use 20MA as the trailing stop for the main trend.
    exit_signal = close < close.average(20)

    # Construct Position
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    # Run Backtest
    report = backtest.sim(position, resample='D', name='VCP Strategy (Minervini)', upload=False)
    return report
