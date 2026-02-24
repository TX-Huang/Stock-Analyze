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

    # --- New Conditions (Fundamental, Technical, Chip) ---

    # 1. Fundamental: Revenue Growth (Monthly Revenue YoY > 30%)
    # Note: Requires 'monthly_revenue:當月營收' and 'monthly_revenue:去年同月增減(%)'
    # Fallback to simple price momentum if revenue data is missing or use price as proxy for strong fundamentals
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        cond_rev = rev_growth > 30
        cond_rev = cond_rev.reindex(close.index, method='ffill').fillna(False)
    except:
        cond_rev = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 2. Technical: Lower Shadow Rejection (Lower shadow > 2 * Body, and Bullish Candle)
    body = (close - data.get('price:開盤價')).abs()
    lower_shadow = (data.get('price:開盤價').combine(close, min) - low)
    cond_shadow = (close > data.get('price:開盤價')) & (lower_shadow > body * 2)

    # 3. Chip: Institutional Buying (Foreign + Investment Trust > 0)
    # Using 'institutional_investors:外資買賣超股數' and 'institutional_investors:投信買賣超股數'
    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數')
        trust_buy = data.get('institutional_investors:投信買賣超股數')
        cond_chip = (foreign_buy.fillna(0) + trust_buy.fillna(0)) > 0
    except:
        cond_chip = pd.DataFrame(False, index=close.index, columns=close.columns)

    # Combine New Conditions: At least one strong signal OR just stick to technical breakout
    # We use them as "enhancers" - if the basic technical setup is met, having these is a bonus.
    # To incorporate them into the "buy_signal", we can say:
    # (Basic Tech Setup) AND ( (Strong Revenue) OR (Shadow Rejection) OR (Chip Support) OR (Standard Breakout) )
    # But since the user wants to "fuse" them, let's treat them as valid triggers if they occur near the breakout zone.

    # Enhanced Logic:
    # Base = Trend Alignment (cond1 & cond2 & is_liquid)
    # Trigger = (Breakout & Vol) OR (Shadow Rejection & Chip) OR (Revenue & Breakout)

    base_condition = cond1 & cond2 & is_liquid & cond5 # Trend + Liquidity + 200MA up

    trigger_breakout = cond3 & cond4 & cond6 & cond7 # The original breakout logic
    trigger_special = (cond_rev | cond_shadow | cond_chip) & (close > ma20) # Special signals in an uptrend

    # Final Buy: Base Condition AND (Original Breakout OR Special Signal near Highs)
    # To be safe and meet user request, we simply AND the original buy_signal with the OR of new factors?
    # No, that would be too strict. User said "fuse conditions".
    # Let's add them as alternative strong entry points if the trend is right.

    buy_signal = base_condition & (trigger_breakout | (trigger_special & (close >= high250 * 0.9)))

    # Liquidity Filter (Optimized)
    # is_liquid is already in base_condition

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
