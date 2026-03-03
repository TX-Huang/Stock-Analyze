from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_long_strategy(api_token):
    from data_provider import sanitize_dataframe

    if api_token:
        finlab.login(api_token)

    # 1. Fetch Data
    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    high = sanitize_dataframe(data.get('price:最高價'), "FinLab_High")
    low = sanitize_dataframe(data.get('price:最低價'), "FinLab_Low")
    vol = sanitize_dataframe(data.get('price:成交股數'), "FinLab_Vol")

    # 2. Indicators
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    high250 = close.rolling(250).max().shift(1)
    low250 = close.rolling(250).min().shift(1)
    vol_avg = vol.rolling(20).mean()

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

    # Liquidity Filter (Added)
    is_liquid = vol_avg > 500000

    # --- New Conditions (Fundamental, Technical, Chip) ---

    # 1. Fundamental: Revenue Growth (Monthly Revenue YoY > 30%)
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        cond_rev = (rev_growth > 30).reindex(close.index, method='ffill').fillna(False)
    except:
        cond_rev = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 2. Technical: Lower Shadow Rejection (Lower shadow > 2 * Body, and Bullish Candle)
    try:
        open_price = data.get('price:開盤價')
        body = (close - open_price).abs()
        lower_shadow = (open_price.combine(close, min) - low)
        cond_shadow = (close > open_price) & (lower_shadow > body * 2)
    except:
        cond_shadow = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 3. Chip: Institutional Buying (Foreign + Investment Trust > 0)
    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數')
        trust_buy = data.get('institutional_investors:投信買賣超股數')
        cond_chip = (foreign_buy.fillna(0) + trust_buy.fillna(0)) > 0
    except:
        cond_chip = pd.DataFrame(False, index=close.index, columns=close.columns)

    # Enhanced Logic:
    base_condition = cond1 & cond2 & is_liquid & cond5 # Trend + Liquidity + 200MA up

    trigger_breakout = cond3 & cond4 & cond6 & cond7 # The original breakout logic
    trigger_special = (cond_rev | cond_shadow | cond_chip) & (close > ma20) # Special signals in an uptrend

    buy_signal = base_condition & (trigger_breakout | (trigger_special & (close >= high250 * 0.9)))

    # 4. Long Exit Condition
    exit_signal = close < ma50

    # 5. Position Construction
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    # Run Backtest
    import logging
    import os
    from data_provider import safe_finlab_sim
    logging.basicConfig(filename="finlab_debug.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    try:
        report = safe_finlab_sim(position, resample='D', name='純做多策略', upload=False)
        return report
    except Exception as e:
        logging.error(f"策略層級崩潰: {str(e)}", exc_info=True)
        raise e
