from finlab import data
from finlab import backtest
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
import finlab

def run_long_short_strategy(api_token):
    from data.provider import sanitize_dataframe
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

    daily_range = (high - low).replace(0, np.nan)
    close_pos = ((close - low) / daily_range).fillna(0)

    # Liquidity
    is_liquid = vol_avg > 500000

    # --- New Conditions (Fundamental, Technical, Chip) ---
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        cond_rev = (rev_growth > 30).reindex(close.index).ffill().fillna(False)
    except Exception: cond_rev = pd.DataFrame(False, index=close.index, columns=close.columns)

    try:
        open_price = data.get('price:開盤價')
        body = (close - open_price).abs()
        lower_shadow = (open_price.combine(close, min) - low)
        cond_shadow = (close > open_price) & (lower_shadow > body * 2)
    except Exception:
        cond_shadow = pd.DataFrame(False, index=close.index, columns=close.columns)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數')
        trust_buy = data.get('institutional_investors:投信買賣超股數')
        cond_chip = (foreign_buy.fillna(0) + trust_buy.fillna(0)) > 0
    except Exception: cond_chip = pd.DataFrame(False, index=close.index, columns=close.columns)

    # --- LONG LOGIC ---
    l_cond1 = (ma50 > ma150) & (ma150 > ma200)
    l_cond2 = (close > ma50) & (close > ma150) & (close > ma200)
    l_cond3 = close > (low250 * 1.30)
    l_cond4 = (close >= high250 * 0.85) & (close <= high250 * 1.05)
    l_cond5 = ma200 > ma200.shift(20)
    l_cond6 = vol > (vol_avg * 1.5)
    l_cond7 = close_pos >= 0.75

    # Enhanced Long Buy
    long_base = l_cond1 & l_cond2 & l_cond5 & is_liquid
    long_trigger_std = l_cond3 & l_cond4 & l_cond6 & l_cond7
    long_trigger_special = (cond_rev | cond_shadow | cond_chip) & (close > ma20)

    buy_signal = long_base & (long_trigger_std | (long_trigger_special & (close >= high250 * 0.9)))
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

    import logging
    import os
    from data.provider import safe_finlab_sim
    logging.basicConfig(filename="finlab_debug.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    try:
        report = safe_finlab_sim(final_pos, resample='D', name='多空策略', upload=False)
        return report
    except Exception as e:
        logging.error(f"策略層級崩潰: {str(e)}", exc_info=True)
        raise e
