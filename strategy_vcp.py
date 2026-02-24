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
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    high250 = close.rolling(250).max().shift(1)
    low250 = close.rolling(250).min().shift(1)
    vol_avg = vol.rolling(50).mean()

    # 3. Mark Minervini's Trend Template
    cond1 = close > ma150
    cond2 = ma150 > ma200
    cond3 = ma200 > ma200.shift(20)
    cond4 = ma50 > ma150
    cond5 = close > ma50

    # 4. Price Structure
    cond6 = close > (low250 * 1.30)
    cond7 = close >= (high250 * 0.75)

    # 5. Volatility Contraction & Dry Up (VCP Essence)
    std = close.rolling(20).std()
    upper = close.rolling(20).mean() + 2 * std
    lower = close.rolling(20).mean() - 2 * std
    bandwidth = (upper - lower) / close.rolling(20).mean()

    # Contraction: Bandwidth is in the lowest 25% of the last 60 days
    is_contracting = bandwidth < bandwidth.rolling(60).quantile(0.25)

    # VCP Dry Up: Volume MUST dry up before breakout.
    # Check if ANY day in the last 10 days had volume < 50% of 50-day average.
    vol_50ma = vol.rolling(50).mean()
    is_dry_up = (vol < vol_50ma * 0.5).rolling(10).max() > 0

    # 6. SMC Structure (Higher Lows)
    # The lowest low of the recent 20 days must be higher than the lowest low of the previous 60 days.
    # This confirms a "Higher Low" structure (Bullish Trend Preservation).
    low_recent = low.rolling(20).min()
    low_past = low.shift(20).rolling(60).min()
    is_higher_low = low_recent > low_past

    # 7. Relative Strength
    stock_ret = close.pct_change(120)
    market_ret = benchmark.pct_change(120)
    rs_rating = stock_ret.sub(market_ret, axis=0) > 0

    # 8. Liquidity Filter
    is_liquid = vol_avg > 500000

    # --- Advanced Conditions (Fundamental, Technical, Chip) ---
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        cond_rev = (rev_growth > 30).reindex(close.index, method='ffill').fillna(False)
    except: cond_rev = pd.DataFrame(False, index=close.index, columns=close.columns)

    body = (close - data.get('price:開盤價')).abs()
    lower_shadow = (data.get('price:開盤價').combine(close, min) - low)
    cond_shadow = (close > data.get('price:開盤價')) & (lower_shadow > body * 2)

    # Institutional Lock-in (Smart Money)
    # Check if Foreign or Trust are NET BUYING during the contraction period (last 10 days avg > 0)
    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy_10d = (foreign_buy + trust_buy).rolling(10).mean()

        # Original chip condition (daily spike)
        cond_chip_spike = (foreign_buy + trust_buy) > 0

        # New: Institutional Lock-in during quiet period
        is_inst_locked_in = inst_net_buy_10d > 0
    except:
        cond_chip_spike = pd.DataFrame(False, index=close.index, columns=close.columns)
        is_inst_locked_in = pd.DataFrame(False, index=close.index, columns=close.columns)

    # 9. Trigger (Breakout)
    # Standard VCP Breakout: Price break 20-day high + Volume spike
    breakout_std = (close > close.rolling(20).max().shift(1)) & (vol > vol.rolling(20).mean() * 1.5)

    # Base Setup: All Minervini Trend Templates + VCP Specifics + SMC Structure
    # Added: is_dry_up (VCP soul), is_higher_low (SMC soul)
    base_setup = (cond1 & cond2 & cond3 & cond4 & cond5 &
                  cond6 & cond7 &
                  is_contracting & is_dry_up & is_higher_low &
                  rs_rating & is_liquid)

    # Trigger Logic:
    # 1. Standard VCP Breakout
    # 2. "Lock-in" Breakout: If institutions are locked in, we accept a smaller breakout volume or price move
    # 3. Special Enhancers: Revenue or Shadow Rejection

    trigger_lock_in = is_inst_locked_in & (close > close.rolling(10).max().shift(1)) & (vol > vol.rolling(20).mean())
    trigger_special = (cond_rev | cond_shadow | cond_chip_spike) & (close > close.rolling(10).max().shift(1))

    final_trigger = breakout_std | trigger_lock_in | trigger_special

    buy_signal = base_setup & final_trigger

    # 9. Exit Strategy (Optimized)
    # Relaxed from 20MA to 50MA to allow trend to ride
    exit_signal = close < ma50

    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[exit_signal] = 0
    position = position.ffill().fillna(0)

    report = backtest.sim(position, resample='D', name='VCP Strategy (Minervini)', upload=False)
    return report
