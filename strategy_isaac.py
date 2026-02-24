from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_isaac_strategy(api_token):
    if api_token:
        finlab.login(api_token)

    # ==========================================
    # 1. Fetch Data
    # ==========================================
    close = data.get('price:收盤價')
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # Benchmark for Market Filter (0050 ETF as proxy for TAIEX)
    benchmark_close = data.get('price:收盤價')['0050']

    # Financials for Small-Cap Strategy
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)') # Monthly Revenue YoY
        rev_current = data.get('monthly_revenue:當月營收')
        capital = data.get('finance_statement:股本') # Capital stock
    except:
        # Fallback if fundamental data missing (use price only)
        rev_growth = pd.DataFrame(0, index=close.index, columns=close.columns)
        rev_current = pd.DataFrame(0, index=close.index, columns=close.columns)
        capital = pd.DataFrame(10000000, index=close.index, columns=close.columns) # Dummy large capital

    # Institutional Data for Short Strategy
    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except:
        inst_net_buy = pd.DataFrame(0, index=close.index, columns=close.columns)

    # ==========================================
    # 2. Indicators & Market Filter
    # ==========================================

    # Moving Averages
    ma20 = close.average(20)
    ma50 = close.average(50) # Quarterly Line (Trend Definition)
    ma60 = close.average(60)
    ma120 = close.average(120) # Half-Year Line

    # Volume MA
    vol_ma5 = vol.average(5)
    vol_ma20 = vol.average(20)

    # RSI
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    my_rsi = rsi(close, 14)

    # Market Regime Filter (Red Light / Green Light)
    # Strategy works best when Market > 60MA (Quarterly)
    # 1% Buffer Zone:
    # Bull > 1.01 * 60MA
    # Bear < 0.99 * 60MA
    bench_ma60 = benchmark_close.average(60)
    is_market_bullish = (benchmark_close > bench_ma60 * 1.01)
    is_market_bearish = (benchmark_close < bench_ma60 * 0.99)

    # ==========================================
    # 3. Signal A: Small-Cap Revenue Surprise (Aggressive Growth)
    # ==========================================

    # A1. Small Capital (< 20 Billion TWD = 200 億 => 2,000,000 k)
    cond_small_cap = capital < 2000000

    # A2. Revenue Explosion
    rev_12m_max = rev_current.rolling(12).max()
    cond_rev_strong = (rev_growth > 30) | (rev_current >= rev_12m_max)
    cond_rev_strong = cond_rev_strong.reindex(close.index, method='ffill').fillna(False)

    # A3. Technical Trend (VCP-lite)
    cond_trend = (close > ma20) & (close > ma60)

    # A4. VCP Dry Up
    is_dry_up = (vol < vol_ma20 * 0.5).rolling(5).max() > 0

    # A5. Breakout
    breakout = (close > close.rolling(20).max().shift(1)) & (vol > vol_ma5 * 1.5)

    # SIGNAL A TRIGGER (Only in Bull Market)
    signal_a = (
        is_market_bullish &
        cond_small_cap &
        cond_rev_strong &
        cond_trend &
        breakout
    )

    # ==========================================
    # 4. Signal B: Mean Reversion (Deep Value / Panic Buy)
    # ==========================================

    # B1. Deep Discount
    cond_oversold_trend = close < ma120

    # B2. RSI Extreme (< 20)
    cond_rsi_panic = my_rsi < 20

    # B3. Reversal Candle (Hammer)
    body = (close - open_).abs()
    lower_shadow = (close.combine(open_, min) - low)
    cond_hammer = lower_shadow > (body * 2)

    # B4. Volume Spike
    cond_vol_panic = vol > (vol_ma20 * 2)

    # SIGNAL B TRIGGER (Bull Market Only - catching dips)
    # Actually, Mean Reversion works best when Market is Bullish but Stock is temporarily dead.
    # In Bear Market, buying dips is dangerous (catching falling knives).
    signal_b = (
        is_market_bullish &
        cond_oversold_trend &
        cond_rsi_panic &
        cond_hammer &
        cond_vol_panic
    )

    # ==========================================
    # 5. Signal C: Bear Hunter (Short Selling)
    # ==========================================

    # C1. Weak Structure
    # Price < 60MA AND Price < 20MA
    cond_weak_structure = (close < ma60) & (close < ma20)

    # C2. Bias Filter (Not too oversold)
    # Bias = (Price - 20MA) / 20MA
    # We want to short when Price is close to MA (Bias > -10%), not when it's already crashed (-30%)
    bias_20 = (close - ma20) / ma20
    cond_bias_ok = bias_20 > -0.10

    # C3. Fundamental/Chip Weakness
    # Revenue Contraction (YoY < 0) OR Heavy Inst Selling
    cond_bad_fund = (rev_growth < 0).reindex(close.index, method='ffill').fillna(False)
    cond_inst_sell = inst_net_buy.rolling(3).sum() < 0 # Net sell over 3 days

    # C4. Rebound Failure (Black Candle eating previous gains? Simplified to Price < Open)
    cond_black_candle = close < open_

    # SIGNAL C TRIGGER (Only in Bear Market)
    signal_c_short = (
        is_market_bearish &
        cond_weak_structure &
        cond_bias_ok &
        (cond_bad_fund | cond_inst_sell) &
        cond_black_candle
    )

    # ==========================================
    # 6. Position Management
    # ==========================================

    # Long Entries
    long_entries = signal_a | signal_b

    # Long Exits
    # 1. Trend Break: Close < 20MA (Aggressive)
    # 2. Market Bearish Turn: Benchmark < 60MA
    # Exception: Deep Value Hold (Signal B)
    is_deep_value = (close < ma120) & (my_rsi < 40)
    long_exits = ((close < ma20) | is_market_bearish) & (~is_deep_value)

    # Short Entries
    short_entries = signal_c_short

    # Short Exits (Cover)
    # 1. Trend Reversal: Close > 20MA
    # 2. Market Bullish Turn: Benchmark > 60MA
    # 3. Profit Taking: RSI < 20 (Oversold)
    short_exits = (close > ma20) | is_market_bullish | (my_rsi < 20)

    # Construct Long Position
    pos_long = pd.DataFrame(np.nan, index=long_entries.index, columns=long_entries.columns)
    pos_long[long_entries] = 1
    pos_long[long_exits] = 0
    pos_long = pos_long.ffill().fillna(0)

    # Construct Short Position
    pos_short = pd.DataFrame(np.nan, index=short_entries.index, columns=short_entries.columns)
    pos_short[short_entries] = -0.5 # Half position for shorts (Risk Control)
    pos_short[short_exits] = 0
    pos_short = pos_short.ffill().fillna(0)

    # Combine (Net Position)
    # Note: A stock can't be Long and Short at same time due to Market Filter mutual exclusion.
    final_pos = pos_long + pos_short

    # Liquidity Filter
    final_pos = final_pos & (vol.average(20) > 1000000)

    # Run Backtest
    report = backtest.sim(final_pos, resample='D', name='Isaac Strategy (All-Weather)', upload=False)
    return report
