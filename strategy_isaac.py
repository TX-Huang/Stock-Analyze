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

    # Financials for Fundamental Shield (Quality/Value)
    try:
        # Quality: EPS & Operating Income
        eps = data.get('finance_statement:每股盈餘')
        op_income = data.get('finance_statement:營業利益')

        # Dynamic Sizing: ROE & Operating Margin
        roe = data.get('finance_statement:權益報酬率')
        op_margin = data.get('finance_statement:營業利益率')

        # Value: PE Ratio
        pe = data.get('price:本益比')
    except:
        eps = pd.DataFrame(0, index=close.index, columns=close.columns)
        op_income = pd.DataFrame(0, index=close.index, columns=close.columns)
        roe = pd.DataFrame(0, index=close.index, columns=close.columns)
        op_margin = pd.DataFrame(0, index=close.index, columns=close.columns)
        pe = pd.DataFrame(100, index=close.index, columns=close.columns)

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

    # A3. Fundamental Shield (Quality & Value) - NEW!
    # Quality: Profitable (Sum of last 4Q EPS > 0) AND Core Business Profitable (Op Income > 0)
    cond_profitable = (eps.rolling(4).sum() > 0) & (op_income > 0)
    cond_profitable = cond_profitable.reindex(close.index, method='ffill').fillna(False)

    # Value: Not too expensive (PE < 30)
    cond_value_safe = (pe < 30) & (pe > 0) # PE > 0 implies earnings > 0 too
    cond_value_safe = cond_value_safe.reindex(close.index, method='ffill').fillna(False)

    # A4. Technical Trend (VCP-lite)
    cond_trend = (close > ma20) & (close > ma60)

    # A5. VCP Dry Up
    is_dry_up = (vol < vol_ma20 * 0.5).rolling(5).max() > 0

    # A6. Breakout
    breakout = (close > close.rolling(20).max().shift(1)) & (vol > vol_ma5 * 1.5)

    # SIGNAL A TRIGGER (Only in Bull Market)
    # Added Shield: cond_profitable & cond_value_safe
    signal_a = (
        is_market_bullish &
        cond_small_cap &
        cond_rev_strong &
        cond_profitable &
        cond_value_safe &
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

    # C3. Fundamental/Chip Weakness (The "Rotten Core" Filter)
    # Revenue Contraction (YoY < 0)
    cond_bad_rev = (rev_growth < 0).reindex(close.index, method='ffill').fillna(False)

    # Profitability Issues: Negative EPS (Sum 4Q < 0) OR Core Business Loss (Op Income < 0)
    cond_unprofitable = (eps.rolling(4).sum() < 0) | (op_income < 0)
    cond_unprofitable = cond_unprofitable.reindex(close.index, method='ffill').fillna(False)

    # Valuation Bubble: PE > 60 (Extreme Overvaluation)
    cond_bubble = (pe > 60).reindex(close.index, method='ffill').fillna(False)

    # Chip Weakness: Institutional Net Selling
    cond_inst_sell = inst_net_buy.rolling(3).sum() < 0

    # Combine Weakness Factors: At least one major flaw (Bad Rev, No Profit, Bubble) + Inst Selling
    # Or just Bad Fundamentals overall
    cond_fundamental_weakness = cond_bad_rev | cond_unprofitable | cond_bubble

    # C4. Rebound Failure (Black Candle eating previous gains? Simplified to Price < Open)
    cond_black_candle = close < open_

    # SIGNAL C TRIGGER (Only in Bear Market)
    # Trigger: Bear Market + Weak Structure + (Fundamental Rot OR Inst Selling) + Black Candle
    signal_c_short = (
        is_market_bearish &
        cond_weak_structure &
        cond_bias_ok &
        (cond_fundamental_weakness | cond_inst_sell) &
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

    # ==========================================
    # 6. Dynamic Position Sizing (The "Bet Heavy" Logic)
    # ==========================================

    # Scoring System for Long Positions
    # Base Score = 1
    score = pd.DataFrame(1, index=close.index, columns=close.columns)

    # Bonus 1: Revenue Explosion (YoY > 50%) -> +1
    cond_rev_super = (rev_growth > 50).reindex(close.index, method='ffill').fillna(False)
    score[cond_rev_super] += 1

    # Bonus 2: Super Profitability (ROE > 20% OR Op Margin > 20%) -> +1
    cond_super_profit = ((roe > 20) | (op_margin > 20)).reindex(close.index, method='ffill').fillna(False)
    score[cond_super_profit] += 1

    # Bonus 3: Institutional Conviction (Net Buy > 0 for 5 days) -> +1
    cond_inst_conviction = (inst_net_buy.rolling(5).min() > 0)
    score[cond_inst_conviction] += 1

    # Cap score at 4? (1+1+1+1 = 4).
    # Finlab backtest uses the value in position dataframe as weight relative to portfolio.
    # If we put 4, and another stock has 1, the first gets 4x allocation of the second.
    # Total portfolio is always 100% invested (if there are stocks).

    # Construct Long Position with Weights
    pos_long = pd.DataFrame(np.nan, index=long_entries.index, columns=long_entries.columns)

    # Initialize entries with calculated score
    pos_long[long_entries] = score[long_entries]

    # Handle Exits (Set to 0)
    pos_long[long_exits] = 0

    # Fill forward: Hold position with the initial entry score?
    # Ideally, we want to hold. ffill() propagates the last valid observation.
    pos_long = pos_long.ffill().fillna(0)

    # Construct Short Position (Fixed 0.5 weight for risk control)
    pos_short = pd.DataFrame(np.nan, index=short_entries.index, columns=short_entries.columns)
    pos_short[short_entries] = -0.5
    pos_short[short_exits] = 0
    pos_short = pos_short.ffill().fillna(0)

    # Combine (Net Position)
    final_pos = pos_long + pos_short

    # Liquidity Filter
    # Set weight to 0 if liquidity is too low (effectively filtering it out)
    liq_filter = (vol.average(20) > 1000000)
    final_pos = final_pos * liq_filter # Zero out illiquid stocks

    # Run Backtest
    report = backtest.sim(final_pos, resample='D', name='Isaac Strategy (All-Weather)', upload=False)
    return report
