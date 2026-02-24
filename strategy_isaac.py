from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_isaac_strategy(api_token, stop_loss=None, take_profit=None):
    if api_token:
        finlab.login(api_token)

    if stop_loss is not None: stop_loss = float(stop_loss)
    if take_profit is not None: take_profit = float(take_profit)

    # ==========================================
    # 1. Fetch Data
    # ==========================================
    close = data.get('price:收盤價')
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    benchmark_close = data.get('price:收盤價')['0050']

    print(f"[DEBUG] Close Shape: {close.shape}")
    print(f"[DEBUG] Benchmark Shape: {benchmark_close.shape}")

    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        rev_current = data.get('monthly_revenue:當月營收')
        capital = data.get('finance_statement:股本')
    except:
        rev_growth = pd.DataFrame(0, index=close.index, columns=close.columns)
        rev_current = pd.DataFrame(0, index=close.index, columns=close.columns)
        capital = pd.DataFrame(10000000, index=close.index, columns=close.columns)

    try:
        eps = data.get('finance_statement:每股盈餘')
        op_income = data.get('finance_statement:營業利益')
        roe = data.get('finance_statement:權益報酬率')
        op_margin = data.get('finance_statement:營業利益率')
        pe = data.get('price:本益比')
    except:
        eps = pd.DataFrame(0, index=close.index, columns=close.columns)
        op_income = pd.DataFrame(0, index=close.index, columns=close.columns)
        roe = pd.DataFrame(0, index=close.index, columns=close.columns)
        op_margin = pd.DataFrame(0, index=close.index, columns=close.columns)
        pe = pd.DataFrame(100, index=close.index, columns=close.columns)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except:
        inst_net_buy = pd.DataFrame(0, index=close.index, columns=close.columns)

    # ==========================================
    # 2. Indicators (Pandas Calculation)
    # ==========================================
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    my_rsi = rsi(close, 14)

    bench_ma60 = benchmark_close.rolling(60).mean()

    # Pre-calculate Helpers
    rev_12m_max = rev_current.rolling(12).max()
    close_max_20 = close.rolling(20).max().shift(1)

    # Fill NaNs for safety
    def safe_df(df):
        return df.reindex(close.index, method='ffill').fillna(0)

    rev_growth = safe_df(rev_growth)
    rev_current = safe_df(rev_current)
    rev_12m_max = safe_df(rev_12m_max)
    capital = safe_df(capital)
    eps_sum = safe_df(eps.rolling(4).sum())
    op_income = safe_df(op_income)
    roe = safe_df(roe)
    op_margin = safe_df(op_margin)
    pe = safe_df(pe)
    inst_net_buy = safe_df(inst_net_buy)

    # ==========================================
    # 3. NumPy Extraction (The Nuclear Option)
    # ==========================================
    # Extract values to bypass Pandas index alignment checks
    v_close = close.values
    v_open = open_.values
    v_low = low.values
    v_vol = vol.values

    v_ma20 = ma20.values
    v_ma60 = ma60.values
    v_ma120 = ma120.values
    v_vol_ma5 = vol_ma5.values
    v_vol_ma20 = vol_ma20.values
    v_rsi = my_rsi.values

    v_rev_growth = rev_growth.values
    v_rev_current = rev_current.values
    v_rev_12m_max = rev_12m_max.values
    v_capital = capital.values
    v_eps_sum = eps_sum.values
    v_op_income = op_income.values
    v_pe = pe.values
    v_roe = roe.values
    v_op_margin = op_margin.values
    v_inst_net_buy = inst_net_buy.values
    v_close_max_20 = close_max_20.values

    # Market Trend Broadcast (Manually)
    v_bench = benchmark_close.values
    v_bench_ma60 = bench_ma60.values

    # Create Market Masks (Shape: Time x 1) -> Broadcast to (Time x Stocks)
    # Reshape to (N, 1) for broadcasting
    v_bullish = (v_bench > v_bench_ma60 * 1.01).reshape(-1, 1)
    v_bearish = (v_bench < v_bench_ma60 * 0.99).reshape(-1, 1)

    print(f"[DEBUG] Market Bullish Days: {np.sum(v_bullish)}")
    print(f"[DEBUG] Market Bearish Days: {np.sum(v_bearish)}")

    # ==========================================
    # 4. Logic Execution (Pure Math)
    # ==========================================

    # --- Signal A: Growth ---
    # Small Cap
    c_small = v_capital < 2000000
    # Rev Strong
    c_rev = (v_rev_growth > 30) | (v_rev_current >= v_rev_12m_max)
    # Profitable
    c_profit = (v_eps_sum > 0) & (v_op_income > 0)
    # Value Safe
    c_value = (v_pe < 30) & (v_pe > 0)
    # Trend
    c_trend = (v_close > v_ma20) & (v_close > v_ma60)
    # Breakout
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 1.5)

    sig_a = v_bullish & c_small & c_rev & c_profit & c_value & c_trend & c_breakout
    print(f"[DEBUG] Signal A Triggers: {np.sum(sig_a)}")

    # --- Signal B: Reversion ---
    c_oversold = v_close < v_ma120
    c_rsi_panic = v_rsi < 20
    c_vol_panic = v_vol > v_vol_ma20 * 2

    # Hammer Logic
    body = np.abs(v_close - v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low
    c_hammer = lower_shadow > (body * 2)

    sig_b = v_bullish & c_oversold & c_rsi_panic & c_hammer & c_vol_panic
    print(f"[DEBUG] Signal B Triggers: {np.sum(sig_b)}")

    # --- Signal C: Short ---
    c_weak = (v_close < v_ma60) & (v_close < v_ma20)
    bias = (v_close - v_ma20) / v_ma20
    c_bias = bias > -0.10

    # Short Filters
    c_bad_rev = v_rev_growth < 0
    c_bad_profit = (v_eps_sum < 0) | (v_op_income < 0)
    c_bubble = v_pe > 60

    # Inst Sell logic (Need rolling sum of numpy array? Or just use pre-calc DF)
    # To keep it simple in numpy, assume inst_net_buy is already rolling sum?
    # No, it was daily. Let's pre-calc rolling sum in Pandas first.
    # Re-fetch rolling inst
    inst_rolling = inst_net_buy.rolling(3).sum().fillna(0).values
    c_inst_sell = inst_rolling < 0

    c_bad_fund = c_bad_rev | c_bad_profit | c_bubble
    c_black = v_close < v_open

    sig_c = v_bearish & c_weak & c_bias & (c_bad_fund | c_inst_sell) & c_black
    print(f"[DEBUG] Signal C Triggers: {np.sum(sig_c)}")

    # ==========================================
    # 5. Position Reconstruction
    # ==========================================

    # Combine signals
    long_entries = sig_a | sig_b
    short_entries = sig_c

    # Exits
    # is_deep_value = (close < ma120) & (my_rsi < 40)
    c_deep = (v_close < v_ma120) & (v_rsi < 40)

    # Long Exit: Close < 20MA OR Bearish, unless Deep Value
    long_exits = ((v_close < v_ma20) | v_bearish) & (~c_deep)

    # Short Exit: Close > 20MA OR Bullish OR RSI < 20
    short_exits = (v_close > v_ma20) | v_bullish | (v_rsi < 20)

    # --- Scoring (Numpy) ---
    score = np.ones_like(v_close)
    score += (v_rev_growth > 50).astype(int)
    score += ((v_roe > 20) | (v_op_margin > 20)).astype(int)

    # Inst Streak (Rolling min > 0)
    # Pre-calc in Pandas
    inst_streak = (inst_net_buy.rolling(5).min() > 0).fillna(False).values
    score += inst_streak.astype(int)

    # --- Final Logic Loop (Simulation) ---
    # We need to simulate holding because vectorized ffill on signals is tricky with multiple states
    # But for Finlab compatibility, we usually return a DataFrame of target weights.

    # Vectorized Position Construction
    # 1. Create Entry Signal Mask (Weight)
    # 2. Create Exit Signal Mask (0)
    # 3. Fill forward

    # To mix Long/Short in one DF:
    # Use 1 for Long, -0.5 for Short

    target_pos = np.zeros_like(v_close)

    # Simple state machine simulation is hard in pure numpy without loop.
    # But Pandas ffill works well. Let's put back to Pandas for ffill.

    df_long = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    df_short = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # Map Numpy back to DF
    # We use boolean indexing on DF directly now that logic is clean
    # Actually, we can just assign based on numpy bool arrays

    df_long[:] = np.where(long_entries, score, np.nan)
    df_long[long_exits] = 0
    df_long = df_long.ffill().fillna(0)

    df_short[:] = np.where(short_entries, -0.5, np.nan)
    df_short[short_exits] = 0
    df_short = df_short.ffill().fillna(0)

    final_pos = df_long + df_short

    # Liquidity Filter
    v_liq = (v_vol_ma20 > 1000000)
    # Apply filter (Zero out if illiquid)
    final_pos = final_pos * v_liq

    # Run Backtest
    if stop_loss is not None or take_profit is not None:
        report = backtest.sim(
            final_pos,
            resample='D',
            name='Isaac Strategy (Stress Test)',
            upload=False,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    else:
        report = backtest.sim(final_pos, resample='D', name='Isaac Strategy (All-Weather)', upload=False)

    return report
