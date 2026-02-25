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
    # Master Data - Defines the Universe and Timeframe
    close = data.get('price:收盤價')

    # Master Index for alignment
    master_index = close.index
    master_columns = close.columns

    # Helper to align everything to (Time x Stocks) or (Time x 1)
    def to_numpy(obj, is_benchmark=False):
        if isinstance(obj, pd.Series):
            # Align index only
            obj = obj.reindex(master_index, method='ffill')
            return obj.fillna(0).values.reshape(-1, 1)
        elif isinstance(obj, pd.DataFrame):
            # Align index. We assume columns match 'close' for stock data.
            # If not, we should reindex columns too, but data.get usually returns full universe.
            # For safety, we can reindex columns too if it's stock data.
            if not is_benchmark:
                obj = obj.reindex(index=master_index, columns=master_columns, method='ffill')
            else:
                obj = obj.reindex(index=master_index, method='ffill')
            return obj.fillna(0).values
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(obj)

    # Fetch other price data
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # Benchmark
    benchmark_close = data.get('price:收盤價')['0050']

    # Fundamentals
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        rev_current = data.get('monthly_revenue:當月營收')
        capital = data.get('finance_statement:股本')
    except:
        rev_growth = pd.DataFrame(0, index=master_index, columns=master_columns)
        rev_current = pd.DataFrame(0, index=master_index, columns=master_columns)
        capital = pd.DataFrame(10000000, index=master_index, columns=master_columns)

    try:
        eps = data.get('finance_statement:每股盈餘')
        op_income = data.get('finance_statement:營業利益')
        roe = data.get('finance_statement:權益報酬率')
        op_margin = data.get('finance_statement:營業利益率')
        pe = data.get('price:本益比')
    except:
        eps = pd.DataFrame(0, index=master_index, columns=master_columns)
        op_income = pd.DataFrame(0, index=master_index, columns=master_columns)
        roe = pd.DataFrame(0, index=master_index, columns=master_columns)
        op_margin = pd.DataFrame(0, index=master_index, columns=master_columns)
        pe = pd.DataFrame(100, index=master_index, columns=master_columns)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except:
        inst_net_buy = pd.DataFrame(0, index=master_index, columns=master_columns)

    # ==========================================
    # 2. Pre-Calculation (Pandas)
    # ==========================================
    # We do rolling calculations in Pandas BEFORE converting to numpy
    # This is easier than implementing rolling in numpy

    # Moving Averages
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean() # Not used?
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # RSI
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    my_rsi = rsi(close, 14)

    # Fundamental Helpers
    rev_12m_max = rev_current.rolling(12).max()
    close_max_20 = close.rolling(20).max().shift(1)
    eps_sum = eps.rolling(4).sum()

    # Institutional Helpers
    inst_rolling_sum = inst_net_buy.rolling(3).sum()
    inst_streak = (inst_net_buy.rolling(5).min() > 0)

    # ==========================================
    # 3. NumPy Conversion (The Nuclear Option)
    # ==========================================
    print("[DEBUG] Converting to NumPy with defensive alignment...")

    # Prices
    v_close = to_numpy(close)
    v_open = to_numpy(open_)
    v_high = to_numpy(high) # Not used?
    v_low = to_numpy(low)
    v_vol = to_numpy(vol)

    # Indicators
    v_ma20 = to_numpy(ma20)
    v_ma60 = to_numpy(ma60)
    v_ma120 = to_numpy(ma120)
    v_vol_ma5 = to_numpy(vol_ma5)
    v_vol_ma20 = to_numpy(vol_ma20)
    v_rsi = to_numpy(my_rsi)
    v_close_max_20 = to_numpy(close_max_20)

    # Fundamentals
    v_rev_growth = to_numpy(rev_growth)
    v_rev_current = to_numpy(rev_current)
    v_rev_12m_max = to_numpy(rev_12m_max)
    v_capital = to_numpy(capital)
    v_eps_sum = to_numpy(eps_sum)
    v_op_income = to_numpy(op_income)
    v_pe = to_numpy(pe)
    v_roe = to_numpy(roe)
    v_op_margin = to_numpy(op_margin)

    # Inst
    v_inst_rolling = to_numpy(inst_rolling_sum)
    v_inst_streak = to_numpy(inst_streak)

    # Market (1D -> Broadcast)
    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    print(f"[DEBUG] Close Shape: {v_close.shape}")
    print(f"[DEBUG] Bench Shape: {v_bench.shape}")

    # ==========================================
    # 4. Logic Execution (Pure Math)
    # ==========================================

    # Market Masks
    # v_bench and v_bench_ma60 are (T, 1)
    v_bullish = (v_bench > v_bench_ma60 * 1.01)
    v_bearish = (v_bench < v_bench_ma60 * 0.99)

    # Liquidity Filter
    v_liq = (v_vol_ma20 > 1000000)

    # --- Signal A: Growth ---
    c_small = v_capital < 2000000 # Unit? Assuming Finlab units
    c_rev = (v_rev_growth > 30) | (v_rev_current >= v_rev_12m_max)
    c_profit = (v_eps_sum > 0) & (v_op_income > 0)
    c_value = (v_pe < 30) & (v_pe > 0)
    c_trend = (v_close > v_ma20) & (v_close > v_ma60)
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 1.5)

    sig_a = v_bullish & c_small & c_rev & c_profit & c_value & c_trend & c_breakout & v_liq
    print(f"[DEBUG] Signal A Triggers: {np.sum(sig_a)}")

    # --- Signal B: Reversion ---
    c_oversold = v_close < v_ma120
    c_rsi_panic = v_rsi < 20
    c_vol_panic = v_vol > v_vol_ma20 * 2

    body = np.abs(v_close - v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low
    c_hammer = lower_shadow > (body * 2)

    sig_b = v_bullish & c_oversold & c_rsi_panic & c_hammer & c_vol_panic & v_liq
    print(f"[DEBUG] Signal B Triggers: {np.sum(sig_b)}")

    # --- Signal C: Short ---
    c_weak = (v_close < v_ma60) & (v_close < v_ma20)
    bias = (v_close - v_ma20) / (v_ma20 + 1e-9) # Prevent div/0
    c_bias = bias > -0.10

    c_bad_rev = v_rev_growth < 0
    c_bad_profit = (v_eps_sum < 0) | (v_op_income < 0)
    c_bubble = v_pe > 60

    c_inst_sell = v_inst_rolling < 0
    c_bad_fund = c_bad_rev | c_bad_profit | c_bubble
    c_black = v_close < v_open

    sig_c = v_bearish & c_weak & c_bias & (c_bad_fund | c_inst_sell) & c_black & v_liq
    print(f"[DEBUG] Signal C Triggers: {np.sum(sig_c)}")

    # ==========================================
    # 5. Position Reconstruction
    # ==========================================

    long_entries = sig_a | sig_b
    short_entries = sig_c

    # Exit Conditions
    c_reversion_hold = (v_close < v_ma60) & (v_rsi < 50)

    long_exits = ((v_close < v_ma20) & (~c_reversion_hold)) | v_bearish
    short_exits = (v_close > v_ma20) | v_bullish | (v_rsi < 20)

    # Score
    score = np.ones_like(v_close)
    score += (v_rev_growth > 50).astype(int)
    score += ((v_roe > 20) | (v_op_margin > 20)).astype(int)
    score += v_inst_streak.astype(int)

    # Construct DF for Simulation
    df_long = pd.DataFrame(np.nan, index=master_index, columns=master_columns)
    df_short = pd.DataFrame(np.nan, index=master_index, columns=master_columns)

    # Apply logic
    # Note: numpy boolean arrays can be used to index DF if shapes match
    # But safer to just use values assignment

    # Initialize with NaNs
    v_pos_long = np.full_like(v_close, np.nan)
    v_pos_short = np.full_like(v_close, np.nan)

    # Entry
    v_pos_long[long_entries] = score[long_entries]
    v_pos_short[short_entries] = -0.5

    # Exit (Overwrite entry if exit coincides? Usually exit happens after entry.
    # But here we are defining signals.
    # Finlab backtest.sim takes a signal DF and holds until signal changes or becomes 0/NaN?
    # Actually Finlab backtest takes "Position Size" or "Signal".
    # If we return a signal that is NaN, it holds previous position?
    # No, typically 0 means exit, NaN means hold? Or NaN means no signal?
    # finlab.backtest.sim documentation says:
    # "entries are positive, exits are 0. NaN means hold."

    # So:
    # Entry: Set to Score (Weight)
    # Exit: Set to 0
    # Hold: Set to NaN

    # We must ensure Exit overrides Entry if both happen same day?
    # If I enter and exit same day, I probably shouldn't enter.
    # So Exits take precedence?
    # Or, Entry happens at Close, Exit happens at Close?
    # If Signal A triggers (Entry), and Exit condition triggers (Close < MA20).
    # Signal A requires Close > MA20. So Long Entry & Long Exit are mutually exclusive mostly.
    # Except v_bearish trigger.
    # If v_bearish is True, Signal A (v_bullish) is False.
    # So Signal A and Market Exit are mutually exclusive.

    # What about Signal B?
    # Signal B is Reversion (Close < MA120).
    # Exit is Close < MA20. (True)
    # But we added "Unless Reversion Hold".
    # If Signal B triggers, we are in Reversion Hold?
    # Signal B: v_rsi < 20.
    # Reversion Hold: v_rsi < 50.
    # So Signal B implies Reversion Hold.
    # So Exit is False.
    # So Entry passes.

    # So we are good.

    v_pos_long[long_exits] = 0
    v_pos_short[short_exits] = 0

    # Assign to DF
    df_long[:] = v_pos_long
    df_short[:] = v_pos_short

    # Ffill to simulate holding
    # "NaN means hold previous state"
    df_long = df_long.ffill().fillna(0)
    df_short = df_short.ffill().fillna(0)

    final_pos = df_long + df_short

    # Run Backtest
    print("[DEBUG] Running Simulation...")
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
