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
    # 1. 數據抓取 (Fetch Data)
    # ==========================================
    # Master Data - 定義宇宙與時間軸
    close = data.get('price:收盤價')

    # [Fix 3: Immutability]
    # Do NOT modify close.columns in place, as it might corrupt Finlab's global cache.
    # Instead, define a separate string index for internal alignment.
    master_index = close.index
    master_columns_str = close.columns.astype(str) # For internal Safe Alignment

    # We will use this to reindex inputs.

    # 輔助函數：將所有資料對齊到 (時間 x 股票) 或 (時間 x 1)，並轉為 NumPy 陣列
    def to_numpy(obj, is_benchmark=False):
        if isinstance(obj, pd.Series):
            # 僅對齊索引
            obj = obj.reindex(master_index, method='ffill')
            return obj.fillna(0).values.reshape(-1, 1)
        elif isinstance(obj, pd.DataFrame):
            # [Fix 3]: Work on a copy to avoid side effects
            # If the input dataframe (e.g. from data.get) has Categorical columns,
            # we must convert them to string to match `master_columns_str` for reindexing.

            # Check if we need to align columns
            if not is_benchmark:
                # We use a copy-based approach for safety
                df_temp = obj.copy()
                df_temp.columns = df_temp.columns.astype(str)

                # Now reindex using the Safe String Index
                df_aligned = df_temp.reindex(index=master_index, columns=master_columns_str, method='ffill')
                return df_aligned.fillna(0).values
            else:
                obj = obj.reindex(index=master_index, method='ffill')
                return obj.fillna(0).values
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(obj)

    # 抓取其他價格數據
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # 大盤基準
    benchmark_close = data.get('price:收盤價')['0050']

    # 基本面數據
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        rev_current = data.get('monthly_revenue:當月營收')
        capital = data.get('finance_statement:股本')
    except:
        rev_growth = pd.DataFrame(0, index=master_index, columns=close.columns)
        rev_current = pd.DataFrame(0, index=master_index, columns=close.columns)
        capital = pd.DataFrame(10000000, index=master_index, columns=close.columns)

    try:
        eps = data.get('finance_statement:每股盈餘')
        op_income = data.get('finance_statement:營業利益')
        roe = data.get('finance_statement:權益報酬率')
        op_margin = data.get('finance_statement:營業利益率')
        pe = data.get('price:本益比')
    except:
        eps = pd.DataFrame(0, index=master_index, columns=close.columns)
        op_income = pd.DataFrame(0, index=master_index, columns=close.columns)
        roe = pd.DataFrame(0, index=master_index, columns=close.columns)
        op_margin = pd.DataFrame(0, index=master_index, columns=close.columns)
        pe = pd.DataFrame(100, index=master_index, columns=close.columns)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except:
        inst_net_buy = pd.DataFrame(0, index=master_index, columns=close.columns)

    # ==========================================
    # 2. 預計算 (Pandas 階段)
    # ==========================================

    # 移動平均線
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # RSI 指標
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    my_rsi = rsi(close, 14)

    # 基本面輔助指標
    rev_12m_max = rev_current.rolling(12).max()
    close_max_20 = close.rolling(20).max().shift(1)
    eps_sum = eps.rolling(4).sum()

    # 籌碼面輔助指標
    inst_rolling_sum = inst_net_buy.rolling(3).sum()
    inst_streak = (inst_net_buy.rolling(5).min() > 0)

    # ==========================================
    # 3. NumPy 轉換 (核心加速與防禦)
    # ==========================================

    # 價格數據
    v_close = to_numpy(close)
    v_open = to_numpy(open_)
    v_high = to_numpy(high)
    v_low = to_numpy(low)
    v_vol = to_numpy(vol)

    # 技術指標
    v_ma20 = to_numpy(ma20)
    v_ma60 = to_numpy(ma60)
    v_ma120 = to_numpy(ma120)
    v_vol_ma5 = to_numpy(vol_ma5)
    v_vol_ma20 = to_numpy(vol_ma20)
    v_rsi = to_numpy(my_rsi)
    v_close_max_20 = to_numpy(close_max_20)

    # 計算歷史重要高點 (Supply Zone Proxy)
    # 使用 Rolling Max 250 天 (約一年)
    high_250 = close.rolling(250).max().shift(1)
    v_high_250 = to_numpy(high_250)

    # 基本面
    v_rev_growth = to_numpy(rev_growth)
    v_rev_current = to_numpy(rev_current)
    v_rev_12m_max = to_numpy(rev_12m_max)
    v_capital = to_numpy(capital)
    v_eps_sum = to_numpy(eps_sum)
    v_op_income = to_numpy(op_income)
    v_pe = to_numpy(pe)
    v_roe = to_numpy(roe)
    v_op_margin = to_numpy(op_margin)

    # 籌碼面
    v_inst_rolling = to_numpy(inst_rolling_sum)
    v_inst_streak = to_numpy(inst_streak)

    # 大盤 (1D -> 廣播)
    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    # ==========================================
    # 4. 策略邏輯執行 (純數學運算)
    # ==========================================

    has_ma20 = v_ma20 > 0
    has_ma60 = v_ma60 > 0
    has_ma120 = v_ma120 > 0

    # 市場狀態濾網
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)
    v_bearish = (v_bench < v_bench_ma60 * 0.99) & (v_bench_ma60 > 0) & (v_bench > 0)

    # 流動性濾網
    v_liq = (v_vol_ma20 > 1000000)

    # [New] Supply Zone Filter (套牢區濾網)
    # 當股價距離重大歷史高點 (250日高) 不到 5% (但在其下方)，視為進入套牢賣壓區，暫停買進。
    # Logic: 0.95 * High250 <= Close < High250
    # 注意：如果 Close >= High250，代表已經突破，則不擋。
    # v_high_250 可能有 NaN (前250天)，需處理 > 0
    c_supply_danger = (v_close >= v_high_250 * 0.95) & (v_close < v_high_250) & (v_high_250 > 0)
    c_safe_supply = ~c_supply_danger

    # --- 訊號 A: 小型成長股 (Growth) ---
    c_small = (v_capital < 2000000) & (v_capital > 0)

    c_rev = (v_rev_growth > 30) | (v_rev_current >= v_rev_12m_max)
    c_profit = (v_eps_sum > 0) & (v_op_income > 0)
    c_value = (v_pe < 30) & (v_pe > 0)

    c_trend = (v_close > v_ma20) & (v_close > v_ma60) & has_ma20 & has_ma60
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 1.5)

    # 加入 Supply Zone Filter
    sig_a = v_bullish & c_small & c_rev & c_profit & c_value & c_trend & c_breakout & v_liq & c_safe_supply

    # --- 訊號 B: 均值回歸 (Reversion) ---
    c_oversold = (v_close < v_ma120) & has_ma120
    c_rsi_panic = v_rsi < 20
    c_vol_panic = v_vol > v_vol_ma20 * 2

    body = np.abs(v_close - v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low
    c_hammer = lower_shadow > (body * 2)

    # 加入 Supply Zone Filter (即使是抄底，如果在壓力區正下方也不買)
    sig_b = v_bullish & c_oversold & c_rsi_panic & c_hammer & c_vol_panic & v_liq & c_safe_supply

    # --- 訊號 C: 放空 (Short) - DISABLED ---
    sig_c = np.zeros_like(v_close, dtype=bool)

    # ==========================================
    # 5. 部位重建 (Position Reconstruction)
    # ==========================================

    long_entries = sig_a | sig_b
    short_entries = sig_c

    c_reversion_hold = (v_close < v_ma60) & (v_rsi < 50)

    long_exits = ((v_close < v_ma60) & (~c_reversion_hold)) | v_bearish
    short_exits = (v_close > v_ma20) | v_bullish | (v_rsi < 20)

    score = np.ones_like(v_close)
    score += (v_rev_growth > 50).astype(int)
    score += ((v_roe > 20) | (v_op_margin > 20)).astype(int)
    score += v_inst_streak.astype(int)

    # 構建模擬用的 DataFrame
    # [Fix 3]: Use the ORIGINAL columns (close.columns) for the return DataFrame.
    # We DO NOT cast this to string, so backtest.sim receives exactly what it gave us.
    # Since all our math was done on numpy arrays (order preserved), this lines up perfectly.
    df_long = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    df_short = pd.DataFrame(np.nan, index=master_index, columns=close.columns)

    # 應用邏輯
    v_pos_long = np.full_like(v_close, np.nan)
    v_pos_short = np.full_like(v_close, np.nan)

    v_pos_long[long_entries] = score[long_entries]
    v_pos_short[short_entries] = -0.5

    v_pos_long[long_exits] = 0
    v_pos_short[short_exits] = 0

    df_long[:] = v_pos_long
    df_short[:] = v_pos_short

    df_long = df_long.ffill().fillna(0)
    df_short = df_short.ffill().fillna(0)

    final_pos = df_long + df_short

    # 執行回測
    if stop_loss is not None or take_profit is not None:
        report = backtest.sim(
            final_pos,
            resample='D',
            name='Isaac 策略 (壓力測試)',
            upload=False,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    else:
        report = backtest.sim(final_pos, resample='D', name='Isaac 策略 (全天候)', upload=False)

    return report
