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

    # 用於對齊的標準索引 (Master Index)
    master_index = close.index
    master_columns = close.columns

    # 輔助函數：將所有資料對齊到 (時間 x 股票) 或 (時間 x 1)，並轉為 NumPy 陣列
    def to_numpy(obj, is_benchmark=False):
        if isinstance(obj, pd.Series):
            # 僅對齊索引
            obj = obj.reindex(master_index, method='ffill')
            return obj.fillna(0).values.reshape(-1, 1)
        elif isinstance(obj, pd.DataFrame):
            # 對齊索引。假設股票數據的欄位與 close 相同。
            # 如果不是，應該重新對齊欄位，但 data.get 通常回傳完整宇宙。
            # 為了安全，如果是股票數據，我們重新對齊欄位。
            if not is_benchmark:
                obj = obj.reindex(index=master_index, columns=master_columns, method='ffill')
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
    # 2. 預計算 (Pandas 階段)
    # ==========================================
    # 在轉換為 NumPy 之前，先在 Pandas 中進行滾動計算
    # 這比在 NumPy 中實現滾動更容易

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

    # 資料完整性檢查 (防止在缺失數據填充為0時進行交易)
    # MA > 0 檢查確保我們有有效的移動平均線 (不是上市初期或缺失數據)
    has_ma20 = v_ma20 > 0
    has_ma60 = v_ma60 > 0
    has_ma120 = v_ma120 > 0

    # 市場狀態濾網
    # [Fix 1]: Ensure v_bench > 0 to prevent false exits on missing data
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)
    v_bearish = (v_bench < v_bench_ma60 * 0.99) & (v_bench_ma60 > 0) & (v_bench > 0)

    # 流動性濾網
    v_liq = (v_vol_ma20 > 1000000)

    # --- 訊號 A: 小型成長股 (Growth) ---
    # 修正：Capital < 2M 檢查也需要 Capital > 0，避免缺失數據被誤判為小型股
    c_small = (v_capital < 2000000) & (v_capital > 0)

    c_rev = (v_rev_growth > 30) | (v_rev_current >= v_rev_12m_max)
    c_profit = (v_eps_sum > 0) & (v_op_income > 0)
    c_value = (v_pe < 30) & (v_pe > 0)

    # 趨勢與均線檢查 (需 > 0)
    c_trend = (v_close > v_ma20) & (v_close > v_ma60) & has_ma20 & has_ma60

    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 1.5)

    sig_a = v_bullish & c_small & c_rev & c_profit & c_value & c_trend & c_breakout & v_liq

    # --- 訊號 B: 均值回歸 (Reversion) ---
    c_oversold = (v_close < v_ma120) & has_ma120
    c_rsi_panic = v_rsi < 20
    c_vol_panic = v_vol > v_vol_ma20 * 2

    body = np.abs(v_close - v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low
    c_hammer = lower_shadow > (body * 2)

    sig_b = v_bullish & c_oversold & c_rsi_panic & c_hammer & c_vol_panic & v_liq

    # --- 訊號 C: 放空 (Short) - DISABLED BY DEFAULT ---
    # [Fix 3]: Temporarily disable shorting to prevent extreme losses (-99% MDD)
    # The logic is commented out by setting sig_c to False
    c_weak = (v_close < v_ma60) & (v_close < v_ma20) & has_ma20 & has_ma60
    bias = (v_close - v_ma20) / (v_ma20 + 1e-9)
    c_bias = bias > -0.10

    c_bad_rev = v_rev_growth < 0
    c_bad_profit = (v_eps_sum < 0) | (v_op_income < 0)
    c_bubble = v_pe > 60

    c_inst_sell = v_inst_rolling < 0
    c_bad_fund = c_bad_rev | c_bad_profit | c_bubble
    c_black = v_close < v_open

    # sig_c = v_bearish & c_weak & c_bias & (c_bad_fund | c_inst_sell) & c_black & v_liq
    sig_c = np.zeros_like(v_close, dtype=bool) # Disable shorting

    # ==========================================
    # 5. 部位重建 (Position Reconstruction)
    # ==========================================

    long_entries = sig_a | sig_b
    short_entries = sig_c

    # 出場條件
    # [Fix 2]: Reversion Hold Logic - Allow bounce until RSI > 50 or Stop Loss hit
    # Hold if: (Price < MA60) AND (RSI < 50)
    c_reversion_hold = (v_close < v_ma60) & (v_rsi < 50)

    # [Fix 4]: Relaxed Trend Exit - Change from MA20 to MA60 to reduce churn
    # Trend Exit: Close < MA60 (Trend Change)
    # Reversion Exit: If NOT holding, exit.

    # Logic:
    # If it was a Trend Trade (Sig A): Exit if Close < MA60 OR Market Bearish
    # If it was a Reversion Trade (Sig B): Exit if RSI > 50 (Take Profit) OR Close < MA60 (Trend continuation down)

    # Simplified Exit: Close < MA60 is the main trend filter.
    # Bearish market forces exit.

    # [Fix 2 applied]: Don't exit if holding due to reversion logic
    long_exits = ((v_close < v_ma60) & (~c_reversion_hold)) | v_bearish
    short_exits = (v_close > v_ma20) | v_bullish | (v_rsi < 20)

    # 評分權重
    score = np.ones_like(v_close)
    score += (v_rev_growth > 50).astype(int)
    score += ((v_roe > 20) | (v_op_margin > 20)).astype(int)
    score += v_inst_streak.astype(int)

    # 構建模擬用的 DataFrame
    df_long = pd.DataFrame(np.nan, index=master_index, columns=master_columns)
    df_short = pd.DataFrame(np.nan, index=master_index, columns=master_columns)

    # 應用邏輯
    # 初始化為 NaN
    v_pos_long = np.full_like(v_close, np.nan)
    v_pos_short = np.full_like(v_close, np.nan)

    # 進場
    v_pos_long[long_entries] = score[long_entries]
    v_pos_short[short_entries] = -0.5

    # 出場 (設置為 0)
    v_pos_long[long_exits] = 0
    v_pos_short[short_exits] = 0

    # 寫回 DataFrame
    df_long[:] = v_pos_long
    df_short[:] = v_pos_short

    # 向前填充 (ffill) 以模擬持倉
    # "NaN 表示保持之前的狀態 (Hold)"
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
