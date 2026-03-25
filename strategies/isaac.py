from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

pd.set_option('future.no_silent_downcasting', True)


# ==========================================
# 工具函數 (來自正規化信號框架)
# ==========================================

def compute_hv(price_series, look_back_period=50):
    """
    Historical Volatility (歷史波動率)
    使用 log return 的滾動標準差計算。
    """
    log_return = np.log(price_series / price_series.shift(1))
    return log_return.rolling(look_back_period).std()


def rolling_mad(series, window, min_periods=None):
    """
    Rolling Median Absolute Deviation (滾動中位數絕對偏差)
    用於 Robust Z-score 計算，比標準差更抗離群值。
    """
    if min_periods is None:
        min_periods = window
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start:i + 1]
        window_data = window_data[~np.isnan(window_data)]
        if len(window_data) >= min_periods:
            med = np.median(window_data)
            out[i] = np.median(np.abs(window_data - med))
    return pd.Series(out, index=series.index)


def _to_numpy(obj, master_index, master_columns_str, sanitize_dataframe, obj_name="Unknown", is_benchmark=False):
    """
    將 pandas 物件對齊到 master_index/master_columns_str 並轉為 numpy array。
    原本定義在 run_isaac_strategy 內的閉包函數，提取為模組級函數。
    """
    if obj is None: return np.nan
    if isinstance(obj, pd.DataFrame):
        obj = sanitize_dataframe(obj, source_name=obj_name)
    if isinstance(obj, pd.Series):
        obj = obj.reindex(master_index).ffill()
        return obj.fillna(0).values.reshape(-1, 1)
    elif isinstance(obj, pd.DataFrame):
        if not is_benchmark:
            df_temp = obj.copy()
            df_temp.columns = df_temp.columns.astype(str)
            df_aligned = df_temp.reindex(index=master_index, columns=master_columns_str).ffill()
            return df_aligned.fillna(0).values
        else:
            obj = obj.reindex(index=master_index).ffill()
            return obj.fillna(0).values
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


def _fetch_data(api_token):
    """
    1. 數據抓取 (Fetch Data)
    登入 FinLab 並取得所有需要的價量與基本面資料。

    Returns:
        dict: 包含所有 DataFrames 和 master_index/master_columns_str
    """
    from data.provider import sanitize_dataframe

    if api_token:
        finlab.login(api_token)

    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")

    # [Fix 3: Immutability] - 不修改 close.columns，避免污染 Finlab 全域快取
    master_index = close.index
    master_columns_str = close.columns.astype(str)

    # 價格數據
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # 大盤基準
    benchmark_close = data.get('price:收盤價')['0050']

    # 基本面數據 — [V3.1 Fix] 分離 try/except 避免一個失敗全部歸零
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
    except Exception:
        rev_growth = pd.DataFrame(0, index=master_index, columns=close.columns)

    try:
        rev_current = data.get('monthly_revenue:當月營收')
    except Exception:
        rev_current = pd.DataFrame(0, index=master_index, columns=close.columns)

    try:
        capital = data.get('finance_statement:股本')
    except Exception:
        capital = pd.DataFrame(10000000, index=master_index, columns=close.columns)

    try:
        eps = data.get('finance_statement:每股盈餘')
    except Exception:
        eps = pd.DataFrame(0, index=master_index, columns=close.columns)
    try:
        op_income = data.get('finance_statement:營業利益')
    except Exception:
        op_income = pd.DataFrame(0, index=master_index, columns=close.columns)
    try:
        roe = data.get('finance_statement:權益報酬率')
    except Exception:
        roe = pd.DataFrame(0, index=master_index, columns=close.columns)
    try:
        op_margin = data.get('finance_statement:營業利益率')
    except Exception:
        op_margin = pd.DataFrame(0, index=master_index, columns=close.columns)
    try:
        pe = data.get('price:本益比')
    except Exception:
        pe = pd.DataFrame(100, index=master_index, columns=close.columns)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except Exception:
        inst_net_buy = pd.DataFrame(0, index=master_index, columns=close.columns)

    return {
        'close': close,
        'master_index': master_index,
        'master_columns_str': master_columns_str,
        'open_': open_,
        'high': high,
        'low': low,
        'vol': vol,
        'benchmark_close': benchmark_close,
        'rev_growth': rev_growth,
        'rev_current': rev_current,
        'capital': capital,
        'eps': eps,
        'op_income': op_income,
        'roe': roe,
        'op_margin': op_margin,
        'pe': pe,
        'inst_net_buy': inst_net_buy,
    }


def _compute_technicals(data_dict):
    """
    2. 預計算 (Pandas 階段) + 3. NumPy 轉換
    計算所有技術指標、K 線型態、並轉為 numpy arrays。

    Returns:
        dict: 包含所有 numpy arrays 和部分 pandas 物件 (用於後續步驟)
    """
    from data.provider import sanitize_dataframe

    close = data_dict['close']
    master_index = data_dict['master_index']
    master_columns_str = data_dict['master_columns_str']
    open_ = data_dict['open_']
    high = data_dict['high']
    low = data_dict['low']
    vol = data_dict['vol']
    benchmark_close = data_dict['benchmark_close']
    rev_growth = data_dict['rev_growth']
    rev_current = data_dict['rev_current']
    eps = data_dict['eps']
    op_income = data_dict['op_income']
    roe = data_dict['roe']
    op_margin = data_dict['op_margin']
    pe = data_dict['pe']
    inst_net_buy = data_dict['inst_net_buy']

    # Helper for numpy conversion (closure over master_index/master_columns_str)
    def to_numpy(obj, obj_name="Unknown", is_benchmark=False):
        return _to_numpy(obj, master_index, master_columns_str, sanitize_dataframe,
                         obj_name=obj_name, is_benchmark=is_benchmark)

    # ==========================================
    # 2. 預計算 (Pandas 階段)
    # ==========================================

    # 移動平均線
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma10 = close.rolling(10).mean()       # [V3.1] Signal C 動能出場用

    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # RSI 指標 (Wilder's RMA - 與 TradingView / Bloomberg 一致)
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        # Wilder's RMA (Recursive Moving Average) = EWM with alpha=1/period
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    my_rsi = rsi(close, 14)

    # [V3.3] ATR (Average True Range) — 動態 Trailing Stop 用
    high_df = sanitize_dataframe(high, "FinLab_High")
    low_df = sanitize_dataframe(low, "FinLab_Low")
    prev_close = close.shift(1)
    tr1 = high_df - low_df
    tr2 = (high_df - prev_close).abs()
    tr3 = (low_df - prev_close).abs()
    # 修正: 用 numpy maximum 對齊
    true_range = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    atr_14 = true_range.rolling(14).mean()

    # 動態 Trailing Stop: 60 日滾動高點 - 3x ATR
    rolling_high_60 = close.rolling(60, min_periods=5).max()
    trail_level = rolling_high_60 - 3.0 * atr_14

    # 基本面輔助指標
    rev_12m_max = rev_current.rolling(12).max()
    close_max_20 = close.rolling(20).max().shift(1)
    eps_sum = eps.rolling(4).sum()

    # 籌碼面輔助指標
    inst_rolling_sum = inst_net_buy.rolling(3).sum()
    inst_streak = (inst_net_buy.rolling(5).min() > 0)

    # Bollinger Bandwidth 波動收縮濾網 (VCP 概念)
    bb_std = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_bandwidth = (bb_upper - bb_lower) / ma20.where(ma20 > 0, np.nan)
    bb_bandwidth = bb_bandwidth.fillna(0)
    bb_contracting = bb_bandwidth < bb_bandwidth.rolling(60).quantile(0.25)

    # [V3.3] K 線型態偵測 (向量化, 用於評分加分)
    # 基於《陰線陽線》— 只取高可信度型態
    prev_close_1 = close.shift(1)
    prev_open_1 = sanitize_dataframe(open_, "Open").shift(1)
    prev_close_2 = close.shift(2)
    prev_open_2 = sanitize_dataframe(open_, "Open").shift(2)
    open_df = sanitize_dataframe(open_, "Open")

    k_body = (close - open_df).abs()
    k_body_1 = (prev_close_1 - prev_open_1).abs()
    k_body_2 = (prev_close_2 - prev_open_2).abs()
    k_avg_body = k_body.rolling(14).mean()

    k_is_bull = close > open_df
    k_is_bear_1 = prev_close_1 < prev_open_1
    k_is_bull_2_bear = prev_close_2 < prev_open_2  # 前天陰線

    # 多頭吞噬: 前陰今陽, 今實體完全包覆前實體
    pat_bull_engulf = (k_is_bear_1 & k_is_bull &
                       (open_df <= prev_close_1) & (close >= prev_open_1) &
                       (k_body > k_body_1))

    # 晨星: 前天長陰 + 昨天小K + 今天長陽收過前天中點
    pat_morning_star = (k_is_bull_2_bear & (k_body_2 > k_avg_body) &
                        (k_body_1 <= k_avg_body * 0.5) &
                        k_is_bull & (k_body > k_avg_body) &
                        (close > (prev_open_2 + prev_close_2) / 2))

    # 空頭吞噬 (出場用): 前陽今陰, 今實體包覆前實體
    k_is_bull_1 = prev_close_1 > prev_open_1
    k_is_bear = close < open_df
    pat_bear_engulf = (k_is_bull_1 & k_is_bear &
                       (open_df >= prev_close_1) & (close <= prev_open_1) &
                       (k_body > k_body_1))

    # 暮星 (出場用): 前天長陽 + 昨天小K + 今天長陰
    k_is_bull_2 = prev_close_2 > prev_open_2
    pat_evening_star = (k_is_bull_2 & (k_body_2 > k_avg_body) &
                        (k_body_1 <= k_avg_body * 0.5) &
                        k_is_bear & (k_body > k_avg_body) &
                        (close < (prev_open_2 + prev_close_2) / 2))

    # [V3.4] Minervini + Edwards-Magee 整合指標 (不依賴 stock_ret_120 的部分)
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    high_252 = close.rolling(252).max()
    low_252 = close.rolling(252).min()
    vol_ma50 = vol.rolling(50).mean()
    # VCP 量縮偵測: 近 10 日內有出現量 < 50日均量一半
    vol_dry_up = (vol < vol_ma50 * 0.5).rolling(10).max()
    # BB 整理偵測 (E&M): 近 10 日內有 BB 收縮到 P20 以下
    bb_consolidation = (bb_bandwidth < bb_bandwidth.rolling(60).quantile(0.20)).rolling(10).max()

    # [V3.2] MACD 指標 (Signal A 確認 + Signal D 空頭確認)
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line

    # [V3.2] 20 日低點 (Signal D 空頭破底用)
    close_min_20 = close.rolling(20).min().shift(1)

    # 相對強度濾網 (120 日報酬率 vs 大盤)
    stock_ret_120 = close.pct_change(120, fill_method=None)
    bench_ret_120 = benchmark_close.pct_change(120, fill_method=None)
    rel_strength = stock_ret_120.sub(bench_ret_120, axis=0) > 0

    # [V3.4] RS Rank: 跨股票百分位排名 (Minervini 核心，依賴 stock_ret_120)
    rs_rank = stock_ret_120.rank(axis=1, pct=True) * 100

    # [V3.3] 多時間框架: 週線趨勢確認
    # 週線 close > 週 MA10 > 週 MA20 = 週線多頭排列
    weekly_close = close.resample('W').last()
    weekly_ma10 = weekly_close.rolling(10).mean()   # 10 週 ≈ 50 交易日
    weekly_ma20 = weekly_close.rolling(20).mean()   # 20 週 ≈ 100 交易日
    weekly_uptrend = (weekly_close > weekly_ma10) & (weekly_ma10 > weekly_ma20)
    # 回填到日線頻率 (週線結果 ffill 到每天)
    weekly_uptrend_daily = weekly_uptrend.reindex(master_index).ffill().fillna(False)

    # [V3.1] Signal C: 短期相對強弱 (20 日報酬率 vs 大盤)
    stock_ret_20 = close.pct_change(20, fill_method=None)
    bench_ret_20 = benchmark_close.pct_change(20, fill_method=None)
    rs_short_20 = stock_ret_20.sub(bench_ret_20, axis=0) > 0

    # [V3.1] Signal C: 連續 3 個月營收 YoY > 20% (動能加速確認)
    # shift(22) ≈ 上個月, shift(44) ≈ 兩個月前 (月營收資料 ffill 到每日)
    rev_yoy_m0 = rev_growth > 20
    rev_yoy_m1 = rev_growth.shift(22) > 20
    rev_yoy_m2 = rev_growth.shift(44) > 20
    rev_momentum_3m = rev_yoy_m0 & rev_yoy_m1 & rev_yoy_m2

    # Historical Volatility (HV) 歷史波動率濾網
    hv = compute_hv(close, 50)
    hv_q80 = hv.rolling(120).quantile(0.8)
    hv_q20 = hv.rolling(120).quantile(0.2)
    hv_normal = (hv >= hv_q20) & (hv <= hv_q80)   # 正常波動區間 (避開極端)
    hv_compressed = hv < hv_q20                      # 波動收縮 (即將爆發)

    # Z-score 信號正規化 (價格偏離均值程度)
    z_std_50 = close.rolling(50).std()
    z_std_50_safe = z_std_50.replace(0, np.nan)
    z_score = (close - ma50) / z_std_50_safe          # 重用已計算的 ma50
    z_score = z_score.fillna(0).replace([np.inf, -np.inf], 0)

    # 反向/槓桿 ETF 黑名單濾網
    import re as _re
    etf_blacklist = pd.Series(False, index=master_columns_str)
    for col in master_columns_str:
        if _re.match(r'^00\d{3,}[RL]?$', col) or _re.match(r'^00\d{3,}', col):
            etf_blacklist[col] = True
    v_etf_ok = ~etf_blacklist.values.reshape(1, -1)

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
    v_ma50 = to_numpy(ma50)
    v_ma60 = to_numpy(ma60)
    v_ma120 = to_numpy(ma120)
    v_vol_ma5 = to_numpy(vol_ma5)
    v_vol_ma20 = to_numpy(vol_ma20)
    v_rsi = to_numpy(my_rsi)
    v_close_max_20 = to_numpy(close_max_20)

    # Supply Zone (250 日高點)
    high_250 = close.rolling(250).max().shift(1)
    v_high_250 = to_numpy(high_250)

    # 新濾網 NumPy 轉換
    v_bb_contracting = to_numpy(bb_contracting)
    v_rel_strength = to_numpy(rel_strength)

    # HV & Z-score NumPy 轉換
    v_hv_normal = to_numpy(hv_normal)
    v_hv_compressed = to_numpy(hv_compressed)
    v_z_score = to_numpy(z_score)

    # [V3.1] Signal C NumPy 轉換
    v_ma10 = to_numpy(ma10)
    v_rs_short = to_numpy(rs_short_20)
    v_rev_momentum = to_numpy(rev_momentum_3m)

    # [V3.3] 週線趨勢 NumPy 轉換
    v_weekly_uptrend = to_numpy(weekly_uptrend_daily)

    # [V3.3] K 線型態 NumPy 轉換
    v_pat_bull_engulf = to_numpy(pat_bull_engulf)
    v_pat_morning_star = to_numpy(pat_morning_star)
    v_pat_bear_engulf = to_numpy(pat_bear_engulf)
    v_pat_evening_star = to_numpy(pat_evening_star)

    # [V3.2] MACD & Signal D NumPy 轉換
    v_macd_hist = to_numpy(macd_hist)
    v_macd_hist_prev = to_numpy(macd_hist.shift(1))
    v_close_min_20 = to_numpy(close_min_20)

    # [V3.4] Minervini + E&M NumPy 轉換
    v_ma150 = to_numpy(ma150)
    v_ma200 = to_numpy(ma200)
    v_high_252 = to_numpy(high_252)
    v_low_252 = to_numpy(low_252)
    v_rs_rank = to_numpy(rs_rank)
    v_vol_dry = to_numpy(vol_dry_up)
    v_bb_consol = to_numpy(bb_consolidation)

    # [V3.3] 動態 Trailing Stop NumPy 轉換
    v_trail_level = to_numpy(trail_level)

    # 基本面
    v_rev_growth = to_numpy(rev_growth)
    v_rev_current = to_numpy(rev_current)
    v_rev_12m_max = to_numpy(rev_12m_max)
    v_capital = to_numpy(data_dict['capital'])
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

    return {
        # numpy arrays
        'v_close': v_close, 'v_open': v_open, 'v_high': v_high, 'v_low': v_low, 'v_vol': v_vol,
        'v_ma10': v_ma10, 'v_ma20': v_ma20, 'v_ma50': v_ma50, 'v_ma60': v_ma60, 'v_ma120': v_ma120,
        'v_ma150': v_ma150, 'v_ma200': v_ma200,
        'v_vol_ma5': v_vol_ma5, 'v_vol_ma20': v_vol_ma20,
        'v_rsi': v_rsi, 'v_close_max_20': v_close_max_20, 'v_high_250': v_high_250,
        'v_bb_contracting': v_bb_contracting, 'v_rel_strength': v_rel_strength,
        'v_hv_normal': v_hv_normal, 'v_hv_compressed': v_hv_compressed, 'v_z_score': v_z_score,
        'v_rs_short': v_rs_short, 'v_rev_momentum': v_rev_momentum,
        'v_weekly_uptrend': v_weekly_uptrend,
        'v_pat_bull_engulf': v_pat_bull_engulf, 'v_pat_morning_star': v_pat_morning_star,
        'v_pat_bear_engulf': v_pat_bear_engulf, 'v_pat_evening_star': v_pat_evening_star,
        'v_macd_hist': v_macd_hist, 'v_macd_hist_prev': v_macd_hist_prev,
        'v_close_min_20': v_close_min_20,
        'v_high_252': v_high_252, 'v_low_252': v_low_252,
        'v_rs_rank': v_rs_rank, 'v_vol_dry': v_vol_dry, 'v_bb_consol': v_bb_consol,
        'v_trail_level': v_trail_level,
        'v_rev_growth': v_rev_growth, 'v_rev_current': v_rev_current, 'v_rev_12m_max': v_rev_12m_max,
        'v_capital': v_capital, 'v_eps_sum': v_eps_sum, 'v_op_income': v_op_income,
        'v_pe': v_pe, 'v_roe': v_roe, 'v_op_margin': v_op_margin,
        'v_inst_rolling': v_inst_rolling, 'v_inst_streak': v_inst_streak,
        'v_bench': v_bench, 'v_bench_ma60': v_bench_ma60,
        'v_etf_ok': v_etf_ok,
    }


def _generate_signals(technicals, data_dict, params, minervini_mode):
    """
    4. 策略邏輯 V3 (Data-Driven 修正)
    生成 Signal A-E 及對應的出場條件與評分。

    Returns:
        dict: 包含所有信號、出場條件、評分 arrays
    """
    # --- 解包可調參數 ---
    p = params or {}
    _volume_mult       = float(p.get('volume_mult',       1.5))
    _supply_danger_pct = float(p.get('supply_danger_pct', 0.97))
    _rsi_threshold     = int(p.get('rsi_threshold',       28))
    _confirm_days      = int(p.get('confirm_days',        0))
    _breadth_filter    = bool(p.get('breadth_filter',     False))
    _disable_d         = bool(p.get('disable_d',          False))
    _atr_exit          = bool(p.get('atr_exit',           False))

    close = data_dict['close']
    master_index = data_dict['master_index']

    # --- 解包 numpy arrays ---
    v_close = technicals['v_close']
    v_open = technicals['v_open']
    v_high = technicals['v_high']
    v_low = technicals['v_low']
    v_vol = technicals['v_vol']
    v_ma10 = technicals['v_ma10']
    v_ma20 = technicals['v_ma20']
    v_ma50 = technicals['v_ma50']
    v_ma60 = technicals['v_ma60']
    v_ma120 = technicals['v_ma120']
    v_ma150 = technicals['v_ma150']
    v_ma200 = technicals['v_ma200']
    v_vol_ma5 = technicals['v_vol_ma5']
    v_vol_ma20 = technicals['v_vol_ma20']
    v_rsi = technicals['v_rsi']
    v_close_max_20 = technicals['v_close_max_20']
    v_high_250 = technicals['v_high_250']
    v_bb_contracting = technicals['v_bb_contracting']
    v_rel_strength = technicals['v_rel_strength']
    v_hv_normal = technicals['v_hv_normal']
    v_hv_compressed = technicals['v_hv_compressed']
    v_z_score = technicals['v_z_score']
    v_rs_short = technicals['v_rs_short']
    v_rev_momentum = technicals['v_rev_momentum']
    v_weekly_uptrend = technicals['v_weekly_uptrend']
    v_macd_hist = technicals['v_macd_hist']
    v_macd_hist_prev = technicals['v_macd_hist_prev']
    v_close_min_20 = technicals['v_close_min_20']
    v_high_252 = technicals['v_high_252']
    v_low_252 = technicals['v_low_252']
    v_rs_rank = technicals['v_rs_rank']
    v_vol_dry = technicals['v_vol_dry']
    v_bb_consol = technicals['v_bb_consol']
    v_trail_level = technicals['v_trail_level']
    v_rev_growth = technicals['v_rev_growth']
    v_eps_sum = technicals['v_eps_sum']
    v_inst_streak = technicals['v_inst_streak']
    v_bench = technicals['v_bench']
    v_bench_ma60 = technicals['v_bench_ma60']
    v_etf_ok = technicals['v_etf_ok']

    # ==========================================
    # 4. 策略邏輯 V3 (Data-Driven 修正)
    # ==========================================
    #
    # V2.1 失敗原因 (from debug log):
    #   c_small: 0   ← fillna(0) 使 NaN 變成 0, 然後 (0>0)=False
    #   c_profit: 0  ← 同上
    #   c_value: 0   ← 同上
    #   Signal B 入場 217 次, 但出場 3.6M 次 → 入場即被出場覆蓋
    #
    # V3 設計原則:
    #   1. Signal A 只用「技術面」條件作為進場門檻 (這些數據不會是 NaN)
    #   2. 基本面條件全部移到「評分系統」作為加分項 (有數據就加分, 沒數據不扣分)
    #   3. 修正進場/出場衝突: 進場信號優先, 出場信號只對「已持倉部位」生效
    #   4. Signal B 使用獨立的出場邏輯 (回復到 MA60 以上就出場)
    # ==========================================

    has_ma10 = v_ma10 > 0
    has_ma20 = v_ma20 > 0
    has_ma50 = v_ma50 > 0
    has_ma60 = v_ma60 > 0
    has_ma120 = v_ma120 > 0
    has_ma150 = v_ma150 > 0
    has_ma200 = v_ma200 > 0

    # 市場狀態濾網
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)
    v_bearish = (v_bench < v_bench_ma60 * 0.99) & (v_bench_ma60 > 0) & (v_bench > 0)

    # 流動性濾網
    v_liq = (v_vol_ma20 > int(p.get('liq_min', 500000)))

    # Supply Zone Filter (套牢區濾網)
    c_supply_danger = (v_close >= v_high_250 * _supply_danger_pct) & (v_close < v_high_250) & (v_high_250 > 0)
    c_safe_supply = ~c_supply_danger

    # 強勢收盤確認 (收盤在日內振幅前 25%)
    v_daily_range = v_high - v_low
    v_daily_range_safe = np.where(v_daily_range == 0, np.nan, v_daily_range)
    v_close_position = (v_close - v_low) / v_daily_range_safe
    v_close_position = np.nan_to_num(v_close_position, nan=0.0)
    c_strong_close = v_close_position >= 0.75

    # 波動收縮確認
    c_bb_tight = v_bb_contracting > 0

    # 相對強度確認
    c_rs = v_rel_strength > 0

    # [V3.4] Minervini 趨勢模板 + Edwards-Magee 新高突破 複合條件
    # Minervini: MA50 > MA150 > MA200 排列 + MA200 上升 + 價格結構
    c_minervini_ma = ((v_close > v_ma50) & (v_ma50 > v_ma150) & (v_ma150 > v_ma200)
                      & has_ma50 & has_ma150 & has_ma200)
    v_ma200_prev = np.zeros_like(v_ma200)
    v_ma200_prev[20:, :] = v_ma200[:-20, :]
    c_ma200_up = (v_ma200 > v_ma200_prev) & (v_ma200_prev > 0)
    c_above_52w_low = v_close > (v_low_252 * 1.25)
    c_near_52w_high = v_close > (v_high_252 * 0.75)
    c_full_minervini = c_minervini_ma & c_ma200_up & c_above_52w_low & c_near_52w_high

    # Edwards-Magee: 52 週新高 + 整理後突破
    c_52w_new_high = v_close >= v_high_252
    c_em_consolidation = v_bb_consol > 0  # 近 10 日有 BB 收縮
    c_rs_rank_70 = v_rs_rank > 70
    c_vol_dry = v_vol_dry > 0  # VCP 量縮

    # === 訊號 A: 技術面突破 (Growth Breakout) ===
    # [V3] 只用技術面作為進場門檻, 基本面全部改為評分加分
    # [V3.1] 加入 HV 正常區間濾網 (避開極端波動/死水行情)
    c_trend = (v_close > v_ma20) & (v_ma20 > v_ma60) & has_ma20 & has_ma60
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * _volume_mult)
    c_hv_ok = v_hv_normal > 0  # 波動率在正常區間 (P20~P80)

    # [V3.4] 三種 Minervini 整合模式
    if minervini_mode == 'gate':
        # 模式 A: Minervini 趨勢模板作為 Signal A 進場門檻
        sig_a = (v_bullish & c_trend & c_breakout & v_liq & c_safe_supply
                 & v_etf_ok & c_hv_ok & c_full_minervini)
    else:
        # 預設 / score / signal_e: Signal A 保持不變
        sig_a = (v_bullish & c_trend & c_breakout & v_liq & c_safe_supply & v_etf_ok & c_hv_ok)

    # === 訊號 B: 均值回歸 (Reversion) ===
    c_oversold = (v_close < v_ma120) & has_ma120
    c_rsi_panic = v_rsi < _rsi_threshold
    c_vol_panic = v_vol > v_vol_ma20 * _volume_mult

    body = np.abs(v_close - v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low
    c_hammer = lower_shadow > (body * 1.5)  # 放寬: 2x → 1.5x

    sig_b = v_bullish & c_oversold & c_rsi_panic & c_hammer & v_liq & v_etf_ok

    # [實驗] 突破確認: 要求連續 N 天都滿足突破條件
    if _confirm_days > 0:
        sig_a_df = pd.DataFrame(sig_a, index=master_index, columns=close.columns)
        sig_a_confirmed = sig_a_df.rolling(_confirm_days, min_periods=_confirm_days).min() > 0
        sig_a = sig_a_confirmed.values

    # [實驗] 大盤寬度過濾: 站上 MA20 的股票比例 < 30% 時暫停開新倉
    if _breadth_filter:
        above_ma20_ratio = (v_close > v_ma20).sum(axis=1) / (v_ma20 > 0).sum(axis=1)
        breadth_ok = (above_ma20_ratio >= 0.30).reshape(-1, 1)
        sig_a = sig_a & breadth_ok
        sig_b = sig_b & breadth_ok

    # === 訊號 C: 動能追蹤 (Momentum Trading) - [V3.1 新增] ===
    # 設計哲學: 針對「無法估值但動能強勁」的轉機股
    # 捨棄 P/E 估值，擁抱相對強弱 + 營收動能加速
    c_rs_short = v_rs_short > 0                                  # 20日報酬 > 大盤
    c_above_ma20 = (v_close > v_ma20) & has_ma20                 # 收盤 > 月線 (動能延續)
    c_rev_momentum = v_rev_momentum > 0                          # 連續3月營收 YoY > 20%

    sig_c = (v_bullish & c_rs_short & c_above_ma20 & c_breakout
             & c_rev_momentum & v_liq & v_etf_ok)

    # Signal C 專用出場條件
    exit_c1 = (v_close < v_ma10) & has_ma10
    c_bearish_body = (v_open - v_close) > (0.015 * v_close)
    c_high_vol_candle = v_vol > v_vol_ma5 * 2.0
    exit_c2 = c_bearish_body & c_high_vol_candle
    exit_c = exit_c1 | exit_c2

    # Signal C 專用評分 (不含 P/E 估值，上限 5)
    score_c = np.ones_like(v_close)
    score_c += c_rs_short.astype(int)                             # +1: 短期相對強弱
    score_c += c_rev_momentum.astype(int)                         # +1: 營收動能加速
    score_c += c_strong_close.astype(int)                         # +1: 強勢收盤
    score_c += (v_vol > v_vol_ma5 * 2.0).astype(int)             # +1: 量能超強
    score_c += (v_hv_compressed > 0).astype(int)                  # +1: 波動收縮蓄勢
    score_c = np.minimum(score_c, 5)                              # 上限 5 (低於 Signal A)

    # === [V3.4] 訊號 E: Minervini+E&M 混合 (signal_e 模式) ===
    if minervini_mode == 'signal_e':
        # 進場: Minervini 趨勢模板 + E&M 新高/整理突破 + 帶量
        c_e_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 2.0)
        sig_e = (c_full_minervini & c_e_breakout & (c_52w_new_high | c_em_consolidation)
                 & c_rs_rank_70 & v_liq & v_etf_ok)
        # 出場: close < MA50 (Minervini 標準)
        exit_e = (v_close < v_ma50) & has_ma50
        # 評分
        score_e = np.ones_like(v_close)
        score_e += c_rs_rank_70.astype(int)              # +1: RS > 70
        score_e += (c_52w_new_high > 0).astype(int)      # +1: 52 週新高
        score_e += (c_vol_dry > 0).astype(int)            # +1: VCP 量縮
        score_e += (v_rev_growth > 20).astype(int)        # +1: 營收 YoY > 20%
        score_e += (v_eps_sum > 0).astype(int)            # +1: EPS > 0
        score_e += (v_inst_streak > 0).astype(int)        # +1: 法人連買
        score_e = np.minimum(score_e, 7)
    else:
        sig_e = np.zeros_like(v_close, dtype=bool)
        exit_e = np.zeros_like(v_close, dtype=bool)
        score_e = np.zeros_like(v_close)

    # === 訊號 D: 熊市放空 (Bear Market Short) - [V3.2 新增] ===
    # 設計哲學: 只在明確空頭市場中，放空技術面破位的弱勢股
    # 保守設計: 最多 2 檔空頭部位，快速停利了結
    # 衝突防護: v_bullish 和 v_bearish 在同一天互斥，合併時多頭優先

    # 空頭趨勢下降排列: 收盤 < MA20 < MA60
    c_d_trend_down = (v_close < v_ma20) & (v_ma20 < v_ma60) & has_ma20 & has_ma60

    # 破底: 收盤跌破 20 日低點 (帶量, 且低於 MA120 確認深度空頭)
    c_d_breakdown = (v_close < v_close_min_20) & (v_close_min_20 > 0) & (v_vol > v_vol_ma5 * 2.0)
    c_d_deep_bear = (v_close < v_ma120) & (v_ma120 > 0)  # 必須低於 MA120 (深度空頭)

    # 弱勢收盤: 收盤在日內振幅底部 25%
    c_d_weak_close = v_close_position <= 0.25

    # MACD 空頭確認: histogram 負且持續惡化
    c_d_macd_bearish = (v_macd_hist < 0) & (v_macd_hist < v_macd_hist_prev)

    sig_d = (v_bearish & c_d_trend_down & c_d_breakdown & c_d_deep_bear
             & c_d_weak_close & c_d_macd_bearish & v_liq & v_etf_ok)

    # [實驗] 關閉 Signal D
    if _disable_d:
        sig_d = np.zeros_like(v_close, dtype=bool)

    # Signal D 出場條件
    # 1. 回復 MA20 以上 (趨勢恢復, 回補)
    exit_d1 = (v_close > v_ma20) & has_ma20
    # 2. RSI 超買 > 75 (可能反彈, 回補出場)
    exit_d2 = v_rsi > 75
    # 3. 爆量長紅 (多頭反攻信號: 漲幅>1.5% + 量>5日均量2x)
    c_d_bullish_body = (v_close - v_open) > (0.015 * v_close)
    c_d_high_vol_bull = v_vol > v_vol_ma5 * 2.0
    exit_d3 = c_d_bullish_body & c_d_high_vol_bull
    # 4. 市場轉多 → 強制回補所有空頭 (防止空頭延續到多頭市場)
    exit_d4 = v_bullish
    exit_d = exit_d1 | exit_d2 | exit_d3 | exit_d4

    # Signal D 評分 (負值代表空頭, 上限 -3, 保守配置)
    score_d = -np.ones_like(v_close)
    score_d -= c_d_weak_close.astype(int)                         # -1: 弱勢收盤
    score_d -= (c_d_macd_bearish > 0).astype(int)                 # -1: MACD 確認
    score_d = np.maximum(score_d, -3)                              # 上限 -3 (保守)

    # [V3] Signal A 出場: 跌破 MA60 (給較大的波動空間)
    exit_a = (v_close < v_ma60) & has_ma60

    # [實驗] ATR 動態出場: close < (60日高點 - 3*ATR14)
    if _atr_exit:
        exit_atr = (v_close < v_trail_level) & (v_trail_level > 0)
        exit_a = exit_a | exit_atr

    # Signal A+B 統一出場
    long_entries_ab = sig_a | sig_b
    long_exits_ab = exit_a

    return {
        'sig_a': sig_a, 'sig_b': sig_b, 'sig_c': sig_c, 'sig_d': sig_d, 'sig_e': sig_e,
        'long_entries_ab': long_entries_ab, 'long_exits_ab': long_exits_ab,
        'exit_c': exit_c, 'exit_d': exit_d, 'exit_e': exit_e,
        'score_c': score_c, 'score_d': score_d, 'score_e': score_e,
        'c_strong_close': c_strong_close, 'c_rs': c_rs, 'c_bb_tight': c_bb_tight,
        'c_full_minervini': c_full_minervini, 'c_rs_rank_70': c_rs_rank_70,
        'c_52w_new_high': c_52w_new_high, 'c_vol_dry': c_vol_dry,
        'v_bullish': v_bullish, 'v_bearish': v_bearish, 'v_liq': v_liq,
        'c_trend': c_trend, 'c_breakout': c_breakout, 'c_hv_ok': c_hv_ok,
        'c_safe_supply': c_safe_supply, 'c_oversold': c_oversold,
        'c_rsi_panic': c_rsi_panic, 'c_hammer': c_hammer,
        'c_rs_short': c_rs_short, 'c_above_ma20': c_above_ma20, 'c_rev_momentum': c_rev_momentum,
        'c_d_trend_down': c_d_trend_down, 'c_d_breakdown': c_d_breakdown,
        'c_d_deep_bear': c_d_deep_bear, 'c_d_macd_bearish': c_d_macd_bearish,
        'c_d_weak_close': c_d_weak_close,
        'exit_c1': (v_close < v_ma10) & has_ma10,
        'exit_c2': ((v_open - v_close) > (0.015 * v_close)) & (v_vol > v_vol_ma5 * 2.0),
    }


def _build_position(signals, technicals, data_dict, params, minervini_mode):
    """
    5. 部位重建 V3.1 (Position Reconstruction) + Dynamic Exposure 資金配置
    從信號生成最終部位 DataFrame。

    Returns:
        dict: 包含 final_pos, hedge_factor, 以及用於 raw_mode 的輔助數據
    """
    p = params or {}
    _min_score         = float(p.get('min_score',         4))
    _max_per_industry  = int(p.get('max_per_industry',    0))
    _time_stop_days    = int(p.get('time_stop_days',      0))
    _trail_stop        = float(p.get('trail_stop',        0.18))

    close = data_dict['close']
    master_index = data_dict['master_index']
    benchmark_close = data_dict['benchmark_close']

    v_close = technicals['v_close']
    v_hv_compressed = technicals['v_hv_compressed']
    v_z_score = technicals['v_z_score']
    v_macd_hist = technicals['v_macd_hist']
    v_weekly_uptrend = technicals['v_weekly_uptrend']
    v_rev_growth = technicals['v_rev_growth']
    v_rev_current = technicals['v_rev_current']
    v_rev_12m_max = technicals['v_rev_12m_max']
    v_eps_sum = technicals['v_eps_sum']
    v_op_income = technicals['v_op_income']
    v_roe = technicals['v_roe']
    v_op_margin = technicals['v_op_margin']
    v_pe = technicals['v_pe']
    v_inst_streak = technicals['v_inst_streak']
    v_etf_ok = technicals['v_etf_ok']

    sig_a = signals['sig_a']
    sig_b = signals['sig_b']
    sig_c = signals['sig_c']
    sig_d = signals['sig_d']
    sig_e = signals['sig_e']
    long_entries_ab = signals['long_entries_ab']
    long_exits_ab = signals['long_exits_ab']
    exit_c = signals['exit_c']
    exit_d = signals['exit_d']
    exit_e = signals['exit_e']
    score_c = signals['score_c']
    score_d = signals['score_d']
    score_e = signals['score_e']
    c_strong_close = signals['c_strong_close']
    c_rs = signals['c_rs']
    c_bb_tight = signals['c_bb_tight']
    c_full_minervini = signals['c_full_minervini']
    c_rs_rank_70 = signals['c_rs_rank_70']
    c_52w_new_high = signals['c_52w_new_high']
    c_vol_dry = signals['c_vol_dry']

    # ==========================================
    # 5. 部位重建 V3.1 (Position Reconstruction)
    # ==========================================
    #
    # [V3 關鍵修正] 進場/出場優先順序
    # 舊邏輯: entries 先寫, exits 後寫 → 同天進出場時, exit 覆蓋 entry → 永遠不持倉
    # 新邏輯: exits 先寫, entries 後寫 → 進場信號優先 (買入信號比出場信號更有信息量)
    #
    # [V3.1 新增] 三軌部位系統:
    # Track AB: Signal A + B (出場: MA60 跌破) — 原有邏輯
    # Track C:  Signal C 動能追蹤 (出場: MA10 跌破 / 爆量長黑) — 更緊出場
    # 合併: np.maximum (任一信號仍看多則持有)
    # ==========================================

    # [V3] 優化評分系統 (基本面全部在這裡, NaN→0 不會失敗)
    score = np.ones_like(v_close)

    # 基本面加分 (有數據就加分, NaN→0 不扣分)
    score += (v_rev_growth > 30).astype(int)                      # +1: 營收成長 > 30%
    score += (v_rev_current >= v_rev_12m_max).astype(int)         # +1: 營收創新高
    score += (v_eps_sum > 0).astype(int)                          # +1: 近四季 EPS 為正
    score += (v_op_income > 0).astype(int)                        # +1: 營業利益為正
    score += ((v_roe > 15) & (v_op_margin > 15)).astype(int)      # +1: 高品質公司
    score += ((v_pe > 0) & (v_pe < 25)).astype(int)               # +1: 合理估值

    # 技術面加分
    score += c_strong_close.astype(int)                            # +1: 強勢收盤
    score += c_rs.astype(int)                                      # +1: 相對強度
    score += c_bb_tight.astype(int)                                # +1: 波動收縮
    score += (v_macd_hist > 0).astype(int)                         # +1: [V3.2] MACD 動能正向
    score += (v_weekly_uptrend > 0).astype(float) * 0.1             # +0.1: [V3.3] 週線多頭排列 (破平用)

    # [V3.4] Minervini + E&M Score 模式加分
    if minervini_mode == 'score':
        score += (c_full_minervini > 0).astype(int) * 2               # +2: Minervini 趨勢模板合規
        score += (c_rs_rank_70 > 0).astype(int)                       # +1: RS Rank > 70
        score += (c_vol_dry > 0).astype(int)                          # +1: VCP 量縮
        score += (c_52w_new_high > 0).astype(int)                     # +1: 52 週新高 (E&M)

    # Z-score & HV 正規化信號加分
    score += ((v_z_score > 1.0) & (v_z_score < 3.0)).astype(int)   # +1: 統計顯著突破 (1~3σ)
    score += (v_z_score < -2.0).astype(int)                         # +1: 極端超賣 (均值回歸機會)
    score += (v_hv_compressed > 0).astype(int)                      # +1: 波動率收縮 (蓄勢待發)

    # 籌碼面加分
    score += v_inst_streak.astype(int)                             # +1: 法人連續買超

    # Signal B (均值回歸) 上限 score=3，降低風險配置
    sig_b_only = sig_b & ~sig_a
    score = np.where(sig_b_only, np.minimum(score, 3), score)

    # === Track AB: Signal A + B 部位 (統一出場: MA60 跌破) ===
    v_pos_ab = np.full_like(v_close, np.nan)
    v_pos_ab[long_exits_ab] = 0                          # Step 1: 出場信號
    v_pos_ab[long_entries_ab] = score[long_entries_ab]    # Step 2: 進場覆蓋

    df_ab = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    df_ab[:] = v_pos_ab
    df_ab = df_ab.ffill().fillna(0)

    # === Track C: Signal C 動能追蹤部位 ===
    v_pos_c = np.full_like(v_close, np.nan)
    v_pos_c[exit_c] = 0                                  # Step 1: 動能出場 (MA10/爆量長黑)
    v_pos_c[sig_c] = score_c[sig_c]                      # Step 2: 動能進場覆蓋

    df_c = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    df_c[:] = v_pos_c
    df_c = df_c.ffill().fillna(0)

    # === [V3.4] Track E: Minervini+E&M 混合部位 (signal_e 模式) ===
    if minervini_mode == 'signal_e':
        v_pos_e = np.full_like(v_close, np.nan)
        v_pos_e[exit_e] = 0
        v_pos_e[sig_e] = score_e[sig_e]
        df_e = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
        df_e[:] = v_pos_e
        df_e = df_e.ffill().fillna(0)
    else:
        df_e = pd.DataFrame(0, index=master_index, columns=close.columns)

    # === Track D: Signal D 空頭部位 - [V3.2 新增] ===
    v_pos_d = np.full_like(v_close, np.nan)
    v_pos_d[exit_d] = 0                                    # Step 1: 空頭出場 (回補)
    v_pos_d[sig_d] = score_d[sig_d]                        # Step 2: 空頭進場 (負值)

    df_d = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    df_d[:] = v_pos_d
    df_d = df_d.ffill().fillna(0)

    # === [V3.2] 多空合併 + Top-N (四軌系統: A + B + C 多頭 + D 空頭) ===
    MAX_CONCURRENT_LONG = 8     # 多頭最多 8 檔 (從 10 降為 8)
    MAX_CONCURRENT_SHORT = 2    # 空頭最多 2 檔 (新增)
    MAX_CONCURRENT_TOTAL = 10   # 總部位不變

    # Step 1: 合併多頭部位 (AB + C + E)
    long_pos = pd.DataFrame(
        np.maximum(np.maximum(df_ab.values, df_c.values), df_e.values),
        index=master_index, columns=close.columns
    )

    # [實驗] Score 門檻過濾: 只保留 score >= min_score 的部位
    if _min_score > 0:
        long_pos[long_pos < _min_score] = 0

    # [實驗] 產業集中度限制: 每產業最多 N 檔
    if _max_per_industry > 0:
        try:
            ind_info = data.get('company_basic_info')
            if ind_info is not None and '產業類別' in ind_info.columns:
                stock_industry = ind_info['產業類別']
            else:
                stock_industry = pd.Series('Unknown', index=close.columns)
        except Exception:
            stock_industry = pd.Series('Unknown', index=close.columns)

        for i in range(len(long_pos)):
            row = long_pos.iloc[i]
            active = row[row > 0].sort_values(ascending=False)
            if len(active) == 0:
                continue
            ind_count = {}
            keep = set()
            for stock_id in active.index:
                ind = stock_industry.get(str(stock_id), 'Unknown')
                ind_count[ind] = ind_count.get(ind, 0) + 1
                if ind_count[ind] <= _max_per_industry:
                    keep.add(stock_id)
            drop = set(active.index) - keep
            if drop:
                long_pos.iloc[i, long_pos.columns.isin(drop)] = 0

    # Step 2: 多頭 Top-N 選股 (Top-8)
    long_rank = long_pos.rank(axis=1, method='first', ascending=False)
    long_top = (long_rank <= MAX_CONCURRENT_LONG) & (long_pos > 0)
    long_pos = long_pos.where(long_top, 0)

    # [V3.7] Signal D 轉避險指標: 不做空個股, 用空頭信號數量減倉多頭
    # 原理: Signal D 選股做空長期虧損, 但作為「市場危險信號」能有效預警
    short_signal_count = (df_d < 0).sum(axis=1)  # 每天有幾檔觸發空頭信號
    hedge_factor = pd.Series(1.0, index=master_index)
    hedge_factor[short_signal_count >= 1] = 0.7   # 1+ 空頭信號 → 多頭減倉 30%
    hedge_factor[short_signal_count >= 2] = 0.4   # 2+ 空頭信號 → 多頭減倉 60%

    long_hedge_mask = long_pos > 0
    long_pos[long_hedge_mask] = long_pos[long_hedge_mask].mul(hedge_factor, axis=0)[long_hedge_mask]

    # Step 3: 最終部位 = 純多頭 (不再持有空頭部位)
    final_pos = long_pos

    # Limit short positions (Signal D) to max 2 concurrent
    MAX_CONCURRENT_SHORT = 2
    short_mask = final_pos < 0
    short_pos_only = final_pos.where(short_mask, 0)
    short_rank = short_pos_only.abs().rank(axis=1, method='first', ascending=False)
    short_top = (short_rank <= MAX_CONCURRENT_SHORT) & short_mask
    final_pos = final_pos.where(~short_mask, 0)  # Zero out all shorts
    final_pos = final_pos + short_pos_only.where(short_top, 0)  # Add back top shorts

    # [V3.2] 防禦性檢查: NaN/Inf 清理
    final_pos = final_pos.replace([np.inf, -np.inf], 0).fillna(0)

    # [實驗] 時間停損: 持倉超過 N 天且未獲利 → 出場
    if _time_stop_days > 0:
        # 計算每個部位的持續天數 (連續非零)
        is_holding = (final_pos != 0).astype(int)
        # 累計持倉天數: 遇到 0 重置
        hold_days = is_holding.copy().astype(float)
        for i in range(1, len(hold_days)):
            mask = is_holding.iloc[i] > 0
            hold_days.iloc[i, mask] = hold_days.iloc[i-1][mask] + 1

        # 取得持倉期間報酬 (簡易: 當前價 vs N天前價)
        close_df = close.reindex(final_pos.index).ffill()
        price_change = close_df / close_df.shift(_time_stop_days) - 1
        # 持倉超過 N 天且報酬 <= 0 → 強制出場
        time_stop_mask = (hold_days >= _time_stop_days) & (price_change <= 0)
        final_pos[time_stop_mask] = 0

    # 信號品質檢查
    if final_pos.abs().sum().sum() == 0:
        import logging as _log
        _log.warning("WARNING: final_pos 全為 0，策略沒有任何信號觸發")

    # ==========================================
    # 5.5 Dynamic Exposure 資金配置 (V3.5 WFO 驗證冠軍)
    # ==========================================
    # 大盤(0050) > MA60 → 全倉 100%
    # MA60 > 大盤 > MA120 → 60%
    # 大盤 < MA120 → 30%
    # 空頭在熊市反而加碼 (乘以 2-exposure)
    bench_series = benchmark_close
    bench_ma60_s = bench_series.rolling(60).mean()
    bench_ma120_s = bench_series.rolling(120).mean()

    exposure = pd.Series(1.0, index=final_pos.index)
    bench_aligned = bench_series.reindex(final_pos.index).ffill()
    bma60_aligned = bench_ma60_s.reindex(final_pos.index).ffill()
    bma120_aligned = bench_ma120_s.reindex(final_pos.index).ffill()

    exposure[bench_aligned <= bma60_aligned] = 0.6
    exposure[bench_aligned <= bma120_aligned] = 0.3
    exposure[bench_aligned > bma60_aligned] = 1.0

    # [V3.6] Score Weight: 高分股多買、低分少買 (score/8, 上限1.5)
    alloc_pos = final_pos.copy()
    alloc_pos[alloc_pos > 0] = (alloc_pos[alloc_pos > 0] / 8.0).clip(lower=0.3, upper=1.5)

    # [V3.7] 純多頭策略: 只做多頭 Dynamic Exposure 調整 (不再有空頭加碼)
    long_mask_alloc = (alloc_pos > 0)
    alloc_pos[long_mask_alloc] = alloc_pos[long_mask_alloc].mul(exposure, axis=0)[long_mask_alloc]

    final_pos = alloc_pos.replace([np.inf, -np.inf], 0).fillna(0)

    return {
        'final_pos': final_pos,
        'hedge_factor': hedge_factor,
        'MAX_CONCURRENT_TOTAL': MAX_CONCURRENT_TOTAL,
    }


def _run_simulation(final_pos, signals, technicals, data_dict, params,
                    minervini_mode, stop_loss, take_profit, sim_start, sim_end,
                    hedge_factor, max_concurrent_total):
    """
    6. 診斷 & 執行回測
    記錄診斷日誌並呼叫 safe_finlab_sim 執行回測。

    Returns:
        backtest report 物件
    """
    p = params or {}
    _trail_stop = float(p.get('trail_stop', 0.18))

    benchmark_close = data_dict['benchmark_close']
    close = data_dict['close']
    master_index = data_dict['master_index']

    v_bullish = signals['v_bullish']
    v_bearish = signals['v_bearish']
    v_liq = signals['v_liq']
    v_etf_ok = technicals['v_etf_ok']
    v_macd_hist = technicals['v_macd_hist']
    v_hv_compressed = technicals['v_hv_compressed']
    v_z_score = technicals['v_z_score']

    sig_a = signals['sig_a']
    sig_b = signals['sig_b']
    sig_c = signals['sig_c']
    sig_d = signals['sig_d']
    long_exits_ab = signals['long_exits_ab']
    exit_c = signals['exit_c']
    exit_d = signals['exit_d']

    # ==========================================
    # 6. 診斷 & 執行回測
    # ==========================================
    import logging
    import os
    import sys

    log_file = "finlab_debug.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.INFO)

    logging.info("=" * 60)
    logging.info(f"--- Isaac V3.7{minervini_mode or ''}: 準備進入 backtest.sim ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    # 信號觸發統計
    sig_a_count = sig_a.sum() if hasattr(sig_a, 'sum') else 0
    sig_b_count = sig_b.sum() if hasattr(sig_b, 'sum') else 0
    sig_c_count = sig_c.sum() if hasattr(sig_c, 'sum') else 0
    sig_d_count = sig_d.sum() if hasattr(sig_d, 'sum') else 0
    exit_ab_count = long_exits_ab.sum() if hasattr(long_exits_ab, 'sum') else 0
    exit_count_c = exit_c.sum() if hasattr(exit_c, 'sum') else 0
    exit_d_count = exit_d.sum() if hasattr(exit_d, 'sum') else 0
    logging.info(f"Signal A 觸發次數 (成長突破): {sig_a_count}")
    logging.info(f"Signal B 觸發次數 (均值回歸): {sig_b_count}")
    logging.info(f"Signal C 觸發次數 (動能追蹤): {sig_c_count}")
    logging.info(f"Signal D 觸發次數 (熊市放空): {sig_d_count}")
    logging.info(f"出場信號 AB (MA60): {exit_ab_count}")
    logging.info(f"出場信號 C (MA10/爆量長黑): {exit_count_c}")
    logging.info(f"出場信號 D (MA20回復/RSI<25/爆量長紅/市場轉多): {exit_d_count}")

    non_zero_days = (final_pos.abs().sum(axis=1) > 0).sum()
    long_days = (final_pos > 0).any(axis=1).sum()
    hedge_active_days = (hedge_factor < 1.0).sum()
    logging.info(f"有持倉的天數: {non_zero_days} / {len(final_pos)}")
    logging.info(f"有多頭持倉天數: {long_days}")
    logging.info(f"空頭避險啟動天數: {hedge_active_days}")

    # 各條件診斷
    c_trend = signals['c_trend']
    c_breakout = signals['c_breakout']
    c_hv_ok = signals['c_hv_ok']
    c_safe_supply = signals['c_safe_supply']
    c_oversold = signals['c_oversold']
    c_rsi_panic = signals['c_rsi_panic']
    c_hammer = signals['c_hammer']
    c_rs_short = signals['c_rs_short']
    c_above_ma20 = signals['c_above_ma20']
    c_rev_momentum = signals['c_rev_momentum']
    c_d_trend_down = signals['c_d_trend_down']
    c_d_breakdown = signals['c_d_breakdown']
    c_d_deep_bear = signals['c_d_deep_bear']
    c_d_macd_bearish = signals['c_d_macd_bearish']

    conditions_debug = {
        'v_bullish': v_bullish.sum(),
        'c_trend': c_trend.sum(),
        'c_breakout': c_breakout.sum(),
        'c_hv_ok (HV正常區間)': c_hv_ok.sum(),
        'v_liq': v_liq.sum(),
        'c_safe_supply': c_safe_supply.sum(),
        'v_etf_ok': v_etf_ok.sum(),
        'c_oversold': c_oversold.sum(),
        'c_rsi_panic': c_rsi_panic.sum(),
        'c_hammer': c_hammer.sum(),
        'c_rs_short (20日RS)': c_rs_short.sum(),
        'c_above_ma20 (月線之上)': c_above_ma20.sum(),
        'c_rev_momentum (3月營收動能)': c_rev_momentum.sum(),
        'c_macd_positive (MACD多頭)': (v_macd_hist > 0).sum(),
        'exit_c1 (跌破MA10)': signals['exit_c1'].sum(),
        'exit_c2 (爆量長黑)': signals['exit_c2'].sum(),
        'v_bearish (空頭市場)': v_bearish.sum(),
        'c_d_trend_down (空頭排列)': c_d_trend_down.sum(),
        'c_d_breakdown (破20日低帶量)': c_d_breakdown.sum(),
        'c_d_deep_bear (低於MA120)': c_d_deep_bear.sum(),
        'c_d_macd_bearish (MACD空頭)': c_d_macd_bearish.sum(),
        'z_score>1 (統計突破)': (v_z_score > 1.0).sum(),
        'z_score<-2 (極端超賣)': (v_z_score < -2.0).sum(),
        'hv_compressed (波動收縮)': (v_hv_compressed > 0).sum(),
    }
    logging.info("--- 各條件觸發統計 ---")
    for cond_name, cond_count in conditions_debug.items():
        logging.info(f"  {cond_name}: {cond_count}")

    # === 功能2: Signal A 進場條件漏斗分析 ===
    funnel = _compute_signal_funnel(signals, technicals, data_dict)
    logging.info("--- Signal A 進場條件漏斗 (逐步篩選) ---")
    print("===== Signal A Entry Funnel =====")
    for name, single, cumulative in funnel:
        line = f"  {name:<35s} | 單獨: {single:>12,} | 累積AND: {cumulative:>10,}"
        logging.info(line)
        print(line)

    try:
        from data.provider import safe_finlab_sim

        sim_kwargs = {
            # [V3.3 修正] 移除 resample='D'：
            # resample='D' 導致 FinLab 將每天視為獨立交易 → 所有持有天數都顯示 1
            # 不設 resample 時，FinLab 根據 position 值的實際變化來判斷交易進出場
            'name': f'Isaac V3.7{minervini_mode or ""}',
            'upload': False,
            'trail_stop': _trail_stop,
            'position_limit': 1.0 / max_concurrent_total,  # 每檔上限 = 1/N (10檔→10%)
            'touched_exit': False,
        }
        if stop_loss is not None:
            sim_kwargs['stop_loss'] = stop_loss
        if take_profit is not None:
            sim_kwargs['take_profit'] = take_profit

        # WFO 窗口: 用 position 切片實現日期範圍限制
        sim_pos = final_pos
        if sim_start is not None or sim_end is not None:
            if sim_start and sim_end:
                sim_pos = final_pos.loc[sim_start:sim_end]
            elif sim_start:
                sim_pos = final_pos.loc[sim_start:]
            elif sim_end:
                sim_pos = final_pos.loc[:sim_end]
            logging.info(f"WFO 窗口切片: {sim_pos.index[0]} ~ {sim_pos.index[-1]} ({len(sim_pos)} 天)")

        report = safe_finlab_sim(sim_pos, **sim_kwargs)

        # 回測後診斷
        try:
            stats = report.get_stats()
            trades = report.get_trades()
            logging.info(f"回測完成 - trades count: {len(trades)}")
            if hasattr(stats, 'get'):
                logging.info(f"  cagr = {stats.get('cagr', 'MISSING')}")
                logging.info(f"  max_drawdown = {stats.get('max_drawdown', 'MISSING')}")
                logging.info(f"  win_ratio = {stats.get('win_ratio', 'MISSING')}")

            # === 功能1: 逐年完整績效表 (Equity + Trade 雙維度) ===
            equity = getattr(report, 'creturn', None)
            if equity is not None and len(equity) > 0:
                header = "===== Yearly Equity Performance ====="
                logging.info(header)
                print(header)
                fmt_h = (f"{'Year':>6} | {'AnnRet':>9} | {'MaxDD':>9} | {'Sharpe':>8} | "
                         f"{'Trades':>6} | {'AvgRet':>9} | {'WinRate':>9} | {'MAE':>9} | {'MFE':>9}")
                logging.info(fmt_h)
                print(fmt_h)
                logging.info("-" * len(fmt_h))
                print("-" * len(fmt_h))

                eq_years = equity.groupby(equity.index.year)
                trades_copy = trades.copy()
                if len(trades) > 0 and 'entry_date' in trades.columns:
                    trades_copy['year'] = pd.to_datetime(trades_copy['entry_date']).dt.year
                else:
                    trades_copy['year'] = []

                all_years = sorted(set(eq_years.groups.keys()))
                for yr in all_years:
                    # Equity-based metrics
                    eq_yr = eq_years.get_group(yr)
                    yr_start = eq_yr.iloc[0]
                    yr_end = eq_yr.iloc[-1]
                    ann_ret = (yr_end / yr_start - 1) * 100
                    # Yearly MDD from equity curve
                    running_max = eq_yr.cummax()
                    dd = (eq_yr - running_max) / running_max
                    yr_mdd = dd.min() * 100
                    # Yearly Sharpe (daily returns annualized)
                    daily_ret = eq_yr.pct_change().dropna()
                    if len(daily_ret) > 1 and daily_ret.std() > 0:
                        yr_sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5)
                    else:
                        yr_sharpe = float('nan')

                    # Trade-based metrics for this year
                    yr_trades = trades_copy[trades_copy['year'] == yr] if len(trades_copy) > 0 else pd.DataFrame()
                    n_trades = len(yr_trades)
                    ret_col = 'return' if 'return' in yr_trades.columns else None
                    mae_col = 'mae' if 'mae' in yr_trades.columns else None
                    mfe_col = 'gmfe' if 'gmfe' in yr_trades.columns else ('bmfe' if 'bmfe' in yr_trades.columns else None)

                    avg_ret = yr_trades[ret_col].mean() * 100 if (ret_col and n_trades > 0) else float('nan')
                    win_rate = (yr_trades[ret_col] > 0).mean() * 100 if (ret_col and n_trades > 0) else float('nan')
                    mae_val = yr_trades[mae_col].mean() * 100 if (mae_col and n_trades > 0) else float('nan')
                    mfe_val = yr_trades[mfe_col].mean() * 100 if (mfe_col and n_trades > 0) else float('nan')

                    line = (f"{yr:>6} | {ann_ret:>8.2f}% | {yr_mdd:>8.2f}% | {yr_sharpe:>8.2f} | "
                            f"{n_trades:>6} | {avg_ret:>8.2f}% | {win_rate:>8.1f}% | {mae_val:>8.2f}% | {mfe_val:>8.2f}%")
                    logging.info(line)
                    print(line)

            # === 功能4: 持倉異動日誌 (Position Change Log) ===
            _log_position_changes(sim_pos, logging)

        except Exception as diag_e:
            logging.warning(f"診斷失敗: {diag_e}")

        logging.info(f"Isaac V3.7{minervini_mode or ''} backtest.sim 執行成功")
        return report

    except Exception as e:
        logging.error(f"策略層級崩潰: {str(e)}", exc_info=True)
        raise e


def _log_position_changes(final_pos, logger):
    """
    功能4: 持倉異動日誌
    逐日比較持倉變化，記錄進場/出場/替換事件到 log。
    只記錄有異動的日期，避免 log 過大。
    """
    prev_holdings = set()
    for i in range(len(final_pos)):
        date = final_pos.index[i]
        row = final_pos.iloc[i]
        curr_holdings = set(row[row > 0].index)

        entered = curr_holdings - prev_holdings
        exited = prev_holdings - curr_holdings
        stayed = curr_holdings & prev_holdings

        if entered or exited:
            date_str = date.strftime('%Y-%m-%d')
            parts = [f"[持倉異動] {date_str} |"]
            if entered and exited:
                # 替換: 有進有出
                entered_str = ','.join(sorted(entered))
                exited_str = ','.join(sorted(exited))
                parts.append(f"替換: OUT({exited_str}) → IN({entered_str})")
            elif entered:
                parts.append(f"新進場: {','.join(sorted(entered))}")
            elif exited:
                parts.append(f"出場: {','.join(sorted(exited))}")
            parts.append(f"| 持倉({len(curr_holdings)}檔): {','.join(sorted(curr_holdings))}")
            logger.info(' '.join(parts))

        prev_holdings = curr_holdings


def _compute_signal_funnel(signals, technicals, data_dict):
    """
    功能2: 進場條件漏斗分析
    逐步疊加 Signal A 的每個條件，記錄每層剩餘的股票數量。
    返回 list of (條件名, 該條件為True的日股數, 逐步AND後剩餘日股數)
    """
    v_etf_ok = technicals['v_etf_ok']
    n_total = technicals['v_close'].shape[1]  # 總股票數

    # Signal A 的條件鏈 (順序與 _generate_signals 一致)
    conditions = [
        ('全部股票', np.ones_like(technicals['v_close'], dtype=bool)),
        ('非ETF (v_etf_ok)', np.broadcast_to(v_etf_ok, technicals['v_close'].shape)),
        ('流動性 > 50萬 (v_liq)', signals['v_liq']),
        ('大盤多頭 (v_bullish)', signals['v_bullish']),
        ('趨勢排列 Close>MA20>MA60', signals['c_trend']),
        ('HV正常區間', signals['c_hv_ok']),
        ('非套牢區 (c_safe_supply)', signals['c_safe_supply']),
        ('20日新高突破+帶量 (c_breakout)', signals['c_breakout']),
    ]

    funnel = []
    cumulative = np.ones_like(technicals['v_close'], dtype=bool)
    for name, cond in conditions:
        cond_bool = np.asarray(cond, dtype=bool)
        # 確保 shape 可以 broadcast
        if cond_bool.shape != cumulative.shape:
            cond_bool = np.broadcast_to(cond_bool, cumulative.shape)
        single_count = cond_bool.sum()
        cumulative = cumulative & cond_bool
        cum_count = cumulative.sum()
        funnel.append((name, int(single_count), int(cum_count)))

    return funnel


def get_current_holdings(api_token, params=None):
    """
    功能3: 現倉追蹤
    返回策略目前持有的股票清單，包含股票名稱、進場時間、目前獲利率。
    """
    result = run_isaac_strategy(api_token, raw_mode=True, params=params)
    final_pos = result['final_pos']
    close = result['close']

    # 最後一天的持倉
    last_date = final_pos.index[-1]
    last_row = final_pos.iloc[-1]
    holdings = last_row[last_row > 0].sort_values(ascending=False)

    if len(holdings) == 0:
        return pd.DataFrame(columns=['股票代碼', '股票名稱', '進場日期', '進場價', '現價', '獲利率%', 'Score'])

    # 取得股票名稱
    try:
        from finlab import data as fdata
        company_info = fdata.get('company_basic_info')
        name_map = company_info['公司簡稱'] if '公司簡稱' in company_info.columns else pd.Series()
    except Exception:
        name_map = pd.Series()

    rows = []
    for stock_id in holdings.index:
        score = holdings[stock_id]
        # 找進場日: 從最後一天往前找，第一個 pos==0 的日期之後就是進場日
        pos_series = final_pos[stock_id]
        # 從尾端往前找到第一個 0 或 NaN 的位置
        nonzero_mask = pos_series > 0
        entry_idx = None
        for j in range(len(pos_series) - 1, -1, -1):
            if not nonzero_mask.iloc[j]:
                entry_idx = j + 1
                break
        if entry_idx is None or entry_idx >= len(pos_series):
            entry_idx = 0  # 從頭就持有

        entry_date = pos_series.index[entry_idx]
        # 進場價 = 進場日的收盤價
        close_aligned = close.reindex(final_pos.index).ffill()
        entry_price = close_aligned.loc[entry_date, stock_id] if stock_id in close_aligned.columns else float('nan')
        current_price = close_aligned.iloc[-1][stock_id] if stock_id in close_aligned.columns else float('nan')
        pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else float('nan')

        stock_name = name_map.get(stock_id, name_map.get(str(stock_id), ''))

        rows.append({
            '股票代碼': stock_id,
            '股票名稱': stock_name,
            '進場日期': entry_date.strftime('%Y-%m-%d'),
            '進場價': round(entry_price, 2),
            '現價': round(current_price, 2),
            '獲利率%': round(pnl_pct, 2),
            'Score': round(score, 1),
        })

    return pd.DataFrame(rows)


def get_signal_funnel(api_token, params=None):
    """
    功能2: 取得進場條件漏斗分析結果
    """
    data_dict = _fetch_data(api_token)
    technicals = _compute_technicals(data_dict)
    p = params or {}
    signals = _generate_signals(technicals, data_dict, p, 'signal_e')
    funnel = _compute_signal_funnel(signals, technicals, data_dict)
    return funnel


def get_replacement_preview(api_token, params=None):
    """
    功能4 (daily report 部分): 今日 vs 昨日持倉比較，預告替換
    """
    result = run_isaac_strategy(api_token, raw_mode=True, params=params)
    final_pos = result['final_pos']
    close = result['close']

    if len(final_pos) < 2:
        return {'today': set(), 'yesterday': set(), 'entered': set(), 'exited': set()}

    today_row = final_pos.iloc[-1]
    yesterday_row = final_pos.iloc[-2]

    today_holdings = set(today_row[today_row > 0].index)
    yesterday_holdings = set(yesterday_row[yesterday_row > 0].index)

    entered = today_holdings - yesterday_holdings
    exited = yesterday_holdings - today_holdings

    # 取得股票名稱
    try:
        from finlab import data as fdata
        company_info = fdata.get('company_basic_info')
        name_map = company_info['公司簡稱'] if '公司簡稱' in company_info.columns else {}
    except Exception:
        name_map = {}

    def _enrich(stock_set, pos_row):
        result = []
        close_last = close.reindex(final_pos.index).ffill().iloc[-1]
        for sid in sorted(stock_set):
            name = name_map.get(sid, name_map.get(str(sid), ''))
            score = pos_row.get(sid, 0)
            price = close_last.get(sid, float('nan'))
            result.append({'代碼': sid, '名稱': name, 'Score': round(float(score), 1), '現價': round(float(price), 2)})
        return result

    return {
        'date': final_pos.index[-1].strftime('%Y-%m-%d'),
        'holdings': _enrich(today_holdings, today_row),
        'entered': _enrich(entered, today_row),
        'exited': _enrich(exited, yesterday_row),
    }


def run_isaac_strategy(api_token, stop_loss=None, take_profit=None, minervini_mode='signal_e',
                       params=None, sim_start=None, sim_end=None, raw_mode=False):
    """
    執行 Isaac V3.7 策略回測。
    V3.7: Signal D 轉避險指標 + Score Weight + Dynamic Exposure

    params (dict, 可選): 覆蓋預設參數
        trail_stop       (float) : 追蹤停損比例，預設 0.18
        rsi_threshold    (int)   : Signal B RSI 超賣門檻，預設 28
        volume_mult      (float) : 突破量能倍率（Signal A/B），預設 1.5
        supply_danger_pct(float) : 供給區安全距離，預設 0.97
        liq_min          (int)   : 流動性門檻（均量股數），預設 500000
    sim_start (str): 回測起始日 (YYYY-MM-DD)，用於 WFO 窗口
    sim_end   (str): 回測結束日 (YYYY-MM-DD)，用於 WFO 窗口
    raw_mode (bool): 若 True，不執行 sim，返回 final_pos 和相關數據供外部配置
    """
    if stop_loss is not None: stop_loss = float(stop_loss)
    if take_profit is not None: take_profit = float(take_profit)

    # --- 可調參數 (WFO 網格搜索的優化目標) ---
    p = params or {}

    # Step 1: 數據抓取
    data_dict = _fetch_data(api_token)

    # Step 2: 技術指標計算
    technicals = _compute_technicals(data_dict)

    # Step 3: 信號生成
    signals = _generate_signals(technicals, data_dict, p, minervini_mode)

    # Step 4: 部位建構
    position_result = _build_position(signals, technicals, data_dict, p, minervini_mode)
    final_pos = position_result['final_pos']
    hedge_factor = position_result['hedge_factor']
    max_concurrent_total = position_result['MAX_CONCURRENT_TOTAL']

    # --- raw_mode: 返回原始數據供外部資金配置測試 ---
    if raw_mode:
        _trail_stop = float(p.get('trail_stop', 0.18))
        sim_pos = final_pos
        if sim_start is not None or sim_end is not None:
            if sim_start and sim_end:
                sim_pos = final_pos.loc[sim_start:sim_end]
            elif sim_start:
                sim_pos = final_pos.loc[sim_start:]
            elif sim_end:
                sim_pos = final_pos.loc[:sim_end]
        return {
            'final_pos': sim_pos,
            'close': data_dict['close'],
            'etf_close': data_dict['benchmark_close'],
            'trail_stop': _trail_stop,
            'max_concurrent': max_concurrent_total,
        }

    # Step 5: 執行回測
    report = _run_simulation(
        final_pos, signals, technicals, data_dict, p,
        minervini_mode, stop_loss, take_profit, sim_start, sim_end,
        hedge_factor, max_concurrent_total,
    )
    return report
