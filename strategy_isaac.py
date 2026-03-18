from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_isaac_strategy(api_token, stop_loss=None, take_profit=None):
    from data_provider import sanitize_dataframe

    if api_token:
        finlab.login(api_token)

    if stop_loss is not None: stop_loss = float(stop_loss)
    if take_profit is not None: take_profit = float(take_profit)

    # ==========================================
    # 1. 數據抓取 (Fetch Data)
    # ==========================================
    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")

    # [Fix 3: Immutability] - 不修改 close.columns，避免污染 Finlab 全域快取
    master_index = close.index
    master_columns_str = close.columns.astype(str)

    def to_numpy(obj, obj_name="Unknown", is_benchmark=False):
        if obj is None: return np.nan
        if isinstance(obj, pd.DataFrame):
            obj = sanitize_dataframe(obj, source_name=obj_name)
        if isinstance(obj, pd.Series):
            obj = obj.reindex(master_index, method='ffill')
            return obj.fillna(0).values.reshape(-1, 1)
        elif isinstance(obj, pd.DataFrame):
            if not is_benchmark:
                df_temp = obj.copy()
                df_temp.columns = df_temp.columns.astype(str)
                df_aligned = df_temp.reindex(index=master_index, columns=master_columns_str, method='ffill')
                return df_aligned.fillna(0).values
            else:
                obj = obj.reindex(index=master_index, method='ffill')
                return obj.fillna(0).values
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(obj)

    # 價格數據
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

    # Bollinger Bandwidth 波動收縮濾網 (VCP 概念)
    bb_std = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_bandwidth = (bb_upper - bb_lower) / ma20
    bb_contracting = bb_bandwidth < bb_bandwidth.rolling(60).quantile(0.25)

    # 相對強度濾網 (120 日報酬率 vs 大盤)
    stock_ret_120 = close.pct_change(120)
    bench_ret_120 = benchmark_close.pct_change(120)
    rel_strength = stock_ret_120.sub(bench_ret_120, axis=0) > 0

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

    has_ma20 = v_ma20 > 0
    has_ma50 = v_ma50 > 0
    has_ma60 = v_ma60 > 0
    has_ma120 = v_ma120 > 0

    # 市場狀態濾網
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)
    v_bearish = (v_bench < v_bench_ma60 * 0.99) & (v_bench_ma60 > 0) & (v_bench > 0)

    # 流動性濾網 (降低門檻: 100萬→50萬股，增加選股範圍)
    v_liq = (v_vol_ma20 > 500000)

    # Supply Zone Filter (套牢區濾網)
    c_supply_danger = (v_close >= v_high_250 * 0.95) & (v_close < v_high_250) & (v_high_250 > 0)
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

    # === 訊號 A: 技術面突破 (Growth Breakout) ===
    # [V3] 只用技術面作為進場門檻, 基本面全部改為評分加分
    c_trend = (v_close > v_ma20) & (v_ma20 > v_ma60) & has_ma20 & has_ma60
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 1.5)

    sig_a = (v_bullish & c_trend & c_breakout & v_liq & c_safe_supply & v_etf_ok)

    # === 訊號 B: 均值回歸 (Reversion) ===
    c_oversold = (v_close < v_ma120) & has_ma120
    c_rsi_panic = v_rsi < 30
    c_vol_panic = v_vol > v_vol_ma20 * 1.5  # 放寬: 2x → 1.5x

    body = np.abs(v_close - v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low
    c_hammer = lower_shadow > (body * 1.5)  # 放寬: 2x → 1.5x

    sig_b = v_bullish & c_oversold & c_rsi_panic & c_hammer & v_liq & v_etf_ok

    # === 訊號 C: 放空 (Short) - DISABLED ===
    sig_c = np.zeros_like(v_close, dtype=bool)

    # ==========================================
    # 5. 部位重建 V3 (Position Reconstruction)
    # ==========================================
    #
    # [V3 關鍵修正] 進場/出場優先順序
    # 舊邏輯: entries 先寫, exits 後寫 → 同天進出場時, exit 覆蓋 entry → 永遠不持倉
    # 新邏輯: exits 先寫, entries 後寫 → 進場信號優先 (買入信號比出場信號更有信息量)
    # ==========================================

    long_entries = sig_a | sig_b
    short_entries = sig_c

    # [V3] Signal A 出場: 跌破 MA60 (給較大的波動空間)
    exit_a = (v_close < v_ma60) & has_ma60

    # [V3] Signal B 出場: 回復到 MA60 以上 (均值回歸完成) 或 RSI > 60
    # 這避免了 V2.1 中 "close < MA50 exit" 與 "close < MA120 entry" 的矛盾
    exit_b = ((v_close > v_ma60) & has_ma60) | (v_rsi > 60)

    # 統一出場 (保守取聯集)
    long_exits = exit_a  # 主要出場信號: 跌破 MA60
    short_exits = (v_close > v_ma20) | v_bullish | (v_rsi < 20)

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

    # 籌碼面加分
    score += v_inst_streak.astype(int)                             # +1: 法人連續買超

    # Signal B (均值回歸) 上限 score=3，降低風險配置
    sig_b_only = sig_b & ~sig_a
    score = np.where(sig_b_only, np.minimum(score, 3), score)

    # 構建模擬用的 DataFrame
    v_pos_long = np.full_like(v_close, np.nan)
    v_pos_short = np.full_like(v_close, np.nan)

    # [V3 關鍵] 先寫出場, 再寫進場 → 進場信號優先
    v_pos_long[long_exits] = 0              # Step 1: 出場信號
    v_pos_long[long_entries] = score[long_entries]  # Step 2: 進場信號覆蓋

    v_pos_short[short_exits] = 0
    v_pos_short[short_entries] = -0.5

    df_long = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    df_short = pd.DataFrame(np.nan, index=master_index, columns=close.columns)

    df_long[:] = v_pos_long
    df_short[:] = v_pos_short

    df_long = df_long.ffill().fillna(0)
    df_short = df_short.ffill().fillna(0)

    final_pos = df_long + df_short

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
    logging.info("--- Isaac V3: 準備進入 backtest.sim ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    # 信號觸發統計
    sig_a_count = sig_a.sum() if hasattr(sig_a, 'sum') else 0
    sig_b_count = sig_b.sum() if hasattr(sig_b, 'sum') else 0
    entry_count = long_entries.sum() if hasattr(long_entries, 'sum') else 0
    exit_count = long_exits.sum() if hasattr(long_exits, 'sum') else 0
    logging.info(f"Signal A 觸發次數: {sig_a_count}")
    logging.info(f"Signal B 觸發次數: {sig_b_count}")
    logging.info(f"進場信號總數: {entry_count}")
    logging.info(f"出場信號總數: {exit_count}")

    non_zero_days = (final_pos.abs().sum(axis=1) > 0).sum()
    logging.info(f"有持倉的天數: {non_zero_days} / {len(final_pos)}")

    # 各條件診斷
    conditions_debug = {
        'v_bullish': v_bullish.sum(),
        'c_trend': c_trend.sum(),
        'c_breakout': c_breakout.sum(),
        'v_liq': v_liq.sum(),
        'c_safe_supply': c_safe_supply.sum(),
        'v_etf_ok': v_etf_ok.sum(),
        'c_oversold': c_oversold.sum(),
        'c_rsi_panic': c_rsi_panic.sum(),
        'c_hammer': c_hammer.sum(),
    }
    logging.info("--- 各條件觸發統計 ---")
    for cond_name, cond_count in conditions_debug.items():
        logging.info(f"  {cond_name}: {cond_count}")

    try:
        from data_provider import safe_finlab_sim

        sim_kwargs = {
            'resample': 'D',
            'name': 'Isaac V3',
            'upload': False,
            'trail_stop': 0.15,
            'position_limit': 0.20,
            'touched_exit': True,
        }
        if stop_loss is not None:
            sim_kwargs['stop_loss'] = stop_loss
        if take_profit is not None:
            sim_kwargs['take_profit'] = take_profit

        report = safe_finlab_sim(final_pos, **sim_kwargs)

        # 回測後診斷
        try:
            stats = report.get_stats()
            trades = report.get_trades()
            logging.info(f"回測完成 - trades count: {len(trades)}")
            if hasattr(stats, 'get'):
                logging.info(f"  cagr = {stats.get('cagr', 'MISSING')}")
                logging.info(f"  max_drawdown = {stats.get('max_drawdown', 'MISSING')}")
                logging.info(f"  win_ratio = {stats.get('win_ratio', 'MISSING')}")
        except Exception as diag_e:
            logging.warning(f"診斷失敗: {diag_e}")

        logging.info("Isaac V3 backtest.sim 執行成功")
        return report

    except Exception as e:
        logging.error(f"策略層級崩潰: {str(e)}", exc_info=True)
        raise e
