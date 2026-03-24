"""
Momentum Trading Strategy (動能追蹤策略) - 獨立版本
=============================================
設計哲學:
  針對「無法估值但動能強勁」的轉機股，捨棄傳統 P/E 估值，
  擁抱相對強弱 (RS) + 營收動能加速 + K 線純粹跟隨。

進場條件:
  1. 大盤多頭 (benchmark > MA60)
  2. 短期相對強弱: 個股近 20 日報酬率 > 大盤
  3. 收盤 > 月線 (MA20) — 動能延續確認
  4. 帶量突破近期高點 (close > 20日高 & vol > 5日均量 1.5x)
  5. 連續 3 個月營收 YoY > 20% — 轉機故事持續中
  6. 流動性 & ETF 排除

出場條件:
  1. 跌破 10 日線 (MA10) — 快速動能停損
  2. 高檔爆量長黑 — 主力出貨信號 (跌幅>1.5% + 量>5日均量2x)

版本: 1.0
"""

from finlab import data
from finlab import backtest
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
import finlab


def compute_hv(price_series, look_back_period=50):
    """Historical Volatility (歷史波動率)"""
    log_return = np.log(price_series / price_series.shift(1))
    return log_return.rolling(look_back_period).std()


def run_momentum_strategy(api_token, stop_loss=None, take_profit=None):
    """
    執行動能追蹤策略回測。

    Args:
        api_token: FinLab API Token
        stop_loss: 固定停損比例 (可選)
        take_profit: 停利比例 (可選)

    Returns:
        FinLab backtest report
    """
    from data.provider import sanitize_dataframe, safe_finlab_sim

    if api_token:
        finlab.login(api_token)

    if stop_loss is not None: stop_loss = float(stop_loss)
    if take_profit is not None: take_profit = float(take_profit)

    # ==========================================
    # 1. 數據抓取 (Fetch Data)
    # ==========================================
    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")

    master_index = close.index
    master_columns_str = close.columns.astype(str)

    def to_numpy(obj, obj_name="Unknown", is_benchmark=False):
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

    # 價格數據
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # 大盤基準
    benchmark_close = data.get('price:收盤價')['0050']

    # 基本面數據 (只需要月營收)
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        rev_current = data.get('monthly_revenue:當月營收')
    except Exception:
        rev_growth = pd.DataFrame(0, index=master_index, columns=close.columns)
        rev_current = pd.DataFrame(0, index=master_index, columns=close.columns)

    # 籌碼面 (加分用)
    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except Exception:
        inst_net_buy = pd.DataFrame(0, index=master_index, columns=close.columns)

    # ==========================================
    # 2. 預計算 (Pandas 階段)
    # ==========================================

    # 移動平均線
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # 短期相對強弱 (20 日報酬率 vs 大盤)
    stock_ret_20 = close.pct_change(20, fill_method=None)
    bench_ret_20 = benchmark_close.pct_change(20, fill_method=None)
    rs_short_20 = stock_ret_20.sub(bench_ret_20, axis=0) > 0

    # 營收動能: 連續 2 個月營收 YoY > 15% (比3個月>20%更寬鬆)
    rev_yoy_m0 = rev_growth > 15
    rev_yoy_m1 = rev_growth.shift(22) > 15    # ~1 個月前
    rev_momentum_3m = rev_yoy_m0 & rev_yoy_m1

    # 20 日高點 (突破用)
    close_max_20 = close.rolling(20).max().shift(1)

    # 營收創新高
    rev_12m_max = rev_current.rolling(12).max()

    # Historical Volatility
    hv = compute_hv(close, 50)
    hv_q20 = hv.rolling(120).quantile(0.2)
    hv_compressed = hv < hv_q20

    # 法人連續買超
    inst_streak = (inst_net_buy.rolling(5).min() > 0)

    # 反向/槓桿 ETF 黑名單
    import re as _re
    etf_blacklist = pd.Series(False, index=master_columns_str)
    for col in master_columns_str:
        if _re.match(r'^00\d{3,}[RL]?$', col) or _re.match(r'^00\d{3,}', col):
            etf_blacklist[col] = True
    v_etf_ok = ~etf_blacklist.values.reshape(1, -1)

    # ==========================================
    # 3. NumPy 轉換
    # ==========================================
    v_close = to_numpy(close)
    v_open = to_numpy(open_)
    v_high = to_numpy(high)
    v_low = to_numpy(low)
    v_vol = to_numpy(vol)

    v_ma10 = to_numpy(ma10)
    v_ma20 = to_numpy(ma20)
    v_ma60 = to_numpy(ma60)
    v_vol_ma5 = to_numpy(vol_ma5)
    v_vol_ma20 = to_numpy(vol_ma20)
    v_close_max_20 = to_numpy(close_max_20)
    v_rs_short = to_numpy(rs_short_20)
    v_rev_momentum = to_numpy(rev_momentum_3m)
    v_hv_compressed = to_numpy(hv_compressed)
    v_inst_streak = to_numpy(inst_streak)
    v_rev_growth = to_numpy(rev_growth)
    v_rev_current = to_numpy(rev_current)
    v_rev_12m_max = to_numpy(rev_12m_max)

    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    # ==========================================
    # 4. 策略邏輯 (Momentum V1.1)
    # ==========================================

    has_ma10 = v_ma10 > 0
    has_ma20 = v_ma20 > 0
    has_ma60 = v_ma60 > 0

    # 市場狀態濾網
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)

    # 流動性濾網
    v_liq = (v_vol_ma20 > 500000)

    # === 進場條件 ===
    c_rs_short = v_rs_short > 0                                   # 20日報酬 > 大盤
    c_above_ma20 = (v_close > v_ma20) & has_ma20                  # 收盤 > 月線
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma5 * 1.5)  # 帶量突破
    c_rev_momentum = v_rev_momentum > 0                           # 連續3月營收 YoY > 20%

    sig_entry = (v_bullish & c_rs_short & c_above_ma20 & c_breakout
                 & c_rev_momentum & v_liq & v_etf_ok)

    # === 出場條件 ===
    # 1. 跌破 20 日線 (月線動能停損 — 比 MA10 寬鬆，避免被洗出)
    exit_ma = (v_close < v_ma20) & has_ma20
    # 2. 高檔爆量長黑 (主力出貨信號)
    c_bearish_body = (v_open - v_close) > (0.02 * v_close)        # 跌幅 > 2%
    c_high_vol_candle = v_vol > v_vol_ma5 * 2.5                   # 量 > 5日均量 2.5x
    exit_bearish = c_bearish_body & c_high_vol_candle

    sig_exit = exit_ma | exit_bearish

    # === 評分系統 (不含 P/E 估值) ===
    score = np.ones_like(v_close)
    score += c_rs_short.astype(int)                                # +1: 短期相對強弱
    score += c_rev_momentum.astype(int)                            # +1: 營收動能加速

    # 強勢收盤確認
    v_daily_range = v_high - v_low
    v_daily_range_safe = np.where(v_daily_range == 0, np.nan, v_daily_range)
    v_close_position = (v_close - v_low) / v_daily_range_safe
    v_close_position = np.nan_to_num(v_close_position, nan=0.0)
    c_strong_close = v_close_position >= 0.75
    score += c_strong_close.astype(int)                            # +1: 強勢收盤

    score += (v_vol > v_vol_ma5 * 2.0).astype(int)                # +1: 量能超強
    score += (v_hv_compressed > 0).astype(int)                     # +1: 波動收縮蓄勢
    score += v_inst_streak.astype(int)                             # +1: 法人連續買超

    # 營收創新高額外加分
    score += (v_rev_current >= v_rev_12m_max).astype(int)          # +1: 營收創新高
    score += (v_rev_growth > 50).astype(int)                       # +1: 營收超高速成長 (>50%)

    score = np.minimum(score, 8)                                   # 上限 8

    # ==========================================
    # 5. 部位重建
    # ==========================================
    v_pos = np.full_like(v_close, np.nan)

    # 先寫出場, 再寫進場 → 進場信號優先
    v_pos[sig_exit] = 0
    v_pos[sig_entry] = score[sig_entry]

    final_pos = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    final_pos[:] = v_pos
    final_pos = final_pos.ffill().fillna(0)

    # 防禦性清理
    final_pos = final_pos.replace([np.inf, -np.inf], 0).fillna(0)

    # Top-N 選股
    MAX_CONCURRENT = 10
    _pos_rank = final_pos.rank(axis=1, method='first', ascending=False)
    _top_n_mask = (_pos_rank <= MAX_CONCURRENT) & (final_pos > 0)
    final_pos = final_pos.where(_top_n_mask, 0)

    # 信號品質檢查
    if final_pos.abs().sum().sum() == 0:
        import logging as _log
        _log.warning("WARNING: final_pos 全為 0，Momentum 策略沒有任何信號觸發")

    # ==========================================
    # 6. 診斷 & 執行回測
    # ==========================================
    import logging
    log_file = "finlab_debug.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.INFO)

    logging.info("=" * 60)
    logging.info("--- Momentum V1.1: 準備進入 backtest.sim ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    entry_count = sig_entry.sum() if hasattr(sig_entry, 'sum') else 0
    exit_count = sig_exit.sum() if hasattr(sig_exit, 'sum') else 0
    logging.info(f"進場信號總數: {entry_count}")
    logging.info(f"出場信號總數: {exit_count}")

    non_zero_days = (final_pos.abs().sum(axis=1) > 0).sum()
    logging.info(f"有持倉的天數: {non_zero_days} / {len(final_pos)}")

    conditions_debug = {
        'v_bullish': v_bullish.sum(),
        'c_rs_short (20日RS)': c_rs_short.sum(),
        'c_above_ma20 (月線之上)': c_above_ma20.sum(),
        'c_breakout (帶量突破)': c_breakout.sum(),
        'c_rev_momentum (3月營收動能)': c_rev_momentum.sum(),
        'v_liq': v_liq.sum(),
        'exit_ma (跌破MA20)': exit_ma.sum(),
        'exit_bearish (爆量長黑)': exit_bearish.sum(),
    }
    logging.info("--- 各條件觸發統計 ---")
    for cond_name, cond_count in conditions_debug.items():
        logging.info(f"  {cond_name}: {cond_count}")

    try:
        sim_kwargs = {
            'name': 'Momentum V1.1',
            'upload': False,
            'trail_stop': 0.18,                    # 動能股波動大，給更多空間 (18%)
            'position_limit': 1.0 / MAX_CONCURRENT,
            'touched_exit': False,
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

        logging.info("Momentum V1.1 backtest.sim 執行成功")
        return report

    except Exception as e:
        logging.error(f"Momentum 策略層級崩潰: {str(e)}", exc_info=True)
        raise e
