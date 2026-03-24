"""
Candlestick Pattern Strategy (陰線陽線型態策略) - 獨立測試版
============================================================
基於《陰線陽線》(Japanese Candlestick Charting Techniques) 的 K 線型態策略。
目的: 測試純 K 線型態的獨立勝率，為整合進 Isaac 提供依據。

多頭進場型態:
  - 晨星 (Morning Star) — 3K 反轉
  - 多頭吞噬 (Bullish Engulfing) — 2K 反轉
  - 錘子 (Hammer) — 單K 反轉
  - 貫穿線 (Piercing Line) — 2K 反轉

空頭出場型態:
  - 暮星 (Evening Star) — 3K 反轉
  - 空頭吞噬 (Bearish Engulfing) — 2K 反轉
  - 流星 (Shooting Star) — 單K 反轉
  - 烏雲罩頂 (Dark Cloud Cover) — 2K 反轉

版本: 1.0
"""

from finlab import data
from finlab import backtest
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
import finlab


def run_candlestick_strategy(api_token):
    from data.provider import sanitize_dataframe, safe_finlab_sim

    if api_token:
        finlab.login(api_token)

    # ==========================================
    # 1. 數據抓取
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

    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')
    benchmark_close = data.get('price:收盤價')['0050']

    # ==========================================
    # 2. NumPy 轉換
    # ==========================================
    v_close = to_numpy(close)
    v_open = to_numpy(open_)
    v_high = to_numpy(high)
    v_low = to_numpy(low)
    v_vol = to_numpy(vol)

    # 前一天 / 前兩天的數據 (shift)
    v_close_1 = np.roll(v_close, 1, axis=0); v_close_1[0] = np.nan
    v_open_1 = np.roll(v_open, 1, axis=0); v_open_1[0] = np.nan
    v_high_1 = np.roll(v_high, 1, axis=0); v_high_1[0] = np.nan
    v_low_1 = np.roll(v_low, 1, axis=0); v_low_1[0] = np.nan

    v_close_2 = np.roll(v_close, 2, axis=0); v_close_2[:2] = np.nan
    v_open_2 = np.roll(v_open, 2, axis=0); v_open_2[:2] = np.nan
    v_high_2 = np.roll(v_high, 2, axis=0); v_high_2[:2] = np.nan

    # 均線和輔助指標
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    bench_ma60 = benchmark_close.rolling(60).mean()

    v_ma20 = to_numpy(ma20)
    v_ma60 = to_numpy(ma60)
    v_vol_ma5 = to_numpy(vol_ma5)
    v_vol_ma20 = to_numpy(vol_ma20)
    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    # ==========================================
    # 3. K 線型態偵測 (向量化)
    # ==========================================

    # 基礎計算
    body = np.abs(v_close - v_open)
    body_1 = np.abs(v_close_1 - v_open_1)
    body_2 = np.abs(v_close_2 - v_open_2)

    upper_shadow = v_high - np.maximum(v_close, v_open)
    lower_shadow = np.minimum(v_close, v_open) - v_low

    is_bull = v_close > v_open      # 陽線
    is_bear = v_close < v_open      # 陰線
    is_bull_1 = v_close_1 > v_open_1
    is_bear_1 = v_close_1 < v_open_1
    is_bull_2 = v_close_2 > v_open_2
    is_bear_2 = v_close_2 < v_open_2

    # 14日平均實體大小
    avg_body = to_numpy(pd.DataFrame(body, index=master_index, columns=close.columns).rolling(14).mean())

    # 趨勢判斷 (10日高低點位置)
    local_high = to_numpy(close.rolling(10).max())
    local_low = to_numpy(close.rolling(10).min())
    range_safe = np.where((local_high - local_low) == 0, 1, local_high - local_low)
    position = (v_close - local_low) / range_safe
    is_downtrend = position <= 0.30   # 在近10日低檔
    is_uptrend = position >= 0.70     # 在近10日高檔

    # === 多頭型態 (Bullish Patterns) ===

    # 1. 錘子 (Hammer): 下影線 ≥ 2x 實體, 上影線 ≤ 0.3x 實體, 處於下跌趨勢
    pat_hammer = (lower_shadow >= body * 2) & (upper_shadow <= body * 0.3) & is_downtrend & (body > 0)

    # 2. 多頭吞噬 (Bullish Engulfing): 前陰今陽, 今實體完全包覆前實體
    pat_bull_engulf = (is_bear_1 & is_bull &
                       (v_open <= v_close_1) & (v_close >= v_open_1) &
                       (body > body_1) & is_downtrend)

    # 3. 晨星 (Morning Star): 前長陰 + 中間小K + 今長陽, 今收盤過前實體中點
    mid_body_small = body_1 <= avg_body * 0.5  # 中間 K 實體小
    prev_big_bear = is_bear_2 & (body_2 > avg_body)  # 前天長陰
    today_big_bull = is_bull & (body > avg_body)  # 今天長陽
    today_close_past_mid = v_close > (v_open_2 + v_close_2) / 2  # 今收過前天中點
    pat_morning_star = prev_big_bear & mid_body_small & today_big_bull & today_close_past_mid & is_downtrend

    # 4. 貫穿線 (Piercing Line): 前長陰, 今開低於前低, 今收過前實體中點
    pat_piercing = (is_bear_1 & (body_1 > avg_body) &
                    is_bull & (v_open < v_low_1) &
                    (v_close > (v_open_1 + v_close_1) / 2) &
                    (v_close < v_open_1) & is_downtrend)

    # 合併多頭信號
    bull_pattern = pat_hammer | pat_bull_engulf | pat_morning_star | pat_piercing

    # === 空頭型態 (Bearish Patterns) ===

    # 1. 流星 (Shooting Star): 上影線 ≥ 2x 實體, 下影線 ≤ 0.3x 實體, 處於上漲趨勢
    pat_shooting_star = (upper_shadow >= body * 2) & (lower_shadow <= body * 0.3) & is_uptrend & (body > 0)

    # 2. 空頭吞噬 (Bearish Engulfing): 前陽今陰, 今實體完全包覆前實體
    pat_bear_engulf = (is_bull_1 & is_bear &
                       (v_open >= v_close_1) & (v_close <= v_open_1) &
                       (body > body_1) & is_uptrend)

    # 3. 暮星 (Evening Star): 前長陽 + 中間小K + 今長陰
    prev_big_bull = is_bull_2 & (body_2 > avg_body)
    today_big_bear = is_bear & (body > avg_body)
    today_close_below_mid = v_close < (v_open_2 + v_close_2) / 2
    pat_evening_star = prev_big_bull & mid_body_small & today_big_bear & today_close_below_mid & is_uptrend

    # 4. 烏雲罩頂 (Dark Cloud Cover): 前長陽, 今開高於前高, 今收跌入前實體中點以下
    pat_dark_cloud = (is_bull_1 & (body_1 > avg_body) &
                      is_bear & (v_open > v_high_1) &
                      (v_close < (v_open_1 + v_close_1) / 2) &
                      (v_close > v_open_1) & is_uptrend)

    # 合併空頭信號
    bear_pattern = pat_shooting_star | pat_bear_engulf | pat_evening_star | pat_dark_cloud

    # ==========================================
    # 4. 策略邏輯
    # ==========================================
    has_ma20 = v_ma20 > 0
    has_ma60 = v_ma60 > 0

    # 市場濾網
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)
    v_liq = v_vol_ma20 > 500000

    # ETF 排除
    import re as _re
    etf_blacklist = pd.Series(False, index=master_columns_str)
    for col in master_columns_str:
        if _re.match(r'^00\d{3,}[RL]?$', col) or _re.match(r'^00\d{3,}', col):
            etf_blacklist[col] = True
    v_etf_ok = ~etf_blacklist.values.reshape(1, -1)

    # 帶量確認 (型態 + 量能放大)
    vol_confirm = v_vol > v_vol_ma5 * 1.2

    # 進場: 多頭型態 + 大盤多頭 + 帶量 + 流動性
    sig_entry = bull_pattern & v_bullish & vol_confirm & v_liq & v_etf_ok

    # 出場: 空頭型態 OR 跌破 MA60
    exit_bear_pat = bear_pattern & vol_confirm  # 空頭型態需帶量確認
    exit_ma60 = (v_close < v_ma60) & has_ma60
    sig_exit = exit_bear_pat | exit_ma60

    # 評分
    score = np.ones_like(v_close)
    score += pat_morning_star.astype(int) * 2   # 晨星 +2 (最可靠)
    score += pat_bull_engulf.astype(int) * 2    # 多頭吞噬 +2
    score += pat_piercing.astype(int)           # 貫穿線 +1
    score += pat_hammer.astype(int)             # 錘子 +1
    score += (v_vol > v_vol_ma5 * 2.0).astype(int)  # 量能超強 +1
    score = np.minimum(score, 5)

    # ==========================================
    # 5. 部位重建
    # ==========================================
    v_pos = np.full_like(v_close, np.nan)
    v_pos[sig_exit] = 0
    v_pos[sig_entry] = score[sig_entry]

    final_pos = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    final_pos[:] = v_pos
    final_pos = final_pos.ffill().fillna(0)
    final_pos = final_pos.replace([np.inf, -np.inf], 0).fillna(0)

    MAX_CONCURRENT = 10
    _pos_rank = final_pos.rank(axis=1, method='first', ascending=False)
    _top_n_mask = (_pos_rank <= MAX_CONCURRENT) & (final_pos > 0)
    final_pos = final_pos.where(_top_n_mask, 0)

    # ==========================================
    # 6. 診斷 & 回測
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
    logging.info("--- Candlestick V1: 回測 ---")

    # 各型態觸發統計
    pat_stats = {
        '錘子 (Hammer)': pat_hammer.sum(),
        '多頭吞噬 (Bull Engulf)': pat_bull_engulf.sum(),
        '晨星 (Morning Star)': pat_morning_star.sum(),
        '貫穿線 (Piercing)': pat_piercing.sum(),
        '流星 (Shooting Star)': pat_shooting_star.sum(),
        '空頭吞噬 (Bear Engulf)': pat_bear_engulf.sum(),
        '暮星 (Evening Star)': pat_evening_star.sum(),
        '烏雲罩頂 (Dark Cloud)': pat_dark_cloud.sum(),
        '多頭型態合計': bull_pattern.sum(),
        '空頭型態合計': bear_pattern.sum(),
        '進場信號': sig_entry.sum(),
        '出場信號': sig_exit.sum(),
    }
    for name, count in pat_stats.items():
        logging.info(f"  {name}: {count}")

    try:
        sim_kwargs = {
            'name': 'Candlestick V1',
            'upload': False,
            'trail_stop': 0.15,
            'position_limit': 1.0 / MAX_CONCURRENT,
            'touched_exit': False,
        }

        report = safe_finlab_sim(final_pos, **sim_kwargs)

        try:
            stats = report.get_stats()
            trades = report.get_trades()
            logging.info(f"trades: {len(trades)}, cagr: {stats.get('cagr')}, win: {stats.get('win_ratio')}")
        except Exception:
            pass

        return report, pat_stats

    except Exception as e:
        logging.error(f"Candlestick 策略崩潰: {e}", exc_info=True)
        raise e
