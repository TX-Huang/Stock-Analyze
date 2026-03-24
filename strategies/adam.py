"""
Adam Theory Strategy (J. Welles Wilder)
=========================================
Mirror Reflection + Trend Following adapted for Taiwan stocks using FinLab data.

Core concept: After a turning point, price mirrors its previous trajectory.
When the mirror projection (reflected past 20 days around the recent pivot high)
is above the current price, the projected path is bullish.

Entry conditions:
  - Mirror projection bullish: mirror_projection > close
  - Trend: close > MA20 & MA20 rising (MA20 > MA20.shift(5))
  - Volume confirmation: vol > vol_ma20 * 1.2
  - Liquidity: vol_ma20 > 500000
  - ETF filter: same regex as Isaac
  - Market filter: 0050 > bench_ma60

Exit: (~mirror_up) | close < MA20 | trail_stop 0.15

Scoring (cap at 5):
  Base 1 + mirror_up (+1) + trend (+1) + vol > 1.5x (+1)
  + close in top 25% of daily range (+1)

MAX_CONCURRENT = 10, position_limit = 0.10
"""

from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab


def run_adam_strategy(api_token, stop_loss=None, take_profit=None):
    from data.provider import sanitize_dataframe

    if api_token:
        finlab.login(api_token)

    if stop_loss is not None: stop_loss = float(stop_loss)
    if take_profit is not None: take_profit = float(take_profit)

    # ==========================================
    # 1. Data Fetch
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

    # Price data
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # Benchmark
    benchmark_close = data.get('price:收盤價')['0050']

    # Fundamentals - separate try/except blocks
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)')
    except Exception:
        rev_growth = pd.DataFrame(0, index=master_index, columns=close.columns)

    try:
        eps = data.get('finance_statement:每股盈餘')
    except Exception:
        eps = pd.DataFrame(0, index=master_index, columns=close.columns)

    try:
        foreign_buy = data.get('institutional_investors:外資買賣超股數').fillna(0)
        trust_buy = data.get('institutional_investors:投信買賣超股數').fillna(0)
        inst_net_buy = foreign_buy + trust_buy
    except Exception:
        inst_net_buy = pd.DataFrame(0, index=master_index, columns=close.columns)

    # ==========================================
    # 2. Pre-compute (Pandas phase)
    # ==========================================

    # Moving averages
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    vol_ma20 = vol.rolling(20).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # --- Adam Theory: Mirror Reflection ---
    # high_20: recent pivot high (20-day rolling max)
    high_20 = close.rolling(20).max()

    # mirror_projection: reflect past 20 days around the pivot high
    # mirror = 2 * pivot_high - price_20_days_ago
    close_shifted_20 = close.shift(20)
    mirror_projection = 2 * high_20 - close_shifted_20

    # Daily range position (for scoring)
    high_df = sanitize_dataframe(high, "FinLab_High")
    low_df = sanitize_dataframe(low, "FinLab_Low")
    open_df = sanitize_dataframe(open_, "FinLab_Open")

    # ETF blacklist
    import re as _re
    etf_blacklist = pd.Series(False, index=master_columns_str)
    for col in master_columns_str:
        if _re.match(r'^00\d{3,}[RL]?$', col) or _re.match(r'^00\d{3,}', col):
            etf_blacklist[col] = True
    v_etf_ok = ~etf_blacklist.values.reshape(1, -1)

    # ==========================================
    # 3. NumPy conversion
    # ==========================================
    v_close = to_numpy(close)
    v_open = to_numpy(open_)
    v_high = to_numpy(high)
    v_low = to_numpy(low)
    v_vol = to_numpy(vol)

    v_ma20 = to_numpy(ma20)
    v_ma20_shifted5 = to_numpy(ma20.shift(5))
    v_ma60 = to_numpy(ma60)
    v_vol_ma20 = to_numpy(vol_ma20)

    v_mirror = to_numpy(mirror_projection)

    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    v_rev_growth = to_numpy(rev_growth)

    # ==========================================
    # 4. Strategy Logic - Adam Theory
    # ==========================================

    has_ma20 = v_ma20 > 0

    # Market filter
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)

    # Liquidity filter
    v_liq = (v_vol_ma20 > 500000)

    # --- Adam Theory Conditions ---
    # Mirror bullish: projected path above current price
    c_mirror_up = (v_mirror > v_close) & (v_mirror > 0) & (v_close > 0)

    # Trend: close > MA20 and MA20 rising
    c_trend = (v_close > v_ma20) & (v_ma20 > v_ma20_shifted5) & has_ma20

    # Volume confirmation
    c_vol = v_vol > v_vol_ma20 * 1.2

    # --- Entry Signal ---
    sig_entry = c_mirror_up & c_trend & c_vol & v_liq & v_etf_ok & v_bullish

    # --- Exit Signal ---
    exit_mirror_down = ~c_mirror_up
    exit_below_ma20 = (v_close < v_ma20) & has_ma20
    exit_sig = exit_mirror_down | exit_below_ma20

    # --- Scoring (cap at 5) ---
    # Daily range position: close in top 25% of daily range
    v_daily_range = v_high - v_low
    v_daily_range_safe = np.where(v_daily_range == 0, np.nan, v_daily_range)
    v_close_position = (v_close - v_low) / v_daily_range_safe
    v_close_position = np.nan_to_num(v_close_position, nan=0.0)
    c_strong_close = v_close_position >= 0.75

    score = np.ones_like(v_close)
    score += c_mirror_up.astype(int)                           # +1: Mirror bullish
    score += c_trend.astype(int)                               # +1: Trend confirmed
    score += (v_vol > v_vol_ma20 * 1.5).astype(int)           # +1: Strong volume (> 1.5x)
    score += c_strong_close.astype(int)                        # +1: Close in top 25%
    score = np.minimum(score, 5)

    # ==========================================
    # 5. Position Reconstruction (exits-first-then-entries)
    # ==========================================
    MAX_CONCURRENT = 10

    v_pos = np.full_like(v_close, np.nan)
    v_pos[exit_sig] = 0                           # Step 1: exits first
    v_pos[sig_entry] = score[sig_entry]           # Step 2: entries override

    df_pos = pd.DataFrame(np.nan, index=master_index, columns=close.columns)
    df_pos[:] = v_pos
    df_pos = df_pos.ffill().fillna(0)

    # Top-N selection
    pos_rank = df_pos.rank(axis=1, method='first', ascending=False)
    top_mask = (pos_rank <= MAX_CONCURRENT) & (df_pos > 0)
    final_pos = df_pos.where(top_mask, 0)

    # Defensive cleanup
    final_pos = final_pos.replace([np.inf, -np.inf], 0).fillna(0)

    if final_pos.abs().sum().sum() == 0:
        import logging as _log
        _log.warning("WARNING: Adam Theory final_pos all zeros - no signals triggered")

    # ==========================================
    # 6. Diagnostics & Backtest
    # ==========================================
    import logging
    import os

    log_file = "finlab_debug.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.INFO)

    logging.info("=" * 60)
    logging.info("--- Adam Theory V1.0: backtest preparation ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    # Signal trigger statistics
    sig_entry_count = sig_entry.sum() if hasattr(sig_entry, 'sum') else 0
    exit_sig_count = exit_sig.sum() if hasattr(exit_sig, 'sum') else 0
    logging.info(f"Entry signal count: {sig_entry_count}")
    logging.info(f"Exit signal count: {exit_sig_count}")

    conditions_debug = {
        'v_bullish': v_bullish.sum(),
        'c_mirror_up (mirror projection bullish)': c_mirror_up.sum(),
        'c_trend (close > MA20 rising)': c_trend.sum(),
        'c_vol (vol > vol_ma20 * 1.2)': c_vol.sum(),
        'v_liq': v_liq.sum(),
        'v_etf_ok': v_etf_ok.sum(),
        'exit_mirror_down': exit_mirror_down.sum(),
        'exit_below_ma20': exit_below_ma20.sum(),
    }
    logging.info("--- Condition trigger statistics ---")
    for cond_name, cond_count in conditions_debug.items():
        logging.info(f"  {cond_name}: {cond_count}")

    non_zero_days = (final_pos.abs().sum(axis=1) > 0).sum()
    logging.info(f"Days with positions: {non_zero_days} / {len(final_pos)}")

    try:
        from data.provider import safe_finlab_sim

        sim_kwargs = {
            'name': 'Adam Theory V1.0',
            'upload': False,
            'trail_stop': 0.15,
            'position_limit': 1.0 / MAX_CONCURRENT,
            'touched_exit': False,
        }
        if stop_loss is not None:
            sim_kwargs['stop_loss'] = stop_loss
        if take_profit is not None:
            sim_kwargs['take_profit'] = take_profit

        report = safe_finlab_sim(final_pos, **sim_kwargs)

        # Post-backtest diagnostics
        try:
            stats = report.get_stats()
            trades = report.get_trades()
            logging.info(f"Backtest complete - trades count: {len(trades)}")
            if hasattr(stats, 'get'):
                logging.info(f"  cagr = {stats.get('cagr', 'MISSING')}")
                logging.info(f"  max_drawdown = {stats.get('max_drawdown', 'MISSING')}")
                logging.info(f"  win_ratio = {stats.get('win_ratio', 'MISSING')}")
        except Exception as diag_e:
            logging.warning(f"Diagnostics failed: {diag_e}")

        logging.info("Adam Theory V1.0 backtest.sim completed successfully")
        return report

    except Exception as e:
        logging.error(f"Strategy-level crash: {str(e)}", exc_info=True)
        raise e
