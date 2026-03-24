"""
Edwards & Magee Technical Analysis of Stock Trends Strategy
=============================================================
Pattern Breakout + Trendlines adapted for Taiwan stocks using FinLab data.

Based on the classic "Technical Analysis of Stock Trends" methodology:
resistance breakouts, new highs, and consolidation-before-breakout patterns.

Entry conditions (ALL must be true):
  - Resistance breakout: close > 20-day max (shifted) with vol > vol_ma20 * 1.5
  - 52-week new high: close >= 252-day rolling max
  - BB consolidation before breakout: bb_bandwidth < bb_bandwidth.rolling(60).quantile(0.20)
    within last 10 days
  - Market filter: 0050 > bench_ma60
  - Liquidity + ETF filter

Exit: close < 20-day min (shifted) (support break) | trail_stop 0.15

Scoring (cap at 6):
  Base 1 + 52w new high (+2) + vol > 2x (+1) + consolidation (+1) + revenue > 0 (+1)

MAX_CONCURRENT = 10, position_limit = 0.10
"""

from finlab import data
from finlab import backtest
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)
import numpy as np
import finlab


def run_edwards_magee_strategy(api_token, stop_loss=None, take_profit=None):
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

    # --- Resistance breakout (20-day) ---
    close_max_20 = close.rolling(20).max().shift(1)

    # --- 52-week (252-day) new high ---
    close_max_252 = close.rolling(252).max()

    # --- Support level (20-day low shifted) ---
    close_min_20 = close.rolling(20).min().shift(1)

    # --- Bollinger Bandwidth consolidation ---
    bb_std = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_bandwidth = (bb_upper - bb_lower) / ma20
    bb_bandwidth = bb_bandwidth.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Consolidation threshold: bandwidth below 20th percentile over 60 days
    bb_bw_q20 = bb_bandwidth.rolling(60).quantile(0.20)
    bb_tight = bb_bandwidth < bb_bw_q20

    # Was consolidated within last 10 days? (recent squeeze before breakout)
    bb_recent_tight = bb_tight.astype(float).rolling(10).max() > 0

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
    v_vol = to_numpy(vol)

    v_ma20 = to_numpy(ma20)
    v_vol_ma20 = to_numpy(vol_ma20)

    v_close_max_20 = to_numpy(close_max_20)
    v_close_max_252 = to_numpy(close_max_252)
    v_close_min_20 = to_numpy(close_min_20)
    v_bb_recent_tight = to_numpy(bb_recent_tight)

    v_rev_growth = to_numpy(rev_growth)

    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    # ==========================================
    # 4. Strategy Logic - Edwards & Magee
    # ==========================================

    # Market filter
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)

    # Liquidity filter
    v_liq = (v_vol_ma20 > 500000)

    # --- Resistance breakout with volume ---
    c_resistance_breakout = (
        (v_close > v_close_max_20) & (v_close_max_20 > 0)
        & (v_vol > v_vol_ma20 * 1.5)
    )

    # --- 52-week new high ---
    c_new_high_52w = (v_close >= v_close_max_252) & (v_close_max_252 > 0)

    # --- BB consolidation before breakout ---
    c_consolidation = v_bb_recent_tight > 0

    # --- Entry Signal ---
    sig_entry = (
        v_bullish & c_resistance_breakout & c_new_high_52w
        & c_consolidation & v_liq & v_etf_ok
    )

    # --- Exit Signal: support break (close < 20-day low) ---
    exit_sig = (v_close < v_close_min_20) & (v_close_min_20 > 0)

    # --- Scoring (cap at 6) ---
    score = np.ones_like(v_close)
    score += c_new_high_52w.astype(int) * 2                    # +2: 52-week new high (strong signal)
    score += (v_vol > v_vol_ma20 * 2.0).astype(int)           # +1: Heavy volume (> 2x)
    score += c_consolidation.astype(int)                       # +1: Recent consolidation
    score += (v_rev_growth > 0).astype(int)                    # +1: Revenue growth > 0
    score = np.minimum(score, 6)

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
        _log.warning("WARNING: Edwards & Magee final_pos all zeros - no signals triggered")

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
    logging.info("--- Edwards & Magee V1.0: backtest preparation ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    # Signal trigger statistics
    sig_entry_count = sig_entry.sum() if hasattr(sig_entry, 'sum') else 0
    exit_sig_count = exit_sig.sum() if hasattr(exit_sig, 'sum') else 0
    logging.info(f"Entry signal count: {sig_entry_count}")
    logging.info(f"Exit signal count: {exit_sig_count}")

    conditions_debug = {
        'v_bullish': v_bullish.sum(),
        'c_resistance_breakout (20d breakout + volume)': c_resistance_breakout.sum(),
        'c_new_high_52w (252-day new high)': c_new_high_52w.sum(),
        'c_consolidation (BB squeeze in last 10 days)': c_consolidation.sum(),
        'v_liq': v_liq.sum(),
        'v_etf_ok': v_etf_ok.sum(),
        'exit_sig (support break)': exit_sig.sum(),
    }
    logging.info("--- Condition trigger statistics ---")
    for cond_name, cond_count in conditions_debug.items():
        logging.info(f"  {cond_name}: {cond_count}")

    non_zero_days = (final_pos.abs().sum(axis=1) > 0).sum()
    logging.info(f"Days with positions: {non_zero_days} / {len(final_pos)}")

    try:
        from data.provider import safe_finlab_sim

        sim_kwargs = {
            'name': 'Edwards & Magee V1.0',
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

        logging.info("Edwards & Magee V1.0 backtest.sim completed successfully")
        return report

    except Exception as e:
        logging.error(f"Strategy-level crash: {str(e)}", exc_info=True)
        raise e
