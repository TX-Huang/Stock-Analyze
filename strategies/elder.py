"""
Elder Triple Screen + Impulse System Strategy
===============================================
Alexander Elder's Triple Screen trading system with Impulse System
adapted for Taiwan stocks using FinLab data.

Screen 1 (Weekly trend): 0050 benchmark weekly MACD histogram rising.
Screen 2 (Daily pullback): Stochastic K(14)/D(3) oversold and turning up.
Screen 3 (Entry trigger): 5-day breakout.

Impulse System:
  - EMA13 direction + MACD histogram direction
  - Green: both rising (bullish impulse)
  - Red: both falling (bearish impulse)

Entry: weekly_bullish & pullback & 5d_breakout & impulse_green & liquidity & ETF_ok
Exit: impulse_red | close < EMA13 | trail_stop 0.15

Scoring (cap at 5):
  Base 1 + impulse_green (+1) + vol > vol_ma5 * 1.5 (+1) + revenue YoY > 20 (+1)

MAX_CONCURRENT = 10, position_limit = 0.10
"""

from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab


def run_elder_strategy(api_token, stop_loss=None, take_profit=None):
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

    # --- Screen 1: Weekly trend (benchmark 0050) ---
    # Resample benchmark to weekly and compute MACD histogram
    bench_weekly = benchmark_close.resample('W').last()
    bench_w_ema12 = bench_weekly.ewm(span=12, adjust=False).mean()
    bench_w_ema26 = bench_weekly.ewm(span=26, adjust=False).mean()
    bench_w_macd_line = bench_w_ema12 - bench_w_ema26
    bench_w_macd_signal = bench_w_macd_line.ewm(span=9, adjust=False).mean()
    bench_w_macd_hist = bench_w_macd_line - bench_w_macd_signal

    # Weekly bullish: MACD histogram rising (improving)
    weekly_bullish = bench_w_macd_hist > bench_w_macd_hist.shift(1)
    # Reindex to daily with ffill
    weekly_bullish_daily = weekly_bullish.reindex(master_index).ffill().fillna(False)

    # --- Screen 2: Stochastic pullback (individual stocks) ---
    low_df = sanitize_dataframe(low, "FinLab_Low")
    high_df = sanitize_dataframe(high, "FinLab_High")

    # Stochastic K(14)
    lowest_14 = low_df.rolling(14).min()
    highest_14 = high_df.rolling(14).max()
    stoch_denom = highest_14 - lowest_14
    stoch_denom = stoch_denom.replace(0, np.nan)
    stoch_k = ((close - lowest_14) / stoch_denom) * 100
    stoch_k = stoch_k.fillna(50)  # neutral when undefined

    # Stochastic D(3) - 3-period SMA of K
    stoch_d = stoch_k.rolling(3).mean()

    # --- Screen 3: 5-day breakout ---
    close_max_5 = close.rolling(5).max().shift(1)

    # --- Impulse System (individual stocks) ---
    ema13 = close.ewm(span=13, adjust=False).mean()

    # MACD for individual stocks (standard 12/26/9)
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line

    # Volume MAs
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # EPS trailing 4Q sum
    eps_sum = eps.rolling(4).sum()

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

    v_ema13 = to_numpy(ema13)
    v_ema13_prev = to_numpy(ema13.shift(1))
    v_macd_hist = to_numpy(macd_hist)
    v_macd_hist_prev = to_numpy(macd_hist.shift(1))

    v_stoch_k = to_numpy(stoch_k)
    v_stoch_k_prev = to_numpy(stoch_k.shift(1))
    v_close_max_5 = to_numpy(close_max_5)

    v_vol_ma5 = to_numpy(vol_ma5)
    v_vol_ma20 = to_numpy(vol_ma20)

    v_rev_growth = to_numpy(rev_growth)

    v_weekly_bullish = to_numpy(weekly_bullish_daily, is_benchmark=True)

    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    # ==========================================
    # 4. Strategy Logic - Elder Triple Screen
    # ==========================================

    # Market filter
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)

    # Liquidity filter
    v_liq = (v_vol_ma20 > 500000)

    # Screen 1: Weekly trend bullish (MACD hist rising)
    c_weekly_bull = v_weekly_bullish > 0

    # Screen 2: Stochastic pullback - oversold and turning up
    c_pullback = (v_stoch_k < 30) & (v_stoch_k > v_stoch_k_prev)

    # Screen 3: 5-day breakout
    c_breakout_5d = (v_close > v_close_max_5) & (v_close_max_5 > 0)

    # Impulse System
    impulse_green = (v_ema13 > v_ema13_prev) & (v_macd_hist > v_macd_hist_prev)
    impulse_red = (v_ema13 < v_ema13_prev) & (v_macd_hist < v_macd_hist_prev)

    # --- Entry Signal ---
    sig_entry = (
        c_weekly_bull & c_pullback & c_breakout_5d & impulse_green
        & v_liq & v_etf_ok
    )

    # --- Exit Signal ---
    exit_sig = impulse_red | (v_close < v_ema13)

    # --- Scoring (cap at 5) ---
    score = np.ones_like(v_close)
    score += impulse_green.astype(int)                         # +1: Impulse green
    score += (v_vol > v_vol_ma5 * 1.5).astype(int)            # +1: Volume surge
    score += (v_rev_growth > 20).astype(int)                   # +1: Revenue YoY > 20%
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
        _log.warning("WARNING: Elder final_pos all zeros - no signals triggered")

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
    logging.info("--- Elder Triple Screen V1.0: backtest preparation ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    # Signal trigger statistics
    sig_entry_count = sig_entry.sum() if hasattr(sig_entry, 'sum') else 0
    exit_sig_count = exit_sig.sum() if hasattr(exit_sig, 'sum') else 0
    logging.info(f"Entry signal count: {sig_entry_count}")
    logging.info(f"Exit signal count: {exit_sig_count}")

    conditions_debug = {
        'c_weekly_bull (Screen 1)': c_weekly_bull.sum(),
        'c_pullback (Screen 2 - Stoch oversold turning up)': c_pullback.sum(),
        'c_breakout_5d (Screen 3 - 5-day breakout)': c_breakout_5d.sum(),
        'impulse_green': impulse_green.sum(),
        'impulse_red': impulse_red.sum(),
        'v_liq': v_liq.sum(),
        'v_etf_ok': v_etf_ok.sum(),
    }
    logging.info("--- Condition trigger statistics ---")
    for cond_name, cond_count in conditions_debug.items():
        logging.info(f"  {cond_name}: {cond_count}")

    non_zero_days = (final_pos.abs().sum(axis=1) > 0).sum()
    logging.info(f"Days with positions: {non_zero_days} / {len(final_pos)}")

    try:
        from data.provider import safe_finlab_sim

        sim_kwargs = {
            'name': 'Elder Triple Screen V1.0',
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

        logging.info("Elder Triple Screen V1.0 backtest.sim completed successfully")
        return report

    except Exception as e:
        logging.error(f"Strategy-level crash: {str(e)}", exc_info=True)
        raise e
