"""
Minervini SEPA (Specific Entry Point Analysis) Strategy
========================================================
Mark Minervini's trend template + VCP (Volatility Contraction Pattern)
adapted for Taiwan stocks using FinLab data.

Entry conditions (ALL must be true):
  - Trend Template: close > MA50 > MA150 > MA200, all MAs > 0
  - MA200 trending up: MA200 > MA200.shift(20)
  - Price structure: close > 252-day low * 1.25, close > 252-day high * 0.75
  - VCP volume dry-up: (vol < vol_ma50 * 0.5).rolling(10).max() > 0
  - Breakout: close > 20-day max (shifted) AND vol > vol_ma50 * 2.0
  - Liquidity: vol_ma20 > 500000
  - ETF filter: same regex as Isaac
  - Market filter: benchmark (0050) > bench_ma60 * 1.01

Scoring (cap at 6):
  Base 1 + RS rank > 70th percentile (+1) + Revenue YoY > 20% (+1)
  + EPS > 0 (+1) + Institutional buying streak (+1) + above MA60 (+1)

Exit: close < MA50 | trail_stop = 0.20
MAX_CONCURRENT = 10, position_limit = 0.10
"""

from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab


def run_minervini_strategy(api_token, stop_loss=None, take_profit=None):
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
    ma50 = close.rolling(50).mean()
    ma60 = close.rolling(60).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    ma200_shifted = ma200.shift(20)

    vol_ma20 = vol.rolling(20).mean()
    vol_ma50 = vol.rolling(50).mean()

    bench_ma60 = benchmark_close.rolling(60).mean()

    # 252-day high/low for price structure
    high_252 = close.rolling(252).max()
    low_252 = close.rolling(252).min()

    # 20-day breakout level (shifted)
    close_max_20 = close.rolling(20).max().shift(1)

    # VCP volume dry-up: at least one day in last 10 days where vol < vol_ma50 * 0.5
    vol_dryup_flag = (vol < vol_ma50 * 0.5).astype(float)
    vcp_dryup = vol_dryup_flag.rolling(10).max() > 0

    # Relative Strength rank (120-day return percentile rank)
    stock_ret_120 = close.pct_change(120)
    rs_rank = stock_ret_120.rank(axis=1, pct=True) * 100

    # EPS trailing 4Q sum
    eps_sum = eps.rolling(4).sum()

    # Institutional buying streak (5 consecutive days of net buying)
    inst_streak = (inst_net_buy.rolling(5).min() > 0)

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

    v_ma50 = to_numpy(ma50)
    v_ma60 = to_numpy(ma60)
    v_ma150 = to_numpy(ma150)
    v_ma200 = to_numpy(ma200)
    v_ma200_shifted = to_numpy(ma200_shifted)
    v_vol_ma20 = to_numpy(vol_ma20)
    v_vol_ma50 = to_numpy(vol_ma50)

    v_high_252 = to_numpy(high_252)
    v_low_252 = to_numpy(low_252)
    v_close_max_20 = to_numpy(close_max_20)
    v_vcp_dryup = to_numpy(vcp_dryup)

    v_rs_rank = to_numpy(rs_rank)
    v_rev_growth = to_numpy(rev_growth)
    v_eps_sum = to_numpy(eps_sum)
    v_inst_streak = to_numpy(inst_streak)

    v_bench = to_numpy(benchmark_close, is_benchmark=True)
    v_bench_ma60 = to_numpy(bench_ma60, is_benchmark=True)

    # ==========================================
    # 4. Strategy Logic - Minervini SEPA
    # ==========================================

    has_ma50 = v_ma50 > 0
    has_ma60 = v_ma60 > 0
    has_ma150 = v_ma150 > 0
    has_ma200 = v_ma200 > 0

    # Market filter
    v_bullish = (v_bench > v_bench_ma60 * 1.01) & (v_bench_ma60 > 0)

    # Liquidity filter
    v_liq = (v_vol_ma20 > 500000)

    # --- Minervini Trend Template ---
    # close > MA50 > MA150 > MA200, all MAs positive
    c_trend_template = (
        (v_close > v_ma50) & (v_ma50 > v_ma150) & (v_ma150 > v_ma200)
        & has_ma50 & has_ma150 & has_ma200
    )

    # MA200 trending up over 20 days
    c_ma200_up = (v_ma200 > v_ma200_shifted) & has_ma200

    # Price structure: within 25% of 252-day high, 25%+ above 252-day low
    c_price_structure = (
        (v_close > v_low_252 * 1.25) & (v_close > v_high_252 * 0.75)
        & (v_low_252 > 0) & (v_high_252 > 0)
    )

    # VCP volume dry-up
    c_vcp_dryup = v_vcp_dryup > 0

    # Breakout: close > 20-day high (shifted) with heavy volume
    c_breakout = (v_close > v_close_max_20) & (v_vol > v_vol_ma50 * 2.0) & (v_close_max_20 > 0)

    # --- Entry Signal ---
    sig_entry = (
        v_bullish & c_trend_template & c_ma200_up & c_price_structure
        & c_vcp_dryup & c_breakout & v_liq & v_etf_ok
    )

    # --- Exit Signal ---
    exit_sig = (v_close < v_ma50) & has_ma50

    # --- Scoring (cap at 6) ---
    score = np.ones_like(v_close)
    score += (v_rs_rank > 70).astype(int)           # +1: RS rank > 70th percentile
    score += (v_rev_growth > 20).astype(int)         # +1: Revenue YoY > 20%
    score += (v_eps_sum > 0).astype(int)             # +1: EPS > 0
    score += v_inst_streak.astype(int)               # +1: Institutional buying streak
    score += ((v_close > v_ma60) & has_ma60).astype(int)  # +1: Above MA60
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
        _log.warning("WARNING: Minervini final_pos all zeros - no signals triggered")

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
    logging.info("--- Minervini SEPA V1.0: backtest preparation ---")
    logging.info(f"final_pos shape: {final_pos.shape}")

    # Signal trigger statistics
    sig_entry_count = sig_entry.sum() if hasattr(sig_entry, 'sum') else 0
    exit_sig_count = exit_sig.sum() if hasattr(exit_sig, 'sum') else 0
    logging.info(f"Entry signal count: {sig_entry_count}")
    logging.info(f"Exit signal count: {exit_sig_count}")

    conditions_debug = {
        'v_bullish': v_bullish.sum(),
        'c_trend_template': c_trend_template.sum(),
        'c_ma200_up': c_ma200_up.sum(),
        'c_price_structure': c_price_structure.sum(),
        'c_vcp_dryup': c_vcp_dryup.sum(),
        'c_breakout': c_breakout.sum(),
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
            'name': 'Minervini SEPA V1.0',
            'upload': False,
            'trail_stop': 0.20,
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

        logging.info("Minervini SEPA V1.0 backtest.sim completed successfully")
        return report

    except Exception as e:
        logging.error(f"Strategy-level crash: {str(e)}", exc_info=True)
        raise e
