"""
Step 5: Monte Carlo Permutation Test
============================================
驗證 Isaac V3.5 策略的統計顯著性。

方法:
1. 用真實策略 (trail_stop=0.18 + Dynamic Exposure) 跑一次基準績效
2. 隨機打亂 position matrix 的進出場時間 N 次
3. 計算 p-value: 真實績效在隨機分布中的排名

若 p < 0.05 → 策略有統計顯著性 (非運氣)
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


def apply_dynamic_exposure(final_pos, etf_close, max_concurrent=10):
    """套用 Dynamic Exposure 配置 (Step 4 冠軍方法)"""
    alloc_pos = final_pos.copy()
    alloc_pos[alloc_pos > 0] = 1
    alloc_pos[alloc_pos < 0] = -1

    if etf_close is not None:
        etf_s = etf_close.iloc[:, 0] if isinstance(etf_close, pd.DataFrame) else etf_close
        etf_s = etf_s.reindex(alloc_pos.index).ffill()
        ma60 = etf_s.rolling(60).mean()
        ma120 = etf_s.rolling(120).mean()

        exposure = pd.Series(1.0, index=alloc_pos.index)
        exposure[etf_s <= ma60] = 0.6
        exposure[etf_s <= ma120] = 0.3
        exposure[etf_s > ma60] = 1.0

        long_mask = (alloc_pos > 0)
        short_mask = (alloc_pos < 0)
        alloc_pos[long_mask] = alloc_pos[long_mask].mul(exposure, axis=0)[long_mask]
        alloc_pos[short_mask] = alloc_pos[short_mask].mul(2.0 - exposure, axis=0)[short_mask]

    return alloc_pos


def shuffle_positions(final_pos, method='time_shift'):
    """
    隨機打亂 position matrix，保留持倉結構但破壞時機。

    method:
        'time_shift'  — 每檔隨機平移 N 天 (保留持倉長度分布)
        'row_shuffle'  — 打亂每天持哪些股票 (保留每天持倉數量)
        'block_shuffle' — 隨機交換時間區塊 (保留短期自相關)
    """
    shuffled = final_pos.copy()

    if method == 'time_shift':
        # 每檔股票隨機向前/後平移 1~252 天
        for col in shuffled.columns:
            series = shuffled[col].values
            if np.any(series != 0):
                shift = np.random.randint(-252, 252)
                shuffled[col] = np.roll(series, shift)

    elif method == 'row_shuffle':
        # 打亂每天的持倉分配 (保留每天持倉股票數)
        vals = shuffled.values.copy()
        for i in range(vals.shape[0]):
            row = vals[i]
            nonzero_idx = np.where(row != 0)[0]
            if len(nonzero_idx) > 0:
                # 隨機選同樣數量的股票
                available = np.arange(vals.shape[1])
                new_idx = np.random.choice(available, size=len(nonzero_idx), replace=False)
                new_row = np.zeros_like(row)
                new_row[new_idx] = row[nonzero_idx]
                vals[i] = new_row
        shuffled = pd.DataFrame(vals, index=final_pos.index, columns=final_pos.columns)

    elif method == 'block_shuffle':
        # 將時間軸切成 20 天區塊，隨機重排
        block_size = 20
        n_rows = len(shuffled)
        n_blocks = n_rows // block_size
        if n_blocks > 1:
            blocks = [shuffled.iloc[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
            remainder = shuffled.iloc[n_blocks*block_size:]
            np.random.shuffle(blocks)
            shuffled = pd.concat(blocks + [remainder], ignore_index=False)
            # 重新設定 index 為原始的日期順序
            shuffled.index = final_pos.index[:len(shuffled)]

    return shuffled


def run_monte_carlo(api_token, n_iterations=500, trail_stop=0.18):
    """
    執行 Monte Carlo Permutation Test。

    n_iterations: 隨機迭代次數 (500 次約需 25 分鐘)
    trail_stop: 使用的 trail_stop 值
    """
    from isaac import run_isaac_strategy
    from data.provider import safe_finlab_sim

    params = {
        'trail_stop': trail_stop,
        'rsi_threshold': 28,
        'volume_mult': 1.5,
        'supply_danger_pct': 0.97,
        'liq_min': 500000,
    }

    # === Step 1: 真實策略績效 ===
    log.info(f"\n{'='*60}")
    log.info(f"Monte Carlo Permutation Test")
    log.info(f"trail_stop={trail_stop}, Dynamic Exposure, N={n_iterations}")
    log.info(f"{'='*60}")

    log.info("\n[Step 1] 計算真實策略績效...")
    raw = run_isaac_strategy(api_token, params=params, raw_mode=True)
    final_pos = raw['final_pos']
    etf_close = raw['etf_close']
    close_df = raw['close']
    max_concurrent = raw.get('max_concurrent', 10)

    # 套用 Dynamic Exposure
    real_pos = apply_dynamic_exposure(final_pos, etf_close, max_concurrent)
    real_pos = real_pos.replace([np.inf, -np.inf], 0).fillna(0)

    sim_kwargs = {
        'name': 'MC_real',
        'upload': False,
        'trail_stop': trail_stop,
        'position_limit': 1.0 / max_concurrent,
        'touched_exit': False,
    }

    real_report = safe_finlab_sim(real_pos, **sim_kwargs)
    real_stats = real_report.get_stats()
    real_cagr = real_stats.get('cagr', 0)
    real_sharpe = real_stats.get('daily_sharpe', 0)
    real_dd = real_stats.get('max_drawdown', 0)
    real_win = real_stats.get('win_ratio', 0)

    log.info(f"  真實績效: CAGR={real_cagr*100:+.1f}% | DD={real_dd*100:.1f}% | "
             f"Sharpe={real_sharpe:.2f} | Win={real_win*100:.1f}%")

    # === Step 2: 隨機打亂 N 次 ===
    log.info(f"\n[Step 2] 執行 {n_iterations} 次隨機排列測試...")
    random_cagrs = []
    random_sharpes = []
    shuffle_methods = ['time_shift', 'row_shuffle', 'block_shuffle']

    for i in range(n_iterations):
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  迭代 {i+1}/{n_iterations}...")

        try:
            # 輪流使用不同的打亂方法
            method = shuffle_methods[i % len(shuffle_methods)]
            shuffled_pos = shuffle_positions(final_pos, method=method)

            # 套用 Dynamic Exposure
            shuffled_alloc = apply_dynamic_exposure(shuffled_pos, etf_close, max_concurrent)
            shuffled_alloc = shuffled_alloc.replace([np.inf, -np.inf], 0).fillna(0)

            sim_kwargs_mc = {
                'name': f'MC_{i}',
                'upload': False,
                'trail_stop': trail_stop,
                'position_limit': 1.0 / max_concurrent,
                'touched_exit': False,
            }

            report = safe_finlab_sim(shuffled_alloc, **sim_kwargs_mc)
            stats = report.get_stats()
            random_cagrs.append(stats.get('cagr', 0))
            random_sharpes.append(stats.get('daily_sharpe', 0))

        except Exception as e:
            if (i + 1) % 50 == 0:
                log.warning(f"  迭代 {i+1} 失敗: {e}")
            random_cagrs.append(0)
            random_sharpes.append(0)

    # === Step 3: 統計分析 ===
    log.info(f"\n[Step 3] 統計分析...")
    random_cagrs = np.array(random_cagrs)
    random_sharpes = np.array(random_sharpes)

    # p-value: 真實績效優於多少比例的隨機結果
    cagr_rank = np.sum(random_cagrs >= real_cagr)
    sharpe_rank = np.sum(random_sharpes >= real_sharpe)
    p_value_cagr = cagr_rank / n_iterations
    p_value_sharpe = sharpe_rank / n_iterations

    # 統計摘要
    log.info(f"\n{'='*60}")
    log.info("Monte Carlo 結果")
    log.info(f"{'='*60}")
    log.info(f"迭代次數: {n_iterations}")
    log.info(f"\n真實策略:")
    log.info(f"  CAGR:   {real_cagr*100:+.1f}%")
    log.info(f"  Sharpe: {real_sharpe:.2f}")
    log.info(f"  MaxDD:  {real_dd*100:.1f}%")
    log.info(f"  Win:    {real_win*100:.1f}%")

    log.info(f"\n隨機分布 (CAGR):")
    log.info(f"  平均: {np.mean(random_cagrs)*100:+.1f}%")
    log.info(f"  中位: {np.median(random_cagrs)*100:+.1f}%")
    log.info(f"  σ:    {np.std(random_cagrs)*100:.1f}%")
    log.info(f"  最大: {np.max(random_cagrs)*100:+.1f}%")
    log.info(f"  最小: {np.min(random_cagrs)*100:+.1f}%")

    log.info(f"\n隨機分布 (Sharpe):")
    log.info(f"  平均: {np.mean(random_sharpes):.2f}")
    log.info(f"  中位: {np.median(random_sharpes):.2f}")
    log.info(f"  σ:    {np.std(random_sharpes):.2f}")

    log.info(f"\np-value (CAGR):   {p_value_cagr:.4f} "
             f"({'✅ 顯著 (p<0.05)' if p_value_cagr < 0.05 else '❌ 不顯著'})")
    log.info(f"p-value (Sharpe): {p_value_sharpe:.4f} "
             f"({'✅ 顯著 (p<0.05)' if p_value_sharpe < 0.05 else '❌ 不顯著'})")

    # Z-score
    if np.std(random_cagrs) > 0:
        z_cagr = (real_cagr - np.mean(random_cagrs)) / np.std(random_cagrs)
    else:
        z_cagr = 0
    if np.std(random_sharpes) > 0:
        z_sharpe = (real_sharpe - np.mean(random_sharpes)) / np.std(random_sharpes)
    else:
        z_sharpe = 0

    log.info(f"\nZ-score (CAGR):   {z_cagr:.2f} "
             f"({'✅ >1.96' if z_cagr > 1.96 else '⚠️ <1.96'})")
    log.info(f"Z-score (Sharpe): {z_sharpe:.2f} "
             f"({'✅ >1.96' if z_sharpe > 1.96 else '⚠️ <1.96'})")

    # 最終判定
    is_significant = p_value_cagr < 0.05 and p_value_sharpe < 0.05
    verdict = "✅ 策略具有統計顯著性 — 非運氣" if is_significant else "⚠️ 策略顯著性不足"
    log.info(f"\n{'='*60}")
    log.info(f"最終判定: {verdict}")
    log.info(f"{'='*60}")

    # 存檔
    result = {
        'test_type': 'monte_carlo_permutation',
        'n_iterations': n_iterations,
        'trail_stop': trail_stop,
        'allocation': 'dynamic_exposure',
        'real_stats': {
            'cagr': real_cagr,
            'max_drawdown': real_dd,
            'daily_sharpe': real_sharpe,
            'win_ratio': real_win,
        },
        'random_distribution': {
            'cagr_mean': float(np.mean(random_cagrs)),
            'cagr_median': float(np.median(random_cagrs)),
            'cagr_std': float(np.std(random_cagrs)),
            'cagr_max': float(np.max(random_cagrs)),
            'cagr_min': float(np.min(random_cagrs)),
            'sharpe_mean': float(np.mean(random_sharpes)),
            'sharpe_median': float(np.median(random_sharpes)),
            'sharpe_std': float(np.std(random_sharpes)),
        },
        'p_value_cagr': p_value_cagr,
        'p_value_sharpe': p_value_sharpe,
        'z_score_cagr': z_cagr,
        'z_score_sharpe': z_sharpe,
        'is_significant': is_significant,
        'verdict': verdict,
        'random_cagrs': random_cagrs.tolist(),
        'random_sharpes': random_sharpes.tolist(),
    }

    path = 'monte_carlo_result.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\n結果已儲存: {path}")

    return result


if __name__ == '__main__':
    import toml

    secrets_path = os.path.join(_PROJECT_ROOT, '.streamlit', 'secrets.toml')
    secrets = toml.load(secrets_path)
    api_token = secrets.get('FINLAB_API_KEY', '')

    if not api_token:
        print("缺少 FINLAB_API_KEY")
        sys.exit(1)

    # 500 次迭代，每次 sim 約 3 秒 → 約 25 分鐘
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_monte_carlo(api_token, n_iterations=n, trail_stop=0.18)
