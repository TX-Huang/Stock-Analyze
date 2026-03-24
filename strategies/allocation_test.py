"""
Step 4: 資金配置方法 A/B 測試
============================================
測試 5 種配置方式，比較 CAGR、MaxDD、Sharpe、Score

方法:
1. Equal Weight (等權重) — 現有 baseline
2. Score Weighted (分數加權) — 依 score 比例配置
3. Inverse Volatility (反向波動率) — 低波動多配
4. Volatility Targeting (波動率目標) — 每檔風險貢獻相等
5. Dynamic Exposure (動態曝險) — 大盤強弱調整總部位
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


def objective_score(stats):
    """WFO 統一評分函數"""
    cagr = stats.get('cagr', 0)
    dd = abs(stats.get('max_drawdown', -1))
    sharpe = stats.get('daily_sharpe', 0)
    win = stats.get('win_ratio', 0)
    if dd == 0:
        dd = 0.001
    return (cagr / dd) * 0.4 + sharpe * 0.4 + win * 0.2



def _run_allocation_with_raw(raw, method='equal'):
    """
    用指定的資金配置方法跑回測 (使用預先計算的 raw data)。
    """
    from data.provider import safe_finlab_sim

    final_pos = raw['final_pos']
    close_df = raw['close']
    etf_close = raw['etf_close']
    trail_stop = raw['trail_stop']

    MAX_CONCURRENT = raw.get('max_concurrent', 10)

    if method == 'equal':
        # 等權重: 把所有非零 position 設為 1 (FinLab 會用 position_limit 均分)
        alloc_pos = final_pos.copy()
        alloc_pos[alloc_pos > 0] = 1
        alloc_pos[alloc_pos < 0] = -1
        position_limit = 1.0 / MAX_CONCURRENT

    elif method == 'score':
        # 分數加權: 保留原始 score，每行正規化成和為 1
        alloc_pos = final_pos.copy()
        # 多頭部分
        long_part = alloc_pos.clip(lower=0)
        row_sum = long_part.sum(axis=1)
        row_sum = row_sum.replace(0, 1)  # 避免除以 0
        long_part = long_part.div(row_sum, axis=0)
        # 空頭部分
        short_part = alloc_pos.clip(upper=0)
        short_sum = short_part.abs().sum(axis=1)
        short_sum = short_sum.replace(0, 1)
        short_part = -short_part.abs().div(short_sum, axis=0)  # 保持負值
        alloc_pos = long_part + short_part
        position_limit = 0.25  # 單檔上限 25%

    elif method == 'inv_vol':
        # 反向波動率: weight_i ∝ 1/σ_i (20日歷史波動率)
        # 向量化版本 — 不逐行迭代
        returns = close_df.pct_change()
        vol_20 = returns.rolling(20).std().reindex_like(final_pos).fillna(0.01)
        vol_20 = vol_20.clip(lower=0.001)

        inv_vol = 1.0 / vol_20

        # 多頭: 只在有持倉的位置計算權重
        long_mask = (final_pos > 0)
        long_inv_vol = inv_vol.where(long_mask, 0)
        long_row_sum = long_inv_vol.sum(axis=1).replace(0, 1)
        long_weights = long_inv_vol.div(long_row_sum, axis=0)

        # 空頭
        short_mask = (final_pos < 0)
        short_inv_vol = inv_vol.where(short_mask, 0)
        short_row_sum = short_inv_vol.sum(axis=1).replace(0, 1)
        short_weights = short_inv_vol.div(short_row_sum, axis=0)
        short_count = short_mask.sum(axis=1).replace(0, 1)
        short_weights = -short_weights.mul(short_count / MAX_CONCURRENT, axis=0)

        alloc_pos = long_weights.where(long_mask, 0) + short_weights.where(short_mask, 0)
        position_limit = 0.30

    elif method == 'risk_parity':
        # 風險平價: weight_i ∝ 1/σ² (簡化版，忽略相關性)
        returns = close_df.pct_change()
        vol_20 = returns.rolling(20).std().reindex_like(final_pos).fillna(0.01)
        vol_20 = vol_20.clip(lower=0.001)

        inv_var = 1.0 / (vol_20 ** 2)

        # 多頭
        long_mask = (final_pos > 0)
        long_iv = inv_var.where(long_mask, 0)
        long_row_sum = long_iv.sum(axis=1).replace(0, 1)
        long_weights = long_iv.div(long_row_sum, axis=0)

        # 空頭
        short_mask = (final_pos < 0)
        short_iv = inv_var.where(short_mask, 0)
        short_row_sum = short_iv.sum(axis=1).replace(0, 1)
        short_weights = short_iv.div(short_row_sum, axis=0)
        short_count = short_mask.sum(axis=1).replace(0, 1)
        short_weights = -short_weights.mul(short_count / MAX_CONCURRENT, axis=0)

        alloc_pos = long_weights.where(long_mask, 0) + short_weights.where(short_mask, 0)
        position_limit = 0.35

    elif method == 'dynamic':
        # 動態曝險: 大盤強 → 全倉, 大盤弱 → 減倉 (向量化)
        alloc_pos = final_pos.copy()
        alloc_pos[alloc_pos > 0] = 1
        alloc_pos[alloc_pos < 0] = -1

        if etf_close is not None:
            etf_s = etf_close.iloc[:, 0] if isinstance(etf_close, pd.DataFrame) else etf_close
            etf_s = etf_s.reindex(alloc_pos.index).ffill()
            ma60 = etf_s.rolling(60).mean()
            ma120 = etf_s.rolling(120).mean()

            # 向量化計算曝險乘數
            exposure = pd.Series(1.0, index=alloc_pos.index)
            exposure[etf_s <= ma60] = 0.6
            exposure[(etf_s <= ma120)] = 0.3
            # 修正: 大於 MA60 的設回 1.0
            exposure[etf_s > ma60] = 1.0

            # 多頭乘以 exposure, 空頭乘以 (2 - exposure)
            long_mask = (alloc_pos > 0)
            short_mask = (alloc_pos < 0)
            alloc_pos[long_mask] = alloc_pos[long_mask].mul(exposure, axis=0)[long_mask]
            alloc_pos[short_mask] = alloc_pos[short_mask].mul(2.0 - exposure, axis=0)[short_mask]

        position_limit = 1.0 / MAX_CONCURRENT
    else:
        raise ValueError(f"未知配置方法: {method}")

    # 清理
    alloc_pos = alloc_pos.replace([np.inf, -np.inf], 0).fillna(0)

    # 跑 sim
    sim_kwargs = {
        'name': f'Isaac_alloc_{method}',
        'upload': False,
        'trail_stop': trail_stop,
        'position_limit': position_limit,
        'touched_exit': False,
    }

    report = safe_finlab_sim(alloc_pos, **sim_kwargs)
    return report


def run_all_allocation_tests(api_token):
    """執行所有 5 種配置方法，比較績效"""
    from isaac import run_isaac_strategy
    from data.provider import safe_finlab_sim

    # 只跑一次策略計算，取得原始 position matrix
    log.info("計算策略信號 (只執行一次)...")
    raw = run_isaac_strategy(api_token, raw_mode=True)
    log.info(f"final_pos shape: {raw['final_pos'].shape}")

    methods = ['equal', 'score', 'inv_vol', 'risk_parity', 'dynamic']
    results = []

    for method in methods:
        log.info(f"\n{'='*60}")
        log.info(f"測試配置方法: {method}")
        log.info(f"{'='*60}")

        try:
            report = _run_allocation_with_raw(raw, method)
            stats = report.get_stats()
            r = {
                'method': method,
                'cagr': stats.get('cagr', 0),
                'max_drawdown': stats.get('max_drawdown', 0),
                'daily_sharpe': stats.get('daily_sharpe', 0),
                'win_ratio': stats.get('win_ratio', 0),
            }
            r['score'] = objective_score(r)
            log.info(f"  CAGR={r['cagr']*100:+.1f}% | DD={r['max_drawdown']*100:.1f}% | "
                     f"Sharpe={r['daily_sharpe']:.2f} | Win={r['win_ratio']*100:.1f}% | "
                     f"Score={r['score']:.4f}")
        except Exception as e:
            log.error(f"  {method} 失敗: {e}", exc_info=True)
            r = {'method': method, 'cagr': 0, 'max_drawdown': -1,
                 'daily_sharpe': 0, 'win_ratio': 0, 'score': 0, 'error': str(e)}

        results.append(r)

    # 排名
    log.info(f"\n{'='*60}")
    log.info("資金配置方法比較")
    log.info(f"{'='*60}")
    log.info(f"{'Method':>14} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>7} | {'Win':>6} | {'Score':>8}")
    log.info("-" * 70)
    results_sorted = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    for i, r in enumerate(results_sorted):
        marker = " ← BEST" if i == 0 else ""
        log.info(f"{r['method']:>14} | {r['cagr']*100:>+7.1f}% | "
                 f"{r['max_drawdown']*100:>7.1f}% | {r['daily_sharpe']:>7.2f} | "
                 f"{r['win_ratio']*100:>5.1f}% | {r['score']:>8.4f}{marker}")

    # 存檔
    path = 'allocation_test_result.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"\n結果已儲存: {path}")

    return results


if __name__ == '__main__':
    import toml

    secrets_path = os.path.join(_PROJECT_ROOT, '.streamlit', 'secrets.toml')
    secrets = toml.load(secrets_path)
    api_token = secrets.get('FINLAB_API_KEY', '')

    if not api_token:
        print("缺少 FINLAB_API_KEY")
        sys.exit(1)

    run_all_allocation_tests(api_token)
