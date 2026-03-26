"""
市場 Regime 策略分析系統
執行: ./python_embed/python.exe scripts/regime_analysis.py

分析五大策略在不同大盤狀態下的表現差異，
建構 Regime-Aware 動態配置引擎。

5-Regime Model:
  strong_bull  — 多頭排列 (0050 > MA60 > MA120, 0050 > MA20)
  weak_bull    — 中期向上但動能減弱
  sideways     — 區間盤整
  weak_bear    — 跌破中期但長期支撐在
  strong_bear  — 空頭排列
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import toml
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize

# 從 analysis/regime.py 共用核心函數和常數
from analysis.regime import (
    STRATEGIES, REGIMES,
    classify_regime, extract_daily_returns,
    backtest_dynamic_portfolio, compute_portfolio_metrics,
)

logging.basicConfig(level=logging.WARNING)

secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
API_TOKEN = secrets.get('FINLAB_API_KEY', secrets.get('FINLAB_API_TOKEN', ''))

if not API_TOKEN:
    print("ERROR: 找不到 FinLab API Token")
    sys.exit(1)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'regime_analysis.json')


# ============================================================
# 1. Regime 分類器 — imported from analysis.regime
# ============================================================
# classify_regime() is imported from analysis.regime


def get_regime_summary(regime_series):
    """Regime 分布統計。"""
    total = len(regime_series)
    counts = regime_series.value_counts()

    # 計算平均持續天數
    durations = {}
    for regime in REGIMES:
        mask = (regime_series == regime)
        if not mask.any():
            durations[regime] = 0
            continue
        # 計算連續段
        changes = mask.astype(int).diff().fillna(0).abs()
        segments = changes.cumsum()
        regime_segments = segments[mask]
        if len(regime_segments) == 0:
            durations[regime] = 0
        else:
            n_segments = regime_segments.nunique()
            durations[regime] = round(mask.sum() / max(n_segments, 1), 1)

    summary = {}
    for regime in REGIMES:
        c = counts.get(regime, 0)
        summary[regime] = {
            'n_days': int(c),
            'pct_days': round(c / total * 100, 1) if total > 0 else 0,
            'avg_duration_days': durations.get(regime, 0),
        }

    # 切換頻率
    transitions = (regime_series != regime_series.shift(1)).sum() - 1
    summary['total_transitions'] = int(max(transitions, 0))
    summary['avg_days_between_transitions'] = (
        round(total / max(transitions, 1), 1)
    )

    return summary


# ============================================================
# 2. 策略執行 (複用 portfolio_analysis 模式)
# ============================================================

def run_strategy(name, module_path, func_name, params):
    """執行單一策略回測。"""
    print(f"\n{'='*60}")
    print(f"  執行: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    import importlib
    mod = importlib.import_module(module_path)
    importlib.reload(mod)
    func = getattr(mod, func_name)
    report = func(API_TOKEN, **params)

    elapsed = time.time() - t0
    print(f"  完成: {name} ({elapsed:.1f}s)")
    return report


# extract_daily_returns() is imported from analysis.regime


# ============================================================
# 3. Regime-Strategy 績效矩陣
# ============================================================

def compute_regime_matrix(returns_dict, regime_series):
    """
    計算每個策略在每個 regime 的績效指標。

    Returns:
        dict — {strategy: {regime: {sharpe, ann_return, ann_vol, mdd, n_days}}}
    """
    matrix = {}

    for name, returns in returns_dict.items():
        # 對齊日期
        common_idx = returns.index.intersection(regime_series.index)
        ret = returns.reindex(common_idx).dropna()
        reg = regime_series.reindex(common_idx)

        strategy_stats = {}
        for regime in REGIMES:
            mask = (reg == regime)
            regime_ret = ret[mask]

            if len(regime_ret) < 20:
                strategy_stats[regime] = {
                    'sharpe': 0, 'ann_return': 0, 'ann_vol': 0,
                    'mdd': 0, 'n_days': int(mask.sum()),
                    'win_rate': 0,
                }
                continue

            ann_ret = regime_ret.mean() * 252
            ann_vol = regime_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            # MDD within regime periods
            cum = (1 + regime_ret).cumprod()
            rolling_max = cum.cummax()
            dd = (cum - rolling_max) / rolling_max
            mdd = float(dd.min())

            win_rate = float((regime_ret > 0).mean())

            strategy_stats[regime] = {
                'sharpe': round(float(sharpe), 3),
                'ann_return': round(float(ann_ret), 4),
                'ann_vol': round(float(ann_vol), 4),
                'mdd': round(mdd, 4),
                'n_days': int(mask.sum()),
                'win_rate': round(win_rate, 3),
            }

        matrix[name] = strategy_stats

    return matrix


# ============================================================
# 4. Regime 最優配置
# ============================================================

def optimize_regime_weights(returns_df, regime_series, allow_cash=True):
    """
    對每個 regime 最大化 portfolio Sharpe ratio。

    Args:
        returns_df: DataFrame — columns = strategy names, index = dates
        regime_series: pd.Series — regime labels
        allow_cash: bool — 是否允許權重總和 < 1.0 (差額 = 現金)

    Returns:
        dict — {regime: {strategy: weight, 'cash': weight}}
    """
    strategies = list(returns_df.columns)
    n = len(strategies)
    regime_weights = {}

    for regime in REGIMES:
        mask = regime_series.reindex(returns_df.index) == regime
        regime_ret = returns_df[mask].dropna()

        if len(regime_ret) < 30:
            # 不足以優化，用等權重
            w = {s: round(1.0 / n, 4) for s in strategies}
            w['cash'] = 0.0
            regime_weights[regime] = w
            continue

        mean_ret = regime_ret.mean().values
        cov = regime_ret.cov().values

        def neg_sharpe(w):
            port_ret = np.dot(w, mean_ret) * 252
            port_vol = np.sqrt(np.dot(w, np.dot(cov, w)) * 252)
            if port_vol < 1e-10:
                return 0
            return -port_ret / port_vol

        # 約束
        if allow_cash:
            constraints = [
                {'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)},  # sum <= 1.0
                {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.1},  # sum >= 0.1
            ]
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            ]

        bounds = [(0.0, 0.5)] * n
        x0 = np.array([1.0 / n] * n)

        try:
            result = minimize(neg_sharpe, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 500})
            w_opt = result.x
        except Exception:
            w_opt = x0

        # 正規化並記錄
        w_dict = {}
        for i, s in enumerate(strategies):
            w_dict[s] = round(float(max(w_opt[i], 0)), 4)
        cash = round(float(max(1.0 - sum(w_opt), 0)), 4)
        w_dict['cash'] = cash
        regime_weights[regime] = w_dict

    return regime_weights


# ============================================================
# 5. 動態組合回測 — imported from analysis.regime
# ============================================================
# backtest_dynamic_portfolio() and compute_portfolio_metrics()
# are imported from analysis.regime


def backtest_walkforward(returns_df, regime_series, train_end='2020-12-31'):
    """
    Walk-Forward 驗證: 用 train 期間最優化權重，在 test 期間測試。

    Returns:
        dict — {train_metrics, test_metrics, regime_weights}
    """
    train_mask = returns_df.index <= train_end
    test_mask = returns_df.index > train_end

    train_df = returns_df[train_mask]
    test_df = returns_df[test_mask]
    train_regime = regime_series[regime_series.index <= train_end]
    test_regime = regime_series[regime_series.index > train_end]

    # 用 train 期間最優化
    regime_weights = optimize_regime_weights(train_df, train_regime)

    # 在 test 期間回測
    test_returns = backtest_dynamic_portfolio(test_df, test_regime, regime_weights)
    test_metrics = compute_portfolio_metrics(test_returns)

    # 也計算 train 期間績效
    train_returns = backtest_dynamic_portfolio(train_df, train_regime, regime_weights)
    train_metrics = compute_portfolio_metrics(train_returns)

    return {
        'regime_weights': regime_weights,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_period': f'2010-01-01 ~ {train_end}',
        'test_period': f'{train_end} ~ end',
    }


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  市場 Regime 策略分析系統")
    print("=" * 60)

    import finlab
    finlab.login(API_TOKEN)
    from data.provider import sanitize_dataframe

    # ── 6.1 取得 0050 benchmark ──
    from finlab import data
    close_all = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    close_all.columns = close_all.columns.astype(str)

    if '0050' not in close_all.columns:
        print("ERROR: 找不到 0050 資料")
        sys.exit(1)

    benchmark = close_all['0050'].dropna()
    benchmark = benchmark.loc['2010-01-01':]
    print(f"\n  0050 benchmark: {len(benchmark)} 交易日 ({benchmark.index[0].strftime('%Y-%m-%d')} ~ {benchmark.index[-1].strftime('%Y-%m-%d')})")

    # ── 6.2 分類 regime ──
    print("\n" + "=" * 60)
    print("  Phase 1: Regime 分類")
    print("=" * 60)

    regime_series = classify_regime(benchmark, debounce_days=5)
    regime_summary = get_regime_summary(regime_series)

    print(f"\n  {'Regime':<15} {'天數':>6} {'佔比':>7} {'平均持續':>10}")
    print("  " + "-" * 42)
    for r in REGIMES:
        s = regime_summary[r]
        print(f"  {r:<15} {s['n_days']:>6} {s['pct_days']:>6.1f}% {s['avg_duration_days']:>9.1f}d")
    print(f"\n  總切換次數: {regime_summary['total_transitions']}")
    print(f"  平均切換間隔: {regime_summary['avg_days_between_transitions']:.1f} 天")

    # ── 6.3 執行五策略回測 ──
    print("\n" + "=" * 60)
    print("  Phase 2: 策略回測")
    print("=" * 60)

    reports = {}
    returns_dict = {}

    for name, (mod_path, func_name, params) in STRATEGIES.items():
        try:
            report = run_strategy(name, mod_path, func_name, params)
            reports[name] = report
            ret = extract_daily_returns(report)
            if ret is not None:
                returns_dict[name] = ret
                print(f"  → 日報酬序列: {len(ret)} 天")
            else:
                print(f"  → WARNING: 無法提取日報酬率")
        except Exception as e:
            print(f"  ERROR: {name} 回測失敗 — {type(e).__name__}: {e}")

    if len(returns_dict) < 2:
        print("\nERROR: 不足 2 個策略有日報酬率")
        return

    # ── 6.4 計算 Regime-Strategy 矩陣 ──
    print("\n" + "=" * 60)
    print("  Phase 3: Regime-Strategy 績效矩陣")
    print("=" * 60)

    matrix = compute_regime_matrix(returns_dict, regime_series)

    # 印出 Sharpe 矩陣
    names = list(matrix.keys())
    print(f"\n  {'策略':<18}", end="")
    for r in REGIMES:
        print(f"{r:>14}", end="")
    print()
    print("  " + "-" * (18 + 14 * len(REGIMES)))
    for name in names:
        print(f"  {name:<18}", end="")
        for r in REGIMES:
            s = matrix[name][r]['sharpe']
            print(f"{s:>14.3f}", end="")
        print()

    # ── 6.5 最優配置 ──
    print("\n" + "=" * 60)
    print("  Phase 4: Regime 最優配置")
    print("=" * 60)

    returns_df = pd.DataFrame(returns_dict).dropna()
    regime_weights = optimize_regime_weights(returns_df, regime_series)

    for regime in REGIMES:
        w = regime_weights[regime]
        print(f"\n  [{regime}]")
        for k, v in w.items():
            if v > 0.001:
                print(f"    {k:<20}: {v*100:.1f}%")

    # ── 6.6 組合比較 ──
    print("\n" + "=" * 60)
    print("  Phase 5: 組合比較")
    print("=" * 60)

    # A) 固定配置
    available = list(returns_df.columns)
    fixed_weights = {}
    fw_map = {'Isaac V3.9': 0.35, 'Will VCP V2.0': 0.10,
              'Mean Reversion': 0.15, 'Value Dividend': 0.25,
              'Pairs Trading': 0.15}
    for s in available:
        fixed_weights[s] = fw_map.get(s, 0)
    total_w = sum(fixed_weights.values())
    if total_w > 0:
        fixed_weights = {k: v/total_w for k, v in fixed_weights.items()}

    fixed_returns = returns_df.dot(
        pd.Series({s: fixed_weights.get(s, 0) for s in returns_df.columns})
    )
    fixed_metrics = compute_portfolio_metrics(fixed_returns)

    # B) Dynamic (in-sample)
    dynamic_is_returns = backtest_dynamic_portfolio(
        returns_df, regime_series, regime_weights
    )
    dynamic_is_metrics = compute_portfolio_metrics(dynamic_is_returns)

    # C) Walk-Forward
    wf_result = backtest_walkforward(returns_df, regime_series, train_end='2020-12-31')

    print(f"\n  {'組合':<30} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Vol':>8}")
    print("  " + "-" * 66)
    for label, m in [('Fixed (Isaac-heavy)', fixed_metrics),
                     ('Dynamic (in-sample)', dynamic_is_metrics),
                     ('Dynamic WF (test period)', wf_result['test_metrics'])]:
        print(f"  {label:<30} {m['cagr']*100:>7.2f}% {m['mdd']*100:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m.get('ann_vol',0)*100:>7.2f}%")

    # ── 6.7 Gap 分析 ──
    print("\n" + "=" * 60)
    print("  Phase 6: Gap 分析")
    print("=" * 60)

    for r in REGIMES:
        all_negative = all(matrix[n][r]['sharpe'] < 0 for n in names)
        best = max(names, key=lambda n: matrix[n][r]['sharpe'])
        best_sharpe = matrix[best][r]['sharpe']
        status = "⚠ ALL NEGATIVE" if all_negative else f"Best: {best} ({best_sharpe:.3f})"
        print(f"  {r:<15}: {status}")

    # ── 6.8 結論 ──
    print("\n" + "=" * 60)
    print("  結論")
    print("=" * 60)

    delta_sharpe = dynamic_is_metrics['sharpe'] - fixed_metrics['sharpe']
    wf_delta = wf_result['test_metrics']['sharpe'] - fixed_metrics['sharpe']

    if wf_result['test_metrics']['sharpe'] > fixed_metrics['sharpe'] * 1.05:
        verdict = "ADOPT — Walk-Forward 動態配置優於固定配置"
    elif wf_result['test_metrics']['sharpe'] > fixed_metrics['sharpe'] * 0.95:
        verdict = "NEUTRAL — 動態配置與固定配置差異不大，固定配置已足夠好"
    else:
        verdict = "REJECT — 動態配置 out-of-sample 劣於固定配置，存在過擬合"

    print(f"  In-sample Sharpe delta: {delta_sharpe:+.3f}")
    print(f"  Walk-forward Sharpe delta: {wf_delta:+.3f}")
    print(f"  判定: {verdict}")

    # ── 6.9 輸出 JSON ──
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'regime_summary': regime_summary,
        'strategy_regime_matrix': matrix,
        'optimal_weights': {
            r: {k: round(v, 4) for k, v in regime_weights[r].items()}
            for r in REGIMES
        },
        'portfolio_comparison': {
            'fixed': fixed_metrics,
            'dynamic_insample': dynamic_is_metrics,
            'dynamic_walkforward_train': wf_result['train_metrics'],
            'dynamic_walkforward_test': wf_result['test_metrics'],
        },
        'walkforward': {
            'train_period': wf_result['train_period'],
            'test_period': wf_result['test_period'],
            'regime_weights_from_train': wf_result['regime_weights'],
        },
        'verdict': verdict,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  結果已儲存至: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
