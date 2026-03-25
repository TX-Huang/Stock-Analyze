"""
Isaac V3.9 Phase 2 + Will VCP V2.1 — 非時間型優化 A/B 測試
執行: ./python_embed/python.exe scripts/strategy_optimization_v3.py

核心思路:
  Isaac: 不改出場, 改進場品質 + 部位管理 + regime filter
  Will VCP: MDD 控制 (持倉數 + regime gate + portfolio DD stop)

判定標準:
  ADOPT: 至少一項指標改善 >0.5%, 且無指標跌破目標門檻
  REJECT: CAGR 下降 >1% 或 MDD 惡化 >2%
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import toml

secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
API_TOKEN = secrets.get('FINLAB_API_KEY', secrets.get('FINLAB_API_TOKEN', ''))

if not API_TOKEN:
    print("ERROR: FINLAB_API_TOKEN not found")
    sys.exit(1)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'optimization_v3_results.json')


def extract_stats(report, label):
    stats = report.get_stats()
    trades = report.get_trades()

    cagr = stats.get('cagr', 0)
    mdd = stats.get('max_drawdown', 0)
    sharpe = stats.get('daily_sharpe', stats.get('sharpe', 0))
    win_ratio = stats.get('win_ratio', 0)
    n_trades = len(trades)

    avg_hold = 0
    risk_reward = 0
    if not trades.empty:
        avg_hold = float(trades['period'].mean())
        avg_win = trades[trades['return'] > 0]['return'].mean() if len(trades[trades['return'] > 0]) > 0 else 0
        avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if len(trades[trades['return'] <= 0]) > 0 else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        'label': label,
        'cagr_pct': round(cagr * 100, 2),
        'mdd_pct': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'win_ratio_pct': round(win_ratio * 100, 1),
        'risk_reward': round(risk_reward, 2),
        'n_trades': n_trades,
        'avg_hold_days': round(avg_hold, 1),
    }


def judge(result, baseline):
    cagr_delta = result['cagr_pct'] - baseline['cagr_pct']
    mdd_delta = result['mdd_pct'] - baseline['mdd_pct']
    sharpe_delta = result['sharpe'] - baseline['sharpe']
    win_delta = result['win_ratio_pct'] - baseline['win_ratio_pct']

    if cagr_delta < -1.0:
        return 'REJECT', f'CAGR {cagr_delta:+.2f}%'
    if mdd_delta < -2.0:
        return 'REJECT', f'MDD {mdd_delta:+.2f}%'

    improvements = []
    if cagr_delta > 0.5:
        improvements.append(f'CAGR {cagr_delta:+.2f}%')
    if mdd_delta > 0.5:
        improvements.append(f'MDD {mdd_delta:+.2f}%')
    if sharpe_delta > 0.05:
        improvements.append(f'Sharpe {sharpe_delta:+.3f}')
    if win_delta > 0.5:
        improvements.append(f'Win% {win_delta:+.1f}%')

    if improvements:
        return 'ADOPT', ', '.join(improvements)
    return 'NEUTRAL', 'No significant change'


def print_comparison(result, baseline):
    verdict, reason = judge(result, baseline)
    print(f"\n  [{result['label']}]")
    print(f"  CAGR:      {result['cagr_pct']:>8.2f}%  (d {result['cagr_pct'] - baseline['cagr_pct']:+.2f}%)")
    print(f"  MDD:       {result['mdd_pct']:>8.2f}%  (d {result['mdd_pct'] - baseline['mdd_pct']:+.2f}%)")
    print(f"  Sharpe:    {result['sharpe']:>8.3f}   (d {result['sharpe'] - baseline['sharpe']:+.3f})")
    print(f"  Win%:      {result['win_ratio_pct']:>8.1f}%  (d {result['win_ratio_pct'] - baseline['win_ratio_pct']:+.1f}%)")
    print(f"  R/R:       {result['risk_reward']:>8.2f}   (d {result['risk_reward'] - baseline['risk_reward']:+.2f})")
    print(f"  Trades:    {result['n_trades']:>8d}   (d {result['n_trades'] - baseline['n_trades']:+d})")
    print(f"  Hold:      {result['avg_hold_days']:>8.1f}d  (d {result['avg_hold_days'] - baseline['avg_hold_days']:+.1f}d)")
    print(f"  >>> {verdict} -- {reason}")
    return verdict, reason


def run_experiment(strategy, label, params):
    t0 = time.time()
    if strategy == 'isaac':
        from strategies.isaac import run_isaac_strategy
        report = run_isaac_strategy(API_TOKEN, params=params)
    elif strategy == 'will_vcp':
        from strategies.will_vcp import run_will_vcp_strategy
        report = run_will_vcp_strategy(API_TOKEN, params=params)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    result = extract_stats(report, label)
    result['elapsed_sec'] = round(time.time() - t0, 1)
    return result


if __name__ == '__main__':
    from datetime import datetime
    print(f"Strategy Optimization V3 (Non-Time Exit) -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    all_results = {'isaac': [], 'will_vcp': []}

    # ============================================================
    # Isaac Baseline (min_score=4, current default)
    # ============================================================
    print("\n>>> Isaac V3.8 Baseline (min_score=4, default)")
    try:
        isaac_baseline = run_experiment('isaac', 'Isaac V3.8 Baseline', {})
        print(f"  CAGR={isaac_baseline['cagr_pct']}% MDD={isaac_baseline['mdd_pct']}% Sharpe={isaac_baseline['sharpe']}")
    except Exception as e:
        print(f"  BASELINE ERROR: {e}")
        sys.exit(1)

    # ============================================================
    # Isaac Experiments — 進場品質 + 部位管理 + Regime
    # ============================================================
    # Note: Isaac V3.5+ 已內建 dynamic_exposure + score_weight，
    #       所以這裡只測 min_score / industry / trail_stop 等可調參數
    isaac_experiments = [
        # 方向一: 提高進場門檻 (目前 min_score=4)
        ('3.1 min_score=5',         {'min_score': 5}),
        ('3.2 min_score=6',         {'min_score': 6}),
        ('3.3 min_score=7',         {'min_score': 7}),
        # 方向二: 產業集中度限制
        ('3.4 industry=3',          {'max_per_industry': 3}),
        ('3.5 industry=2',          {'max_per_industry': 2}),
        # 方向三: Trail Stop 調整 (目前 0.18)
        ('3.6 trail=0.20',          {'trail_stop': 0.20}),
        ('3.7 trail=0.25',          {'trail_stop': 0.25}),
        # 組合實驗: 最有潛力的組合
        ('3.8 score5+industry3',    {'min_score': 5, 'max_per_industry': 3}),
        ('3.9 score6+industry3',    {'min_score': 6, 'max_per_industry': 3}),
        ('3.10 score6+trail20',     {'min_score': 6, 'trail_stop': 0.20}),
        ('3.11 score5+trail20+ind3', {'min_score': 5, 'trail_stop': 0.20, 'max_per_industry': 3}),
    ]

    for label, params in isaac_experiments:
        print(f"\n{'=' * 70}")
        print(f"Isaac Experiment: {label}")
        print("=" * 70)
        try:
            result = run_experiment('isaac', label, params)
            verdict, reason = print_comparison(result, isaac_baseline)
            result['verdict'] = verdict
            result['reason'] = reason
            all_results['isaac'].append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results['isaac'].append({'label': label, 'error': str(e)})

    # ============================================================
    # Will VCP Baseline (trail_stop=0.25, V2.0)
    # ============================================================
    print(f"\n{'=' * 70}")
    print(">>> Will VCP V2.0 Baseline (trail_stop=0.25)")
    try:
        vcp_baseline = run_experiment('will_vcp', 'Will VCP V2.0 Baseline', {'trail_stop': 0.25})
        print(f"  CAGR={vcp_baseline['cagr_pct']}% MDD={vcp_baseline['mdd_pct']}% Sharpe={vcp_baseline['sharpe']}")
    except Exception as e:
        print(f"  BASELINE ERROR: {e}")
        sys.exit(1)

    # ============================================================
    # Will VCP Experiments — MDD 控制
    # ============================================================
    vcp_experiments = [
        # 方向一: 減少最大持倉數
        ('4.1 max_pos=8',          {'trail_stop': 0.25, 'max_positions': 8}),
        ('4.2 max_pos=6',          {'trail_stop': 0.25, 'max_positions': 6}),
        # 方向二: 大盤 Regime filter (已有 dynamic_exposure)
        ('4.3 regime_exposure',    {'trail_stop': 0.25, 'dynamic_exposure': True}),
        # 方向三: 組合最佳
        ('4.4 pos8+regime',        {'trail_stop': 0.25, 'max_positions': 8, 'dynamic_exposure': True}),
        ('4.5 pos6+regime',        {'trail_stop': 0.25, 'max_positions': 6, 'dynamic_exposure': True}),
        # 方向四: 更寬 trail_stop + 減倉
        ('4.6 trail30+pos8',       {'trail_stop': 0.30, 'max_positions': 8}),
        ('4.7 trail30+pos8+regime', {'trail_stop': 0.30, 'max_positions': 8, 'dynamic_exposure': True}),
    ]

    for label, params in vcp_experiments:
        print(f"\n{'=' * 70}")
        print(f"Will VCP Experiment: {label}")
        print("=" * 70)
        try:
            result = run_experiment('will_vcp', label, params)
            verdict, reason = print_comparison(result, vcp_baseline)
            result['verdict'] = verdict
            result['reason'] = reason
            all_results['will_vcp'].append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results['will_vcp'].append({'label': label, 'error': str(e)})

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    for strategy, baseline in [('isaac', isaac_baseline), ('will_vcp', vcp_baseline)]:
        print(f"\n--- {strategy.upper()} ---")
        header = f"{'Experiment':<28} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Win%':>6} {'Verdict':>10}"
        print(header)
        print("-" * len(header))
        print(f"{'Baseline':<28} {baseline['cagr_pct']:>7.2f} {baseline['mdd_pct']:>7.2f} {baseline['sharpe']:>7.3f} {baseline['win_ratio_pct']:>6.1f} {'---':>10}")
        for exp in all_results[strategy]:
            if 'error' in exp:
                print(f"{exp['label']:<28} {'ERROR':>7}")
            else:
                print(f"{exp['label']:<28} {exp['cagr_pct']:>7.2f} {exp['mdd_pct']:>7.2f} {exp['sharpe']:>7.3f} {exp['win_ratio_pct']:>6.1f} {exp['verdict']:>10}")

    # Adopted items
    for strategy in ['isaac', 'will_vcp']:
        adopted = [e for e in all_results[strategy] if e.get('verdict') == 'ADOPT']
        if adopted:
            print(f"\n{strategy.upper()} ADOPTED:")
            for a in adopted:
                print(f"  [V] {a['label']}: {a['reason']}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'isaac_baseline': isaac_baseline,
        'will_vcp_baseline': vcp_baseline,
        'results': all_results,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
