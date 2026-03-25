"""
Isaac V3.9 + Will VCP V2.0 A/B 測試
執行: ./python_embed/python.exe scripts/strategy_optimization_v2.py

判定標準:
  ADOPT: 至少一項指標改善 >0.5%, 且無指標跌破目標門檻
  REJECT: 任何關鍵指標惡化 >2%, 或 CAGR 下降 >1%
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import toml

secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
API_TOKEN = secrets.get('FINLAB_API_KEY', secrets.get('FINLAB_API_TOKEN', ''))

if not API_TOKEN:
    print("ERROR: FINLAB_API_TOKEN not found")
    sys.exit(1)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'optimization_v2_results.json')


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
        return 'REJECT', f'CAGR -{abs(cagr_delta):.2f}%'
    if mdd_delta < -2.0:
        return 'REJECT', f'MDD -{abs(mdd_delta):.2f}%'

    improvements = []
    if cagr_delta > 0.5:
        improvements.append(f'CAGR +{cagr_delta:.2f}%')
    if mdd_delta > 0.5:
        improvements.append(f'MDD +{mdd_delta:.2f}%')
    if sharpe_delta > 0.05:
        improvements.append(f'Sharpe +{sharpe_delta:.3f}')
    if win_delta > 0.5:
        improvements.append(f'Win% +{win_delta:.1f}%')

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
    print(f"Strategy Optimization V2 -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    all_results = {'isaac': [], 'will_vcp': []}

    # ============================================================
    # Isaac Baselines
    # ============================================================
    print("\n>>> Isaac V3.8 Baseline (min_score=6)")
    try:
        isaac_baseline = run_experiment('isaac', 'Isaac V3.8 Baseline', {'min_score': 6})
        print(f"  CAGR={isaac_baseline['cagr_pct']}% MDD={isaac_baseline['mdd_pct']}% Sharpe={isaac_baseline['sharpe']}")
    except Exception as e:
        print(f"  BASELINE ERROR: {e}")
        sys.exit(1)

    # ============================================================
    # Isaac Experiments
    # ============================================================
    isaac_experiments = [
        ('1.1 time_stop=15',   {'min_score': 6, 'time_stop_days': 15}),
        ('1.1 time_stop=10',   {'min_score': 6, 'time_stop_days': 10}),
        ('1.1 time_stop=20',   {'min_score': 6, 'time_stop_days': 20}),
        ('1.2 early_exit=3/-2%', {'min_score': 6, 'early_exit_days': 3, 'early_exit_threshold': -0.02}),
        ('1.2 early_exit=3/-3%', {'min_score': 6, 'early_exit_days': 3, 'early_exit_threshold': -0.03}),
        ('1.2 early_exit=5/-3%', {'min_score': 6, 'early_exit_days': 5, 'early_exit_threshold': -0.03}),
        ('1.3 adaptive_exit',    {'min_score': 6, 'adaptive_exit': True}),
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
    # Will VCP Baseline
    # ============================================================
    print(f"\n{'=' * 70}")
    print(">>> Will VCP V1.0 Baseline")
    try:
        vcp_baseline = run_experiment('will_vcp', 'Will VCP V1.0 Baseline', {})
        print(f"  CAGR={vcp_baseline['cagr_pct']}% MDD={vcp_baseline['mdd_pct']}% Sharpe={vcp_baseline['sharpe']}")
    except Exception as e:
        print(f"  BASELINE ERROR: {e}")
        sys.exit(1)

    # ============================================================
    # Will VCP Experiments
    # ============================================================
    vcp_experiments = [
        ('2.1 dynamic_exposure', {'dynamic_exposure': True}),
        ('2.2 confirm_2d',       {'confirm_days': 2}),
        ('2.3 trail_stop=0.20',  {'trail_stop': 0.20}),
        ('2.3 trail_stop=0.25',  {'trail_stop': 0.25}),
        ('2.4 time_stop=10',     {'time_stop_days': 10}),
        ('2.4 time_stop=15',     {'time_stop_days': 15}),
        ('2.5 near_high=0.90',   {'close_near_high_pct': 0.90}),
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
