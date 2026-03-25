"""
Isaac V3.7 參數實驗 — Phase 2 A/B 測試
執行: ./python_embed/python.exe scripts/isaac_experiments.py

輸出: data/isaac_experiments.json

判定標準:
  ADOPT: 至少一項指標改善 >0.5%，且無指標跌破目標門檻
  REJECT: 任何關鍵指標惡化 >2%，或 CAGR 下降 >1%
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
    print("ERROR: 找不到 FinLab API Token")
    sys.exit(1)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'isaac_experiments.json')

# 基準值 (Phase 1 取得)
BASELINE = {
    'label': 'Isaac V3.7 (Baseline)',
    'cagr_pct': 20.41,
    'mdd_pct': -30.87,
    'sharpe': 0.949,
    'win_ratio_pct': 45.8,
    'risk_reward': 1.84,
    'n_trades': 2602,
    'avg_hold_days': 13.2,
}


def extract_stats(report, label):
    """從回測報告提取 7 項關鍵指標"""
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


def judge(result, baseline=None):
    """判定 ADOPT / REJECT / NEUTRAL"""
    if baseline is None:
        baseline = BASELINE

    cagr_delta = result['cagr_pct'] - baseline['cagr_pct']
    mdd_delta = result['mdd_pct'] - baseline['mdd_pct']  # less negative = better
    sharpe_delta = result['sharpe'] - baseline['sharpe']
    win_delta = result['win_ratio_pct'] - baseline['win_ratio_pct']

    # REJECT: CAGR 下降 >1% 或 MDD 惡化 >2%
    if cagr_delta < -1.0:
        return 'REJECT', f'CAGR 下降 {cagr_delta:.2f}%'
    if mdd_delta < -2.0:
        return 'REJECT', f'MDD 惡化 {mdd_delta:.2f}%'

    # ADOPT: 任一指標改善 >0.5%
    improvements = []
    if cagr_delta > 0.5:
        improvements.append(f'CAGR +{cagr_delta:.2f}%')
    if mdd_delta > 0.5:  # less negative = improvement
        improvements.append(f'MDD 改善 {mdd_delta:.2f}%')
    if sharpe_delta > 0.05:
        improvements.append(f'Sharpe +{sharpe_delta:.3f}')
    if win_delta > 0.5:
        improvements.append(f'Win% +{win_delta:.1f}%')

    if improvements:
        return 'ADOPT', ', '.join(improvements)

    return 'NEUTRAL', '無顯著改善或惡化'


def print_comparison(result, baseline=None):
    """印出與基準的比較"""
    if baseline is None:
        baseline = BASELINE
    verdict, reason = judge(result, baseline)

    print(f"\n  [{result['label']}]")
    print(f"  CAGR:      {result['cagr_pct']:>8.2f}%  (Δ {result['cagr_pct'] - baseline['cagr_pct']:+.2f}%)")
    print(f"  MDD:       {result['mdd_pct']:>8.2f}%  (Δ {result['mdd_pct'] - baseline['mdd_pct']:+.2f}%)")
    print(f"  Sharpe:    {result['sharpe']:>8.3f}   (Δ {result['sharpe'] - baseline['sharpe']:+.3f})")
    print(f"  Win%:      {result['win_ratio_pct']:>8.1f}%  (Δ {result['win_ratio_pct'] - baseline['win_ratio_pct']:+.1f}%)")
    print(f"  Risk/Rew:  {result['risk_reward']:>8.2f}   (Δ {result['risk_reward'] - baseline['risk_reward']:+.2f})")
    print(f"  Trades:    {result['n_trades']:>8d}   (Δ {result['n_trades'] - baseline['n_trades']:+d})")
    print(f"  Avg Hold:  {result['avg_hold_days']:>8.1f}d  (Δ {result['avg_hold_days'] - baseline['avg_hold_days']:+.1f}d)")
    print(f"  >>> 判定: {verdict} — {reason}")
    return verdict, reason


def run_experiment(label, params):
    """執行單一實驗"""
    from strategies.isaac import run_isaac_strategy
    report = run_isaac_strategy(API_TOKEN, params=params)
    result = extract_stats(report, label)
    return result


if __name__ == '__main__':
    from datetime import datetime
    print(f"Isaac V3.7 參數實驗 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print(f"基準: CAGR {BASELINE['cagr_pct']}% | MDD {BASELINE['mdd_pct']}% | "
          f"Sharpe {BASELINE['sharpe']} | Win {BASELINE['win_ratio_pct']}%")
    print("=" * 70)

    experiments = []

    # ============================================================
    # 2.1 大盤寬度過濾
    # ============================================================
    print("\n" + "=" * 70)
    print("實驗 2.1: breadth_filter=True")
    print("=" * 70)
    t0 = time.time()
    try:
        result = run_experiment('2.1 breadth_filter', {'breadth_filter': True})
        result['elapsed_sec'] = round(time.time() - t0, 1)
        verdict, reason = print_comparison(result)
        result['verdict'] = verdict
        result['reason'] = reason
        experiments.append(result)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        experiments.append({'label': '2.1 breadth_filter', 'error': str(e)})

    # ============================================================
    # 2.2 Score 門檻 sweep
    # ============================================================
    for min_score in [3, 4, 5, 6, 7]:
        print(f"\n{'=' * 70}")
        print(f"實驗 2.2: min_score={min_score}")
        print("=" * 70)
        t0 = time.time()
        try:
            result = run_experiment(f'2.2 min_score={min_score}', {'min_score': min_score})
            result['elapsed_sec'] = round(time.time() - t0, 1)
            verdict, reason = print_comparison(result)
            result['verdict'] = verdict
            result['reason'] = reason
            experiments.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            experiments.append({'label': f'2.2 min_score={min_score}', 'error': str(e)})

    # ============================================================
    # 2.3 產業集中度限制 sweep
    # ============================================================
    for max_ind in [2, 3, 4, 5]:
        print(f"\n{'=' * 70}")
        print(f"實驗 2.3: max_per_industry={max_ind}")
        print("=" * 70)
        t0 = time.time()
        try:
            result = run_experiment(f'2.3 max_per_industry={max_ind}', {'max_per_industry': max_ind})
            result['elapsed_sec'] = round(time.time() - t0, 1)
            verdict, reason = print_comparison(result)
            result['verdict'] = verdict
            result['reason'] = reason
            experiments.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            experiments.append({'label': f'2.3 max_per_industry={max_ind}', 'error': str(e)})

    # ============================================================
    # 2.4 突破確認 sweep
    # ============================================================
    for confirm in [1, 2, 3]:
        print(f"\n{'=' * 70}")
        print(f"實驗 2.4: confirm_days={confirm}")
        print("=" * 70)
        t0 = time.time()
        try:
            result = run_experiment(f'2.4 confirm_days={confirm}', {'confirm_days': confirm})
            result['elapsed_sec'] = round(time.time() - t0, 1)
            verdict, reason = print_comparison(result)
            result['verdict'] = verdict
            result['reason'] = reason
            experiments.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            experiments.append({'label': f'2.4 confirm_days={confirm}', 'error': str(e)})

    # ============================================================
    # 2.5 時間停損 sweep
    # ============================================================
    for days in [30, 45, 60, 90]:
        print(f"\n{'=' * 70}")
        print(f"實驗 2.5: time_stop_days={days}")
        print("=" * 70)
        t0 = time.time()
        try:
            result = run_experiment(f'2.5 time_stop_days={days}', {'time_stop_days': days})
            result['elapsed_sec'] = round(time.time() - t0, 1)
            verdict, reason = print_comparison(result)
            result['verdict'] = verdict
            result['reason'] = reason
            experiments.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            experiments.append({'label': f'2.5 time_stop_days={days}', 'error': str(e)})

    # ============================================================
    # 總結
    # ============================================================
    print("\n" + "=" * 70)
    print("實驗總結")
    print("=" * 70)
    header = f"{'Experiment':<30} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'Win%':>6} {'Verdict':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Baseline':.<30} {BASELINE['cagr_pct']:>7.2f} {BASELINE['mdd_pct']:>7.2f} {BASELINE['sharpe']:>7.3f} {BASELINE['win_ratio_pct']:>6.1f} {'---':>10}")
    for exp in experiments:
        if 'error' in exp:
            print(f"{exp['label']:<30} {'ERROR':>7}")
        else:
            print(f"{exp['label']:<30} {exp['cagr_pct']:>7.2f} {exp['mdd_pct']:>7.2f} {exp['sharpe']:>7.3f} {exp['win_ratio_pct']:>6.1f} {exp['verdict']:>10}")

    adopted = [e for e in experiments if e.get('verdict') == 'ADOPT']
    print(f"\nADOPT: {len(adopted)} items")
    for a in adopted:
        print(f"  [V] {a['label']}: {a['reason']}")

    # 儲存
    output = {
        'timestamp': datetime.now().isoformat(),
        'baseline': BASELINE,
        'experiments': experiments,
        'adopted': [a['label'] for a in adopted],
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
