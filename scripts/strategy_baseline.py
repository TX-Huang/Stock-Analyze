"""
三策略基準比較 — Isaac V3.7 / Will VCP V1.0 / VCP V1.1
執行: ./python_embed/python.exe scripts/strategy_baseline.py

輸出: data/strategy_baselines.json
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

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'strategy_baselines.json')


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

    result = {
        'label': label,
        'cagr_pct': round(cagr * 100, 2),
        'mdd_pct': round(mdd * 100, 2),
        'sharpe': round(sharpe, 3),
        'win_ratio_pct': round(win_ratio * 100, 1),
        'risk_reward': round(risk_reward, 2),
        'n_trades': n_trades,
        'avg_hold_days': round(avg_hold, 1),
    }

    print(f"\n  [{label}]")
    print(f"  CAGR:      {result['cagr_pct']:>8.2f}%")
    print(f"  MDD:       {result['mdd_pct']:>8.2f}%")
    print(f"  Sharpe:    {result['sharpe']:>8.3f}")
    print(f"  Win%:      {result['win_ratio_pct']:>8.1f}%")
    print(f"  Risk/Rew:  {result['risk_reward']:>8.2f}")
    print(f"  Trades:    {result['n_trades']:>8d}")
    print(f"  Avg Hold:  {result['avg_hold_days']:>8.1f} days")

    return result


def run_isaac():
    print("\n" + "=" * 60)
    print("1) Isaac V3.7")
    print("=" * 60)
    from strategies.isaac import run_isaac_strategy
    report = run_isaac_strategy(API_TOKEN)
    return extract_stats(report, 'Isaac V3.7')


def run_will_vcp():
    print("\n" + "=" * 60)
    print("2) Will VCP V1.0")
    print("=" * 60)
    from strategies.will_vcp import run_will_vcp_strategy
    report = run_will_vcp_strategy(API_TOKEN)
    return extract_stats(report, 'Will VCP V1.0')


def run_vcp():
    print("\n" + "=" * 60)
    print("3) VCP V1.1 (bug fixed)")
    print("=" * 60)
    from strategies.vcp import run_vcp_strategy
    report = run_vcp_strategy(API_TOKEN)
    return extract_stats(report, 'VCP V1.1')


if __name__ == '__main__':
    from datetime import datetime
    print(f"策略基準比較 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results = []
    errors = []

    for name, fn in [('Isaac V3.7', run_isaac), ('Will VCP V1.0', run_will_vcp), ('VCP V1.1', run_vcp)]:
        t0 = time.time()
        try:
            result = fn()
            result['elapsed_sec'] = round(time.time() - t0, 1)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR running {name}: {e}")
            import traceback
            traceback.print_exc()
            errors.append({'label': name, 'error': str(e)})

    # 結果摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Strategy':<20} {'CAGR%':>8} {'MDD%':>8} {'Sharpe':>8} {'Win%':>8} {'R/R':>6} {'Trades':>7} {'Hold':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['label']:<20} {r['cagr_pct']:>8.2f} {r['mdd_pct']:>8.2f} {r['sharpe']:>8.3f} "
              f"{r['win_ratio_pct']:>8.1f} {r['risk_reward']:>6.2f} {r['n_trades']:>7d} {r['avg_hold_days']:>6.1f}")

    # 儲存
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'errors': errors,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
