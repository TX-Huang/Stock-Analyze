"""
Isaac V3.5 優化測試: ATR出場 / Signal D分析 / 時間停損
一次跑完所有變體並比較績效
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import toml
import warnings
warnings.filterwarnings('ignore')

secrets = toml.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.streamlit', 'secrets.toml'))
API_TOKEN = secrets.get('FINLAB_API_KEY', '')

from strategies.isaac import run_isaac_strategy

# 測試變體定義 — Round 2
VARIANTS = {
    # 基準
    'V3.5 Baseline': {},

    # 測試 1: Score 門檻過濾
    'Min Score >= 2': {'min_score': 2},
    'Min Score >= 3': {'min_score': 3},
    'Min Score >= 4': {'min_score': 4},

    # 測試 2: 突破確認 (N+1 日)
    'Confirm 2d': {'confirm_days': 2},
    'Confirm 3d': {'confirm_days': 3},

    # 測試 3: 產業集中度限制
    'Max 2 per Industry': {'max_per_industry': 2},
    'Max 3 per Industry': {'max_per_industry': 3},

    # 測試 4: 大盤寬度過濾
    'Breadth Filter': {'breadth_filter': True},
}

def run_variant(name, params):
    """執行一個變體並回傳績效指標"""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Params: {params}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        report = run_isaac_strategy(API_TOKEN, params=params)
        stats = report.get_stats()
        trades = report.get_trades()
        elapsed = time.time() - t0
        result = {
            'name': name,
            'cagr': stats.get('cagr', 0),
            'max_dd': stats.get('max_drawdown', 0),
            'sharpe': stats.get('daily_sharpe', 0),
            'sortino': stats.get('daily_sortino', 0),
            'calmar': stats.get('calmar', 0),
            'win_ratio': stats.get('win_ratio', 0),
            'total_return': stats.get('total_return', 0),
            'trades': len(trades),
            'time_sec': round(elapsed, 1),
        }
        # Score: (cagr/|dd|)*0.4 + sharpe*0.4 + win*0.2
        dd = abs(result['max_dd']) if result['max_dd'] != 0 else 1
        result['score'] = round(
            (result['cagr'] / dd) * 0.4 + result['sharpe'] * 0.4 + result['win_ratio'] * 0.2, 4
        )
        print(f"  -> CAGR: {result['cagr']*100:+.2f}% | MaxDD: {result['max_dd']*100:.2f}% | "
              f"Sharpe: {result['sharpe']:.3f} | Win: {result['win_ratio']*100:.1f}% | "
              f"Score: {result['score']:.4f} | {elapsed:.0f}s")
        return result
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return {'name': name, 'error': str(e)}


if __name__ == '__main__':
    results = []
    for name, params in VARIANTS.items():
        r = run_variant(name, params)
        results.append(r)

    # 彙整比較表
    print("\n\n")
    print("=" * 120)
    print("  OPTIMIZATION TEST RESULTS COMPARISON")
    print("=" * 120)
    header = (f"{'Variant':<30} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>7} | "
              f"{'Sortino':>8} | {'Win%':>6} | {'Trades':>6} | {'Score':>8}")
    print(header)
    print("-" * 120)

    baseline = None
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<30} | {'ERROR':>8}")
            continue
        if baseline is None:
            baseline = r
        diff_cagr = (r['cagr'] - baseline['cagr']) * 100
        diff_dd = (r['max_dd'] - baseline['max_dd']) * 100
        diff_score = r['score'] - baseline['score']
        line = (f"{r['name']:<30} | {r['cagr']*100:>+7.2f}% | {r['max_dd']*100:>7.2f}% | "
                f"{r['sharpe']:>7.3f} | {r['sortino']:>8.3f} | {r['win_ratio']*100:>5.1f}% | "
                f"{r['trades']:>6} | {r['score']:>8.4f}")
        if r != baseline:
            line += f"  (CAGR {diff_cagr:+.2f}%, DD {diff_dd:+.2f}%, Score {diff_score:+.4f})"
        else:
            line += "  [BASELINE]"
        print(line)

    print("=" * 120)

    # 儲存結果
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'opt_test_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
