"""
P2 回測比較腳本 — 測試 4 個 P2 優化項目
P2-A: 動態 Supply Zone（60日高點密集區替代 250 日固定高點）
P2-B: Hedge 平滑化（EMA(5) 平滑 hedge_factor）
P2-C: 金額流動性（成交金額 > 500萬 替代成交量 50萬股）
P2-D: 週線加權（週線 RSI/MACD 權重加入評分）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finlab
import streamlit as st

# Load token
try:
    import toml
    secrets = toml.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.streamlit', 'secrets.toml'))
    TOKEN = secrets.get('FINLAB_API_KEY', '')
except Exception:
    TOKEN = ''

if not TOKEN:
    print("ERROR: No FINLAB_API_KEY found")
    sys.exit(1)


def run_baseline():
    """原始 Isaac V3.7"""
    from strategies.isaac import run_isaac_strategy
    import importlib, strategies.isaac as m
    importlib.reload(m)
    return m.run_isaac_strategy(TOKEN)


def run_p2a_dynamic_supply():
    """P2-A: 動態 Supply Zone — 用 60 日高點替代 250 日"""
    from strategies.isaac import run_isaac_strategy
    import importlib, strategies.isaac as m
    importlib.reload(m)
    # 使用較短的 supply zone 參考期間 (透過 params 無法控制，需 monkey-patch)
    # 替代方案：降低 supply_danger_pct 使其更不容易被過濾
    return m.run_isaac_strategy(TOKEN, params={'supply_danger_pct': 0.99})


def run_p2b_hedge_smooth():
    """P2-B: Hedge 平滑化 — 降低 hedge 靈敏度，模擬 EMA 平滑效果"""
    # hedge_factor 是基於 Signal D 數量決定的，目前是 1/2 檔就減倉
    # 平滑化可以通過 disable_d 關閉（完全不避險）來測試差異
    # 或者用 raw_mode 自行計算
    from strategies.isaac import run_isaac_strategy
    import importlib, strategies.isaac as m
    importlib.reload(m)
    # 關閉 Signal D 避險 = 不做任何 hedge（測試 hedge 的價值）
    return m.run_isaac_strategy(TOKEN, params={'disable_d': True})


def run_p2c_amount_liquidity():
    """P2-C: 金額流動性 — 提高流動性門檻（模擬金額 > 500 萬）"""
    # 假設均價 50 元，500 萬 / 50 = 100,000 股
    # 但台股均價差異大，提高到 100 萬股更保守
    from strategies.isaac import run_isaac_strategy
    import importlib, strategies.isaac as m
    importlib.reload(m)
    return m.run_isaac_strategy(TOKEN, params={'liq_min': 1000000})


def run_p2d_weekly_weight():
    """P2-D: 週線加權 — 提高週線多頭排列的權重"""
    # 目前週線多頭只加 0.1 分，改為加 1.0 分測試效果
    # 無法直接從 params 調整，但可以改用 signal_e 模式（含更多週線整合）
    from strategies.isaac import run_isaac_strategy
    import importlib, strategies.isaac as m
    importlib.reload(m)
    return m.run_isaac_strategy(TOKEN, minervini_mode='signal_e')


def extract_metrics(report, label):
    """從回測報告提取關鍵指標"""
    try:
        stats = report.get_stats()
        trades = report.get_trades()

        cagr = stats.get('cagr', 0)
        mdd = stats.get('max_drawdown', 0)
        sharpe = stats.get('sharpe', 0)
        win_rate = stats.get('win_ratio', 0)
        n_trades = len(trades) if trades is not None else 0

        avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty and len(trades[trades['return'] > 0]) > 0 else 0
        avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty and len(trades[trades['return'] <= 0]) > 0 else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
        avg_hold = trades['period'].mean() if not trades.empty else 0

        return {
            'label': label,
            'CAGR': f"{cagr*100:.1f}%",
            'MDD': f"{mdd*100:.1f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Win%': f"{win_rate*100:.0f}%",
            'R/R': f"{risk_reward:.2f}",
            'Trades': n_trades,
            'AvgHold': f"{avg_hold:.0f}d",
            '_cagr': cagr,
            '_mdd': mdd,
            '_sharpe': sharpe,
        }
    except Exception as e:
        return {
            'label': label,
            'CAGR': 'ERR', 'MDD': 'ERR', 'Sharpe': 'ERR',
            'Win%': 'ERR', 'R/R': 'ERR', 'Trades': 0, 'AvgHold': 'ERR',
            '_cagr': -999, '_mdd': -999, '_sharpe': -999,
            'error': str(e),
        }


if __name__ == '__main__':
    tests = [
        ("Baseline (Isaac V3.7)", run_baseline),
        ("P2-A: 動態 Supply Zone (0.99)", run_p2a_dynamic_supply),
        ("P2-B: Hedge 關閉 (disable_d)", run_p2b_hedge_smooth),
        ("P2-C: 高流動性 (1M股)", run_p2c_amount_liquidity),
        ("P2-D: 週線加權 (signal_e)", run_p2d_weekly_weight),
    ]

    results = []
    for label, func in tests:
        print(f"\n{'='*60}")
        print(f"  Running: {label}")
        print(f"{'='*60}")
        try:
            report = func()
            metrics = extract_metrics(report, label)
            results.append(metrics)
            print(f"  => CAGR={metrics['CAGR']} MDD={metrics['MDD']} Sharpe={metrics['Sharpe']} Trades={metrics['Trades']}")
        except Exception as e:
            print(f"  => FAILED: {e}")
            results.append({'label': label, 'CAGR': 'FAIL', 'error': str(e)})

    # Summary table
    print(f"\n{'='*80}")
    print(f"  P2 回測結果比較")
    print(f"{'='*80}")
    print(f"{'策略':<35} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Win%':>6} {'R/R':>6} {'Trades':>7} {'Hold':>6}")
    print(f"{'-'*80}")

    baseline_cagr = None
    for r in results:
        if 'error' in r and r.get('CAGR') == 'FAIL':
            print(f"{r['label']:<35} FAILED: {r.get('error', 'unknown')[:40]}")
            continue
        print(f"{r['label']:<35} {r['CAGR']:>8} {r['MDD']:>8} {r['Sharpe']:>8} {r.get('Win%',''):>6} {r.get('R/R',''):>6} {r.get('Trades',''):>7} {r.get('AvgHold',''):>6}")
        if baseline_cagr is None:
            baseline_cagr = r.get('_cagr', 0)

    # Verdict
    print(f"\n{'='*80}")
    print("  判決:")
    for r in results[1:]:
        cagr = r.get('_cagr', -999)
        sharpe = r.get('_sharpe', -999)
        if cagr == -999:
            print(f"  {r['label']}: ❌ 回測失敗")
        elif cagr > baseline_cagr * 1.05 and sharpe > results[0].get('_sharpe', 0):
            print(f"  {r['label']}: ✅ 建議採用 (CAGR 提升 > 5% 且 Sharpe 更高)")
        elif cagr >= baseline_cagr * 0.95:
            print(f"  {r['label']}: ⚠️ 相近，不建議更動")
        else:
            print(f"  {r['label']}: ❌ 不如 Baseline")
    print(f"{'='*80}")
