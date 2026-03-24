"""
Isaac V3.5 進階優化 A/B 測試
測試方向:
  1. 部分停利 (profit_take_half)
  2. Score 加權持倉 (score_weight)
  3. 回檔進場 (pullback_entry)
  4. 信號貢獻度分析 (signal_contribution)
  5. 多週期確認 - 月線 (monthly_confirm)
  6. 組合測試 (combo)
"""
import sys, os, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import toml
import numpy as np
import pandas as pd

secrets = toml.load(os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml'))
API_TOKEN = secrets.get('FINLAB_KEY', '') or secrets.get('FINLAB_API_KEY', '')

from isaac import run_isaac_strategy
from data.provider import safe_finlab_sim


def run_baseline():
    """基準: V3.5 + min_score>=4 (目前預設)"""
    report = run_isaac_strategy(API_TOKEN)
    return extract_stats(report, 'baseline')


def run_with_params(name, params):
    """用指定參數跑回測"""
    report = run_isaac_strategy(API_TOKEN, params=params)
    return extract_stats(report, name)


def run_score_weight():
    """Score 加權持倉: 高分多買低分少買"""
    raw = run_isaac_strategy(API_TOKEN, raw_mode=True)
    final_pos = raw['final_pos']
    close = raw['close']
    benchmark_close = raw['etf_close']

    # 將 score 轉為加權 (score/max_score_per_row)
    # 正常 score 範圍 4~15, 用 score/10 作為權重
    alloc_pos = final_pos.copy()
    long_mask = alloc_pos > 0
    short_mask = alloc_pos < 0

    # 多頭: score 越高權重越大 (score/8, 上限1.5)
    alloc_pos[long_mask] = (alloc_pos[long_mask] / 8.0).clip(upper=1.5)
    alloc_pos[short_mask] = -1

    # Dynamic Exposure
    alloc_pos = apply_dynamic_exposure(alloc_pos, benchmark_close)

    report = safe_finlab_sim(
        alloc_pos,
        name='ScoreWeight',
        upload=False,
        trail_stop=raw['trail_stop'],
        position_limit=1.0 / raw['max_concurrent'],
        touched_exit=False,
    )
    return extract_stats(report, 'score_weight')


def run_pullback_entry(pullback_pct=0.02):
    """回檔進場: 信號觸發後等股價回檔 N% 才進場"""
    raw = run_isaac_strategy(API_TOKEN, raw_mode=True)
    final_pos = raw['final_pos']
    close = raw['close']
    benchmark_close = raw['etf_close']

    close_aligned = close.reindex(final_pos.index).ffill()

    # 找出新進場點 (前一天 0, 今天 > 0)
    was_zero = final_pos.shift(1).fillna(0) == 0
    new_entry = (final_pos > 0) & was_zero

    # 進場日的收盤價
    entry_price = close_aligned.where(new_entry)
    entry_price = entry_price.ffill()  # 持續記錄進場價

    # 計算從進場價的回檔幅度
    pullback = (close_aligned - entry_price) / entry_price

    # 如果新進場且未回檔到目標 → 延遲進場 (設為0)
    # 只在新進場的前5天內檢查回檔
    hold_days = (final_pos > 0).astype(int)
    for i in range(1, len(hold_days)):
        mask = hold_days.iloc[i] > 0
        hold_days.iloc[i, mask] = hold_days.iloc[i-1][mask] + 1

    # 新進場且持倉<=3天且未回檔 → 暫不進場
    delay_mask = (hold_days <= 3) & (hold_days > 0) & (pullback > -pullback_pct) & new_entry.cummax()

    adjusted_pos = final_pos.copy()
    adjusted_pos[delay_mask & (adjusted_pos > 0)] = 0

    # Dynamic Exposure
    alloc_pos = adjusted_pos.copy()
    alloc_pos[alloc_pos > 0] = 1
    alloc_pos[alloc_pos < 0] = -1
    alloc_pos = apply_dynamic_exposure(alloc_pos, benchmark_close)

    report = safe_finlab_sim(
        alloc_pos,
        name='PullbackEntry',
        upload=False,
        trail_stop=raw['trail_stop'],
        position_limit=1.0 / raw['max_concurrent'],
        touched_exit=False,
    )
    return extract_stats(report, f'pullback_{int(pullback_pct*100)}pct')


def run_partial_profit(profit_threshold=0.07):
    """部分停利: 浮盈達 N% 時減倉一半"""
    raw = run_isaac_strategy(API_TOKEN, raw_mode=True)
    final_pos = raw['final_pos']
    close = raw['close']
    benchmark_close = raw['etf_close']

    close_aligned = close.reindex(final_pos.index).ffill()

    # 找新進場日的價格
    was_zero = final_pos.shift(1).fillna(0) == 0
    new_entry = (final_pos > 0) & was_zero

    # 持續追蹤每個持倉的進場價格
    adjusted_pos = final_pos.copy()

    # 逐列追蹤 entry_price (向量化困難, 用 column-wise)
    for col in final_pos.columns:
        pos_col = final_pos[col]
        close_col = close_aligned[col] if col in close_aligned.columns else None
        if close_col is None:
            continue

        entry_px = np.nan
        half_taken = False
        for i in range(len(pos_col)):
            if pos_col.iloc[i] > 0:
                if i == 0 or pos_col.iloc[i-1] <= 0:
                    # 新進場
                    entry_px = close_col.iloc[i]
                    half_taken = False
                elif not np.isnan(entry_px) and not half_taken:
                    # 檢查浮盈
                    pnl = (close_col.iloc[i] - entry_px) / entry_px
                    if pnl >= profit_threshold:
                        adjusted_pos.iloc[i, final_pos.columns.get_loc(col)] *= 0.5
                        half_taken = True
            else:
                entry_px = np.nan
                half_taken = False

    # Dynamic Exposure
    alloc_pos = adjusted_pos.copy()
    alloc_pos[alloc_pos > 0] = alloc_pos[alloc_pos > 0].clip(upper=1)
    alloc_pos[alloc_pos < 0] = -1
    alloc_pos = apply_dynamic_exposure(alloc_pos, benchmark_close)

    report = safe_finlab_sim(
        alloc_pos,
        name='PartialProfit',
        upload=False,
        trail_stop=raw['trail_stop'],
        position_limit=1.0 / raw['max_concurrent'],
        touched_exit=False,
    )
    return extract_stats(report, f'partial_profit_{int(profit_threshold*100)}pct')


def run_monthly_confirm():
    """多週期確認: 月線趨勢向上才進場"""
    raw = run_isaac_strategy(API_TOKEN, raw_mode=True)
    final_pos = raw['final_pos']
    close = raw['close']
    benchmark_close = raw['etf_close']

    close_aligned = close.reindex(final_pos.index).ffill()

    # 月線 (MA20) 方向: 最近5天 MA20 斜率 > 0 才進場
    ma20 = close_aligned.rolling(20, min_periods=10).mean()
    ma20_slope = ma20 - ma20.shift(5)
    monthly_up = ma20_slope > 0

    # 只保留月線向上的多頭部位
    adjusted_pos = final_pos.copy()
    adjusted_pos[(adjusted_pos > 0) & ~monthly_up] = 0

    # Dynamic Exposure
    alloc_pos = adjusted_pos.copy()
    alloc_pos[alloc_pos > 0] = 1
    alloc_pos[alloc_pos < 0] = -1
    alloc_pos = apply_dynamic_exposure(alloc_pos, benchmark_close)

    report = safe_finlab_sim(
        alloc_pos,
        name='MonthlyConfirm',
        upload=False,
        trail_stop=raw['trail_stop'],
        position_limit=1.0 / raw['max_concurrent'],
        touched_exit=False,
    )
    return extract_stats(report, 'monthly_confirm')


def run_signal_contribution():
    """信號貢獻度分析: 分別跑 AB-only, C-only, E-only, D-only"""
    results = []

    # 只跑 AB (關閉 C, E, D) — 需要改 isaac.py 或用 raw_mode 分析
    # 簡化方案: 用 report.get_trades() 加上 signal 標記來分析
    report = run_isaac_strategy(API_TOKEN)
    trades = report.get_trades()

    if len(trades) == 0:
        return {'signal_analysis': 'no trades'}

    stats = report.get_stats()

    # 依據 entry score 分段分析
    analysis = {}

    # Score 分段
    for lo, hi, label in [(1, 3, 'low(1-3)'), (4, 6, 'mid(4-6)'), (7, 10, 'high(7-10)'), (10, 99, 'elite(10+)')]:
        mask = (trades['return'].notna())  # 基礎 mask
        # 用 position 值近似 score (FinLab trades 的 position 欄位)
        if 'position' in trades.columns:
            mask = mask & (trades['position'] >= lo) & (trades['position'] <= hi)

        subset = trades[mask]
        if len(subset) > 0:
            analysis[label] = {
                'trades': len(subset),
                'avg_ret': f"{subset['return'].mean()*100:.2f}%",
                'med_ret': f"{subset['return'].median()*100:.2f}%",
                'win_rate': f"{(subset['return']>0).mean()*100:.1f}%",
                'total_pnl': f"{subset['return'].sum()*100:.1f}%",
            }

    # 多頭 vs 空頭
    if 'position' in trades.columns:
        long_trades = trades[trades['position'] > 0]
        short_trades = trades[trades['position'] < 0]
        analysis['long_all'] = {
            'trades': len(long_trades),
            'avg_ret': f"{long_trades['return'].mean()*100:.2f}%" if len(long_trades) > 0 else 'N/A',
            'win_rate': f"{(long_trades['return']>0).mean()*100:.1f}%" if len(long_trades) > 0 else 'N/A',
        }
        analysis['short_all'] = {
            'trades': len(short_trades),
            'avg_ret': f"{short_trades['return'].mean()*100:.2f}%" if len(short_trades) > 0 else 'N/A',
            'win_rate': f"{(short_trades['return']>0).mean()*100:.1f}%" if len(short_trades) > 0 else 'N/A',
        }

    # 按持倉天數分析
    if 'period' in trades.columns:
        for lo, hi, label in [(1, 5, '1-5d'), (6, 20, '6-20d'), (21, 60, '21-60d'), (61, 999, '60d+')]:
            subset = trades[(trades['period'] >= lo) & (trades['period'] <= hi)]
            if len(subset) > 0:
                analysis[f'period_{label}'] = {
                    'trades': len(subset),
                    'avg_ret': f"{subset['return'].mean()*100:.2f}%",
                    'win_rate': f"{(subset['return']>0).mean()*100:.1f}%",
                }

    return {
        'type': 'signal_analysis',
        'overall': {
            'cagr': f"{stats['cagr']*100:.2f}%",
            'sharpe': f"{stats['daily_sharpe']:.4f}",
            'trades': len(trades),
        },
        'by_score': {k: v for k, v in analysis.items() if not k.startswith('period_') and k not in ('long_all', 'short_all')},
        'long_vs_short': {k: v for k, v in analysis.items() if k in ('long_all', 'short_all')},
        'by_period': {k: v for k, v in analysis.items() if k.startswith('period_')},
    }


def apply_dynamic_exposure(alloc_pos, benchmark_close):
    """共用 Dynamic Exposure"""
    bench_ma60 = benchmark_close.rolling(60).mean()
    bench_ma120 = benchmark_close.rolling(120).mean()

    exposure = pd.Series(1.0, index=alloc_pos.index)
    bench_a = benchmark_close.reindex(alloc_pos.index).ffill()
    bma60_a = bench_ma60.reindex(alloc_pos.index).ffill()
    bma120_a = bench_ma120.reindex(alloc_pos.index).ffill()

    exposure[bench_a <= bma60_a] = 0.6
    exposure[bench_a <= bma120_a] = 0.3
    exposure[bench_a > bma60_a] = 1.0

    long_m = alloc_pos > 0
    short_m = alloc_pos < 0
    alloc_pos[long_m] = alloc_pos[long_m].mul(exposure, axis=0)[long_m]
    alloc_pos[short_m] = alloc_pos[short_m].mul(2.0 - exposure, axis=0)[short_m]

    return alloc_pos


def extract_stats(report, name):
    """從 report 提取關鍵指標"""
    stats = report.get_stats()
    trades = report.get_trades()
    return {
        'name': name,
        'cagr': stats.get('cagr', 0),
        'max_dd': stats.get('max_drawdown', 0),
        'sharpe': stats.get('daily_sharpe', 0),
        'sortino': stats.get('daily_sortino', 0),
        'calmar': stats.get('calmar', 0),
        'win_ratio': stats.get('win_ratio', 0),
        'total_return': stats.get('total_return', 0),
        'trades': len(trades),
    }


def print_comparison(results):
    """輸出比較表"""
    print("\n" + "=" * 120)
    print("  Isaac V3.5 進階優化 A/B 測試結果")
    print("=" * 120)
    header = f"  {'Name':<25} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>8} | {'Sortino':>8} | {'Calmar':>8} | {'WinRate':>8} | {'Trades':>7}"
    print(header)
    print("-" * 120)

    for r in results:
        if isinstance(r, dict) and 'cagr' in r:
            marker = " ★" if r['name'] == 'baseline' else ""
            print(f"  {r['name']:<25} | {r['cagr']*100:>+7.2f}% | {r['max_dd']*100:>7.2f}% | {r['sharpe']:>8.4f} | {r['sortino']:>8.4f} | {r['calmar']:>8.4f} | {r['win_ratio']*100:>7.2f}% | {r['trades']:>7}{marker}")

    print("=" * 120)

    # Rank by score
    scored = []
    for r in results:
        if isinstance(r, dict) and 'cagr' in r:
            s = (r['cagr'] / abs(r['max_dd'])) * 0.4 + r['sharpe'] * 0.4 + r['win_ratio'] * 0.2
            scored.append((r['name'], s))

    scored.sort(key=lambda x: x[1], reverse=True)
    print("\n  📊 Score 排名 (CAGR/DD*0.4 + Sharpe*0.4 + WinRate*0.2):")
    for i, (name, sc) in enumerate(scored):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f" {i+1}."
        print(f"    {medal} {name}: {sc:.4f}")


if __name__ == '__main__':
    results = []

    # 0. Baseline
    print("\n[0/7] Running baseline (V3.5 + min_score>=4)...")
    t0 = time.time()
    results.append(run_baseline())
    print(f"  Done in {time.time()-t0:.0f}s")

    # 1. 部分停利 (7%, 10%, 15%)
    for pct in [0.07, 0.10, 0.15]:
        print(f"\n[1] Running partial_profit ({int(pct*100)}%)...")
        t0 = time.time()
        results.append(run_partial_profit(profit_threshold=pct))
        print(f"  Done in {time.time()-t0:.0f}s")

    # 2. Score 加權
    print("\n[2] Running score_weight...")
    t0 = time.time()
    results.append(run_score_weight())
    print(f"  Done in {time.time()-t0:.0f}s")

    # 3. 回檔進場 (2%, 3%)
    for pct in [0.02, 0.03]:
        print(f"\n[3] Running pullback_entry ({int(pct*100)}%)...")
        t0 = time.time()
        results.append(run_pullback_entry(pullback_pct=pct))
        print(f"  Done in {time.time()-t0:.0f}s")

    # 4. 信號貢獻度分析
    print("\n[4] Running signal_contribution analysis...")
    t0 = time.time()
    sig_analysis = run_signal_contribution()
    print(f"  Done in {time.time()-t0:.0f}s")

    # 5. 月線確認
    print("\n[5] Running monthly_confirm...")
    t0 = time.time()
    results.append(run_monthly_confirm())
    print(f"  Done in {time.time()-t0:.0f}s")

    # Print comparison
    print_comparison(results)

    # Print signal analysis
    print("\n" + "=" * 80)
    print("  信號貢獻度分析")
    print("=" * 80)

    if 'by_score' in sig_analysis:
        print("\n  📊 依 Score 分段:")
        for k, v in sig_analysis['by_score'].items():
            print(f"    {k:>15}: trades={v['trades']:>4}, avg_ret={v['avg_ret']:>8}, win_rate={v['win_rate']:>6}, total_pnl={v.get('total_pnl','N/A'):>8}")

    if 'long_vs_short' in sig_analysis:
        print("\n  📊 多頭 vs 空頭:")
        for k, v in sig_analysis['long_vs_short'].items():
            print(f"    {k:>15}: trades={v['trades']:>4}, avg_ret={v['avg_ret']:>8}, win_rate={v['win_rate']:>6}")

    if 'by_period' in sig_analysis:
        print("\n  📊 依持倉天數:")
        for k, v in sig_analysis['by_period'].items():
            print(f"    {k:>15}: trades={v['trades']:>4}, avg_ret={v['avg_ret']:>8}, win_rate={v['win_rate']:>6}")

    # Save results
    output = {
        'comparison': results,
        'signal_analysis': sig_analysis,
    }
    with open(os.path.join(os.path.dirname(__file__), '..', 'advanced_test_result.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n✅ 結果已存入 advanced_test_result.json")
