"""V3.7 Overfitting Validation: WFO + Monte Carlo"""
import sys, os, warnings, time, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import toml, numpy as np, pandas as pd

secrets = toml.load(os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml'))
api_token = secrets.get('FINLAB_KEY', '') or secrets.get('FINLAB_API_KEY', '')
from strategies.isaac import run_isaac_strategy
from data.provider import safe_finlab_sim

# ============================================================
# Part 1: Walk-Forward Optimization (WFO)
# 3-year train / 6-month test / 6-month roll
# ============================================================
print("=" * 80)
print("  Part 1: Walk-Forward Optimization (WFO)")
print("=" * 80)

def generate_wfo_windows(start_year=2014, end_year=2026, train_years=3, test_months=6, roll_months=6):
    windows = []
    current = pd.Timestamp(f'{start_year}-01-01')
    end = pd.Timestamp(f'{end_year}-07-01')
    while current + pd.DateOffset(years=train_years, months=test_months) <= end:
        train_start = current
        train_end = current + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)
        windows.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        current += pd.DateOffset(months=roll_months)
    return windows

def run_window(raw, window, benchmark_close):
    """Run a single WFO window using pre-computed raw data"""
    fp = raw['final_pos']

    # IS (In-Sample / Train)
    is_pos = fp.loc[window['train_start']:window['train_end']].copy()
    if len(is_pos) < 20:
        return None
    is_alloc = is_pos.copy()
    is_alloc[is_alloc > 0] = (is_alloc[is_alloc > 0] / 8.0).clip(upper=1.5)

    # Apply dynamic exposure
    bench_ma60 = benchmark_close.rolling(60).mean()
    bench_ma120 = benchmark_close.rolling(120).mean()
    exposure_is = pd.Series(1.0, index=is_alloc.index)
    ba = benchmark_close.reindex(is_alloc.index).ffill()
    b60 = bench_ma60.reindex(is_alloc.index).ffill()
    b120 = bench_ma120.reindex(is_alloc.index).ffill()
    exposure_is[ba <= b60] = 0.6
    exposure_is[ba <= b120] = 0.3
    exposure_is[ba > b60] = 1.0
    lm = is_alloc > 0
    is_alloc[lm] = is_alloc[lm].mul(exposure_is, axis=0)[lm]

    try:
        is_report = safe_finlab_sim(is_alloc, name='wfo_is', upload=False,
            trail_stop=raw['trail_stop'], position_limit=1.0/raw['max_concurrent'], touched_exit=False)
        is_stats = is_report.get_stats()
    except Exception:
        return None

    # OOS (Out-of-Sample / Test)
    oos_pos = fp.loc[window['test_start']:window['test_end']].copy()
    if len(oos_pos) < 20:
        return None
    oos_alloc = oos_pos.copy()
    oos_alloc[oos_alloc > 0] = (oos_alloc[oos_alloc > 0] / 8.0).clip(upper=1.5)

    exposure_oos = pd.Series(1.0, index=oos_alloc.index)
    ba2 = benchmark_close.reindex(oos_alloc.index).ffill()
    b602 = bench_ma60.reindex(oos_alloc.index).ffill()
    b1202 = bench_ma120.reindex(oos_alloc.index).ffill()
    exposure_oos[ba2 <= b602] = 0.6
    exposure_oos[ba2 <= b1202] = 0.3
    exposure_oos[ba2 > b602] = 1.0
    lm2 = oos_alloc > 0
    oos_alloc[lm2] = oos_alloc[lm2].mul(exposure_oos, axis=0)[lm2]

    try:
        oos_report = safe_finlab_sim(oos_alloc, name='wfo_oos', upload=False,
            trail_stop=raw['trail_stop'], position_limit=1.0/raw['max_concurrent'], touched_exit=False)
        oos_stats = oos_report.get_stats()
    except Exception:
        return None

    return {
        'window': f"{window['train_start'][:7]}~{window['test_end'][:7]}",
        'is_cagr': is_stats['cagr'],
        'is_sharpe': is_stats['daily_sharpe'],
        'oos_cagr': oos_stats['cagr'],
        'oos_sharpe': oos_stats['daily_sharpe'],
        'oos_maxdd': oos_stats['max_drawdown'],
    }

# Load raw data once
print("\nLoading raw data...")
t0 = time.time()
raw = run_isaac_strategy(api_token, raw_mode=True)
benchmark_close = raw['etf_close']
print(f"  Loaded in {time.time()-t0:.0f}s")

# Generate windows and run
windows = generate_wfo_windows()
print(f"\n  Running {len(windows)} WFO windows...")

wfo_results = []
for i, w in enumerate(windows):
    t1 = time.time()
    r = run_window(raw, w, benchmark_close)
    if r:
        wfo_results.append(r)
        print(f"  [{i+1}/{len(windows)}] {r['window']}: IS_CAGR={r['is_cagr']*100:+.1f}% OOS_CAGR={r['oos_cagr']*100:+.1f}% ({time.time()-t1:.0f}s)")
    else:
        print(f"  [{i+1}/{len(windows)}] {w['train_start'][:7]}~{w['test_end'][:7]}: SKIPPED")

# WFO Summary
if wfo_results:
    is_cagrs = [r['is_cagr'] for r in wfo_results]
    oos_cagrs = [r['oos_cagr'] for r in wfo_results]
    oos_positive = sum(1 for c in oos_cagrs if c > 0)
    overfitting_index = sum(1 for ic, oc in zip(is_cagrs, oos_cagrs) if oc < 0 and ic > 0) / max(len(wfo_results), 1)

    print(f"\n  WFO Summary:")
    print(f"    Windows: {len(wfo_results)}")
    print(f"    IS avg CAGR:  {np.mean(is_cagrs)*100:+.2f}%")
    print(f"    OOS avg CAGR: {np.mean(oos_cagrs)*100:+.2f}%")
    print(f"    OOS positive: {oos_positive}/{len(oos_cagrs)} ({oos_positive/len(oos_cagrs)*100:.0f}%)")
    print(f"    Overfitting Index: {overfitting_index:.2f} (0=good, 1=bad)")
    print(f"    IS/OOS ratio: {np.mean(is_cagrs)/max(np.mean(oos_cagrs),0.001):.2f} (closer to 1 = less overfit)")

# ============================================================
# Part 2: Monte Carlo Permutation Test (200 iterations for speed)
# ============================================================
print("\n" + "=" * 80)
print("  Part 2: Monte Carlo Permutation Test (200 iterations)")
print("=" * 80)

# Get actual strategy result
print("\nRunning actual strategy...")
t0 = time.time()
report_actual = run_isaac_strategy(api_token)
actual_stats = report_actual.get_stats()
actual_cagr = actual_stats['cagr']
actual_sharpe = actual_stats['daily_sharpe']
print(f"  Actual CAGR={actual_cagr*100:+.2f}%, Sharpe={actual_sharpe:.4f} ({time.time()-t0:.0f}s)")

# Shuffle positions and re-run
N_ITER = 200
random_cagrs = []
random_sharpes = []

print(f"\nRunning {N_ITER} random permutations...")
fp_actual = raw['final_pos'].copy()

# Apply score weight + dynamic exposure once for the position matrix
alloc_actual = fp_actual.copy()
alloc_actual[alloc_actual > 0] = (alloc_actual[alloc_actual > 0] / 8.0).clip(upper=1.5)
bench_ma60 = benchmark_close.rolling(60).mean()
bench_ma120 = benchmark_close.rolling(120).mean()
exposure = pd.Series(1.0, index=alloc_actual.index)
ba = benchmark_close.reindex(alloc_actual.index).ffill()
b60 = bench_ma60.reindex(alloc_actual.index).ffill()
b120 = bench_ma120.reindex(alloc_actual.index).ffill()
exposure[ba <= b60] = 0.6
exposure[ba <= b120] = 0.3
exposure[ba > b60] = 1.0
lm = alloc_actual > 0
alloc_actual[lm] = alloc_actual[lm].mul(exposure, axis=0)[lm]

shuffle_methods = ['time_shift', 'row_shuffle', 'block_shuffle']

for i in range(N_ITER):
    method = shuffle_methods[i % 3]
    shuffled = alloc_actual.copy()

    if method == 'time_shift':
        shift = np.random.randint(20, len(shuffled) - 20)
        shuffled = shuffled.shift(shift).fillna(0)
    elif method == 'row_shuffle':
        idx = np.random.permutation(len(shuffled))
        shuffled = pd.DataFrame(shuffled.values[idx], index=shuffled.index, columns=shuffled.columns)
    elif method == 'block_shuffle':
        block_size = np.random.randint(20, 60)
        n_blocks = len(shuffled) // block_size
        if n_blocks > 1:
            blocks = [shuffled.iloc[j*block_size:(j+1)*block_size] for j in range(n_blocks)]
            np.random.shuffle(blocks)
            shuffled = pd.concat(blocks).reset_index(drop=True)
            shuffled.index = alloc_actual.index[:len(shuffled)]

    try:
        r = safe_finlab_sim(shuffled, name=f'mc_{i}', upload=False,
            trail_stop=raw['trail_stop'], position_limit=1.0/raw['max_concurrent'], touched_exit=False)
        s = r.get_stats()
        random_cagrs.append(s['cagr'])
        random_sharpes.append(s['daily_sharpe'])
    except Exception:
        pass

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_ITER} done...")

# Monte Carlo Results
random_cagrs = np.array(random_cagrs)
random_sharpes = np.array(random_sharpes)

p_cagr = (random_cagrs >= actual_cagr).sum() / len(random_cagrs)
p_sharpe = (random_sharpes >= actual_sharpe).sum() / len(random_sharpes)
z_cagr = (actual_cagr - random_cagrs.mean()) / max(random_cagrs.std(), 0.001)

print(f"\n  Monte Carlo Results:")
print(f"    Actual CAGR:   {actual_cagr*100:+.2f}%")
print(f"    Random avg:    {random_cagrs.mean()*100:+.2f}%")
print(f"    Random std:    {random_cagrs.std()*100:.2f}%")
print(f"    p-value (CAGR):   {p_cagr:.4f}")
print(f"    p-value (Sharpe): {p_sharpe:.4f}")
print(f"    z-score:          {z_cagr:.2f}")

if p_cagr < 0.01:
    verdict = "STRONG: p<0.01, strategy is statistically significant"
elif p_cagr < 0.05:
    verdict = "PASS: p<0.05, strategy is statistically significant"
else:
    verdict = "FAIL: p>=0.05, cannot reject null hypothesis"
print(f"    Verdict: {verdict}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 80)
print("  V3.7 Validation Summary")
print("=" * 80)
print(f"  CAGR:              {actual_cagr*100:+.2f}%")
print(f"  MaxDD:             {actual_stats['max_drawdown']*100:.2f}%")
print(f"  Sharpe:            {actual_sharpe:.4f}")
if wfo_results:
    print(f"  WFO OOS avg CAGR: {np.mean(oos_cagrs)*100:+.2f}%")
    print(f"  WFO Overfit Index: {overfitting_index:.2f}")
    print(f"  WFO OOS positive:  {oos_positive}/{len(oos_cagrs)}")
print(f"  MC p-value (CAGR): {p_cagr:.4f}")
print(f"  MC p-value (Sharpe): {p_sharpe:.4f}")
print(f"  MC z-score:        {z_cagr:.2f}")
print(f"  Verdict:           {verdict}")
print("=" * 80)

# Save results
result = {
    'version': 'V3.7',
    'cagr': actual_cagr,
    'max_dd': actual_stats['max_drawdown'],
    'sharpe': actual_sharpe,
    'wfo_oos_avg_cagr': float(np.mean(oos_cagrs)) if wfo_results else None,
    'wfo_overfit_index': overfitting_index if wfo_results else None,
    'wfo_oos_positive_ratio': oos_positive / len(oos_cagrs) if wfo_results else None,
    'mc_p_cagr': float(p_cagr),
    'mc_p_sharpe': float(p_sharpe),
    'mc_z_score': float(z_cagr),
    'verdict': verdict,
}
with open('v37_validation_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nResults saved to v37_validation_result.json")
