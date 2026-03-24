"""Round 3 Optimization Test: Signal D fix + Min Hold + Trail Tighten"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import toml, numpy as np, pandas as pd

secrets = toml.load(os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml'))
api_token = secrets.get('FINLAB_KEY', '') or secrets.get('FINLAB_API_KEY', '')
from strategies.isaac import run_isaac_strategy
from data.provider import safe_finlab_sim

results = []

def get_stats_dict(name, report):
    stats = report.get_stats()
    trades = report.get_trades()
    return {
        'name': name,
        'cagr': stats['cagr'],
        'max_dd': stats['max_drawdown'],
        'sharpe': stats['daily_sharpe'],
        'sortino': stats['daily_sortino'],
        'calmar': stats['calmar'],
        'win_ratio': stats['win_ratio'],
        'trades': len(trades),
    }

def apply_dynamic_exposure(alloc_pos, benchmark_close, short_boost=True):
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
    if short_boost:
        alloc_pos[short_m] = alloc_pos[short_m].mul(2.0 - exposure, axis=0)[short_m]
    return alloc_pos

def score_weight_pos(alloc_pos):
    lm = alloc_pos > 0
    alloc_pos[lm] = (alloc_pos[lm] / 8.0).clip(upper=1.5)
    return alloc_pos

# Cache raw data
print("Loading raw data (cached for reuse)...")
t_start = time.time()
raw_full = run_isaac_strategy(api_token, raw_mode=True)
raw_no_d = run_isaac_strategy(api_token, raw_mode=True, params={'disable_d': True})
print(f"  Raw data loaded in {time.time()-t_start:.0f}s")

benchmark_close = raw_full['etf_close']
trail_stop = raw_full['trail_stop']
pos_limit = 1.0 / raw_full['max_concurrent']

# ======== 0. BASELINE ========
print('\n[0/7] Baseline (V3.6 Score Weight)...')
t0 = time.time()
report = run_isaac_strategy(api_token)
results.append(get_stats_dict('baseline', report))
print(f'  Done {time.time()-t0:.0f}s')

# ======== Signal D Optimization ========

# A. Bear-market-only shorts (ETF < MA120)
print('[1/7] Short_A: bear_market_only...')
t0 = time.time()
fp_a = raw_full['final_pos'].copy()
bench_ma120 = benchmark_close.rolling(120).mean()
bench_a = benchmark_close.reindex(fp_a.index).ffill()
bma120_a = bench_ma120.reindex(fp_a.index).ffill()
bear_mask = bench_a < bma120_a

short_m = fp_a < 0
# Remove shorts in non-bear markets
for col in fp_a.columns:
    mask = (fp_a[col] < 0) & (~bear_mask)
    fp_a.loc[mask, col] = 0

alloc_a = fp_a.copy()
alloc_a = score_weight_pos(alloc_a)
alloc_a[alloc_a < 0] = -1
alloc_a = apply_dynamic_exposure(alloc_a, benchmark_close)
r_a = safe_finlab_sim(alloc_a, name='short_A_bear_only', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('short_A_bear_only', r_a))
print(f'  Done {time.time()-t0:.0f}s')

# B. No shorts, use short signal count to reduce long exposure
print('[2/7] Short_B: short_as_hedge...')
t0 = time.time()
fp_b = raw_no_d['final_pos'].copy()
short_signal_count = (raw_full['final_pos'] < 0).sum(axis=1)
hedge_factor = pd.Series(1.0, index=fp_b.index)
sc_aligned = short_signal_count.reindex(fp_b.index, fill_value=0)
hedge_factor[sc_aligned >= 1] = 0.7
hedge_factor[sc_aligned >= 2] = 0.4

long_b = fp_b > 0
fp_b[long_b] = fp_b[long_b].mul(hedge_factor, axis=0)[long_b]
alloc_b = fp_b.copy()
alloc_b = score_weight_pos(alloc_b)
alloc_b[alloc_b < 0] = 0
alloc_b = apply_dynamic_exposure(alloc_b, benchmark_close, short_boost=False)
r_b = safe_finlab_sim(alloc_b, name='short_B_hedge', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('short_B_hedge', r_b))
print(f'  Done {time.time()-t0:.0f}s')

# C. Disable shorts entirely, clean dynamic exposure
print('[3/7] Short_C: no_shorts_clean...')
t0 = time.time()
fp_c = raw_no_d['final_pos'].copy()
alloc_c = fp_c.copy()
alloc_c = score_weight_pos(alloc_c)
alloc_c[alloc_c < 0] = 0
alloc_c = apply_dynamic_exposure(alloc_c, benchmark_close, short_boost=False)
r_c = safe_finlab_sim(alloc_c, name='short_C_no_shorts', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('short_C_no_shorts', r_c))
print(f'  Done {time.time()-t0:.0f}s')

# ======== Direction 2: Min holding period ========
print('[4/7] Min hold 5 days...')
t0 = time.time()
fp_h5 = raw_full['final_pos'].copy()
for col in fp_h5.columns:
    vals = fp_h5[col].values.copy()
    i = 0
    while i < len(vals):
        if vals[i] != 0:
            entry_val = vals[i]
            for j in range(i+1, min(i+5, len(vals))):
                if vals[j] == 0:
                    vals[j] = entry_val
            k = i+1
            while k < len(vals) and vals[k] != 0:
                k += 1
            i = k
        else:
            i += 1
    fp_h5[col] = vals

alloc_h5 = fp_h5.copy()
alloc_h5 = score_weight_pos(alloc_h5)
alloc_h5[alloc_h5 < 0] = -1
alloc_h5 = apply_dynamic_exposure(alloc_h5, benchmark_close)
r_h5 = safe_finlab_sim(alloc_h5, name='min_hold_5d', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('min_hold_5d', r_h5))
print(f'  Done {time.time()-t0:.0f}s')

print('[5/7] Min hold 10 days...')
t0 = time.time()
fp_h10 = raw_full['final_pos'].copy()
for col in fp_h10.columns:
    vals = fp_h10[col].values.copy()
    i = 0
    while i < len(vals):
        if vals[i] != 0:
            entry_val = vals[i]
            for j in range(i+1, min(i+10, len(vals))):
                if vals[j] == 0:
                    vals[j] = entry_val
            k = i+1
            while k < len(vals) and vals[k] != 0:
                k += 1
            i = k
        else:
            i += 1
    fp_h10[col] = vals

alloc_h10 = fp_h10.copy()
alloc_h10 = score_weight_pos(alloc_h10)
alloc_h10[alloc_h10 < 0] = -1
alloc_h10 = apply_dynamic_exposure(alloc_h10, benchmark_close)
r_h10 = safe_finlab_sim(alloc_h10, name='min_hold_10d', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('min_hold_10d', r_h10))
print(f'  Done {time.time()-t0:.0f}s')

# ======== Direction 3: Tighter trail_stop ========
print('[6/7] trail_stop 0.14...')
t0 = time.time()
r_ts14 = run_isaac_strategy(api_token, params={'trail_stop': 0.14})
results.append(get_stats_dict('trail_stop_0.14', r_ts14))
print(f'  Done {time.time()-t0:.0f}s')

# ======== Print Results ========
print()
print('=' * 130)
print('  Round 3 Optimization Results')
print('=' * 130)
header = f"  {'Name':<25} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>8} | {'Sortino':>8} | {'Calmar':>8} | {'WinRate':>8} | {'Trades':>6}"
print(header)
print('-' * 130)
for r in results:
    tag = ' <<' if r['name'] == 'baseline' else ''
    line = f"  {r['name']:<25} | {r['cagr']*100:>+7.2f}% | {r['max_dd']*100:>7.2f}% | {r['sharpe']:>8.4f} | {r['sortino']:>8.4f} | {r['calmar']:>8.4f} | {r['win_ratio']*100:>7.2f}% | {r['trades']:>6}{tag}"
    print(line)
print('=' * 130)

# Score ranking
scored = []
for r in results:
    s = (r['cagr'] / abs(r['max_dd'])) * 0.4 + r['sharpe'] * 0.4 + r['win_ratio'] * 0.2
    scored.append((r['name'], s, r['cagr'], r['max_dd'], r['sharpe']))
scored.sort(key=lambda x: x[1], reverse=True)
print('\n  Score Ranking:')
for i, (n, s, c, d, sh) in enumerate(scored):
    rank = i + 1
    print(f'    #{rank}: {n:<25} Score={s:.4f}  (CAGR={c*100:+.2f}%, DD={d*100:.2f}%, Sharpe={sh:.4f})')
