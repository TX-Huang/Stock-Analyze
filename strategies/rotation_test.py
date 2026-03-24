"""Rotation Test: Keep-strong-drop-weak optimization"""
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

def apply_v37_alloc(fp, benchmark_close):
    """Apply V3.7 allocation: Score Weight + Dynamic Exposure (no short boost)"""
    alloc = fp.copy()
    alloc[alloc > 0] = (alloc[alloc > 0] / 8.0).clip(upper=1.5)

    bench_ma60 = benchmark_close.rolling(60).mean()
    bench_ma120 = benchmark_close.rolling(120).mean()
    exposure = pd.Series(1.0, index=alloc.index)
    ba = benchmark_close.reindex(alloc.index).ffill()
    b60 = bench_ma60.reindex(alloc.index).ffill()
    b120 = bench_ma120.reindex(alloc.index).ffill()
    exposure[ba <= b60] = 0.6
    exposure[ba <= b120] = 0.3
    exposure[ba > b60] = 1.0
    lm = alloc > 0
    alloc[lm] = alloc[lm].mul(exposure, axis=0)[lm]
    return alloc

# Load raw data
print("Loading raw data...")
t0 = time.time()
raw = run_isaac_strategy(api_token, raw_mode=True)
benchmark_close = raw['etf_close']
trail_stop = raw['trail_stop']
pos_limit = 1.0 / raw['max_concurrent']
MAX_LONG = 8
print(f"  Loaded in {time.time()-t0:.0f}s")

# Get the raw score matrix (before top-N selection) and the hedge factor
# We need to re-extract the "all candidates" score matrix
# raw['final_pos'] already has hedge_factor and top-N applied
# We need the pre-top-N scores to do rotation properly

# For rotation, we work with final_pos which has scores as values
# We'll rebuild rotation logic on top of that
fp = raw['final_pos']  # This has top-8 applied + hedge factor

# ======== 0. BASELINE ========
print('\n[0/4] Baseline (V3.7)...')
t0 = time.time()
report = run_isaac_strategy(api_token)
results.append(get_stats_dict('baseline_v37', report))
print(f'  Done {time.time()-t0:.0f}s')

# ======== A. Daily re-rank Top-8 ========
# The current system already does daily Top-8 selection via rank()
# But it uses signal persistence (ffill) - once a signal fires, score stays until exit
# "Daily re-rank" means: each day, only keep the 8 highest-scoring ACTIVE positions
# This is essentially what the current system does, but let's make it stricter:
# If a stock's score drops below the 8th highest available candidate, swap it out

print('[1/4] Variant A: Daily re-rank (force top-8 each day)...')
t0 = time.time()

# fp already has the position matrix with scores
# Current system: once entered, held until exit signal (score preserved via ffill)
# Variant A: each day, hard-select top-8 by current score value
# This means a stock with score=4 will be replaced by a new stock with score=6

fp_a = fp.copy()
for i in range(len(fp_a)):
    row = fp_a.iloc[i]
    active = row[row > 0].sort_values(ascending=False)
    if len(active) > MAX_LONG:
        # Keep only top-8, zero out the rest
        keep = active.index[:MAX_LONG]
        drop = active.index[MAX_LONG:]
        fp_a.iloc[i][drop] = 0

alloc_a = apply_v37_alloc(fp_a, benchmark_close)
r_a = safe_finlab_sim(alloc_a, name='daily_rerank', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('A_daily_rerank', r_a))
print(f'  Done {time.time()-t0:.0f}s')

# ======== B. Threshold replacement (new score > min_held + 2) ========
print('[2/4] Variant B: Threshold replacement (gap >= 2)...')
t0 = time.time()

fp_b = fp.copy()
for i in range(len(fp_b)):
    row = fp_b.iloc[i]
    held = row[row > 0].sort_values(ascending=False)

    if len(held) <= MAX_LONG:
        continue

    # Find the minimum score among current top-8
    top8 = held.iloc[:MAX_LONG]
    min_score_held = top8.iloc[-1]  # lowest score in top-8

    # Check candidates outside top-8
    candidates = held.iloc[MAX_LONG:]

    # Replace only if candidate score > min_held + 2
    for stock_id in candidates.index:
        cand_score = candidates[stock_id]
        if cand_score > min_score_held + 2:
            # Find the weakest held stock and swap
            weakest = top8.idxmin()
            fp_b.iloc[i][weakest] = 0
            # top8 is already reflected in fp_b, the candidate keeps its score
            # Update tracking
            top8 = top8.drop(weakest)
            min_score_held = top8.min() if len(top8) > 0 else 0
        else:
            break  # candidates are sorted, no more will qualify

alloc_b = apply_v37_alloc(fp_b, benchmark_close)
r_b = safe_finlab_sim(alloc_b, name='threshold_replace', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('B_threshold_gap2', r_b))
print(f'  Done {time.time()-t0:.0f}s')

# ======== C. Weekly re-rank (every Friday) ========
print('[3/4] Variant C: Weekly re-rank (Friday only)...')
t0 = time.time()

fp_c = fp.copy()
# Get day-of-week for each index date
dow = pd.to_datetime(fp_c.index).dayofweek  # Monday=0, Friday=4

# On non-Friday: keep yesterday's positions (don't allow new entries to displace)
# On Friday: full re-rank top-8
prev_held = set()
for i in range(len(fp_c)):
    row = fp_c.iloc[i]
    active = row[row > 0]

    if dow[i] == 4:  # Friday - full re-rank
        ranked = active.sort_values(ascending=False)
        if len(ranked) > MAX_LONG:
            drop = ranked.index[MAX_LONG:]
            fp_c.iloc[i][drop] = 0
        prev_held = set(ranked.index[:MAX_LONG])
    else:
        # Non-Friday: prioritize keeping existing positions
        # Only drop if clearly exited (score = 0, handled by signal logic)
        if len(active) > MAX_LONG:
            # Keep previously held stocks, fill remaining with best new
            keep_existing = [s for s in prev_held if s in active.index]
            new_candidates = [s for s in active.sort_values(ascending=False).index if s not in prev_held]

            slots_left = MAX_LONG - len(keep_existing)
            keep_new = new_candidates[:max(slots_left, 0)]
            keep_all = set(keep_existing + keep_new)

            for stock_id in active.index:
                if stock_id not in keep_all:
                    fp_c.iloc[i][stock_id] = 0

            prev_held = keep_all

alloc_c = apply_v37_alloc(fp_c, benchmark_close)
r_c = safe_finlab_sim(alloc_c, name='weekly_rerank', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('C_weekly_rerank', r_c))
print(f'  Done {time.time()-t0:.0f}s')

# Also test variant B with gap=1 (more aggressive rotation)
print('[4/4] Variant B2: Threshold replacement (gap >= 1)...')
t0 = time.time()

fp_b2 = fp.copy()
for i in range(len(fp_b2)):
    row = fp_b2.iloc[i]
    held = row[row > 0].sort_values(ascending=False)

    if len(held) <= MAX_LONG:
        continue

    top8 = held.iloc[:MAX_LONG]
    min_score_held = top8.iloc[-1]
    candidates = held.iloc[MAX_LONG:]

    for stock_id in candidates.index:
        cand_score = candidates[stock_id]
        if cand_score > min_score_held + 1:
            weakest = top8.idxmin()
            fp_b2.iloc[i][weakest] = 0
            top8 = top8.drop(weakest)
            min_score_held = top8.min() if len(top8) > 0 else 0
        else:
            break

alloc_b2 = apply_v37_alloc(fp_b2, benchmark_close)
r_b2 = safe_finlab_sim(alloc_b2, name='threshold_gap1', upload=False,
    trail_stop=trail_stop, position_limit=pos_limit, touched_exit=False)
results.append(get_stats_dict('B2_threshold_gap1', r_b2))
print(f'  Done {time.time()-t0:.0f}s')

# ======== Print Results ========
print()
print('=' * 130)
print('  Rotation Optimization Results (Keep Strong, Drop Weak)')
print('=' * 130)
header = f"  {'Name':<25} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>8} | {'Sortino':>8} | {'Calmar':>8} | {'WinRate':>8} | {'Trades':>6}"
print(header)
print('-' * 130)
for r in results:
    tag = ' <<' if r['name'] == 'baseline_v37' else ''
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

# Trade count comparison (rotation = higher turnover)
print('\n  Turnover comparison:')
for r in results:
    print(f'    {r["name"]:<25}: {r["trades"]} trades')
