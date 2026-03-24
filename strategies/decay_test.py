"""Score Decay Test: Make old positions naturally lose priority to new high-score entries"""
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

def apply_v37_pipeline(fp, df_d, benchmark_close, max_concurrent):
    """Apply full V3.7 pipeline: hedge + score weight + dynamic exposure"""
    # Signal D hedge
    short_signal_count = (df_d < 0).sum(axis=1)
    hedge_factor = pd.Series(1.0, index=fp.index)
    hedge_factor[short_signal_count >= 1] = 0.7
    hedge_factor[short_signal_count >= 2] = 0.4
    lm = fp > 0
    fp[lm] = fp[lm].mul(hedge_factor, axis=0)[lm]

    # Top-8
    long_rank = fp.rank(axis=1, method='first', ascending=False)
    long_top = (long_rank <= 8) & (fp > 0)
    fp = fp.where(long_top, 0)

    # Score Weight
    alloc = fp.copy()
    alloc[alloc > 0] = (alloc[alloc > 0] / 8.0).clip(upper=1.5)

    # Dynamic Exposure
    bench_ma60 = benchmark_close.rolling(60).mean()
    bench_ma120 = benchmark_close.rolling(120).mean()
    exposure = pd.Series(1.0, index=alloc.index)
    ba = benchmark_close.reindex(alloc.index).ffill()
    b60 = bench_ma60.reindex(alloc.index).ffill()
    b120 = bench_ma120.reindex(alloc.index).ffill()
    exposure[ba <= b60] = 0.6
    exposure[ba <= b120] = 0.3
    exposure[ba > b60] = 1.0
    lm2 = alloc > 0
    alloc[lm2] = alloc[lm2].mul(exposure, axis=0)[lm2]
    return alloc

# We need to get the raw track data BEFORE top-N and BEFORE hedge
# To do this, we need to access the internal track dataframes
# Let's modify the approach: get raw_mode data which gives us final_pos after top-N
# Instead, we'll directly modify the track ffill behavior

print("Loading strategy internals...")
t0 = time.time()

# Run full strategy for baseline
report_baseline = run_isaac_strategy(api_token)
results.append(get_stats_dict('baseline_v37', report_baseline))
print(f"  Baseline done in {time.time()-t0:.0f}s")

# Now we need access to the track DataFrames BEFORE merge
# We'll re-implement the track merge with decay applied
# First get raw components by importing and running the signal generation part

# Actually, the cleanest way is to get raw_mode output which gives us
# the merged long_pos + short_pos BEFORE allocation
# But we need the individual tracks to apply decay properly

# Alternative approach: apply decay to the final_pos directly
# Since final_pos values are scores (via ffill), we can detect "held" days
# and reduce score over time

raw = run_isaac_strategy(api_token, raw_mode=True)
raw_no_d = run_isaac_strategy(api_token, raw_mode=True, params={'disable_d': True})
benchmark_close = raw['etf_close']
trail_stop = raw['trail_stop']
pos_limit = 1.0 / raw['max_concurrent']

# Get df_d for hedge calculation
raw_with_d = run_isaac_strategy(api_token, raw_mode=True)
# We need df_d - run with disable_d=False to get it
# Actually raw_mode returns final_pos which already has hedge applied
# Let's get the position data before hedge by using disable_d versions

# Simpler approach: get the no-hedge version and apply decay, then apply hedge
# raw_no_d gives us positions without Signal D influence at all
# raw (with D) gives us positions with D merged in

# For decay testing, work with the raw final_pos from the full pipeline
# The final_pos has scores as values, with ffill keeping them constant
# We'll apply decay to simulate score reduction over holding time

def apply_score_decay(final_pos, decay_rate, min_score=1.0):
    """Apply exponential decay to held positions' scores.

    Each day a position is held, its score is multiplied by (1 - decay_rate).
    When score drops below min_score, it becomes vulnerable to replacement.

    Args:
        final_pos: Position matrix with scores as values
        decay_rate: Daily decay rate (e.g., 0.02 = 2% per day)
        min_score: Floor value for decayed score
    """
    result = final_pos.copy()

    for col in result.columns:
        vals = result[col].values.copy().astype(float)
        holding_days = 0
        entry_score = 0

        for i in range(len(vals)):
            if vals[i] > 0:
                if holding_days == 0 or (i > 0 and vals[i-1] <= 0):
                    # New entry
                    entry_score = vals[i]
                    holding_days = 0

                holding_days += 1
                # Apply decay: score * (1 - rate)^days
                decayed = entry_score * ((1 - decay_rate) ** holding_days)
                vals[i] = max(decayed, min_score)
            else:
                holding_days = 0
                entry_score = 0

        result[col] = vals

    return result

# Get the position matrix BEFORE hedge and top-N from the full run
# raw['final_pos'] already has hedge + top-N applied
# We need the pre-processed version

# Since we can't easily get pre-top-N data from raw_mode,
# let's use raw_no_d (no signal D at all) as the base long positions
# Then apply decay -> top-N -> hedge -> allocation

# Actually, raw['final_pos'] is AFTER top-8 and AFTER hedge
# The hedge reduces scores via multiplication, not the same as entry score
# Let's just work with what we have: apply decay to the final merged position matrix

# The key insight: in the current code, top-8 selection happens AFTER merge
# If we apply decay to the merged scores BEFORE top-8, new high-score entries
# will naturally replace old decayed entries

# To do this properly, we need to get the merged score matrix BEFORE top-8
# Let's hack it: use raw_no_d's final_pos which is the long_pos before hedge

# Actually raw_no_d['final_pos'] = long_pos (with top-8 already applied)
# We can't get pre-top-8 from raw_mode...

# Best approach: modify the test to apply decay on the SCORE values
# before the position matrix is built (i.e., during ffill)
# This requires rebuilding the tracks. Let's simulate it:

# Get the "entry signals" by detecting where score changes (new entry)
# Between entries, apply decay

print("\nApplying decay variants...")

# Use raw['final_pos'] which has hedge + top-8 applied
# Apply decay AFTER: this will cause decayed positions to have lower allocation
# via score weight (score/8), effectively reducing position size over time
# This won't cause stock replacement (that's top-8 which already happened)
# BUT it will affect the Score Weight allocation

# For true rotation, we need the PRE-top-8 merged scores
# Let's reconstruct: run with a very large MAX_CONCURRENT to get ALL candidates

# Workaround: use the fact that raw_mode returns final_pos after top-8
# We can get ALL positions by temporarily increasing MAX_CONCURRENT
# Not possible via params...

# CLEANEST APPROACH: modify the decay at the Track level
# Rebuild tracks with decay, then merge and top-8

# Since this requires modifying isaac.py internals, let's add a 'score_decay' param

print("\n  Building decay variants by modifying track ffill behavior...")

# We'll simulate by working with the position matrix we have
# Even though top-8 is already applied, the decay will change the RANKING
# of positions on days where there are >8 candidates
# Since we can't change that, let's test a different approach:
# HALF-LIFE DECAY on the alloc weights (not the selection, but the sizing)

def apply_weight_decay(alloc_pos, half_life_days):
    """Reduce position weight based on holding duration.
    Positions that have been held longer get smaller allocation.
    This encourages natural rotation via rebalancing."""
    result = alloc_pos.copy()
    decay = np.log(2) / half_life_days

    for col in result.columns:
        vals = result[col].values.copy().astype(float)
        holding_days = 0

        for i in range(len(vals)):
            if vals[i] > 0:
                holding_days += 1
                weight = np.exp(-decay * holding_days)
                vals[i] = vals[i] * max(weight, 0.3)  # Floor at 30% of original weight
            else:
                holding_days = 0

        result[col] = vals

    return result

# Test different decay rates on score before top-8
# Since we can't modify top-8 selection from outside,
# let's add score_decay as a param to isaac.py

# For now, test weight decay on the final allocation
fp_base = raw['final_pos'].copy()

# Rebuild the full pipeline with decay applied at different stages

# Approach: apply decay to the final_pos scores, then re-do top-8 + alloc
# This IS valid because it changes which stocks survive top-8 each day

for decay_name, decay_rate in [
    ('decay_1pct', 0.01),   # 1% per day, half-life ~70 days
    ('decay_2pct', 0.02),   # 2% per day, half-life ~35 days
    ('decay_5pct', 0.05),   # 5% per day, half-life ~14 days
    ('decay_10pct', 0.10),  # 10% per day, half-life ~7 days
    ('decay_15pct', 0.15),  # 15% per day, half-life ~4.5 days
]:
    print(f"\n  Testing {decay_name}...")
    t1 = time.time()

    # We need to get ALL candidate positions (not just top-8)
    # Use raw_no_d which has long positions only (no hedge yet)
    # But it's still top-8 filtered...

    # KEY INSIGHT: The raw_mode already applies top-8 in isaac.py
    # To test decay properly, we need positions BEFORE top-8
    # Let's use a workaround: get raw with very high min_score=0 to get more candidates
    # Actually that won't help either.

    # FINAL APPROACH: Apply decay to final_pos values,
    # then re-apply top-8 selection. Since final_pos already has top-8,
    # the only effect is when decayed scores drop below new entries that
    # would have been #9, #10 etc (which are 0 in our data).
    # This means decay on final_pos is USELESS for rotation.

    # WE MUST get pre-top-8 data. Let me check if we can pass max_concurrent via params.
    break  # Skip this approach, need to modify isaac.py

# Better approach: add score_decay param to isaac.py and test from there
print("\n  Need to add score_decay support to isaac.py...")
print("  Adding param and re-testing...")

# Read and modify isaac.py to support score_decay
import importlib

# Add score_decay param support
isaac_path = os.path.join(os.path.dirname(__file__), 'isaac.py')
with open(isaac_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Check if score_decay already added
if 'score_decay' not in code:
    # Add param parsing
    code = code.replace(
        "_disable_d         = bool(p.get('disable_d',          False))",
        "_disable_d         = bool(p.get('disable_d',          False))\n"
        "    _score_decay        = float(p.get('score_decay',        0.0))    # Daily score decay rate"
    )

    # Add decay to each track's ffill
    # After df_ab ffill, apply decay
    old_ab = "    df_ab = df_ab.ffill().fillna(0)"
    new_ab = """    df_ab = df_ab.ffill().fillna(0)

    # [Experiment] Score Decay: reduce held scores over time
    if _score_decay > 0:
        for col in df_ab.columns:
            vals = df_ab[col].values.copy()
            hold_days = 0
            entry_val = 0
            for ii in range(len(vals)):
                if vals[ii] > 0:
                    if hold_days == 0:
                        entry_val = vals[ii]
                    hold_days += 1
                    vals[ii] = max(entry_val * ((1 - _score_decay) ** hold_days), 1.0)
                else:
                    hold_days = 0
                    entry_val = 0
            df_ab[col] = vals"""
    code = code.replace(old_ab, new_ab)

    # Same for df_c
    old_c = "    df_c = df_c.ffill().fillna(0)"
    new_c = """    df_c = df_c.ffill().fillna(0)
    if _score_decay > 0:
        for col in df_c.columns:
            vals = df_c[col].values.copy()
            hold_days = 0
            entry_val = 0
            for ii in range(len(vals)):
                if vals[ii] > 0:
                    if hold_days == 0:
                        entry_val = vals[ii]
                    hold_days += 1
                    vals[ii] = max(entry_val * ((1 - _score_decay) ** hold_days), 1.0)
                else:
                    hold_days = 0
                    entry_val = 0
            df_c[col] = vals"""
    code = code.replace(old_c, new_c)

    # Same for df_e
    old_e = "        df_e = df_e.ffill().fillna(0)"
    new_e = """        df_e = df_e.ffill().fillna(0)
        if _score_decay > 0:
            for col in df_e.columns:
                vals = df_e[col].values.copy()
                hold_days = 0
                entry_val = 0
                for ii in range(len(vals)):
                    if vals[ii] > 0:
                        if hold_days == 0:
                            entry_val = vals[ii]
                        hold_days += 1
                        vals[ii] = max(entry_val * ((1 - _score_decay) ** hold_days), 1.0)
                    else:
                        hold_days = 0
                        entry_val = 0
                df_e[col] = vals"""
    code = code.replace(old_e, new_e)

    with open(isaac_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print("  isaac.py updated with score_decay support")

# Reload module
if 'strategies.isaac' in sys.modules:
    del sys.modules['strategies.isaac']
from strategies.isaac import run_isaac_strategy as run_isaac_v2

# Now test different decay rates
decay_variants = [
    ('decay_1pct_day', 0.01),    # half-life ~70 days
    ('decay_2pct_day', 0.02),    # half-life ~35 days
    ('decay_5pct_day', 0.05),    # half-life ~14 days
    ('decay_10pct_day', 0.10),   # half-life ~7 days
    ('decay_20pct_day', 0.20),   # half-life ~3 days (aggressive)
]

for name, rate in decay_variants:
    print(f"\n[{name}] decay_rate={rate} (half-life ~{int(np.log(2)/rate)} days)...")
    t1 = time.time()
    try:
        r = run_isaac_v2(api_token, params={'score_decay': rate})
        results.append(get_stats_dict(name, r))
        print(f"  Done {time.time()-t1:.0f}s")
    except Exception as e:
        print(f"  ERROR: {e}")

# ======== Print Results ========
print()
print('=' * 130)
print('  Score Decay Rotation Results')
print('=' * 130)
header = f"  {'Name':<25} | {'CAGR':>8} | {'MaxDD':>8} | {'Sharpe':>8} | {'Sortino':>8} | {'Calmar':>8} | {'WinRate':>8} | {'Trades':>6}"
print(header)
print('-' * 130)
for r in results:
    tag = ' <<' if r['name'] == 'baseline_v37' else ''
    line = f"  {r['name']:<25} | {r['cagr']*100:>+7.2f}% | {r['max_dd']*100:>7.2f}% | {r['sharpe']:>8.4f} | {r['sortino']:>8.4f} | {r['calmar']:>8.4f} | {r['win_ratio']*100:>7.2f}% | {r['trades']:>6}{tag}"
    print(line)
print('=' * 130)

scored = []
for r in results:
    s = (r['cagr'] / abs(r['max_dd'])) * 0.4 + r['sharpe'] * 0.4 + r['win_ratio'] * 0.2
    scored.append((r['name'], s, r['cagr'], r['max_dd'], r['sharpe'], r['trades']))
scored.sort(key=lambda x: x[1], reverse=True)
print('\n  Score Ranking:')
for i, (n, s, c, d, sh, t) in enumerate(scored):
    rank = i + 1
    print(f'    #{rank}: {n:<25} Score={s:.4f}  (CAGR={c*100:+.2f}%, DD={d*100:.2f}%, Sharpe={sh:.4f}, Trades={t})')
