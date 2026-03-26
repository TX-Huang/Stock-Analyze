"""
Isaac V4.0 — Regime 動態配置元策略 (Top-2 Sharpe-Weighted)

基於市場 Regime 分類，動態調整五大子策略權重。
每 regime 配置 Sharpe 最高的兩個策略，按 Sharpe 比例加權。

WF Test (2021+): CAGR 31.21%, MDD -17.90%, CAGR/MDD 1.74x, Sharpe 1.846

配置表:
  strong_bull → VD 52% + Isaac 48%    (Sharpe 1.395/1.270)
  weak_bull   → VD 58% + PT 42%      (Sharpe 1.947/1.385)
  sideways    → Isaac 50% + VD 50%   (Sharpe 1.139/1.157)
  weak_bear   → Isaac 50% + VCP 50%  (Sharpe 2.004/1.994)
  strong_bear → PT 53% + VD 47%      (Sharpe 1.540/1.348)
"""
from strategies.isaac_v4_base import run_v4_strategy, get_v4_regime_allocation

STRATEGY_NAME = "Isaac V4.0 Regime 動態配置"
_CACHE_PREFIX = 'v4'
_VARIANT = 'top2'

# ============================================================
# Top-2 Sharpe-weighted allocation per regime
# Source: regime_analysis.json strategy_regime_matrix
# Tested 2026-03-26, variant D — best CAGR/MDD among balanced configs
# ============================================================
_WEIGHTS = {
    'strong_bull': {
        'Isaac V3.9': 0.48, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.52, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bull': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.58, 'Pairs Trading': 0.42, 'cash': 0.0,
    },
    'sideways': {
        'Isaac V3.9': 0.50, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.50, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bear': {
        'Isaac V3.9': 0.50, 'Will VCP V2.0': 0.50, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'strong_bear': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.47, 'Pairs Trading': 0.53, 'cash': 0.0,
    },
}


def run_strategy(api_token, **kwargs):
    """標準入口 — Streamlit UI 呼叫。"""
    return run_v4_strategy(api_token, _WEIGHTS, STRATEGY_NAME, _CACHE_PREFIX, _VARIANT, kwargs)


def get_current_regime_allocation(api_token):
    """取得今日配置。"""
    return get_v4_regime_allocation(api_token, _WEIGHTS, _VARIANT)
