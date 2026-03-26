"""
Isaac V4.1 Razor — 單策略極簡 Regime 配置

每個市場狀態只配置「風報比最佳」的單一策略 100%。
如同 Occam's Razor — 最簡單的配置，最好的風報比。

WF Test (2021+): CAGR 30.41%, MDD -13.01%, CAGR/MDD 2.34x, Sharpe 1.785

配置表:
  strong_bull → Value Dividend 100%   (Ret/MDD 1.01x) — 多頭穩吃
  weak_bull   → Value Dividend 100%   (Ret/MDD 1.84x) — 弱多防守
  sideways    → Isaac V3.9 100%       (Ret/MDD 1.22x) — 盤整突破
  weak_bear   → Will VCP V2.0 100%    (Ret/MDD 4.30x) — 反彈獵場
  strong_bear → Pairs Trading 100%    (Ret/MDD 1.40x) — 空頭避險
"""
from strategies.isaac_v4_base import run_v4_strategy, get_v4_regime_allocation

STRATEGY_NAME = "Isaac V4.1 Razor 極簡配置"
_CACHE_PREFIX = 'v4r'
_VARIANT = 'razor'

# ============================================================
# Razor Allocation: 每 regime 只選 CAGR/MDD 最佳的單一策略
# Tested 2026-03-26, variant H — best CAGR/MDD (2.34x)
# ============================================================
_WEIGHTS = {
    'strong_bull': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 1.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bull': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 1.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'sideways': {
        'Isaac V3.9': 1.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bear': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 1.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'strong_bear': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 1.0, 'cash': 0.0,
    },
}


def run_strategy(api_token, **kwargs):
    """標準入口 — Streamlit UI 呼叫。"""
    return run_v4_strategy(api_token, _WEIGHTS, STRATEGY_NAME, _CACHE_PREFIX, _VARIANT, kwargs)


def get_current_regime_allocation(api_token):
    """取得今日 Razor 配置。"""
    return get_v4_regime_allocation(api_token, _WEIGHTS, _VARIANT)
