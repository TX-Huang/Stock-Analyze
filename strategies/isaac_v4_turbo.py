"""
Isaac V4.2 Turbo — 最大報酬率 Regime 配置

多頭/盤整/弱空全力進攻（VCP + Isaac），空頭防守（VD + PT）。
追求最大 CAGR，接受較高 MDD。

WF Test (2021+): CAGR 45.72%, MDD -24.68%, CAGR/MDD 1.85x, Sharpe 1.609

配置表:
  strong_bull → VCP 55% + Isaac 45%     — 多頭全力進攻
  weak_bull   → VCP 50% + Isaac 50%     — 弱多雙引擎
  sideways    → VCP 50% + Isaac 50%     — 盤整突破雙策略
  weak_bear   → VCP 58% + Isaac 42%     — 反彈獵場加碼
  strong_bear → VD 50% + PT 50%         — 空頭防守切換

空頭防守是「免費午餐」: 移除後 CAGR 反而下降 0.5%, MDD 增加 0.5%。
"""
from strategies.isaac_v4_base import run_v4_strategy, get_v4_regime_allocation, PARAM_SCHEMA

STRATEGY_NAME = "Isaac V4.2 Turbo 最大報酬"
_CACHE_PREFIX = 'v4t'
_VARIANT = 'turbo'

# ============================================================
# Turbo Allocation: max CAGR per regime, strong_bear 切防守
# Tested 2026-03-26, variant M — highest practical CAGR with best
# CAGR/MDD among aggressive configs (1.85x)
# ============================================================
_WEIGHTS = {
    'strong_bull': {
        'Isaac V3.9': 0.45, 'Will VCP V2.0': 0.55, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bull': {
        'Isaac V3.9': 0.50, 'Will VCP V2.0': 0.50, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'sideways': {
        'Isaac V3.9': 0.50, 'Will VCP V2.0': 0.50, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bear': {
        'Isaac V3.9': 0.42, 'Will VCP V2.0': 0.58, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'strong_bear': {
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.50, 'Pairs Trading': 0.50, 'cash': 0.0,
    },
}


def run_strategy(api_token, **kwargs):
    """標準入口 — Streamlit UI 呼叫。"""
    return run_v4_strategy(api_token, _WEIGHTS, STRATEGY_NAME, _CACHE_PREFIX, _VARIANT, kwargs)


def get_current_regime_allocation(api_token):
    """取得今日 Turbo 配置。"""
    return get_v4_regime_allocation(api_token, _WEIGHTS, _VARIANT)
