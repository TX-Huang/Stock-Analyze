"""
Isaac V4.2 Turbo — 最大報酬率 Regime 配置

多頭/盤整/弱空全力進攻（VCP + Isaac），空頭防守（VD + PT）。
追求最大 CAGR，接受較高 MDD。

WF Test (2021+): CAGR 45.72%, MDD -24.68%, CAGR/MDD 1.85x, Sharpe 1.609
對比 V4.0:       CAGR 31.21%, MDD -17.90%, CAGR/MDD 1.74x, Sharpe 1.846
對比 V4.1 Razor: CAGR 30.41%, MDD -13.01%, CAGR/MDD 2.34x, Sharpe 1.785

取捨: +14.5% CAGR，但 MDD 從 -17.9% 升到 -24.7%

配置表:
  strong_bull → VCP 55% + Isaac 45%     — 多頭全力進攻
  weak_bull   → VCP 50% + Isaac 50%     — 弱多雙引擎
  sideways    → VCP 50% + Isaac 50%     — 盤整突破雙策略
  weak_bear   → VCP 58% + Isaac 42%     — 反彈獵場加碼
  strong_bear → VD 50% + PT 50%         — 空頭防守切換
"""
import os
import sys
import time
import hashlib
import pickle
import importlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.regime import (
    STRATEGIES, REGIMES,
    classify_regime, extract_daily_returns,
    backtest_dynamic_portfolio, compute_portfolio_metrics,
    load_regime_weights, get_current_regime,
)
from strategies.regime_report import build_regime_report

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'cache'
)

STRATEGY_NAME = "Isaac V4.2 Turbo 最大報酬"

# ============================================================
# Turbo Allocation: 多頭進攻，空頭防守
# ============================================================
# Source: regime_analysis.json strategy_regime_matrix
# Selection criterion: max CAGR per regime, strong_bear 切防守
#
# Tested 2026-03-26, compared against 5 max-return variants:
#   I) VCP all         — WF CAGR 43.0%, MDD -37.8%, CAGR/MDD 1.14x (MDD 太高)
#   J) Top-2 Ret-wt    — WF CAGR 43.3%, MDD -28.6%, CAGR/MDD 1.51x
#   K) Aggr Smart      — WF CAGR 47.6%, MDD -34.4%, CAGR/MDD 1.38x (MDD 太高)
#   L) VCP+Isaac all   — WF CAGR 45.8%, MDD -28.6%, CAGR/MDD 1.60x
#   M) VCP+Isaac bull, VD+PT bear — WF CAGR 45.7%, MDD -24.7%, CAGR/MDD 1.85x ← ADOPTED
#
# M wins: highest practical CAGR with best CAGR/MDD among aggressive configs.
# Key insight: VCP dominates returns in all regimes except strong_bear,
# but strong_bear VCP has -35% MDD → switch to VD+PT for bear hedge.

_TURBO_WEIGHTS = {
    'strong_bull': {
        # VCP 33.5% + Isaac 26.4% — 兩大成長引擎
        'Isaac V3.9': 0.45, 'Will VCP V2.0': 0.55, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bull': {
        # VCP 23.3% + Isaac 22.1% — 弱多雙引擎等權
        'Isaac V3.9': 0.50, 'Will VCP V2.0': 0.50, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'sideways': {
        # VCP 20.2% + Isaac 19.8% — 盤整突破雙策略
        'Isaac V3.9': 0.50, 'Will VCP V2.0': 0.50, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bear': {
        # VCP 56.0% + Isaac 40.5% — 反彈最佳狩獵場
        'Isaac V3.9': 0.42, 'Will VCP V2.0': 0.58, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'strong_bear': {
        # VD 16.7% + PT 10.2% — 空頭切防守（VCP MDD -35% 不可接受）
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.50, 'Pairs Trading': 0.50, 'cash': 0.0,
    },
}


# ============================================================
# Public API
# ============================================================

def run_strategy(api_token, **kwargs):
    """標準入口 — Streamlit UI 呼叫。"""
    return run_turbo_strategy(api_token, params=kwargs)


def run_turbo_strategy(api_token, params=None):
    """
    Isaac V4.2 Turbo — 最大報酬率 Regime 配置

    流程:
    1. 載入 Turbo 權重
    2. 跑子策略回測 (try/except，≥2 成功即可)
    3. 取 0050 → classify_regime()
    4. backtest_dynamic_portfolio() 混合日報酬
    5. 組裝 RegimeBlendedReport
    """
    import finlab
    from data.provider import sanitize_dataframe

    t_start = time.time()
    logger.info("Isaac V4.2 Turbo 啟動...")

    # 1. Turbo 權重
    regime_weights = {r: dict(w) for r, w in _TURBO_WEIGHTS.items()}
    logger.info(f"已載入 {len(regime_weights)} 個 regime 的 Turbo 權重")

    # 2. FinLab login + 取得 0050
    finlab.login(api_token)
    from finlab import data
    close_all = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    close_all.columns = close_all.columns.astype(str)

    if '0050' not in close_all.columns:
        raise ValueError("0050 not found in price data")

    benchmark = close_all['0050'].dropna()

    # 3. 跑子策略 (只需要被 Turbo 配置到的策略)
    needed = set()
    for w in regime_weights.values():
        for k, v in w.items():
            if v > 0.01 and k != 'cash':
                needed.add(k)
    logger.info(f"  Turbo 需要 {len(needed)} 個策略: {needed}")

    returns_dict = {}
    sub_reports = {}
    failed = []

    for name, (module_path, func_name, strat_params) in STRATEGIES.items():
        if name not in needed:
            continue
        report, returns = _run_substrategy(
            name, module_path, func_name, strat_params, api_token
        )
        if returns is not None:
            returns_dict[name] = returns
            sub_reports[name] = report
            logger.info(f"  OK {name}: {len(returns)} days")
        else:
            failed.append(name)
            logger.warning(f"  FAIL {name}")

    n_success = len(returns_dict)
    if n_success < 2:
        raise RuntimeError(
            f"Isaac V4.2 Turbo: only {n_success} strategies succeeded "
            f"(failed: {', '.join(failed)}), need at least 2"
        )

    if failed:
        logger.warning(f"Turbo: {len(failed)} failed ({', '.join(failed)}), renormalizing")
        regime_weights = _renormalize_weights(regime_weights, failed)

    # 4. 組合
    returns_df = pd.DataFrame(returns_dict)

    # 5. Regime 分類
    regime_series = classify_regime(benchmark, debounce_days=5)
    current_regime = regime_series.iloc[-1]
    current_date = regime_series.index[-1].strftime('%Y-%m-%d')
    logger.info(f"Regime: {current_regime} ({current_date})")

    # 6. 動態組合回測
    portfolio_returns = backtest_dynamic_portfolio(
        returns_df, regime_series, regime_weights, transition_days=5
    )

    # 7. 組裝報告
    current_weights = regime_weights.get(current_regime, {})
    regime_info = {
        'current_regime': current_regime,
        'weights': current_weights,
        'date': current_date,
        'failed_strategies': failed,
        'n_strategies': n_success,
        'variant': 'turbo',
    }

    report = build_regime_report(
        portfolio_returns=portfolio_returns,
        benchmark=benchmark,
        sub_reports=sub_reports,
        regime_info=regime_info,
    )

    elapsed = time.time() - t_start
    stats = report.get_stats()
    logger.info(
        f"Isaac V4.2 Turbo done ({elapsed:.1f}s) — "
        f"CAGR: {stats['cagr']*100:.2f}%, "
        f"MDD: {stats['max_drawdown']*100:.2f}%, "
        f"Sharpe: {stats['daily_sharpe']:.3f}"
    )

    return report


def get_current_regime_allocation(api_token):
    """取得今日 Turbo 配置。"""
    current = get_current_regime(api_token)
    weights = _TURBO_WEIGHTS.get(current, {})
    active = {k: v for k, v in weights.items() if v > 0.01 and k != 'cash'}
    return {
        'regime': current,
        'weights': active,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'variant': 'turbo',
    }


# ============================================================
# Internal helpers
# ============================================================

def _run_substrategy(name, module_path, func_name, params, api_token):
    """執行單一子策略，失敗返回 (None, None)。優先快取。"""
    cache_key = _cache_key(name)
    cached = _load_cache(cache_key)
    if cached is not None:
        returns, report_data = cached
        logger.info(f"  cache {name}: {len(returns)} days")
        return _CachedReportStub(report_data), returns

    try:
        t0 = time.time()
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        report = func(api_token, **params)
        returns = extract_daily_returns(report)
        elapsed = time.time() - t0

        if returns is None or len(returns) < 10:
            return None, None

        logger.info(f"  {name}: backtest done ({elapsed:.1f}s)")
        try:
            report_data = {'trades': report.get_trades()}
        except Exception:
            report_data = {'trades': pd.DataFrame()}
        _save_cache(cache_key, (returns, report_data))
        return report, returns

    except Exception as e:
        logger.error(f"  {name} failed: {e}")
        return None, None


def _renormalize_weights(regime_weights, failed_strategies):
    """移除失敗策略的權重，重新正規化。"""
    new_weights = {}
    for regime, weights in regime_weights.items():
        new_w = {}
        total = 0
        for k, v in weights.items():
            if k not in failed_strategies and k != 'cash':
                new_w[k] = v
                total += v
        cash = weights.get('cash', 0)
        failed_weight = sum(weights.get(f, 0) for f in failed_strategies)
        if total > 0 and failed_weight > 0:
            for k in new_w:
                new_w[k] = new_w[k] * (1 - cash) / total
        new_w['cash'] = cash
        new_weights[regime] = new_w
    return new_weights


class _CachedReportStub:
    def __init__(self, report_data):
        self._data = report_data
        self.position = None

    def get_trades(self):
        return self._data.get('trades', pd.DataFrame())

    def get_stats(self):
        return {}


def _cache_key(strategy_name):
    today = datetime.now().strftime('%Y%m%d')
    raw = f"{strategy_name}_{today}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"v4t_returns_{h}"


def _cache_path(key):
    return os.path.join(_CACHE_DIR, f"{key}.pkl")


def _load_cache(key):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(key, data):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(path := _cache_path(key), 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")
