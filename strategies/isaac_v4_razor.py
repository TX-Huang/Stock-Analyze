"""
Isaac V4.1 Razor — 單策略極簡 Regime 配置

每個市場狀態只配置「風報比最佳」的單一策略 100%。
如同 Occam's Razor — 最簡單的配置，最好的風報比。

WF Test (2021+): CAGR 30.41%, MDD -13.01%, CAGR/MDD 2.34x, Sharpe 1.785
對比 V4.0:       CAGR 31.21%, MDD -17.90%, CAGR/MDD 1.74x, Sharpe 1.846

取捨: 犧牲 ~1% CAGR，換來 MDD 從 -17.9% 降到 -13.0%（降幅 27%）

配置表:
  strong_bull → Value Dividend 100%   (Ret/MDD 1.01x) — 多頭穩吃
  weak_bull   → Value Dividend 100%   (Ret/MDD 1.84x) — 弱多防守
  sideways    → Isaac V3.9 100%       (Ret/MDD 1.22x) — 盤整突破
  weak_bear   → Will VCP V2.0 100%    (Ret/MDD 4.30x) — 反彈獵場
  strong_bear → Pairs Trading 100%    (Ret/MDD 1.40x) — 空頭避險
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

STRATEGY_NAME = "Isaac V4.1 Razor 極簡配置"

# ============================================================
# Razor Allocation: 每 regime 只選風報比 (CAGR/MDD) 最佳的單一策略
# ============================================================
# Source: regime_analysis.json strategy_regime_matrix
# Selection criterion: max( |ann_return / mdd| ) per regime
#
# Tested 2026-03-26, compared against 7 other allocation variants:
#   D) Top-2 Sharpe-wt  — WF CAGR/MDD 1.74x (CAGR 31.2%, MDD -17.9%)
#   H) Razor (this)     — WF CAGR/MDD 2.34x (CAGR 30.4%, MDD -13.0%) ← best risk-adjusted
#
# Razor wins on CAGR/MDD by +34%, sacrificing only 0.8% CAGR for 4.9% less MDD.

_RAZOR_WEIGHTS = {
    'strong_bull': {
        # VD: Ret/MDD=1.01x (Sharpe 1.395, Ret 18.1%, MDD -18.0%)
        # Isaac: Ret/MDD=0.93x — close but VD's MDD is 10% lower
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 1.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bull': {
        # VD: Ret/MDD=1.84x (Sharpe 1.947, Ret 22.6%, MDD -12.3%)
        # PT: Ret/MDD=1.12x — VD dominates on both return and risk-adjusted
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 1.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'sideways': {
        # Isaac: Ret/MDD=1.22x (Sharpe 1.139, Ret 19.8%, MDD -16.2%)
        # VD: Ret/MDD=1.11x — Isaac captures breakout in range-bound markets
        'Isaac V3.9': 1.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'weak_bear': {
        # VCP: Ret/MDD=4.30x (Sharpe 1.994, Ret 56.0%, MDD -13.0%)
        # Isaac: Ret/MDD=3.67x — VCP edges out on oversold bounce plays
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 1.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 0.0, 'cash': 0.0,
    },
    'strong_bear': {
        # PT: Ret/MDD=1.40x (Sharpe 1.540, Ret 10.2%, MDD -7.3%)
        # VD: Ret/MDD=1.21x — PT is market-neutral, best bear hedge
        'Isaac V3.9': 0.0, 'Will VCP V2.0': 0.0, 'Mean Reversion': 0.0,
        'Value Dividend': 0.0, 'Pairs Trading': 1.0, 'cash': 0.0,
    },
}


# ============================================================
# Public API
# ============================================================

def run_strategy(api_token, **kwargs):
    """標準入口 — Streamlit UI 呼叫。"""
    return run_razor_strategy(api_token, params=kwargs)


def run_razor_strategy(api_token, params=None):
    """
    Isaac V4.1 Razor — 單策略極簡 Regime 配置

    流程:
    1. 載入 Razor 權重
    2. 跑 5 子策略回測 (try/except，≥2 成功即可)
    3. 取 0050 → classify_regime()
    4. backtest_dynamic_portfolio() 混合日報酬
    5. 組裝 RegimeBlendedReport
    """
    import finlab
    from data.provider import sanitize_dataframe

    t_start = time.time()
    logger.info("Isaac V4.1 Razor 啟動...")

    # 1. Razor 權重
    regime_weights = {r: dict(w) for r, w in _RAZOR_WEIGHTS.items()}
    logger.info(f"已載入 {len(regime_weights)} 個 regime 的 Razor 權重")

    # 2. FinLab login + 取得 0050
    finlab.login(api_token)
    from finlab import data
    close_all = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    close_all.columns = close_all.columns.astype(str)

    if '0050' not in close_all.columns:
        raise ValueError("0050 not found in price data")

    benchmark = close_all['0050'].dropna()

    # 3. 跑子策略 (只需要被 Razor 配置到的策略)
    needed = set()
    for w in regime_weights.values():
        for k, v in w.items():
            if v > 0.01 and k != 'cash':
                needed.add(k)
    logger.info(f"  Razor 需要 {len(needed)} 個策略: {needed}")

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
            f"Isaac V4.1 Razor: only {n_success} strategies succeeded "
            f"(failed: {', '.join(failed)}), need at least 2"
        )

    if failed:
        logger.warning(f"Razor: {len(failed)} failed ({', '.join(failed)}), renormalizing")
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
        'variant': 'razor',
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
        f"Isaac V4.1 Razor done ({elapsed:.1f}s) — "
        f"CAGR: {stats['cagr']*100:.2f}%, "
        f"MDD: {stats['max_drawdown']*100:.2f}%, "
        f"Sharpe: {stats['daily_sharpe']:.3f}"
    )

    return report


def get_current_regime_allocation(api_token):
    """取得今日 Razor 配置。"""
    current = get_current_regime(api_token)
    weights = _RAZOR_WEIGHTS.get(current, {})
    active = {k: v for k, v in weights.items() if v > 0.01 and k != 'cash'}
    return {
        'regime': current,
        'weights': active,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'variant': 'razor',
    }


# ============================================================
# Internal helpers (shared with isaac_v4.py pattern)
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
    return f"v4r_returns_{h}"


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
