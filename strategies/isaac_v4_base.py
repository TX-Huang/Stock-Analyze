"""
Isaac V4.x 共用基礎模組

所有 V4.x 策略（V4.0, V4.1 Razor, V4.2 Turbo）共用的：
- 子策略執行 + 快取
- 權重正規化
- Regime 分類 + 動態組合回測
- 報告組裝

各策略檔只需定義：STRATEGY_NAME, _WEIGHTS, cache_prefix
"""
import os
import time
import hashlib
import pickle
import importlib
import logging
from datetime import datetime

import pandas as pd

from analysis.regime import (
    STRATEGIES, classify_regime, extract_daily_returns,
    backtest_dynamic_portfolio, get_current_regime,
)
from strategies.regime_report import build_regime_report

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'cache'
)


# ============================================================
# Public: 通用策略執行器
# ============================================================

def run_v4_strategy(api_token, weights_dict, strategy_name, cache_prefix, variant, params=None):
    """
    通用 V4.x 策略執行流程。

    Args:
        api_token: FinLab API token
        weights_dict: {regime: {strategy: weight}} 配置表
        strategy_name: 策略顯示名稱
        cache_prefix: 快取 key 前綴 (e.g. 'v4', 'v4r', 'v4t')
        variant: 變體標識 (e.g. 'top2', 'razor', 'turbo')
        params: 額外參數 (unused, for API compatibility)

    Returns:
        RegimeBlendedReport
    """
    import finlab
    from data.provider import sanitize_dataframe

    t_start = time.time()
    logger.info(f"{strategy_name} 啟動...")

    # 1. 權重
    regime_weights = {r: dict(w) for r, w in weights_dict.items()}

    # 2. FinLab login + 取得 0050
    finlab.login(api_token)
    from finlab import data
    close_all = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    close_all.columns = close_all.columns.astype(str)

    if '0050' not in close_all.columns:
        raise ValueError("0050 not found in price data")

    benchmark = close_all['0050'].dropna()

    # 3. 跑子策略 (只跑被配置到的)
    needed = set()
    for w in regime_weights.values():
        for k, v in w.items():
            if v > 0.01 and k != 'cash':
                needed.add(k)
    logger.info(f"  需要 {len(needed)} 個策略: {needed}")

    returns_dict = {}
    sub_reports = {}
    failed = []

    for name, (module_path, func_name, strat_params) in STRATEGIES.items():
        if name not in needed:
            continue
        report, returns = _run_substrategy(
            name, module_path, func_name, strat_params, api_token, cache_prefix
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
            f"{strategy_name}: only {n_success} strategies succeeded "
            f"(failed: {', '.join(failed)}), need at least 2"
        )

    if failed:
        logger.warning(f"{len(failed)} failed ({', '.join(failed)}), renormalizing")
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
        'variant': variant,
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
        f"{strategy_name} done ({elapsed:.1f}s) — "
        f"CAGR: {stats['cagr']*100:.2f}%, "
        f"MDD: {stats['max_drawdown']*100:.2f}%, "
        f"Sharpe: {stats['daily_sharpe']:.3f}"
    )

    return report


def get_v4_regime_allocation(api_token, weights_dict, variant):
    """通用取得今日配置。"""
    current = get_current_regime(api_token)
    weights = weights_dict.get(current, {})
    active = {k: v for k, v in weights.items() if v > 0.01 and k != 'cash'}
    return {
        'regime': current,
        'weights': active,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'variant': variant,
    }


# ============================================================
# Internal helpers
# ============================================================

def _run_substrategy(name, module_path, func_name, params, api_token, cache_prefix):
    """執行單一子策略，失敗返回 (None, None)。優先快取。"""
    cache_key = _cache_key(name, cache_prefix)
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
    """快取模式下的 minimal report stub。"""
    def __init__(self, report_data):
        self._data = report_data
        self.position = None

    def get_trades(self):
        return self._data.get('trades', pd.DataFrame())

    def get_stats(self):
        return {}


# ============================================================
# Caching
# ============================================================

def _cache_key(strategy_name, prefix='v4'):
    today = datetime.now().strftime('%Y%m%d')
    raw = f"{strategy_name}_{today}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{prefix}_returns_{h}"


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
