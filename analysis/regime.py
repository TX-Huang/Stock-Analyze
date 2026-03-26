"""
市場 Regime 核心函數模組

提供 5-Regime 分類器、動態組合回測、績效指標計算等可重用函數。
供 strategies/isaac_v4.py 和 scripts/regime_analysis.py 共用。

5-Regime Model:
  strong_bull  — 多頭排列 (0050 > MA60 > MA120, 0050 > MA20)
  weak_bull    — 中期向上但動能減弱
  sideways     — 區間盤整
  weak_bear    — 跌破中期但長期支撐在
  strong_bear  — 空頭排列
"""
import os
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 集中定義策略映射 — 避免多處重複
# ============================================================
STRATEGIES = {
    'Isaac V3.9':     ('strategies.isaac', 'run_isaac_strategy', {}),
    'Will VCP V2.0':  ('strategies.will_vcp', 'run_strategy', {}),
    'Mean Reversion': ('strategies.mean_reversion', 'run_strategy', {
        'mode': 'classic', 'uptrend_filter': True,
        'deviation_threshold': -0.08, 'exit_ma': 10,
    }),
    'Value Dividend': ('strategies.value_dividend', 'run_strategy', {}),
    'Pairs Trading':  ('strategies.pairs_trading', 'run_strategy', {}),
}

REGIMES = ['strong_bull', 'weak_bull', 'sideways', 'weak_bear', 'strong_bear']

# Default JSON path
_DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'regime_analysis.json'
)


# ============================================================
# 1. Regime 分類器
# ============================================================

def classify_regime(benchmark, debounce_days=5):
    """
    使用 0050 收盤價分類市場 regime。

    Args:
        benchmark: pd.Series — 0050 日收盤價
        debounce_days: 防抖動天數 (regime 必須連續 N 天確認才切換)

    Returns:
        pd.Series — 每日 regime 標籤
    """
    ma20 = benchmark.rolling(20, min_periods=20).mean()
    ma60 = benchmark.rolling(60, min_periods=60).mean()
    ma120 = benchmark.rolling(120, min_periods=120).mean()

    # MA60 斜率 (20 日變化率)
    ma60_slope = ma60.pct_change(20)

    raw_regime = pd.Series('unknown', index=benchmark.index)

    # strong_bull: 多頭排列
    strong_bull = (
        (benchmark > ma60) &
        (ma60 > ma120) &
        (benchmark > ma20) &
        ma20.notna() & ma60.notna() & ma120.notna()
    )

    # strong_bear: 空頭排列
    strong_bear = (
        (benchmark < ma60) &
        (benchmark < ma120) &
        (ma60 < ma120) &
        ma60.notna() & ma120.notna()
    )

    # weak_bear: 跌破中期但長期支撐在
    weak_bear = (
        (benchmark < ma60) &
        (benchmark > ma120) &
        ma60.notna() & ma120.notna() &
        ~strong_bear
    )

    # sideways: 接近 MA60 且斜率平坦
    sideways = (
        (abs(benchmark / ma60 - 1) < 0.03) &
        (abs(ma60_slope) < 0.02) &
        ma60.notna() &
        ~strong_bull & ~strong_bear & ~weak_bear
    )

    # weak_bull: 在 MA60 之上但不是 strong_bull
    weak_bull = (
        (benchmark > ma60) &
        ma60.notna() &
        ~strong_bull & ~sideways
    )

    raw_regime[strong_bull] = 'strong_bull'
    raw_regime[weak_bull] = 'weak_bull'
    raw_regime[sideways] = 'sideways'
    raw_regime[weak_bear] = 'weak_bear'
    raw_regime[strong_bear] = 'strong_bear'

    # 防抖動: regime 必須連續 debounce_days 天才確認切換
    smoothed = raw_regime.copy()
    current_regime = 'unknown'
    pending_regime = None
    pending_count = 0

    for i in range(len(smoothed)):
        r = raw_regime.iloc[i]
        if r == current_regime:
            pending_regime = None
            pending_count = 0
            smoothed.iloc[i] = current_regime
        elif r == pending_regime:
            pending_count += 1
            if pending_count >= debounce_days:
                current_regime = r
                pending_regime = None
                pending_count = 0
            smoothed.iloc[i] = current_regime
        else:
            pending_regime = r
            pending_count = 1
            smoothed.iloc[i] = current_regime

    return smoothed


# ============================================================
# 2. 日報酬率提取
# ============================================================

def extract_daily_returns(report):
    """從 FinLab report 提取每日報酬率序列。"""
    try:
        cr = report.creturn
        if isinstance(cr, pd.Series) and len(cr) > 1:
            returns = cr.pct_change().dropna()
            returns.name = None
            return returns
    except Exception:
        pass
    try:
        dcr = report.daily_creturn
        if isinstance(dcr, pd.Series) and len(dcr) > 1:
            returns = dcr.pct_change().dropna()
            returns.name = None
            return returns
    except Exception:
        pass
    return None


# ============================================================
# 3. 動態組合回測
# ============================================================

def backtest_dynamic_portfolio(returns_df, regime_series, regime_weights,
                                transition_days=5):
    """
    用 regime-switching 權重回測動態組合。

    Args:
        returns_df: DataFrame — 各策略日報酬
        regime_series: pd.Series — regime labels
        regime_weights: dict — {regime: {strategy: weight}}
        transition_days: int — 切換過渡天數

    Returns:
        pd.Series — 動態組合的日報酬
    """
    strategies = [c for c in returns_df.columns]
    common_idx = returns_df.index.intersection(regime_series.index)
    ret_aligned = returns_df.reindex(common_idx).fillna(0)
    reg_aligned = regime_series.reindex(common_idx)

    # 建立每日權重矩陣
    n = len(common_idx)
    weight_matrix = np.zeros((n, len(strategies)))

    prev_weights = None
    transition_counter = 0
    target_weights = None

    for i in range(n):
        regime = reg_aligned.iloc[i]
        rw = regime_weights.get(regime, regime_weights.get('sideways', {}))
        current_target = np.array([rw.get(s, 0) for s in strategies])

        if prev_weights is None:
            weight_matrix[i] = current_target
            prev_weights = current_target
            target_weights = current_target
            continue

        # 檢查是否有 regime 切換
        if not np.allclose(current_target, target_weights, atol=1e-4):
            target_weights = current_target
            transition_counter = 0

        if transition_counter < transition_days:
            # 線性混合
            alpha = (transition_counter + 1) / transition_days
            weight_matrix[i] = prev_weights * (1 - alpha) + target_weights * alpha
            transition_counter += 1
        else:
            weight_matrix[i] = target_weights

        prev_weights = weight_matrix[i].copy()

    # 計算組合日報酬
    portfolio_returns = (ret_aligned.values * weight_matrix).sum(axis=1)
    return pd.Series(portfolio_returns, index=common_idx)


# ============================================================
# 4. 績效指標計算
# ============================================================

def compute_portfolio_metrics(portfolio_returns):
    """計算組合績效指標。"""
    if len(portfolio_returns) < 10:
        return {'cagr': 0, 'mdd': 0, 'sharpe': 0, 'ann_vol': 0, 'n_days': 0}

    total_return = (1 + portfolio_returns).prod()
    n_years = len(portfolio_returns) / 252
    cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    mdd = float(drawdown.min())

    sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
              if portfolio_returns.std() > 0 else 0)
    ann_vol = float(portfolio_returns.std() * np.sqrt(252))

    return {
        'cagr': round(float(cagr), 4),
        'mdd': round(mdd, 4),
        'sharpe': round(float(sharpe), 3),
        'ann_vol': round(ann_vol, 4),
        'n_days': int(len(portfolio_returns)),
    }


# ============================================================
# 5. Walk-Forward 權重載入
# ============================================================

def load_regime_weights(json_path=None):
    """
    從 regime_analysis.json 讀取 Walk-Forward 訓練期權重。

    注意: 讀取 walkforward.regime_weights_from_train（非 optimal_weights）。
    optimal_weights 是 in-sample 過擬合的結果。

    Returns:
        dict — {regime: {strategy: weight, 'cash': weight}}
    """
    if json_path is None:
        json_path = _DEFAULT_JSON

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Regime analysis JSON not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用 Walk-Forward 訓練期權重，非 in-sample optimal_weights
    wf = data.get('walkforward', {})
    weights = wf.get('regime_weights_from_train', {})

    if not weights:
        raise ValueError("No Walk-Forward regime weights found in JSON")

    logger.info(f"Loaded WF regime weights from {json_path}")
    return weights


def get_current_regime(api_token):
    """
    取得今日的市場 regime（輕量級，不跑策略回測）。

    Args:
        api_token: FinLab API token

    Returns:
        str — 當前 regime 標籤
    """
    import finlab
    from data.provider import sanitize_dataframe

    finlab.login(api_token)
    from finlab import data
    close_all = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    close_all.columns = close_all.columns.astype(str)

    if '0050' not in close_all.columns:
        raise ValueError("0050 not found in price data")

    benchmark = close_all['0050'].dropna()
    # 只需要最近 150 天即可分類
    benchmark = benchmark.iloc[-200:]

    regime_series = classify_regime(benchmark, debounce_days=5)
    current = regime_series.iloc[-1]

    logger.info(f"Current regime: {current} (as of {regime_series.index[-1].strftime('%Y-%m-%d')})")
    return current
