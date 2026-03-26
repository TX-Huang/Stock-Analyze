"""NT$ 損益計算引擎 — 將 FinLab 回測的 creturn 轉換為個人化投資模擬結果。"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def simulate_lumpsum(creturn, capital):
    """
    一次投入模擬：將 creturn (累積報酬率序列) 轉換為 NT$ 淨值序列。

    Args:
        creturn: pd.Series, FinLab 回測的 cumulative return (1.0-based)
        capital: float, 投入本金 (NTD)

    Returns:
        pd.Series of NT$ equity values
    """
    if creturn is None or creturn.empty:
        return pd.Series(dtype=float)

    # Normalize to 1.0-based if needed
    first_val = creturn.iloc[0]
    if first_val != 0:
        normalized = creturn / first_val
    else:
        normalized = creturn

    return normalized * capital


def simulate_dca(creturn, dca_monthly, start_date=None, end_date=None):
    """
    定期定額 (DCA) 模擬。

    公式: total_value[t] = Σ(monthly_contribution[i] × creturn[t] / creturn[i])
    每月第一個交易日投入 dca_monthly 金額。

    Args:
        creturn: pd.Series, FinLab 回測的 cumulative return (1.0-based)
        dca_monthly: float, 每月投入金額 (NTD)
        start_date: optional start date filter
        end_date: optional end date filter

    Returns:
        dict with:
        - equity: pd.Series of NT$ total market value
        - invested: pd.Series of NT$ cumulative invested capital
        - irr: float, internal rate of return (annualized)
    """
    if creturn is None or creturn.empty or dca_monthly <= 0:
        return {
            'equity': pd.Series(dtype=float),
            'invested': pd.Series(dtype=float),
            'irr': 0.0,
        }

    cr = creturn.copy()
    if start_date:
        cr = cr.loc[start_date:]
    if end_date:
        cr = cr.loc[:end_date]
    if cr.empty:
        return {
            'equity': pd.Series(dtype=float),
            'invested': pd.Series(dtype=float),
            'irr': 0.0,
        }

    # Identify first trading day of each month
    monthly_first = cr.groupby(cr.index.to_period('M')).apply(lambda x: x.index[0])

    # Track contributions
    contributions = []  # list of (date, amount, creturn_at_date)
    equity = pd.Series(0.0, index=cr.index)
    invested = pd.Series(0.0, index=cr.index)

    cum_invested = 0.0

    for dt in cr.index:
        # Check if this is a monthly contribution date
        if dt in monthly_first.values:
            contributions.append((dt, dca_monthly, cr.loc[dt]))
            cum_invested += dca_monthly

        # Calculate total value: sum of each contribution's current value
        total_value = 0.0
        for _contrib_dt, amount, cr_at_contrib in contributions:
            if cr.loc[dt] != 0 and cr_at_contrib != 0:
                total_value += amount * (cr.loc[dt] / cr_at_contrib)

        equity.loc[dt] = total_value
        invested.loc[dt] = cum_invested

    # Calculate IRR (simplified: use annualized return based on total)
    irr = _calculate_irr(contributions, equity.iloc[-1] if len(equity) > 0 else 0, cr.index)

    return {
        'equity': equity,
        'invested': invested,
        'irr': irr,
    }


def _calculate_irr(contributions, final_value, date_index):
    """Approximate IRR using simple annualization."""
    if not contributions or final_value <= 0:
        return 0.0

    total_invested = sum(amt for _, amt, _ in contributions)
    if total_invested <= 0:
        return 0.0

    # Weighted average investment period in years
    last_date = date_index[-1]
    weighted_years = 0.0
    for dt, amt, _ in contributions:
        days = (last_date - dt).days
        weighted_years += amt * (days / 365.25)

    avg_years = weighted_years / total_invested if total_invested > 0 else 1.0
    if avg_years <= 0:
        return 0.0

    # Money-weighted return
    total_return = final_value / total_invested
    try:
        irr = total_return ** (1.0 / avg_years) - 1.0
    except (ZeroDivisionError, ValueError):
        irr = 0.0

    return irr


def yearly_pnl(equity_series, capital=None):
    """
    計算逐年損益表。

    Args:
        equity_series: pd.Series of NT$ equity values (from simulate_lumpsum or dca)
        capital: float, initial capital (used for first year start value).
                 If None, uses first value of equity_series.

    Returns:
        pd.DataFrame with columns: year, start_value, end_value, pnl_ntd, return_pct
    """
    if equity_series is None or equity_series.empty:
        return pd.DataFrame(columns=['year', 'start_value', 'end_value', 'pnl_ntd', 'return_pct'])

    rows = []
    years = sorted(equity_series.index.year.unique())

    for i, year in enumerate(years):
        year_data = equity_series[equity_series.index.year == year]
        if year_data.empty:
            continue

        end_value = year_data.iloc[-1]

        if i == 0:
            start_value = capital if capital is not None else year_data.iloc[0]
        else:
            # Start value = previous year's end value
            prev_year_data = equity_series[equity_series.index.year == years[i - 1]]
            start_value = prev_year_data.iloc[-1] if not prev_year_data.empty else year_data.iloc[0]

        pnl = end_value - start_value
        ret_pct = (pnl / start_value * 100) if start_value != 0 else 0.0

        rows.append({
            'year': year,
            'start_value': round(start_value, 0),
            'end_value': round(end_value, 0),
            'pnl_ntd': round(pnl, 0),
            'return_pct': round(ret_pct, 1),
        })

    return pd.DataFrame(rows)


def summary_stats(equity_series, capital, cost_result=None):
    """
    計算 NT$ 摘要統計。

    Args:
        equity_series: pd.Series of NT$ equity values
        capital: float, initial capital
        cost_result: dict from cost_analysis.analyze_trading_costs() (optional)

    Returns:
        dict with final_value, total_pnl, total_return_pct, total_cost, net_pnl
    """
    if equity_series is None or equity_series.empty:
        return {
            'final_value': capital,
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'total_cost': 0.0,
            'net_pnl': 0.0,
        }

    final_value = equity_series.iloc[-1]
    total_pnl = final_value - capital
    total_return_pct = (total_pnl / capital * 100) if capital > 0 else 0.0

    total_cost = 0.0
    if cost_result and cost_result.get('total_cost', 0) > 0:
        # Scale cost by capital ratio
        total_cost = cost_result['total_cost']

    net_pnl = total_pnl - total_cost

    return {
        'final_value': round(final_value, 0),
        'total_pnl': round(total_pnl, 0),
        'total_return_pct': round(total_return_pct, 1),
        'total_cost': round(total_cost, 0),
        'net_pnl': round(net_pnl, 0),
    }


def validate_settings(capital, start_date, end_date, mode, dca_monthly=0):
    """
    驗證使用者設定。

    Returns:
        (is_valid, error_message)
    """
    if capital < 100_000:
        return False, "投資金額最低 10 萬元"

    if start_date and end_date and start_date >= end_date:
        return False, "起始日期必須早於結束日期"

    if mode == 'dca' and dca_monthly <= 0:
        return False, "定期定額模式需設定每月投入金額"

    return True, ""
