"""
RegimeBlendedReport — FinLab 相容報告包裝

將 Return-Level Blending 的結果包裝成 FinLab report 相容的介面，
供 ui/backtest_dashboard.py 和 ui/pages/ 使用。

Dashboard 存取的 attributes (audited from backtest_dashboard.py):
- report.get_trades()     → 交易 DataFrame
- report.get_stats()      → 績效 dict
- report.position         → 部位 DataFrame (None for return-blend)
  ↳ L476: report.position.index[-1] — wrapped in try/except, safe with None
"""
import pandas as pd
import numpy as np
from datetime import datetime


class RegimeBlendedReport:
    """FinLab-compatible report wrapper for regime-switched portfolio."""

    def __init__(self, creturn, benchmark, trades_df, stats_dict, regime_info):
        """
        Args:
            creturn: pd.Series — 累積報酬 (1-based, i.e. starts at 1.0)
            benchmark: pd.Series — 0050 收盤價
            trades_df: pd.DataFrame — 合併的子策略交易
            stats_dict: dict — 績效指標
            regime_info: dict — {'current_regime': str, 'weights': dict, 'date': str}
        """
        self.creturn = creturn
        self.benchmark = benchmark
        self.position = None          # Return-blend 無原始部位
        self._trades = trades_df
        self._stats = stats_dict
        self.regime_info = regime_info

    def get_trades(self) -> pd.DataFrame:
        """返回合併的子策略交易列表。"""
        return self._trades

    def get_stats(self) -> dict:
        """返回績效指標 dict。

        Keys: cagr, max_drawdown, daily_sharpe, sharpe, win_ratio,
              ann_vol, n_days, regime_info
        """
        return self._stats


def build_regime_report(portfolio_returns, benchmark, sub_reports,
                        regime_info, strategy_names=None):
    """
    從動態組合日報酬建構 RegimeBlendedReport。

    Args:
        portfolio_returns: pd.Series — 動態組合日報酬
        benchmark: pd.Series — 0050 收盤價
        sub_reports: dict — {strategy_name: report} 子策略報告
        regime_info: dict — regime 相關資訊
        strategy_names: list — 子策略名稱 (for ordering)

    Returns:
        RegimeBlendedReport
    """
    # 1. 累積報酬
    creturn = (1 + portfolio_returns).cumprod()
    creturn.name = 'Isaac V4.0'

    # 2. 績效指標
    n_days = len(portfolio_returns)
    if n_days < 10:
        stats = {
            'cagr': 0, 'max_drawdown': 0, 'daily_sharpe': 0,
            'sharpe': 0, 'win_ratio': 0, 'ann_vol': 0, 'n_days': 0,
        }
    else:
        total_return = float((1 + portfolio_returns).prod())
        n_years = n_days / 252
        cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        mdd = float(drawdown.min())

        std = portfolio_returns.std()
        daily_sharpe = float(portfolio_returns.mean() / std * np.sqrt(252)) if std > 0 else 0
        ann_vol = float(std * np.sqrt(252))
        win_ratio = float((portfolio_returns > 0).sum() / n_days)

        stats = {
            'cagr': round(float(cagr), 4),
            'max_drawdown': round(mdd, 4),
            'daily_sharpe': round(daily_sharpe, 3),
            'sharpe': round(daily_sharpe, 3),          # alias
            'win_ratio': round(win_ratio, 4),
            'ann_vol': round(ann_vol, 4),
            'n_days': n_days,
        }

    # 3. 合併交易
    all_trades = []
    for name, report in sub_reports.items():
        try:
            trades = report.get_trades()
            if trades is not None and len(trades) > 0:
                t = trades.copy()
                t['strategy'] = name
                all_trades.append(t)
        except Exception:
            continue

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        # 按進場日期排序
        date_col = None
        for col in ['entry_date', '進場日期', 'entry']:
            if col in trades_df.columns:
                date_col = col
                break
        if date_col:
            trades_df = trades_df.sort_values(date_col, ascending=False)
    else:
        trades_df = pd.DataFrame()

    # 4. 組裝
    return RegimeBlendedReport(
        creturn=creturn,
        benchmark=benchmark,
        trades_df=trades_df,
        stats_dict=stats,
        regime_info=regime_info,
    )
