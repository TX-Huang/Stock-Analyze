"""
交易品質深度分析模組
==================
提供三層分析 Feature 供策略迭代優化使用:
  Layer 1: 交易品質 (Efficiency, Edge Ratio, Profit Factor, Expectancy)
  Layer 2: 時間與市場環境 (Regime, Seasonality, Rolling Sharpe, DD Recovery)
  Layer 3: 連續性與穩定度 (Streaks, Underwater Days, Return Autocorrelation)
"""
import numpy as np
import pandas as pd


# ============================================================
# Layer 1: 交易品質分析
# ============================================================

def compute_trade_efficiency(trades):
    """
    Trade Efficiency = return / gmfe
    衡量你捕捉到多少最大有利波動。
    1.0 = 完美出場 (在最高點出場)
    < 0.5 = 出場太早，錯失大半利潤
    < 0 = 進場方向對但最終虧損 (沒守住利潤)
    """
    if trades.empty or 'gmfe' not in trades.columns:
        return pd.Series(dtype=float)
    gmfe = trades['gmfe'].replace(0, np.nan)
    efficiency = trades['return'] / gmfe
    return efficiency.clip(-2, 2)  # 限制極端值


def compute_edge_ratio(trades):
    """
    Edge Ratio = gmfe / abs(mae)
    每筆交易的風報邊際。
    > 1.0 = 有正期望值邊際
    < 1.0 = 風險大於報酬，信號品質差
    """
    if trades.empty or 'gmfe' not in trades.columns or 'mae' not in trades.columns:
        return pd.Series(dtype=float)
    abs_mae = trades['mae'].abs().replace(0, np.nan)
    return (trades['gmfe'] / abs_mae).clip(0, 20)


def compute_profit_factor(trades):
    """
    Profit Factor = 總獲利 / 總虧損
    > 1.5 = 優秀, > 2.0 = 卓越
    """
    if trades.empty or 'return' not in trades.columns:
        return 0.0
    wins = trades[trades['return'] > 0]['return'].sum()
    losses = abs(trades[trades['return'] <= 0]['return'].sum())
    return round(wins / losses, 3) if losses > 0 else float('inf')


def compute_expectancy(trades):
    """
    Expectancy = avg_win * win_rate - avg_loss * lose_rate
    每筆交易的期望報酬 (應為正值)
    """
    if trades.empty or 'return' not in trades.columns:
        return 0.0
    wins = trades[trades['return'] > 0]['return']
    losses = trades[trades['return'] <= 0]['return']
    n = len(trades)
    if n == 0:
        return 0.0
    win_rate = len(wins) / n
    lose_rate = len(losses) / n
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    return round(avg_win * win_rate - avg_loss * lose_rate, 6)


def compute_signal_attribution(trades):
    """
    按信號來源分析績效 (需要 trades 有 signal 欄位)。
    若無 signal 欄位，嘗試從 entry_sig_date 等欄位推斷。

    Returns: dict of {signal: {n_trades, win_rate, avg_return, profit_factor, edge_ratio}}
    """
    # 嘗試找信號欄位
    signal_col = None
    for col in trades.columns:
        if 'signal' in col.lower() or 'sig_type' in col.lower():
            signal_col = col
            break

    if signal_col is None:
        return None

    result = {}
    for sig, group in trades.groupby(signal_col):
        wins = group[group['return'] > 0]
        losses = group[group['return'] <= 0]
        pf = wins['return'].sum() / abs(losses['return'].sum()) if len(losses) > 0 and losses['return'].sum() != 0 else float('inf')
        er = compute_edge_ratio(group).median() if len(group) > 0 else 0

        result[str(sig)] = {
            'n_trades': len(group),
            'win_rate_pct': round(len(wins) / len(group) * 100, 1) if len(group) > 0 else 0,
            'avg_return_pct': round(group['return'].mean() * 100, 2),
            'profit_factor': round(pf, 2),
            'median_edge_ratio': round(er, 2),
            'avg_hold_days': round(group['period'].mean(), 1) if 'period' in group.columns else 0,
        }
    return result


# ============================================================
# Layer 2: 時間與市場環境
# ============================================================

def compute_market_regime_stats(trades, benchmark_series=None):
    """
    按大盤狀態分析每筆交易的績效。
    regime: bull (benchmark > MA60), bear (< MA60), sideways (within 2%)

    Args:
        trades: trades DataFrame with entry_date
        benchmark_series: 大盤收盤價 Series (index=datetime)

    Returns: dict of {regime: stats}
    """
    if trades.empty or benchmark_series is None or len(benchmark_series) == 0:
        return None

    ma60 = benchmark_series.rolling(60).mean()

    regimes = []
    for _, row in trades.iterrows():
        entry = row.get('entry_date')
        if pd.isna(entry):
            regimes.append('unknown')
            continue
        entry = pd.Timestamp(entry)
        # 找最近的日期
        idx = benchmark_series.index.searchsorted(entry)
        if idx >= len(benchmark_series):
            idx = len(benchmark_series) - 1
        bench_val = benchmark_series.iloc[idx]
        ma_val = ma60.iloc[idx] if idx < len(ma60) and not pd.isna(ma60.iloc[idx]) else bench_val

        if ma_val == 0 or pd.isna(ma_val):
            regimes.append('unknown')
        elif bench_val > ma_val * 1.02:
            regimes.append('bull')
        elif bench_val < ma_val * 0.98:
            regimes.append('bear')
        else:
            regimes.append('sideways')

    trades_with_regime = trades.copy()
    trades_with_regime['regime'] = regimes

    result = {}
    for regime, group in trades_with_regime.groupby('regime'):
        if regime == 'unknown':
            continue
        wins = group[group['return'] > 0]
        result[regime] = {
            'n_trades': len(group),
            'win_rate_pct': round(len(wins) / len(group) * 100, 1) if len(group) > 0 else 0,
            'avg_return_pct': round(group['return'].mean() * 100, 2),
            'profit_factor': compute_profit_factor(group),
            'avg_hold_days': round(group['period'].mean(), 1) if 'period' in group.columns else 0,
        }
    return result


def compute_monthly_seasonality(trades):
    """
    按月份統計交易勝率和平均報酬。
    Returns: DataFrame with columns [month, n_trades, win_rate, avg_return]
    """
    if trades.empty or 'entry_date' not in trades.columns:
        return pd.DataFrame()

    t = trades.copy()
    t['month'] = pd.to_datetime(t['entry_date']).dt.month

    rows = []
    for m in range(1, 13):
        group = t[t['month'] == m]
        if len(group) == 0:
            rows.append({'month': m, 'count': 0, 'win_rate': 0, 'avg_return': 0})
            continue
        wins = group[group['return'] > 0]
        rows.append({
            'month': m,
            'count': len(group),
            'win_rate': round(len(wins) / len(group) * 100, 1),
            'avg_return': round(group['return'].mean(), 6),
        })
    return pd.DataFrame(rows)


def compute_rolling_sharpe(equity, window=60):
    """
    滾動 Sharpe Ratio (預設 60 個交易日)。
    Returns: Series
    """
    if equity is None or len(equity) < window:
        return pd.Series(dtype=float)
    daily_ret = equity.pct_change().dropna()
    rolling_mean = daily_ret.rolling(window).mean()
    rolling_std = daily_ret.rolling(window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (rolling_mean / rolling_std * np.sqrt(252)).dropna()


def compute_drawdown_recovery(equity):
    """
    計算每次回撤的深度和恢復天數。
    Returns: list of {start, trough, end, depth_pct, recovery_days, total_days}
    """
    if equity is None or len(equity) < 2:
        return []

    peak = equity.cummax()
    dd = (equity - peak) / peak

    # 找回撤區間
    in_dd = False
    events = []
    current = {}

    for i in range(len(dd)):
        if dd.iloc[i] < -0.01 and not in_dd:
            in_dd = True
            current = {
                'start': equity.index[i],
                'start_idx': i,
                'trough_val': dd.iloc[i],
                'trough_date': equity.index[i],
            }
        elif in_dd:
            if dd.iloc[i] < current['trough_val']:
                current['trough_val'] = dd.iloc[i]
                current['trough_date'] = equity.index[i]
            if dd.iloc[i] >= 0:
                in_dd = False
                current['end'] = equity.index[i]
                current['depth_pct'] = round(current['trough_val'] * 100, 2)
                trough_idx = equity.index.get_loc(current['trough_date'])
                current['drawdown_days'] = (current['trough_date'] - current['start']).days
                current['recovery_days'] = (current['end'] - current['trough_date']).days
                current['total_days'] = (current['end'] - current['start']).days
                events.append({
                    'start': current['start'].strftime('%Y-%m-%d'),
                    'trough': current['trough_date'].strftime('%Y-%m-%d'),
                    'end': current['end'].strftime('%Y-%m-%d'),
                    'depth_pct': current['depth_pct'],
                    'drawdown_days': current['drawdown_days'],
                    'recovery_days': current['recovery_days'],
                    'total_days': current['total_days'],
                })

    # 按深度排序
    events.sort(key=lambda x: x['depth_pct'])
    return events


# ============================================================
# Layer 3: 連續性與穩定度
# ============================================================

def compute_streaks(trades):
    """
    計算連勝/連敗序列。
    Returns: {
        max_win_streak, max_loss_streak,
        avg_win_streak, avg_loss_streak,
        current_streak, current_streak_type
    }
    """
    if trades.empty or 'return' not in trades.columns:
        return {}

    wins = (trades['return'] > 0).astype(int).values

    # 計算所有連續序列
    win_streaks = []
    loss_streaks = []
    current = 1

    for i in range(1, len(wins)):
        if wins[i] == wins[i-1]:
            current += 1
        else:
            if wins[i-1] == 1:
                win_streaks.append(current)
            else:
                loss_streaks.append(current)
            current = 1

    # 最後一段
    if len(wins) > 0:
        if wins[-1] == 1:
            win_streaks.append(current)
        else:
            loss_streaks.append(current)

    current_type = 'win' if len(wins) > 0 and wins[-1] == 1 else 'loss'

    return {
        'max_win_streak': max(win_streaks) if win_streaks else 0,
        'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
        'avg_win_streak': round(np.mean(win_streaks), 1) if win_streaks else 0,
        'avg_loss_streak': round(np.mean(loss_streaks), 1) if loss_streaks else 0,
        'current_streak': current,
        'current_streak_type': current_type,
    }


def compute_underwater_days(equity):
    """
    計算最長水下天數 (equity 低於前高的連續天數)。
    Returns: {max_underwater_days, avg_underwater_days, current_underwater_days}
    """
    if equity is None or len(equity) < 2:
        return {}

    peak = equity.cummax()
    is_underwater = equity < peak

    # 找連續水下區間
    periods = []
    current = 0
    for i in range(len(is_underwater)):
        if is_underwater.iloc[i]:
            current += 1
        else:
            if current > 0:
                periods.append(current)
            current = 0

    # 如果最後仍在水下
    current_uw = current

    return {
        'max_underwater_days': max(periods) if periods else current_uw,
        'avg_underwater_days': round(np.mean(periods), 1) if periods else 0,
        'current_underwater_days': current_uw,
        'n_dd_periods': len(periods) + (1 if current_uw > 0 else 0),
    }


def compute_return_autocorrelation(trades, lags=None):
    """
    交易報酬的自相關係數。
    > 0 表示動量效應 (贏了還會贏)
    < 0 表示均值回歸 (贏了下次容易輸)
    ≈ 0 表示交易間獨立 (理想狀態)
    """
    if trades.empty or 'return' not in trades.columns or len(trades) < 20:
        return {}

    returns = trades['return'].values
    if lags is None:
        lags = [1, 2, 3, 5, 10]

    result = {}
    for lag in lags:
        if lag >= len(returns):
            continue
        x = returns[lag:]
        y = returns[:-lag]
        # 過濾 NaN/Inf
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(corr):
            result[f'lag_{lag}'] = round(corr, 4)
    return result


# ============================================================
# 整合: 完整分析報告
# ============================================================

def generate_full_analytics(report, benchmark_series=None):
    """
    從回測 report 產出完整交易分析報告。

    Args:
        report: FinLab backtest report object
        benchmark_series: 大盤收盤價 Series (optional, 自動載入 0050)

    Returns: dict with all analytics
    """
    trades = report.get_trades()
    equity = getattr(report, 'creturn', None)

    if benchmark_series is None:
        benchmark_series = getattr(report, 'benchmark', None)

    # 自動載入 0050 作為 benchmark
    if benchmark_series is None or (hasattr(benchmark_series, '__len__') and len(benchmark_series) == 0):
        try:
            import finlab
            from finlab import data as fdata
            bench_close = fdata.get('price:收盤價')
            if '0050' in bench_close.columns:
                benchmark_series = bench_close['0050'].dropna()
        except Exception:
            pass

    analytics = {}

    # --- Layer 1: 交易品質 ---
    analytics['trade_efficiency'] = {
        'values': compute_trade_efficiency(trades).describe().to_dict() if not trades.empty else {},
        'median': round(float(compute_trade_efficiency(trades).median()), 3) if not trades.empty and len(compute_trade_efficiency(trades).dropna()) > 0 else 0,
    }

    edge_ratios = compute_edge_ratio(trades)
    analytics['edge_ratio'] = {
        'values': edge_ratios.describe().to_dict() if not trades.empty else {},
        'median': round(float(edge_ratios.median()), 3) if not trades.empty and len(edge_ratios.dropna()) > 0 else 0,
        'pct_above_1': round(float((edge_ratios > 1).mean() * 100), 1) if len(edge_ratios.dropna()) > 0 else 0,
    }

    analytics['profit_factor'] = compute_profit_factor(trades)
    analytics['expectancy'] = compute_expectancy(trades)
    analytics['expectancy_pct'] = round(analytics['expectancy'] * 100, 3)

    signal_attr = compute_signal_attribution(trades)
    if signal_attr:
        analytics['signal_attribution'] = signal_attr

    # --- Layer 2: 時間與市場環境 ---
    analytics['market_regime'] = compute_market_regime_stats(trades, benchmark_series)
    analytics['monthly_seasonality'] = compute_monthly_seasonality(trades).to_dict('records')

    if equity is not None:
        rolling_sh = compute_rolling_sharpe(equity)
        analytics['rolling_sharpe_summary'] = {
            'mean': round(float(rolling_sh.mean()), 3) if len(rolling_sh) > 0 else 0,
            'min': round(float(rolling_sh.min()), 3) if len(rolling_sh) > 0 else 0,
            'max': round(float(rolling_sh.max()), 3) if len(rolling_sh) > 0 else 0,
            'pct_negative': round(float((rolling_sh < 0).mean() * 100), 1) if len(rolling_sh) > 0 else 0,
        }

        dd_events = compute_drawdown_recovery(equity)
        analytics['drawdown_recovery'] = dd_events[:10]  # Top 10 deepest
        if dd_events:
            recovery_days = [e['recovery_days'] for e in dd_events if e['recovery_days'] > 0]
            analytics['avg_recovery_days'] = round(np.mean(recovery_days), 1) if recovery_days else 0
            analytics['max_recovery_days'] = max(recovery_days) if recovery_days else 0

    # --- Layer 3: 連續性與穩定度 ---
    analytics['streaks'] = compute_streaks(trades)
    if equity is not None:
        analytics['underwater'] = compute_underwater_days(equity)
    analytics['return_autocorrelation'] = compute_return_autocorrelation(trades)

    return analytics
