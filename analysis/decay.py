"""策略衰退偵測 — 監控 live 績效是否偏離回測預期。"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_strategy_decay(live_trades, backtest_stats, window=20):
    """
    比較近期實際交易績效與回測預期，偵測策略衰退。

    Args:
        live_trades: list of dicts with 'pnl_pct' (近期實際交易)
        backtest_stats: dict with 'cagr', 'win_ratio', 'daily_sharpe', 'max_drawdown'
        window: 用幾筆交易來評估

    Returns:
        {
            'is_decaying': bool,
            'decay_score': int (0-5, higher=worse),
            'signals': list of str,
            'metrics': {
                'live_win_rate': float,
                'expected_win_rate': float,
                'live_avg_return': float,
                'live_consecutive_losses': int,
                'live_sharpe': float,
            }
        }
    """
    if not live_trades or len(live_trades) < 5:
        return {
            'is_decaying': False,
            'decay_score': 0,
            'signals': ['交易筆數不足，無法評估'],
            'metrics': {},
        }

    recent = live_trades[-window:] if len(live_trades) > window else live_trades
    pnls = [t.get('pnl_pct', 0) for t in recent]

    # Live metrics
    wins = sum(1 for p in pnls if p > 0)
    live_win_rate = wins / len(pnls) * 100
    live_avg_return = np.mean(pnls)
    live_std = float(np.std(pnls)) if len(pnls) > 1 else 0
    # Annualize using actual sample size when fewer than 252 trades available
    live_sharpe = (live_avg_return / live_std * np.sqrt(min(len(pnls), 252))) if live_std > 0.001 else 0

    # Consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in pnls:
        if p < 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Expected metrics from backtest
    expected_win = backtest_stats.get('win_ratio', 0.5) * 100
    expected_sharpe = backtest_stats.get('daily_sharpe', 1.0)

    # Decay signals
    score = 0
    signals = []

    # 1. Win rate significantly below expected
    win_diff = expected_win - live_win_rate
    if win_diff > 20:
        score += 2
        signals.append(f"勝率大幅低於預期: {live_win_rate:.1f}% vs 回測 {expected_win:.1f}%")
    elif win_diff > 10:
        score += 1
        signals.append(f"勝率低於預期: {live_win_rate:.1f}% vs 回測 {expected_win:.1f}%")

    # 2. Consecutive losses
    if max_consec_loss >= 8:
        score += 2
        signals.append(f"連續虧損 {max_consec_loss} 筆（異常）")
    elif max_consec_loss >= 5:
        score += 1
        signals.append(f"連續虧損 {max_consec_loss} 筆")

    # 3. Average return negative
    if live_avg_return < -2:
        score += 1
        signals.append(f"平均報酬為負: {live_avg_return:.2f}%")

    # 4. Sharpe ratio collapse
    if expected_sharpe > 0.5 and live_sharpe < 0:
        score += 1
        signals.append(f"Sharpe 崩潰: {live_sharpe:.2f} vs 回測 {expected_sharpe:.2f}")

    if not signals:
        signals.append("策略表現正常")

    return {
        'is_decaying': score >= 3,
        'decay_score': min(score, 5),
        'signals': signals,
        'metrics': {
            'live_win_rate': live_win_rate,
            'expected_win_rate': expected_win,
            'live_avg_return': live_avg_return,
            'live_consecutive_losses': max_consec_loss,
            'live_sharpe': live_sharpe,
        }
    }
