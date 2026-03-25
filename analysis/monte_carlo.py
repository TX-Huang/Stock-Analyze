"""Monte Carlo 模擬 — 回測結果的統計驗證。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def run_monte_carlo(trades_df, n_simulations=1000, n_trades=None, initial_capital=1000000, seed=None):
    """
    用歷史交易的報酬分佈進行 Monte Carlo 模擬。

    Args:
        trades_df: DataFrame with 'return' column (per-trade return %)
        n_simulations: number of simulation paths
        n_trades: trades per path (default = same as historical)
        initial_capital: starting capital

    Returns:
        {
            'paths': np.array (n_simulations x n_trades),  # equity curves
            'final_values': np.array,
            'median_return': float,
            'percentile_5': float,   # worst 5% scenario
            'percentile_95': float,  # best 5% scenario
            'prob_profit': float,    # probability of profit
            'prob_double': float,    # probability of doubling capital
            'max_drawdowns': np.array,  # max drawdown per path
            'median_mdd': float,
            'worst_mdd': float,
        }
    """
    if trades_df is None or trades_df.empty or 'return' not in trades_df.columns:
        return None

    returns = trades_df['return'].dropna().values / 100  # Convert % to decimal
    if len(returns) < 10:
        return None

    if n_trades is None:
        n_trades = len(returns)

    if seed is not None:
        np.random.seed(seed)

    # Sample with replacement from historical returns
    sampled = np.random.choice(returns, size=(n_simulations, n_trades), replace=True)

    # Build equity curves
    cumulative = np.cumprod(1 + sampled, axis=1) * initial_capital
    paths = np.column_stack([np.full(n_simulations, initial_capital), cumulative])

    final_values = paths[:, -1]
    total_returns = (final_values / initial_capital - 1) * 100

    # Calculate max drawdown for each path
    max_drawdowns = np.zeros(n_simulations)
    for i in range(n_simulations):
        peak = np.maximum.accumulate(paths[i])
        dd = (paths[i] - peak) / peak
        max_drawdowns[i] = float(dd.min()) * 100

    return {
        'paths': paths,
        'final_values': final_values,
        'total_returns': total_returns,
        'median_return': float(np.median(total_returns)),
        'percentile_5': float(np.percentile(total_returns, 5)),
        'percentile_95': float(np.percentile(total_returns, 95)),
        'prob_profit': float(np.mean(final_values > initial_capital)),
        'prob_double': float(np.mean(final_values > initial_capital * 2)),
        'max_drawdowns': max_drawdowns,
        'median_mdd': float(np.median(max_drawdowns)),
        'worst_mdd': float(np.min(max_drawdowns)),
    }


def render_monte_carlo_chart(mc_result, title="Monte Carlo 模擬"):
    """Render Monte Carlo equity curves."""
    if mc_result is None:
        return None

    paths = mc_result['paths']
    n_sims = len(paths)
    n_trades = paths.shape[1]
    x = list(range(n_trades))

    fig = go.Figure()

    # Plot sample paths (50 random paths for visibility)
    sample_idx = np.random.choice(n_sims, min(50, n_sims), replace=False)
    for i in sample_idx:
        final_ret = (paths[i, -1] / paths[i, 0] - 1) * 100
        color = 'rgba(239,68,68,0.15)' if final_ret > 0 else 'rgba(34,197,94,0.15)'
        fig.add_trace(go.Scatter(
            x=x, y=paths[i], mode='lines',
            line=dict(width=0.5, color=color),
            showlegend=False, hoverinfo='skip',
        ))

    # Percentile bands
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig.add_trace(go.Scatter(x=x, y=p50, mode='lines',
                              line=dict(width=2, color='#00f0ff'),
                              name='中位數'))
    fig.add_trace(go.Scatter(x=x, y=p5, mode='lines',
                              line=dict(width=1, dash='dot', color='#ef4444'),
                              name='5th pctl (最差)'))
    fig.add_trace(go.Scatter(x=x, y=p95, mode='lines',
                              line=dict(width=1, dash='dot', color='#22c55e'),
                              name='95th pctl (最佳)'))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        title=dict(text=title, font=dict(size=14, color='#e2e8f0')),
        xaxis_title='交易次數',
        yaxis_title='資金 (NTD)',
        yaxis=dict(tickformat=',.0f'),
        margin=dict(t=55, l=60, r=20, b=40),
        font=dict(family='Noto Sans TC, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return fig


def render_monte_carlo_distribution(mc_result):
    """Render final returns distribution histogram."""
    if mc_result is None:
        return None

    returns = mc_result['total_returns']

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker_color='rgba(0,240,255,0.5)',
        marker_line=dict(width=0.5, color='rgba(0,240,255,0.8)'),
        name='報酬分佈',
    ))

    # Add vertical lines for percentiles
    fig.add_vline(x=mc_result['median_return'], line=dict(color='#00f0ff', width=2),
                  annotation_text=f"中位數 {mc_result['median_return']:.1f}%")
    fig.add_vline(x=mc_result['percentile_5'], line=dict(color='#ef4444', width=1, dash='dash'),
                  annotation_text=f"5th {mc_result['percentile_5']:.1f}%")
    fig.add_vline(x=0, line=dict(color='#64748b', width=1, dash='dot'))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        title=dict(text='最終報酬分佈', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='報酬率 (%)',
        yaxis_title='次數',
        margin=dict(t=40, l=60, r=20, b=40),
        font=dict(family='Noto Sans TC, sans-serif'),
        showlegend=False,
    )

    return fig
