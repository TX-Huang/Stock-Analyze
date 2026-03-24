"""績效歸因分析 — 各持股對組合損益的貢獻。"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def calculate_attribution(positions, provider):
    """
    計算各持股對今日組合損益的貢獻。

    Args:
        positions: list of position dicts with 'ticker', 'name', 'shares', 'entry_price'
        provider: DataProvider for current prices

    Returns:
        list of dicts sorted by contribution (largest absolute first)
    """
    results = []
    total_pnl = 0

    for pos in positions:
        ticker = pos.get('ticker', '')
        entry_price = pos.get('entry_price', 0)
        shares = pos.get('shares', 0)
        name = pos.get('name', ticker)

        if not ticker or entry_price <= 0 or shares <= 0:
            continue

        try:
            df = provider.get_historical_data(str(ticker), period="5d", interval="1d")
            if df is None or df.empty or len(df) < 2:
                continue

            current = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2])

            # Today's P&L for this position
            daily_pnl = (current - prev) * shares
            # Total unrealized P&L
            total_unrealized = (current - entry_price) * shares
            # Return %
            daily_return_pct = (current - prev) / prev * 100
            total_return_pct = (current - entry_price) / entry_price * 100

            total_pnl += daily_pnl

            results.append({
                'ticker': ticker,
                'name': name,
                'shares': shares,
                'entry_price': entry_price,
                'current_price': current,
                'prev_close': prev,
                'daily_pnl': daily_pnl,
                'daily_return_pct': daily_return_pct,
                'total_unrealized': total_unrealized,
                'total_return_pct': total_return_pct,
                'position_value': current * shares,
            })
        except Exception:
            continue

    # Calculate contribution weight
    for r in results:
        r['contribution_pct'] = (r['daily_pnl'] / abs(total_pnl) * 100) if total_pnl != 0 else 0

    # Sort by absolute daily P&L (largest impact first)
    results.sort(key=lambda r: abs(r['daily_pnl']), reverse=True)

    return results, total_pnl


def render_attribution_chart(attribution_results):
    """Render waterfall chart of P&L attribution."""
    if not attribution_results:
        return None

    names = [r['name'] for r in attribution_results]
    pnls = [r['daily_pnl'] for r in attribution_results]
    colors = ['#ef4444' if p >= 0 else '#22c55e' for p in pnls]  # Taiwan: red up, green down

    fig = go.Figure(go.Bar(
        x=names,
        y=pnls,
        marker_color=colors,
        text=[f"{p:+,.0f}" for p in pnls],
        textposition='outside',
        textfont=dict(size=10, family='JetBrains Mono, monospace'),
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        title=dict(text='今日損益歸因', font=dict(size=14, color='#e2e8f0')),
        yaxis=dict(title='損益 (NTD)', tickformat=',.0f'),
        xaxis=dict(title=''),
        margin=dict(t=40, l=60, r=20, b=40),
        font=dict(family='Noto Sans TC, sans-serif'),
        showlegend=False,
    )

    return fig
