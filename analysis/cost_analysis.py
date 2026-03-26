"""交易成本分析 — 手續費、證交稅、滑價估算。"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

# ─── Taiwan Stock Trading Cost Constants ───

COMMISSION_RATE = 0.001425       # 法定手續費率 0.1425%
DISCOUNT_RATE = 0.35             # 線上券商折扣 (約 3.5 折)
COMMISSION_RATE_DISCOUNTED = COMMISSION_RATE * DISCOUNT_RATE  # ~0.05%
TAX_RATE_STOCK = 0.003           # 證交稅 0.3% (一般股票，賣方)
TAX_RATE_ETF_DAYTRADE = 0.001   # 證交稅 0.1% (ETF 當沖)
SLIPPAGE_PER_SIDE = 0.001       # 估計滑價 0.1% per side
DEFAULT_CAPITAL_PER_TRADE = 1_000_000  # 預設每筆交易金額 NTD


# ─── Plotly Theme Helper ───

def _apply_dark_theme(fig):
    """Apply standard plotly_dark theme to figure."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Noto Sans TC, sans-serif'),
    )
    return fig


# ─── Per-Trade Cost Calculation ───

def _estimate_trade_cost(
    entry_price,
    exit_price,
    capital=DEFAULT_CAPITAL_PER_TRADE,
    commission_rate=COMMISSION_RATE_DISCOUNTED,
    tax_rate=TAX_RATE_STOCK,
    slippage_rate=SLIPPAGE_PER_SIDE,
):
    """
    Estimate costs for a single round-trip trade.

    Returns:
        dict with commission_buy, commission_sell, tax, slippage, total_cost,
        cost_pct (as fraction of capital).
    """
    if entry_price <= 0 or exit_price <= 0 or capital <= 0:
        return {
            'commission_buy': 0.0,
            'commission_sell': 0.0,
            'tax': 0.0,
            'slippage': 0.0,
            'total_cost': 0.0,
            'cost_pct': 0.0,
        }

    shares = capital / entry_price

    commission_buy = capital * commission_rate
    sell_value = shares * exit_price
    commission_sell = sell_value * commission_rate

    # 證交稅 only on sell side
    tax = sell_value * tax_rate

    # Slippage on both sides
    slippage = capital * slippage_rate + sell_value * slippage_rate

    total_cost = commission_buy + commission_sell + tax + slippage
    cost_pct = total_cost / capital if capital > 0 else 0.0

    return {
        'commission_buy': round(commission_buy, 2),
        'commission_sell': round(commission_sell, 2),
        'tax': round(tax, 2),
        'slippage': round(slippage, 2),
        'total_cost': round(total_cost, 2),
        'cost_pct': round(cost_pct, 6),
    }


# ─── Main Analysis ───

def analyze_trading_costs(trades_df, capital=DEFAULT_CAPITAL_PER_TRADE):
    """
    Analyze trading costs from a DataFrame of closed trades.

    Args:
        trades_df: DataFrame with columns like 'entry_price', 'exit_price',
                   'stock_id', 'return', 'period'
                   (This is the format from FinLab's report.get_trades())

    Returns:
        dict with:
        - total_trades: int
        - total_commission: float (estimated total commission NTD)
        - total_tax: float (estimated total tax NTD)
        - total_cost: float (commission + tax + slippage)
        - avg_cost_per_trade_pct: float (average cost as % of trade value)
        - cost_drag_annualized: float (annualized cost drag %)
        - gross_return: float (return before costs)
        - net_return: float (return after costs)
        - cost_ratio: float (costs as % of gross profit)
        - turnover_rate: float (annualized turnover)
        - avg_holding_days: float
        - by_trade: list of per-trade cost breakdowns
    """
    empty_result = {
        'total_trades': 0,
        'total_commission': 0.0,
        'total_tax': 0.0,
        'total_slippage': 0.0,
        'total_cost': 0.0,
        'avg_cost_per_trade_pct': 0.0,
        'cost_drag_annualized': 0.0,
        'gross_return': 0.0,
        'net_return': 0.0,
        'cost_ratio': 0.0,
        'turnover_rate': 0.0,
        'avg_holding_days': 0.0,
        'by_trade': [],
    }

    # ── Edge case: empty or None ──
    if trades_df is None or not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
        logger.warning("analyze_trading_costs: empty or invalid trades DataFrame")
        return empty_result

    df = trades_df.copy()
    n_trades = len(df)

    # ── Resolve entry / exit prices ──
    has_entry = 'entry_price' in df.columns
    has_exit = 'exit_price' in df.columns

    if not has_entry:
        df['entry_price'] = capital  # use 1:1 proxy
    if not has_exit:
        if 'return' in df.columns:
            df['exit_price'] = df['entry_price'] * (1 + df['return'])
        else:
            df['exit_price'] = df['entry_price']

    # ── Holding period (days) ──
    if 'period' in df.columns:
        df['holding_days'] = pd.to_numeric(df['period'], errors='coerce').fillna(0)
    elif 'entry_date' in df.columns and 'exit_date' in df.columns:
        df['holding_days'] = (
            pd.to_datetime(df['exit_date']) - pd.to_datetime(df['entry_date'])
        ).dt.days
    else:
        df['holding_days'] = 0

    # ── Per-trade cost breakdown ──
    by_trade = []
    total_commission = 0.0
    total_tax = 0.0
    total_slippage = 0.0
    total_cost = 0.0

    for idx, row in df.iterrows():
        entry_p = float(row.get('entry_price', DEFAULT_CAPITAL_PER_TRADE))
        exit_p = float(row.get('exit_price', entry_p))

        cost = _estimate_trade_cost(entry_p, exit_p, capital=capital)
        cost['stock_id'] = row.get('stock_id', 'N/A')
        cost['entry_date'] = str(row.get('entry_date', ''))
        cost['holding_days'] = float(row.get('holding_days', 0))
        cost['entry_price'] = entry_p
        cost['exit_price'] = exit_p

        by_trade.append(cost)
        total_commission += cost['commission_buy'] + cost['commission_sell']
        total_tax += cost['tax']
        total_slippage += cost['slippage']
        total_cost += cost['total_cost']

    # ── Aggregate metrics ──
    avg_cost_pct = np.mean([t['cost_pct'] for t in by_trade]) if by_trade else 0.0
    avg_holding = df['holding_days'].mean() if 'holding_days' in df.columns else 0.0
    avg_holding = float(avg_holding) if not np.isnan(avg_holding) else 0.0

    # Gross return (from 'return' column or computed)
    if 'return' in df.columns:
        gross_returns = pd.to_numeric(df['return'], errors='coerce').fillna(0)
        gross_return = float(gross_returns.sum())
    else:
        gross_return = 0.0

    # Net return = gross minus cost drag
    total_cost_pct = sum(t['cost_pct'] for t in by_trade)
    net_return = gross_return - total_cost_pct

    # Cost ratio (costs as % of gross profit)
    gross_profit_ntd = gross_return * capital
    cost_ratio = (total_cost / gross_profit_ntd) if gross_profit_ntd > 0 else 0.0

    # Annualized turnover
    total_days = df['holding_days'].sum()
    total_days = float(total_days) if not np.isnan(total_days) else 0.0
    trading_years = total_days / 252 if total_days > 0 else 1.0
    turnover_rate = n_trades / trading_years if trading_years > 0 else 0.0

    # Annualized cost drag
    cost_drag_annualized = avg_cost_pct * turnover_rate if turnover_rate > 0 else 0.0

    return {
        'total_trades': n_trades,
        'total_commission': round(total_commission, 2),
        'total_tax': round(total_tax, 2),
        'total_slippage': round(total_slippage, 2),
        'total_cost': round(total_cost, 2),
        'avg_cost_per_trade_pct': round(avg_cost_pct * 100, 4),
        'cost_drag_annualized': round(cost_drag_annualized * 100, 4),
        'gross_return': round(gross_return, 6),
        'net_return': round(net_return, 6),
        'cost_ratio': round(cost_ratio, 4),
        'turnover_rate': round(turnover_rate, 2),
        'avg_holding_days': round(avg_holding, 1),
        'by_trade': by_trade,
    }


# ─── Charts ───

def render_cost_chart(cost_result):
    """
    Render a Plotly pie chart showing cost breakdown:
    commission vs tax vs slippage.

    Args:
        cost_result: dict returned by analyze_trading_costs()

    Returns:
        plotly Figure or None if no cost data.
    """
    if not cost_result or cost_result.get('total_cost', 0) == 0:
        logger.info("render_cost_chart: no cost data to display")
        return None

    labels = ['手續費 (Commission)', '證交稅 (Tax)', '滑價 (Slippage)']
    values = [
        cost_result['total_commission'],
        cost_result['total_tax'],
        cost_result['total_slippage'],
    ]

    # Filter out zero categories
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not filtered:
        return None

    labels, values = zip(*filtered)

    fig = go.Figure(data=[go.Pie(
        labels=list(labels),
        values=list(values),
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=['#636EFA', '#EF553B', '#FFA15A'],
            line=dict(color='rgba(255,255,255,0.3)', width=1),
        ),
    )])

    fig.update_layout(
        title=dict(
            text='交易成本結構 (Cost Breakdown)',
            x=0.5,
            font=dict(size=16),
        ),
        annotations=[dict(
            text=f'總成本<br>NT${cost_result["total_cost"]:,.0f}',
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False,
        )],
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
    )

    _apply_dark_theme(fig)
    return fig


def render_cost_over_time(cost_result):
    """
    Render cumulative cost drag over time as a line chart.

    Args:
        cost_result: dict returned by analyze_trading_costs()

    Returns:
        plotly Figure or None if no trade data.
    """
    by_trade = cost_result.get('by_trade', [])
    if not by_trade:
        logger.info("render_cost_over_time: no trade data to display")
        return None

    # Build cumulative cost series
    cum_cost = []
    cum_cost_pct = []
    x_values = []
    running_cost = 0.0
    running_cost_pct = 0.0

    for i, t in enumerate(by_trade):
        running_cost += t['total_cost']
        running_cost_pct += t['cost_pct'] * 100  # convert to %
        cum_cost.append(running_cost)
        cum_cost_pct.append(running_cost_pct)
        x_values.append(t.get('entry_date', ''))

    # Use dates for x-axis if available, fallback to trade numbers
    has_dates = x_values and x_values[0] and x_values[0] != ''
    if has_dates:
        try:
            import pandas as _pd
            x_axis = _pd.to_datetime(x_values)
            x_title = '交易日期'
        except Exception:
            x_axis = list(range(1, len(by_trade) + 1))
            x_title = '交易序號 (Trade #)'
    else:
        x_axis = list(range(1, len(by_trade) + 1))
        x_title = '交易序號 (Trade #)'

    fig = go.Figure()

    # Cumulative cost NTD (left y-axis)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=cum_cost,
        mode='lines',
        name='累計成本 (NTD)',
        line=dict(color='#ef4444', width=2),
        yaxis='y',
    ))

    # Cumulative cost % (right y-axis)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=cum_cost_pct,
        mode='lines',
        name='累計成本拖累 (%)',
        line=dict(color='#f59e0b', width=2, dash='dot'),
        yaxis='y2',
    ))

    fig.update_layout(
        title=dict(
            text='累計交易成本趨勢',
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title=x_title),
        yaxis=dict(
            title=dict(text='累計成本 (NTD)', font=dict(color='#ef4444')),
            tickfont=dict(color='#ef4444'),
            side='left',
        ),
        yaxis2=dict(
            title=dict(text='累計成本拖累 (%)', font=dict(color='#f59e0b')),
            tickfont=dict(color='#f59e0b'),
            overlaying='y',
            side='right',
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        margin=dict(l=10, r=60, t=60, b=60),
    )

    _apply_dark_theme(fig)
    return fig
