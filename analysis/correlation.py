"""持股相關性分析。"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def calculate_correlation_matrix(tickers, provider, period="6mo"):
    """
    計算多檔股票的相關性矩陣。

    Returns:
        pd.DataFrame: correlation matrix (N x N)
        or None if insufficient data
    """
    returns_dict = {}

    for ticker in tickers:
        try:
            df = provider.get_historical_data(str(ticker), period=period, interval="1d")
            if df is not None and not df.empty and len(df) >= 30:
                returns_dict[str(ticker)] = df['Close'].pct_change().dropna()
        except Exception:
            continue

    if len(returns_dict) < 2:
        return None

    # Align dates
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    if len(returns_df) < 20:
        return None

    return returns_df.corr()


def render_correlation_heatmap(corr_matrix, title="持股相關性矩陣"):
    """Render correlation matrix as Plotly heatmap."""
    tickers = list(corr_matrix.columns)

    # Mask diagonal
    mask = np.eye(len(tickers), dtype=bool)
    values = corr_matrix.values.copy()

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=tickers,
        y=tickers,
        colorscale=[
            [0, '#22c55e'],      # -1: green (negative corr = good diversification)
            [0.5, '#1e293b'],    # 0: dark
            [1, '#ef4444'],      # +1: red (high correlation = concentration risk)
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in values],
        texttemplate="%{text}",
        textfont=dict(size=11, family='JetBrains Mono, monospace'),
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="相關係數", side="right"),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#e2e8f0')),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(350, len(tickers) * 45),
        margin=dict(t=40, l=10, r=10, b=10),
        font=dict(family='Noto Sans TC, sans-serif'),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
    )

    return fig


def get_concentration_risk(corr_matrix, threshold=0.7):
    """
    分析持股集中度風險。

    Returns:
        {
            'high_corr_pairs': [(t1, t2, corr), ...],  # 高相關配對
            'avg_correlation': float,
            'max_correlation': float,
            'risk_level': str,  # 'low', 'medium', 'high'
            'suggestion': str,
        }
    """
    tickers = list(corr_matrix.columns)
    n = len(tickers)

    high_pairs = []
    all_corrs = []

    for i in range(n):
        for j in range(i+1, n):
            c = float(corr_matrix.iloc[i, j])
            all_corrs.append(c)
            if c >= threshold:
                high_pairs.append((tickers[i], tickers[j], c))

    high_pairs.sort(key=lambda x: x[2], reverse=True)
    avg_corr = float(np.mean(all_corrs)) if all_corrs else 0
    max_corr = float(np.max(all_corrs)) if all_corrs else 0

    if len(high_pairs) >= 3 or avg_corr > 0.6:
        risk = 'high'
        suggestion = '持股高度相關，建議增加不同產業的標的以分散風險。'
    elif len(high_pairs) >= 1 or avg_corr > 0.4:
        risk = 'medium'
        suggestion = '部分持股相關性偏高，可考慮調整配置。'
    else:
        risk = 'low'
        suggestion = '持股分散度良好。'

    return {
        'high_corr_pairs': high_pairs,
        'avg_correlation': avg_corr,
        'max_correlation': max_corr,
        'risk_level': risk,
        'suggestion': suggestion,
    }
