"""板塊熱力圖 — 產業族群輪動分析。"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ui.components import cyber_header, cyber_spinner


# Taiwan stock sectors
TW_SECTORS = {
    '半導體': ['2330', '2303', '2454', '3711', '2379', '6770', '3034', '2408'],
    '電子零組件': ['2317', '2382', '3231', '2327', '6285'],
    '光電': ['3008', '2409', '3481', '2393'],
    'AI/雲端': ['2345', '3037', '6669', '4977', '2376'],
    '金融': ['2881', '2882', '2884', '2886', '2891', '2892'],
    '傳產': ['1301', '1303', '1326', '2002', '2105'],
    '航運': ['2603', '2609', '2615', '5765'],
    '生技': ['4743', '6446', '4142', '4726'],
    '營建': ['2504', '2520', '2542', '5534'],
    '觀光/食品': ['2723', '1216', '1227', '2912'],
}


def render(_embedded=False):
    if not _embedded:
        cyber_header("板塊熱力圖", "產業族群輪動 | 強弱對比")

    # Period selector
    period_map = {'1日': '1d', '5日': '5d', '1月': '1mo', '3月': '3mo'}
    period_label = st.radio("期間", list(period_map.keys()), horizontal=True, index=1)
    period = period_map[period_label]

    try:
        from data.provider import get_data_provider
        provider = get_data_provider("auto", market_type="TW")
    except Exception as e:
        st.error(f"無法初始化資料源: {e}")
        return

    # Fetch returns for all sectors
    with cyber_spinner("LOADING", "板塊數據載入中..."):
        sector_returns = {}
        stock_returns = {}

        for sector, tickers in TW_SECTORS.items():
            returns = []
            for ticker in tickers:
                try:
                    df = provider.get_historical_data(ticker, period="3mo", interval="1d")
                    if df is not None and not df.empty and len(df) >= 2:
                        if period == '1d':
                            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                        elif period == '5d':
                            lookback = min(5, len(df) - 1)
                            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback-1] - 1) * 100
                        elif period == '1mo':
                            lookback = min(20, len(df) - 1)
                            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback-1] - 1) * 100
                        else:  # 3mo
                            ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

                        returns.append(float(ret))
                        stock_returns[f"{ticker}"] = {
                            'sector': sector,
                            'return': float(ret),
                            'price': float(df['Close'].iloc[-1]),
                        }
                except Exception:
                    continue

            if returns:
                sector_returns[sector] = np.mean(returns)

        if not sector_returns:
            st.warning("無法取得板塊數據")
            return

    # Build treemap
    sectors = list(sector_returns.keys())
    values = [abs(sector_returns[s]) + 0.1 for s in sectors]  # treemap needs positive values
    colors = [sector_returns[s] for s in sectors]
    texts = [f"{s}<br>{sector_returns[s]:+.2f}%" for s in sectors]

    fig = go.Figure(go.Treemap(
        labels=sectors,
        parents=[""] * len(sectors),
        values=values,
        text=texts,
        textinfo="text",
        marker=dict(
            colors=colors,
            colorscale=[[0, '#22c55e'], [0.5, '#1e293b'], [1, '#ef4444']],  # 綠跌紅漲
            cmid=0,
            line=dict(width=2, color='#0f172a'),
        ),
        textfont=dict(size=14, family='Noto Sans TC, sans-serif'),
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(t=10, l=10, r=10, b=10),
        font=dict(family='Noto Sans TC, sans-serif'),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sector ranking table
    st.markdown('<p class="sec-header">板塊排名</p>', unsafe_allow_html=True)
    sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)

    from ui.theme import _tw_color
    rows_html = "".join(
        f'<tr><td style="text-align:left;font-weight:700">{i+1}. {s}</td>'
        f'<td style="text-align:right">{_tw_color(r, "+.2f")}%</td>'
        f'<td style="text-align:right;color:#64748b">{len(TW_SECTORS[s])} 檔</td></tr>'
        for i, (s, r) in enumerate(sorted_sectors)
    )

    st.markdown(
        f'<table class="pro-table"><thead><tr>'
        f'<th style="text-align:left">板塊</th><th>漲跌幅</th><th>成分股</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )
