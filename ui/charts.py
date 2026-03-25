import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import graphviz

from analysis.trend import calculate_pattern_convergence
from utils.helpers import validate_ticker, get_date_from_index


def render_supply_chain_graph(keyword, structure, market):
    if not structure:
        return
    try:
        dot = graphviz.Digraph(comment=keyword)
        dot.attr(rankdir='LR')
        dot.attr('node', fontname='Noto Sans CJK TC')
        dot.node('ROOT', keyword, shape='doubleoctagon', style='filled', fillcolor='#f3f4f6', fontcolor='#111827', fontsize='16')
        for part, tickers in structure.items():
            part_id = f"PART_{part}"
            dot.node(part_id, part, shape='box', style='filled', fillcolor='#dbeafe', fontcolor='#1e40af')
            dot.edge('ROOT', part_id)
            ticker_iter = tickers.items() if isinstance(tickers, dict) else [(t, t) for t in tickers]
            for t, t_name in ticker_iter:
                if not validate_ticker(t, market):
                    continue
                t_clean = t.replace(".TW", "").replace(".TWO", "")
                name = t_name if t_name != t else st.session_state.dynamic_name_map.get(t_clean, t_clean)
                stock_label = f"{name}\n({t_clean})"
                stock_id = f"STOCK_{t_clean}"
                dot.node(stock_id, stock_label, shape='ellipse', style='filled', fillcolor='#f9fafb', fontcolor='#374151')
                dot.edge(part_id, stock_id)
        st.graphviz_chart(dot)
    except Exception as e:
        st.warning("⚠️ 無法繪製供應鏈圖 (可能是電腦未安裝 Graphviz 軟體)，改為顯示文字清單：")
        st.write(structure)


def render_trend_chart(df, patterns, market, is_box=False, height=600, is_weekly=False, candle_patterns=None, entry_price=None):
    try:
        rows = 2

        if "台股" in market:
            ma_s = 'MA5'; ma_l = 'MA20'; s_win = 5; l_win = 20
        else:
            ma_s = 'MA20'; ma_l = 'MA50'; s_win = 20; l_win = 50
        df[ma_s] = df['Close'].rolling(s_win).mean()
        df[ma_l] = df['Close'].rolling(l_win).mean()

        n = 10
        df['peaks'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
        df['troughs'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2], subplot_titles=("價格與壓力/支撐", "成交量"))

        # Taiwan standard: Red for Up (increasing), Green for Down (decreasing)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'), row=1, col=1)

        if st.session_state.chart_settings.get('ma', True):
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_s], line=dict(color='orange', width=1), name=f'{ma_s}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_l], line=dict(color='blue', width=1), name=f'{ma_l}'), row=1, col=1)

        if st.session_state.chart_settings.get('bbands', False) and 'BB_Upper' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB上軌'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)', name='BB下軌'), row=1, col=1)

        y_min = df['Low'].min() * 0.95
        y_max = df['High'].max() * 1.05
        fig.update_yaxes(range=[y_min, y_max], fixedrange=False, row=1, col=1)

        # Draw Support and Resistance lines
        if st.session_state.chart_settings.get('trendline', True):
            peaks, troughs = df['peaks'].dropna(), df['troughs'].dropna()

            zone_data = calculate_pattern_convergence(df, peaks, troughs)
            if zone_data:
                z_start_date = get_date_from_index(zone_data['x_zone_start'], df, is_weekly)
                z_end_date = get_date_from_index(zone_data['x_zone_end'], df, is_weekly)
                apex_date = get_date_from_index(zone_data['x_int'], df, is_weekly)

                fig.add_vrect(
                    x0=z_start_date, x1=z_end_date,
                    fillcolor="rgba(255, 165, 0, 0.2)", layer="below", line_width=0,
                    annotation_text="轉折熱區", annotation_position="top left"
                )

                if zone_data['x_int'] > len(df) * 0.5:
                    fig.add_trace(go.Scatter(
                        x=[apex_date], y=[zone_data['y_int']],
                        mode='markers', marker=dict(color="purple", size=8, symbol="star"),
                        name='預期收斂點 (Apex)'
                    ), row=1, col=1)

            if len(peaks) >= 2:
                p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]
                p1_val, p2_val = peaks.iloc[-2], peaks.iloc[-1]
                fig.add_trace(go.Scatter(x=[p1_idx, p2_idx], y=[p1_val, p2_val], mode='lines', line=dict(color="#22c55e", width=1.5, dash="dash"), name='壓力線'), row=1, col=1)
                x1, x2 = df.index.get_loc(p1_idx), df.index.get_loc(p2_idx)
                if x2 != x1:
                    slope = (p2_val - p1_val) / (x2 - x1)
                    end_idx_for_line = max(int(zone_data['x_int']) + 2 if zone_data else len(df)-1, len(df)-1)
                    end_date = get_date_from_index(end_idx_for_line, df, is_weekly)
                    proj = p2_val + slope * (end_idx_for_line - x2)
                    fig.add_trace(go.Scatter(x=[p2_idx, end_date], y=[p2_val, proj], mode='lines', line=dict(color="#22c55e", width=1, dash="dot"), name='壓力線延伸'), row=1, col=1)

            if len(troughs) >= 2:
                t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]
                t1_val, t2_val = troughs.iloc[-2], troughs.iloc[-1]
                fig.add_trace(go.Scatter(x=[t1_idx, t2_idx], y=[t1_val, t2_val], mode='lines', line=dict(color="#ef4444", width=1.5, dash="dash"), name='支撐線'), row=1, col=1)
                x1, x2 = df.index.get_loc(t1_idx), df.index.get_loc(t2_idx)
                if x2 != x1:
                    slope = (t2_val - t1_val) / (x2 - x1)
                    end_idx_for_line = max(int(zone_data['x_int']) + 2 if zone_data else len(df)-1, len(df)-1)
                    end_date = get_date_from_index(end_idx_for_line, df, is_weekly)
                    proj = t2_val + slope * (end_idx_for_line - x2)
                    fig.add_trace(go.Scatter(x=[t2_idx, end_date], y=[t2_val, proj], mode='lines', line=dict(color="#ef4444", width=1, dash="dot"), name='支撐線延伸'), row=1, col=1)

        # Gap Visualization (跳空) — 向量化偵測 + 批次繪製
        if st.session_state.chart_settings.get('gaps', True):
            prev_high = df['High'].shift(1)
            prev_low = df['Low'].shift(1)

            # 向上跳空: 今低 > 昨高 × 1.005 (紅)
            gap_up = df['Low'] > prev_high * 1.005
            for idx in df.index[gap_up]:
                i = df.index.get_loc(idx)
                fig.add_shape(type="rect", x0=df.index[i-1], x1=idx,
                              y0=prev_high.iloc[i], y1=df['Low'].iloc[i],
                              line=dict(width=0), fillcolor="rgba(239, 68, 68, 0.3)", row=1, col=1)

            # 向下跳空: 今高 < 昨低 × 0.995 (綠)
            gap_dn = df['High'] < prev_low * 0.995
            for idx in df.index[gap_dn]:
                i = df.index.get_loc(idx)
                fig.add_shape(type="rect", x0=df.index[i-1], x1=idx,
                              y0=df['High'].iloc[i], y1=prev_low.iloc[i],
                              line=dict(width=0), fillcolor="rgba(34, 197, 94, 0.3)", row=1, col=1)

        # Candle Patterns Annotations
        if st.session_state.chart_settings.get('candle_patterns', True) and candle_patterns:
            annotation_counts = {}
            for p in candle_patterns:
                date = p['date']
                is_bearish = p['type'] == 'Bearish'

                key = (date, is_bearish)
                count = annotation_counts.get(key, 0)
                annotation_counts[key] = count + 1

                y_val = df.loc[date, 'High'] * 1.02 if is_bearish else df.loc[date, 'Low'] * 0.98

                font_color = "#22c55e" if is_bearish else "#ef4444"  # 台股: 看跌=綠, 看漲=紅
                clean_name = p['name']

                if len(p.get('points', [])) > 1:
                    offset_y = (count * 20) if not is_bearish else (-count * 20)
                    fig.add_annotation(
                        x=date, y=y_val,
                        yshift=offset_y,
                        text=f"「{clean_name}」",
                        showarrow=False,
                        font=dict(color=font_color, size=12, weight="bold"),
                        row=1, col=1
                    )
                else:
                    ay_val = (-30 - count*25) if is_bearish else (30 + count*25)
                    fig.add_annotation(
                        x=date, y=y_val,
                        text=clean_name,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor=font_color,
                        ax=0, ay=ay_val,
                        font=dict(color=font_color, size=12, weight="bold"),
                        row=1, col=1
                    )

        # Entry Price Line (for position charts)
        if entry_price is not None and entry_price > 0:
            fig.add_hline(
                y=entry_price, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                annotation_text=f"Entry: {entry_price:,.1f}",
                annotation_position="right",
                annotation_font=dict(color="#f59e0b", size=10),
                row=1, col=1,
            )

        # Volume
        colors = np.where(df['Close'].values >= df['Open'].values, '#ef4444', '#22c55e').tolist()
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)

        # Initial View Range (2 Months)
        view_len = min(60, len(df))
        start_date = df.index[-view_len]
        end_date = df.index[-1]

        fig.update_xaxes(
            range=[start_date, end_date],
            minallowed=df.index[0],
            maxallowed=df.index[-1],
            row=1, col=1
        )
        fig.update_xaxes(
            range=[start_date, end_date],
            minallowed=df.index[0],
            maxallowed=df.index[-1],
            row=2, col=1
        )

        fig.update_layout(
            height=height, margin=dict(l=10, r=10, t=50, b=10),
            xaxis_rangeslider_visible=False, showlegend=True,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.8)',
            font=dict(family='Noto Sans TC, 微軟正黑體, Microsoft JhengHei, sans-serif', size=11, color='#94a3b8'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        )
        fig.update_xaxes(gridcolor='#1e293b', zerolinecolor='#334155')
        fig.update_yaxes(gridcolor='#1e293b', zerolinecolor='#334155')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"繪圖錯誤: {e}")


def render_position_chart(df, patterns, market, entry_price=None, height=260,
                          is_weekly=False, candle_patterns=None, chart_settings=None):
    """
    精簡版持倉線圖 — 專為多圖並列設計
    - 無 Volume subplot (單 row)
    - 無 legend (由外層統一顯示一次)
    - 無 subplot title
    - 縮小 annotation 字體
    - 預設 2 個月視窗
    """
    try:
        if chart_settings is None:
            chart_settings = st.session_state.get('chart_settings', {})

        if "台股" in market:
            ma_s, ma_l, s_win, l_win = 'MA5', 'MA20', 5, 20
        else:
            ma_s, ma_l, s_win, l_win = 'MA20', 'MA50', 20, 50
        df[ma_s] = df['Close'].rolling(s_win).mean()
        df[ma_l] = df['Close'].rolling(l_win).mean()

        n = 10
        df['peaks'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
        df['troughs'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']

        # 單 row — 不含 Volume
        fig = go.Figure()

        # K 線 (台股慣例: 紅漲綠跌)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing_line_color='#ef4444', decreasing_line_color='#22c55e',
            showlegend=False,
        ))

        # MA
        if chart_settings.get('ma', True):
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_s], line=dict(color='orange', width=1), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df[ma_l], line=dict(color='#3b82f6', width=1), showlegend=False))

        # Bollinger Bands
        if chart_settings.get('bbands', False) and 'BB_Upper' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=0.8, dash='dot'), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=0.8, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.08)', showlegend=False))

        # Y range
        y_min, y_max = df['Low'].min() * 0.95, df['High'].max() * 1.05
        fig.update_yaxes(range=[y_min, y_max], fixedrange=False)

        # 支撐壓力線
        if chart_settings.get('trendline', True):
            peaks, troughs = df['peaks'].dropna(), df['troughs'].dropna()
            zone_data = calculate_pattern_convergence(df, peaks, troughs)

            if zone_data:
                z_start = get_date_from_index(zone_data['x_zone_start'], df, is_weekly)
                z_end = get_date_from_index(zone_data['x_zone_end'], df, is_weekly)
                fig.add_vrect(x0=z_start, x1=z_end, fillcolor="rgba(255,165,0,0.15)", layer="below", line_width=0)

            if len(peaks) >= 2:
                p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]
                fig.add_trace(go.Scatter(x=[p1_idx, p2_idx], y=[peaks.iloc[-2], peaks.iloc[-1]],
                              mode='lines', line=dict(color="#22c55e", width=1.2, dash="dash"), showlegend=False))
                x1, x2 = df.index.get_loc(p1_idx), df.index.get_loc(p2_idx)
                if x2 != x1:
                    slope = (peaks.iloc[-1] - peaks.iloc[-2]) / (x2 - x1)
                    end_i = max(int(zone_data['x_int']) + 2 if zone_data else len(df)-1, len(df)-1)
                    fig.add_trace(go.Scatter(x=[p2_idx, get_date_from_index(end_i, df, is_weekly)],
                                  y=[peaks.iloc[-1], peaks.iloc[-1] + slope*(end_i-x2)],
                                  mode='lines', line=dict(color="#22c55e", width=0.8, dash="dot"), showlegend=False))

            if len(troughs) >= 2:
                t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]
                fig.add_trace(go.Scatter(x=[t1_idx, t2_idx], y=[troughs.iloc[-2], troughs.iloc[-1]],
                              mode='lines', line=dict(color="#ef4444", width=1.2, dash="dash"), showlegend=False))
                x1, x2 = df.index.get_loc(t1_idx), df.index.get_loc(t2_idx)
                if x2 != x1:
                    slope = (troughs.iloc[-1] - troughs.iloc[-2]) / (x2 - x1)
                    end_i = max(int(zone_data['x_int']) + 2 if zone_data else len(df)-1, len(df)-1)
                    fig.add_trace(go.Scatter(x=[t2_idx, get_date_from_index(end_i, df, is_weekly)],
                                  y=[troughs.iloc[-1], troughs.iloc[-1] + slope*(end_i-x2)],
                                  mode='lines', line=dict(color="#ef4444", width=0.8, dash="dot"), showlegend=False))

        # 跳空缺口 (只畫 shape，不加延伸線以減少雜訊)
        if chart_settings.get('gaps', True):
            prev_h, prev_l = df['High'].shift(1), df['Low'].shift(1)
            for i in range(1, len(df)):
                if df['Low'].iloc[i] > prev_h.iloc[i] * 1.005:
                    fig.add_shape(type="rect", x0=df.index[i-1], x1=df.index[i],
                                  y0=prev_h.iloc[i], y1=df['Low'].iloc[i],
                                  line=dict(width=0), fillcolor="rgba(239,68,68,0.25)")
                if df['High'].iloc[i] < prev_l.iloc[i] * 0.995:
                    fig.add_shape(type="rect", x0=df.index[i-1], x1=df.index[i],
                                  y0=df['High'].iloc[i], y1=prev_l.iloc[i],
                                  line=dict(width=0), fillcolor="rgba(34,197,94,0.25)")

        # K線型態 (精簡版 — 只顯示最近 5 個, 小字體)
        if chart_settings.get('candle_patterns', True) and candle_patterns:
            recent = candle_patterns[-5:]
            for p in recent:
                date = p['date']
                is_bear = p['type'] == 'Bearish'
                y_val = df.loc[date, 'High'] * 1.01 if is_bear else df.loc[date, 'Low'] * 0.99
                clr = "#22c55e" if is_bear else "#ef4444"  # 台股: 看跌=綠, 看漲=紅
                fig.add_annotation(x=date, y=y_val, text=p['name'], showarrow=True,
                                   arrowhead=2, arrowsize=0.8, arrowwidth=1, arrowcolor=clr,
                                   ax=0, ay=-20 if is_bear else 20,
                                   font=dict(color=clr, size=9), bgcolor="rgba(0,0,0,0.5)")

        # Entry Price 進場線
        if entry_price is not None and entry_price > 0:
            fig.add_hline(y=entry_price, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                          annotation_text=f"▸ {entry_price:,.0f}", annotation_position="left",
                          annotation_font=dict(color="#f59e0b", size=9))

        # 預設 2 個月視窗
        view_len = min(44, len(df))
        fig.update_xaxes(
            range=[df.index[-view_len], df.index[-1]],
            minallowed=df.index[0], maxallowed=df.index[-1],
            gridcolor='#1e293b', zerolinecolor='#334155',
            showticklabels=True, tickfont=dict(size=9),
        )
        fig.update_yaxes(
            gridcolor='#1e293b', zerolinecolor='#334155',
            tickfont=dict(size=9), side='right',
        )

        fig.update_layout(
            height=height,
            margin=dict(l=4, r=4, t=4, b=4),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            font=dict(family='Noto Sans TC, 微軟正黑體, Microsoft JhengHei, sans-serif', size=9, color='#94a3b8'),
            dragmode='pan',
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True,
            'displayModeBar': False,
        })
    except Exception as e:
        st.error(f"圖表錯誤: {e}")
