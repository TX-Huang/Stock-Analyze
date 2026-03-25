import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ui.components import cyber_kpi_strip, cyber_header, cyber_alert
from analysis.trade_analytics import (
    compute_trade_efficiency, compute_edge_ratio, compute_profit_factor,
    compute_expectancy, compute_monthly_seasonality, compute_rolling_sharpe,
    compute_streaks, compute_underwater_days, compute_drawdown_recovery,
    compute_market_regime_stats,
)


def _build_yearly_table(equity, trades):
    """從 equity curve + trades 建構逐年績效 DataFrame"""
    if equity is None or len(equity) == 0:
        return pd.DataFrame()

    eq_years = equity.groupby(equity.index.year)
    trades_copy = trades.copy()
    if len(trades) > 0 and 'entry_date' in trades.columns:
        trades_copy['year'] = pd.to_datetime(trades_copy['entry_date']).dt.year
    else:
        trades_copy['year'] = []

    rows = []
    for yr in sorted(eq_years.groups.keys()):
        eq_yr = eq_years.get_group(yr)
        ann_ret = (eq_yr.iloc[-1] / eq_yr.iloc[0] - 1) * 100
        running_max = eq_yr.cummax()
        dd = (eq_yr - running_max) / running_max
        yr_mdd = dd.min() * 100
        daily_ret = eq_yr.pct_change().dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            yr_sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5)
        else:
            yr_sharpe = float('nan')

        yr_trades = trades_copy[trades_copy['year'] == yr] if len(trades_copy) > 0 else pd.DataFrame()
        n = len(yr_trades)
        ret_col = 'return' if 'return' in yr_trades.columns else None
        mae_col = 'mae' if 'mae' in yr_trades.columns else None
        mfe_col = 'gmfe' if 'gmfe' in yr_trades.columns else ('bmfe' if 'bmfe' in yr_trades.columns else None)

        rows.append({
            '年度': yr,
            '年報酬%': round(ann_ret, 2),
            '年MDD%': round(yr_mdd, 2),
            'Sharpe': round(yr_sharpe, 2),
            '交易數': n,
            '平均報酬%': round(yr_trades[ret_col].mean() * 100, 2) if (ret_col and n > 0) else None,
            '勝率%': round((yr_trades[ret_col] > 0).mean() * 100, 1) if (ret_col and n > 0) else None,
            'MAE%': round(yr_trades[mae_col].mean() * 100, 2) if (mae_col and n > 0) else None,
            'MFE%': round(yr_trades[mfe_col].mean() * 100, 2) if (mfe_col and n > 0) else None,
        })

    return pd.DataFrame(rows)


def _build_position_changes(position):
    """從 position DataFrame 建構持倉異動紀錄"""
    if position is None or len(position) == 0:
        return pd.DataFrame()

    records = []
    prev_holdings = set()
    for i in range(len(position)):
        date = position.index[i]
        row = position.iloc[i]
        curr_holdings = set(row[row > 0].index)

        entered = curr_holdings - prev_holdings
        exited = prev_holdings - curr_holdings

        if entered or exited:
            if entered and exited:
                action = '替換'
                detail = f"OUT({','.join(sorted(exited))}) → IN({','.join(sorted(entered))})"
            elif entered:
                action = '新進場'
                detail = ','.join(sorted(entered))
            else:
                action = '出場'
                detail = ','.join(sorted(exited))

            records.append({
                '日期': date.strftime('%Y-%m-%d'),
                '異動': action,
                '明細': detail,
                '持倉數': len(curr_holdings),
                '持倉清單': ','.join(sorted(curr_holdings)),
            })

        prev_holdings = curr_holdings

    return pd.DataFrame(records)


def _render_trade_analytics_tab(trades, equity, benchmark, strategy_name):
    """交易品質分析 Tab — 三層深度分析"""
    cyber_header("交易品質深度分析", subtitle=strategy_name)

    if trades.empty:
        cyber_alert("無交易", "無交易資料可分析", level="info")
        return

    # === Layer 1: 核心品質指標 ===
    st.markdown("#### Layer 1: 交易品質")

    efficiency = compute_trade_efficiency(trades)
    edge_ratios = compute_edge_ratio(trades)
    pf = compute_profit_factor(trades)
    expectancy = compute_expectancy(trades)

    eff_median = float(efficiency.median()) if len(efficiency.dropna()) > 0 else 0
    er_median = float(edge_ratios.median()) if len(edge_ratios.dropna()) > 0 else 0
    er_above_1 = float((edge_ratios > 1).mean() * 100) if len(edge_ratios.dropna()) > 0 else 0

    cyber_kpi_strip([
        {'label': 'Profit Factor', 'value': f"{pf:.2f}",
         'color': '#22c55e' if pf > 1.5 else '#f59e0b', 'accent': '#22c55e'},
        {'label': 'Expectancy', 'value': f"{expectancy*100:.2f}%",
         'color': '#22c55e' if expectancy > 0 else '#ef4444', 'accent': '#00f0ff'},
        {'label': 'Trade Efficiency', 'value': f"{eff_median:.1%}",
         'color': '#22c55e' if eff_median > 0.3 else '#f59e0b', 'accent': '#8b5cf6'},
        {'label': 'Edge Ratio (med)', 'value': f"{er_median:.2f}",
         'color': '#22c55e' if er_median > 1.0 else '#ef4444', 'accent': '#00f0ff'},
        {'label': 'Edge > 1.0', 'value': f"{er_above_1:.0f}%",
         'accent': '#22c55e'},
    ])

    # Efficiency & Edge Ratio 分佈圖
    col1, col2 = st.columns(2)
    with col1:
        if len(efficiency.dropna()) > 0:
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Histogram(
                x=efficiency.dropna().clip(-1, 1.5).values,
                nbinsx=50, marker_color='#8b5cf6', opacity=0.7,
            ))
            fig_eff.add_vline(x=eff_median, line_dash="dash", line_color="#00f0ff",
                             annotation_text=f"中位數 {eff_median:.2f}")
            fig_eff.update_layout(
                title="Trade Efficiency 分佈 (return/MFE)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Efficiency", yaxis_title="次數",
            )
            st.plotly_chart(fig_eff, use_container_width=True)

    with col2:
        if len(edge_ratios.dropna()) > 0:
            fig_er = go.Figure()
            fig_er.add_trace(go.Histogram(
                x=edge_ratios.dropna().clip(0, 5).values,
                nbinsx=50, marker_color='#00f0ff', opacity=0.7,
            ))
            fig_er.add_vline(x=1.0, line_dash="dash", line_color="#ef4444",
                             annotation_text="損益平衡線 1.0")
            fig_er.update_layout(
                title="Edge Ratio 分佈 (MFE/|MAE|)",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Edge Ratio", yaxis_title="次數",
            )
            st.plotly_chart(fig_er, use_container_width=True)

    # === Layer 2: 時間與環境 ===
    st.markdown("#### Layer 2: 時間與市場環境")

    col1, col2 = st.columns(2)

    with col1:
        # 月份季節性
        seasonality = compute_monthly_seasonality(trades)
        if not seasonality.empty:
            month_labels = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
            avg_ret_pct = seasonality['avg_return'] * 100
            colors = ['#22c55e' if v >= 0 else '#ef4444' for v in avg_ret_pct]

            fig_season = make_subplots(specs=[[{"secondary_y": True}]])
            fig_season.add_trace(go.Bar(
                x=month_labels, y=avg_ret_pct,
                marker_color=colors, name='平均報酬%',
                text=[f"{v:.2f}%" for v in avg_ret_pct],
                textposition='outside',
            ), secondary_y=False)
            fig_season.add_trace(go.Scatter(
                x=month_labels, y=seasonality['win_rate'],
                mode='lines+markers', name='勝率%',
                line=dict(color='#00f0ff', width=2),
            ), secondary_y=True)
            fig_season.update_layout(
                title="月份季節性分析", height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation='h', y=-0.15),
            )
            fig_season.update_yaxes(title_text="平均報酬%", secondary_y=False)
            fig_season.update_yaxes(title_text="勝率%", secondary_y=True)
            st.plotly_chart(fig_season, use_container_width=True)

    with col2:
        # 市場環境分析
        regime_stats = compute_market_regime_stats(trades, benchmark)
        if regime_stats:
            regime_df = pd.DataFrame([
                {'市場狀態': {'bull': '多頭', 'bear': '空頭', 'sideways': '盤整'}.get(k, k),
                 '交易數': v['n_trades'],
                 '勝率%': v['win_rate_pct'],
                 '平均報酬%': v['avg_return_pct'],
                 'Profit Factor': v['profit_factor'],
                 '平均持有天': v['avg_hold_days']}
                for k, v in regime_stats.items()
            ])
            st.markdown("**大盤環境績效比較**")
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

    # Rolling Sharpe
    if equity is not None:
        rolling_sh = compute_rolling_sharpe(equity, window=60)
        if len(rolling_sh) > 0:
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(
                x=rolling_sh.index, y=rolling_sh.values,
                mode='lines', name='Rolling Sharpe (60d)',
                line=dict(color='#00f0ff', width=1.5),
            ))
            fig_rs.add_hline(y=0, line_dash="dash", line_color="#ef4444", opacity=0.5)
            fig_rs.add_hline(y=1.0, line_dash="dot", line_color="#22c55e", opacity=0.5,
                            annotation_text="Sharpe=1.0")

            # 標記負 Sharpe 區域
            neg_pct = (rolling_sh < 0).mean() * 100
            fig_rs.update_layout(
                title=f"Rolling Sharpe (60d) — 負值時段佔 {neg_pct:.0f}%",
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                yaxis_title="Sharpe Ratio",
            )
            st.plotly_chart(fig_rs, use_container_width=True)

    # === Layer 3: 連續性與穩定度 ===
    st.markdown("#### Layer 3: 連續性與穩定度")

    col1, col2, col3 = st.columns(3)

    streaks = compute_streaks(trades)
    with col1:
        st.metric("最長連勝", f"{streaks.get('max_win_streak', 0)} 筆")
        st.metric("平均連勝", f"{streaks.get('avg_win_streak', 0)} 筆")
    with col2:
        st.metric("最長連敗", f"{streaks.get('max_loss_streak', 0)} 筆")
        st.metric("平均連敗", f"{streaks.get('avg_loss_streak', 0)} 筆")

    if equity is not None:
        uw = compute_underwater_days(equity)
        with col3:
            st.metric("最長水下天數", f"{uw.get('max_underwater_days', 0)} 天")
            st.metric("目前水下天數", f"{uw.get('current_underwater_days', 0)} 天")

    # Drawdown Recovery 表格
    if equity is not None:
        dd_events = compute_drawdown_recovery(equity)
        if dd_events:
            st.markdown("**Top 10 回撤恢復分析**")
            dd_df = pd.DataFrame(dd_events[:10])
            dd_df.columns = ['起始日', '谷底日', '恢復日', '深度%', '下跌天數', '恢復天數', '總天數']
            st.dataframe(dd_df, use_container_width=True, hide_index=True, height=300)


def render_backtest_dashboard(report, strategy_name="custom"):
    """
    通用型回測儀表板渲染函數
    """
    equity = getattr(report, 'creturn', None)
    benchmark = getattr(report, 'benchmark', None)
    drawdown = equity / equity.cummax() - 1 if equity is not None else None
    trades = report.get_trades()
    stats = report.get_stats()
    position = getattr(report, 'position', None)

    cagr = stats.get('cagr', 0)
    mdd = stats.get('max_drawdown', 0)
    win_rate = stats.get('win_ratio', 0)

    avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty else 0
    avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty else 0
    risk_reward = avg_win / avg_loss if avg_loss != 0 else 0

    avg_hold_win = trades[trades['return'] > 0]['period'].mean() if not trades.empty else 0
    avg_hold_loss = trades[trades['return'] <= 0]['period'].mean() if not trades.empty else 0

    exposure = stats.get('exposure', (equity != equity.shift(1)).mean() if equity is not None else 0)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "實戰戰情室",
        "資金曲線",
        "逐年績效",
        "交易品質分析",
        "持倉異動",
        "交易明細",
    ])

    with tab1:
        cyber_header("核心五大戰略指標", subtitle=strategy_name)
        cyber_kpi_strip([
            {'label': 'MDD', 'value': f"{mdd*100:.1f}%",
             'color': '#ef4444' if mdd < -0.2 else '#f59e0b', 'accent': '#ef4444'},
            {'label': '勝率/風報', 'value': f"{win_rate*100:.0f}% | {risk_reward:.1f}",
             'accent': '#00f0ff'},
            {'label': 'CAGR', 'value': f"{cagr*100:.1f}%",
             'color': '#22c55e' if cagr > 0 else '#ef4444', 'accent': '#22c55e'},
            {'label': '持倉 (贏/輸)', 'value': f"{avg_hold_win:.0f}/{avg_hold_loss:.0f}天",
             'accent': '#8b5cf6'},
            {'label': '曝險', 'value': f"{exposure*100:.0f}%",
             'accent': '#00f0ff'},
        ])

        # 現倉追蹤 (從 position 最後一天取得)
        if position is not None and len(position) > 0:
            cyber_header("目前持倉")
            last_row = position.iloc[-1]
            current = last_row[last_row > 0].sort_values(ascending=False)
            if len(current) > 0:
                # 找每檔的進場日
                holding_rows = []
                for stock_id in current.index:
                    pos_series = position[stock_id]
                    nonzero_mask = pos_series > 0
                    entry_idx = 0
                    for j in range(len(pos_series) - 1, -1, -1):
                        if not nonzero_mask.iloc[j]:
                            entry_idx = j + 1
                            break
                    if entry_idx >= len(pos_series):
                        entry_idx = 0
                    entry_date = pos_series.index[entry_idx]
                    holding_rows.append({
                        '股票代碼': stock_id,
                        '進場日期': entry_date.strftime('%Y-%m-%d'),
                        'Score': round(float(current[stock_id]), 1),
                    })
                st.dataframe(pd.DataFrame(holding_rows), use_container_width=True, hide_index=True)
            else:
                cyber_alert("空手狀態", "目前無持倉 — 策略處於觀望中", level="info")

    with tab2:
        if equity is not None:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("資產權益曲線 (Equity Curve)", "資金回撤 (Drawdown)"),
                              vertical_spacing=0.1, row_heights=[0.7, 0.3])

            fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                                   mode='lines', name='策略報酬',
                                   line=dict(color='#22c55e', width=2)), row=1, col=1)

            if benchmark is not None:
                 fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values,
                                   mode='lines', name='大盤基準',
                                   line=dict(color='gray', width=1, dash='dot')), row=1, col=1)

            if drawdown is not None:
                fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                       mode='lines', name='回撤幅度',
                                       line=dict(color='#ef4444', width=1), fill='tozeroy'), row=2, col=1)

            fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cyber_header("逐年績效總覽")
        yearly_df = _build_yearly_table(equity, trades)
        if not yearly_df.empty:
            # 年報酬柱狀圖 (Primary — 5 秒掛接)
            fig_yr = go.Figure()
            colors = ['#22c55e' if v >= 0 else '#ef4444' for v in yearly_df['年報酬%']]
            fig_yr.add_trace(go.Bar(
                x=yearly_df['年度'], y=yearly_df['年報酬%'],
                marker_color=colors, text=[f"{v:.1f}%" for v in yearly_df['年報酬%']],
                textposition='outside',
            ))
            fig_yr.update_layout(title="逐年報酬率", height=350, margin=dict(l=10, r=10, t=40, b=10),
                                yaxis_title="報酬率 %")
            st.plotly_chart(fig_yr, use_container_width=True)

            # 逐年績效表格 (Secondary — 精確數字)
            def _color_ret(val):
                if pd.isna(val):
                    return ''
                if isinstance(val, (int, float)):
                    return 'color: #ef4444' if val > 0 else 'color: #22c55e' if val < 0 else ''
                return ''

            st.dataframe(
                yearly_df.style.format({
                    '年報酬%': '{:.2f}%',
                    '年MDD%': '{:.2f}%',
                    'Sharpe': '{:.2f}',
                    '平均報酬%': '{:.2f}%',
                    '勝率%': '{:.1f}%',
                    'MAE%': '{:.2f}%',
                    'MFE%': '{:.2f}%',
                }, na_rep="—").map(_color_ret, subset=['年報酬%', '年MDD%']),
                use_container_width=True,
                hide_index=True,
                height=400,
            )
        else:
            cyber_alert("缺少資料", "缺少 equity curve — 請先執行回測", level="warn")

    with tab4:
        _render_trade_analytics_tab(trades, equity, benchmark, strategy_name)

    with tab5:
        cyber_header("持倉異動紀錄")
        changes_df = _build_position_changes(position)
        if not changes_df.empty:
            # 篩選年份
            changes_df['_year'] = pd.to_datetime(changes_df['日期']).dt.year
            available_years = sorted(changes_df['_year'].unique(), reverse=True)
            selected_year = st.selectbox("篩選年份", available_years, index=0)
            filtered = changes_df[changes_df['_year'] == selected_year].drop(columns=['_year'])

            st.info(f"{selected_year} 年共 {len(filtered)} 次持倉異動")

            def _color_action(val):
                if val == '替換':
                    return 'color: #f59e0b'
                elif val == '新進場':
                    return 'color: #22c55e'
                elif val == '出場':
                    return 'color: #ef4444'
                return ''

            st.dataframe(
                filtered.style.map(_color_action, subset=['異動']),
                use_container_width=True,
                hide_index=True,
                height=600,
            )
        else:
            cyber_alert("無異動", "此策略尚無持倉異動紀錄", level="info")

    with tab6:
        cyber_header("詳細交易紀錄")
        if not trades.empty:
            rename_map = {
                "stock_id": "股票代碼",
                "entry_date": "進場日期",
                "exit_date": "出場日期",
                "entry_price": "進場價",
                "exit_price": "出場價",
                "return": "報酬率",
                "mae": "最大不利(MAE)",
                "mfe": "最大有利(MFE)",
                "period": "持有天數"
            }

            trades_display = trades.copy()
            trades_display.rename(columns=rename_map, inplace=True)

            if '進場日期' in trades_display.columns:
                trades_display['進場日期'] = pd.to_datetime(trades_display['進場日期'])
            if '出場日期' in trades_display.columns:
                trades_display['出場日期'] = pd.to_datetime(trades_display['出場日期'], errors='coerce')

            try:
                today = report.position.index[-1]
            except Exception:
                today = datetime.now()

            # 向量化計算持有天數
            if '出場日期' in trades_display.columns and '進場日期' in trades_display.columns:
                has_exit = pd.notna(trades_display['出場日期'])
                has_entry = pd.notna(trades_display['進場日期'])
                trades_display.loc[has_exit, '持有天數'] = (trades_display.loc[has_exit, '出場日期'] - trades_display.loc[has_exit, '進場日期']).dt.days + 1
                trades_display.loc[~has_exit & has_entry, '持有天數'] = (today - trades_display.loc[~has_exit & has_entry, '進場日期']).dt.days + 1
                trades_display['持有天數'] = trades_display['持有天數'].fillna(0).astype(int)

            trades_filtered = trades_display

            # 下載 + 分頁控制
            csv_all = trades_filtered.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="下載完整交易明細 (.csv)",
                data=csv_all,
                file_name=f'trade_log_{strategy_name}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

            items_per_page = 1000
            total_items = len(trades_filtered)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            page = st.number_input("頁數", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)

            st.caption(f"顯示第 {start_idx + 1} 至 {end_idx} 筆交易 (共 {total_items} 筆)")

            available_cols = ['股票代碼', '進場日期', '出場日期', '進場價', '出場價', '報酬率', '持有天數', '最大不利(MAE)', '最大有利(MFE)']
            cols_to_show = [c for c in available_cols if c in trades_filtered.columns]

            trades_final = trades_filtered[cols_to_show].sort_values("進場日期", ascending=False).iloc[start_idx:end_idx]

            def highlight_ret(val):
                if pd.isna(val):
                    return ''
                if isinstance(val, (int, float)):
                    return 'color: #ef4444' if val > 0 else 'color: #22c55e'
                return ''

            st.dataframe(
                trades_final.style.format({
                    '報酬率': '{:.2%}',
                    '最大不利(MAE)': '{:.2%}',
                    '最大有利(MFE)': '{:.2%}',
                    '進場價': '{:.2f}',
                    '出場價': '{:.2f}'
                }, na_rep="N/A").map(highlight_ret, subset=['報酬率']),
                use_container_width=True,
                height=600
            )
        else:
            cyber_alert("無交易", "無交易紀錄 — 策略可能無觸發信號", level="info")
