import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ui.components import custom_metric


def render_backtest_dashboard(report, strategy_name="custom"):
    """
    通用型回測儀表板渲染函數
    """
    equity = getattr(report, 'creturn', None)
    benchmark = getattr(report, 'benchmark', None)
    drawdown = equity / equity.cummax() - 1 if equity is not None else None
    trades = report.get_trades()
    stats = report.get_stats()

    cagr = stats.get('cagr', 0)
    mdd = stats.get('max_drawdown', 0)
    win_rate = stats.get('win_ratio', 0)

    avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty else 0
    avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty else 0
    risk_reward = avg_win / avg_loss if avg_loss != 0 else 0

    avg_hold_win = trades[trades['return'] > 0]['period'].mean() if not trades.empty else 0
    avg_hold_loss = trades[trades['return'] <= 0]['period'].mean() if not trades.empty else 0

    exposure = (equity != equity.shift(1)).mean() if equity is not None else 0

    tab1, tab2, tab3 = st.tabs(["📊 實戰戰情室 (Metrics)", "📈 資金曲線 (Chart)", "📋 交易明細 (Log)"])

    with tab1:
        st.markdown("### 🏆 核心五大戰略指標")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🛡️ MDD", f"{mdd*100:.1f}%", "心理極限")
        c2.metric("⚖️ 勝率/風報", f"{win_rate*100:.0f}% | {risk_reward:.1f}", "獲利引擎")
        c3.metric("📈 CAGR", f"{cagr*100:.1f}%", "複利速度")
        c4.metric("⏳ 持倉 (贏/輸)", f"{avg_hold_win:.0f}/{avg_hold_loss:.0f}天", "資金效率")
        c5.metric("🛡️ 曝險", f"{exposure*100:.0f}%", "避險能力")

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

            fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📋 詳細交易紀錄")
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

            def calculate_holding(row):
                if pd.notna(row.get('出場日期')):
                    return (row['出場日期'] - row['進場日期']).days + 1
                elif pd.notna(row.get('進場日期')):
                    return (today - row['進場日期']).days + 1
                return row.get('持有天數', 0)

            trades_display['持有天數'] = trades_display.apply(calculate_holding, axis=1)

            trades_filtered = trades_display

            csv = trades_filtered.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載完整交易明細 (.csv)",
                data=csv,
                file_name=f'trade_log_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

            items_per_page = 1000
            total_items = len(trades_filtered)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            page = st.number_input("頁數 (Page)", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)

            st.info(f"顯示第 {start_idx + 1} 至 {end_idx} 筆交易 (共 {total_items} 筆)")

            available_cols = ['股票代碼', '進場日期', '出場日期', '進場價', '出場價', '報酬率', '持有天數', '最大不利(MAE)', '最大有利(MFE)']
            cols_to_show = [c for c in available_cols if c in trades_filtered.columns]

            trades_final = trades_filtered[cols_to_show].sort_values("進場日期", ascending=False).iloc[start_idx:end_idx]

            csv = trades_final.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載交易明細 (.csv)",
                data=csv,
                file_name=f'trade_log_{strategy_name}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

            def highlight_ret(val):
                color = ''
                if pd.isna(val):
                    return ''
                if isinstance(val, (int, float)):
                    color = 'color: #ef4444' if val > 0 else 'color: #22c55e'
                return color

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
            st.info("無交易紀錄")
