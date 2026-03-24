"""Monte Carlo simulation panel for strategy robustness analysis."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

from ui.theme import _plotly_dark_layout
from ui.components import cyber_kpi_strip

logger = logging.getLogger(__name__)

# -- Cyberpunk palette --
CYAN = "#00d4ff"
CYAN_DIM = "rgba(0,212,255,0.3)"
RED = "#ef4444"
GREEN = "#22c55e"
AMBER = "#f59e0b"
PURPLE = "#a855f7"
SLATE = "#94a3b8"
DARK_BG = "rgba(5,8,16,0.9)"


def render_monte_carlo(trades_df=None, stats=None):
    """Render the Monte Carlo simulation panel.

    Args:
        trades_df: DataFrame of trades from backtest (with 'return' column)
        stats: Dict of backtest statistics
    """
    st.markdown("### :game_die: 蒙地卡羅模擬分析")
    st.caption("透過隨機重組歷史交易，評估策略在不同情境下的表現穩健度")

    # --- Controls ---
    col1, col2, col3 = st.columns(3)
    with col1:
        n_simulations = st.slider("模擬次數", 100, 10000, 1000, step=100, key="mc_n_sims")
    with col2:
        confidence = st.selectbox(
            "信賴區間", [0.90, 0.95, 0.99], index=1,
            format_func=lambda x: f"{x * 100:.0f}%", key="mc_confidence",
        )
    with col3:
        initial_capital = st.number_input(
            "初始資金 (NTD)", value=1_000_000, step=100_000, format="%d", key="mc_capital",
        )

    # --- Check data availability ---
    if trades_df is None or trades_df.empty:
        st.info("請先在「量化回測」頁面執行回測，取得交易紀錄後再進行蒙地卡羅分析。")
        _render_demo_mode(n_simulations, confidence, initial_capital)
        return

    if st.button(":rocket: 執行蒙地卡羅模擬", type="primary", key="mc_run"):
        _execute_and_render(trades_df, n_simulations, confidence, initial_capital)
    elif "mc_results" in st.session_state:
        # Re-render cached results
        _render_results(st.session_state["mc_results"], confidence, initial_capital)


def _render_demo_mode(n_sims, confidence, initial_capital):
    """Show a demo simulation with synthetic data so the panel is not empty."""
    st.markdown("---")
    st.markdown("**:test_tube: 示範模式** — 使用合成交易資料展示面板功能")

    if st.button(":sparkles: 執行示範模擬", key="mc_demo"):
        np.random.seed(42)
        demo_returns = np.random.normal(0.8, 5.0, 200)  # mean 0.8%, std 5%
        demo_df = pd.DataFrame({"return": demo_returns})
        _execute_and_render(demo_df, n_sims, confidence, initial_capital)


def _execute_and_render(trades_df, n_sims, confidence, initial_capital):
    """Run the simulation and render results."""
    with st.spinner("蒙地卡羅模擬執行中..."):
        results = _run_simulation(trades_df, n_sims, initial_capital)
        st.session_state["mc_results"] = results
    _render_results(results, confidence, initial_capital)


# ================================================================
# Core simulation
# ================================================================

def _run_simulation(trades_df, n_sims, initial_capital):
    """Run Monte Carlo simulation by bootstrapping trade returns."""
    # Extract returns (in decimal form)
    returns = _extract_returns(trades_df)
    n_trades = len(returns)

    # Bootstrap: randomly sample trades with replacement
    sampled = np.random.choice(returns, size=(n_sims, n_trades), replace=True)
    cumulative = np.cumprod(1 + sampled, axis=1) * initial_capital
    paths = np.column_stack([np.full(n_sims, initial_capital), cumulative])

    final_values = paths[:, -1]
    total_returns_pct = (final_values / initial_capital - 1) * 100

    # Max drawdown per path
    max_drawdowns = np.zeros(n_sims)
    for i in range(n_sims):
        peak = np.maximum.accumulate(paths[i])
        dd = (paths[i] - peak) / peak
        max_drawdowns[i] = float(dd.min()) * 100

    # Annualised CAGR estimate (assume ~250 trading days, avg period 15 days)
    years_est = n_trades * 15 / 250  # rough estimate
    if years_est > 0:
        cagr_values = ((final_values / initial_capital) ** (1 / years_est) - 1) * 100
    else:
        cagr_values = total_returns_pct

    return {
        "paths": paths,
        "final_values": final_values,
        "total_returns": total_returns_pct,
        "cagr_values": cagr_values,
        "max_drawdowns": max_drawdowns,
        "n_sims": n_sims,
        "n_trades": n_trades,
    }


def _extract_returns(trades_df):
    """Extract per-trade returns as decimals from various column conventions."""
    if "return" in trades_df.columns:
        raw = trades_df["return"].dropna().values
        # Heuristic: if values look like percentages (most > 1 or < -1), convert
        if np.median(np.abs(raw[raw != 0])) > 0.5:
            return raw / 100.0
        return raw
    if "profit_pct" in trades_df.columns:
        return trades_df["profit_pct"].dropna().values / 100.0
    if "exit_price" in trades_df.columns and "entry_price" in trades_df.columns:
        ep = trades_df["entry_price"]
        xp = trades_df["exit_price"]
        return ((xp - ep) / ep).dropna().values
    raise ValueError("trades_df 缺少 return / profit_pct / entry_price+exit_price 欄位")


# ================================================================
# Rendering
# ================================================================

def _render_results(results, confidence, initial_capital):
    """Render simulation results with KPIs, fan chart, histograms."""
    paths = results["paths"]
    final_values = results["final_values"]
    total_returns = results["total_returns"]
    cagr_values = results["cagr_values"]
    max_drawdowns = results["max_drawdowns"]
    n_sims = results["n_sims"]
    n_trades = results["n_trades"]

    alpha = (1 - confidence) / 2  # tail probability
    lo_pct = alpha * 100
    hi_pct = (1 - alpha) * 100

    # --- KPI strip ---
    median_cagr = float(np.median(cagr_values))
    prob_profit = float(np.mean(final_values > initial_capital)) * 100
    prob_ruin = float(np.mean(final_values < initial_capital * 0.5)) * 100
    worst_case = float(np.percentile(total_returns, lo_pct))
    best_case = float(np.percentile(total_returns, hi_pct))
    median_mdd = float(np.median(max_drawdowns))

    cyber_kpi_strip([
        {"label": "中位數 CAGR", "value": f"{median_cagr:+.1f}%",
         "color": GREEN if median_cagr > 0 else RED, "accent": CYAN},
        {"label": "獲利機率", "value": f"{prob_profit:.1f}%",
         "color": GREEN if prob_profit > 50 else RED, "accent": GREEN},
        {"label": "破產機率 (<50%)", "value": f"{prob_ruin:.1f}%",
         "color": GREEN if prob_ruin < 5 else RED, "accent": RED},
        {"label": f"最差情境 ({lo_pct:.0f}th)", "value": f"{worst_case:+.1f}%",
         "color": RED, "accent": AMBER},
        {"label": f"最佳情境 ({hi_pct:.0f}th)", "value": f"{best_case:+.1f}%",
         "color": GREEN, "accent": PURPLE},
        {"label": "中位數 MDD", "value": f"{median_mdd:.1f}%",
         "color": RED, "accent": RED},
    ])

    st.markdown(f"<div style='text-align:right;color:#64748b;font-size:0.75rem;'>"
                f"{n_sims:,} 次模擬 | {n_trades:,} 筆交易/路徑</div>",
                unsafe_allow_html=True)

    # --- Fan chart ---
    _render_fan_chart(paths, confidence)

    # --- Two-column histograms ---
    c1, c2 = st.columns(2)
    with c1:
        _render_return_histogram(total_returns, confidence)
    with c2:
        _render_drawdown_histogram(max_drawdowns, confidence)

    # --- Statistics table ---
    _render_stats_table(results, confidence, initial_capital)


def _render_fan_chart(paths, confidence):
    """Equity fan chart with percentile bands."""
    n_steps = paths.shape[1]
    x = list(range(n_steps))

    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    # 5-95 band (lightest)
    fig.add_trace(go.Scatter(
        x=x, y=p95, mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,212,255,0.08)",
        name="5th-95th", hoverinfo="skip",
    ))

    # 25-75 band (medium)
    fig.add_trace(go.Scatter(
        x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p25, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(0,212,255,0.18)",
        name="25th-75th", hoverinfo="skip",
    ))

    # Percentile lines
    fig.add_trace(go.Scatter(
        x=x, y=p5, mode="lines",
        line=dict(width=1, dash="dot", color=RED),
        name="5th (最差)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p95, mode="lines",
        line=dict(width=1, dash="dot", color=GREEN),
        name="95th (最佳)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines",
        line=dict(width=2.5, color=CYAN),
        name="中位數 (50th)",
    ))

    _plotly_dark_layout(fig, height=420,
                        title=dict(text="資金曲線扇形圖 (Equity Fan Chart)",
                                   font=dict(size=14, color="#e2e8f0")),
                        xaxis_title="交易序號",
                        yaxis_title="資金 (NTD)",
                        yaxis=dict(tickformat=",.0f"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="center", x=0.5))

    st.plotly_chart(fig, use_container_width=True, key="mc_fan_chart")


def _render_return_histogram(total_returns, confidence):
    """Final return distribution histogram."""
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(total_returns, alpha * 100))
    hi = float(np.percentile(total_returns, (1 - alpha) * 100))
    median_ret = float(np.median(total_returns))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=total_returns, nbinsx=60,
        marker_color=CYAN_DIM,
        marker_line=dict(width=0.5, color=CYAN),
        name="報酬分佈",
    ))

    fig.add_vline(x=median_ret, line=dict(color=CYAN, width=2),
                  annotation_text=f"中位數 {median_ret:.1f}%",
                  annotation_font_color=CYAN)
    fig.add_vline(x=lo, line=dict(color=RED, width=1, dash="dash"),
                  annotation_text=f"{alpha*100:.0f}th {lo:.1f}%",
                  annotation_font_color=RED)
    fig.add_vline(x=0, line=dict(color=SLATE, width=1, dash="dot"))

    _plotly_dark_layout(fig, height=320,
                        title=dict(text="最終報酬率分佈",
                                   font=dict(size=13, color="#e2e8f0")),
                        xaxis_title="總報酬率 (%)",
                        yaxis_title="次數",
                        showlegend=False)

    st.plotly_chart(fig, use_container_width=True, key="mc_return_hist")


def _render_drawdown_histogram(max_drawdowns, confidence):
    """Max drawdown distribution histogram."""
    alpha = (1 - confidence) / 2
    worst_dd = float(np.percentile(max_drawdowns, alpha * 100))
    median_dd = float(np.median(max_drawdowns))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=max_drawdowns, nbinsx=60,
        marker_color="rgba(239,68,68,0.35)",
        marker_line=dict(width=0.5, color=RED),
        name="MDD 分佈",
    ))

    fig.add_vline(x=median_dd, line=dict(color=AMBER, width=2),
                  annotation_text=f"中位數 {median_dd:.1f}%",
                  annotation_font_color=AMBER)
    fig.add_vline(x=worst_dd, line=dict(color=RED, width=1.5, dash="dash"),
                  annotation_text=f"最差 {worst_dd:.1f}%",
                  annotation_font_color=RED)

    _plotly_dark_layout(fig, height=320,
                        title=dict(text="最大回撤 (MDD) 分佈",
                                   font=dict(size=13, color="#e2e8f0")),
                        xaxis_title="最大回撤 (%)",
                        yaxis_title="次數",
                        showlegend=False)

    st.plotly_chart(fig, use_container_width=True, key="mc_dd_hist")


def _render_stats_table(results, confidence, initial_capital):
    """Detailed statistics table."""
    final_values = results["final_values"]
    total_returns = results["total_returns"]
    max_drawdowns = results["max_drawdowns"]
    cagr_values = results["cagr_values"]

    alpha = (1 - confidence) / 2
    lo_pct = alpha * 100
    hi_pct = (1 - alpha) * 100

    with st.expander("詳細統計數據", expanded=False):
        data = {
            "指標": [
                "模擬次數",
                "每路徑交易數",
                f"中位數最終資金",
                f"{lo_pct:.0f}th 百分位資金",
                f"{hi_pct:.0f}th 百分位資金",
                "平均報酬率",
                "中位數報酬率",
                "報酬率標準差",
                "中位數 CAGR",
                "獲利機率 (>0%)",
                "翻倍機率 (>100%)",
                "破產機率 (<-50%)",
                "中位數 MDD",
                f"最差 MDD ({lo_pct:.0f}th)",
                "平均 MDD",
            ],
            "數值": [
                f"{results['n_sims']:,}",
                f"{results['n_trades']:,}",
                f"${np.median(final_values):,.0f}",
                f"${np.percentile(final_values, lo_pct):,.0f}",
                f"${np.percentile(final_values, hi_pct):,.0f}",
                f"{np.mean(total_returns):+.2f}%",
                f"{np.median(total_returns):+.2f}%",
                f"{np.std(total_returns):.2f}%",
                f"{np.median(cagr_values):+.2f}%",
                f"{np.mean(final_values > initial_capital) * 100:.1f}%",
                f"{np.mean(total_returns > 100) * 100:.1f}%",
                f"{np.mean(total_returns < -50) * 100:.1f}%",
                f"{np.median(max_drawdowns):.1f}%",
                f"{np.percentile(max_drawdowns, lo_pct):.1f}%",
                f"{np.mean(max_drawdowns):.1f}%",
            ],
        }
        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True, hide_index=True,
        )
