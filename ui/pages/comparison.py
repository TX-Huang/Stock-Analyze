"""Strategy A/B comparison panel."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import os
import ast
import importlib

from ui.theme import _plotly_dark_layout
from ui.components import cyber_kpi_strip

logger = logging.getLogger(__name__)

# -- Cyberpunk palette --
CYAN = "#00d4ff"
RED = "#ef4444"
GREEN = "#22c55e"
AMBER = "#f59e0b"
PURPLE = "#a855f7"
SLATE = "#94a3b8"

# Built-in strategy registry: display_name -> (module_path, function_name)
_BUILTIN_STRATEGIES = {
    "Isaac V3.7": ("strategies.isaac", "run_isaac_strategy"),
    "Isaac V4.0": ("strategies.isaac_v4", "run_strategy"),
    "Isaac V4.1 Razor": ("strategies.isaac_v4_razor", "run_strategy"),
    "Isaac V4.2 Turbo": ("strategies.isaac_v4_turbo", "run_strategy"),
    "VCP": ("strategies.vcp", "run_vcp_strategy"),
    "Will VCP": ("strategies.will_vcp", "run_will_vcp_strategy"),
    "Minervini": ("strategies.minervini", "run_minervini_strategy"),
    "Momentum": ("strategies.momentum", "run_momentum_strategy"),
    "Elder": ("strategies.elder", "run_elder_strategy"),
    "Candlestick": ("strategies.candlestick", "run_candlestick_strategy"),
}


def _extract_strategy_name(tree):
    """
    從 AST 中尋找頂層 STRATEGY_NAME = "..." 賦值，回傳字串或 None。
    """
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STRATEGY_NAME":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return None


def _discover_custom_strategies():
    """
    掃描 strategies/custom/ 目錄，找出包含 run_strategy() 或 run_*_strategy() 的 .py 檔案。
    讀取檔案中的 STRATEGY_NAME 作為顯示名稱，若無則使用檔名。
    回傳 dict: display_name -> (module_path, function_name)
    """
    custom = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    custom_dir = os.path.join(base_dir, "strategies", "custom")

    if not os.path.isdir(custom_dir):
        return custom

    # 確保 strategies/custom/__init__.py 存在
    init_path = os.path.join(custom_dir, "__init__.py")
    if not os.path.exists(init_path):
        try:
            with open(init_path, "w") as f:
                f.write("")
        except Exception:
            pass

    for fname in sorted(os.listdir(custom_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue

        fpath = os.path.join(custom_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

            # 優先找 run_strategy，其次找 run_*_strategy
            if "run_strategy" in func_names:
                func_name = "run_strategy"
            else:
                candidates = [fn for fn in func_names if fn.startswith("run_") and fn.endswith("_strategy")]
                if candidates:
                    func_name = candidates[0]
                else:
                    continue

            module_name = fname.replace(".py", "")
            mod_path = f"strategies.custom.{module_name}"

            # 優先使用檔案中定義的 STRATEGY_NAME，否則以檔名為顯示名
            strat_name = _extract_strategy_name(tree)
            if strat_name:
                display_name = f"\U0001F4C2 {strat_name}"
            else:
                display_name = f"\U0001F4C2 {module_name}"

            # 避免與內建策略名稱衝突
            if display_name in _BUILTIN_STRATEGIES:
                display_name = f"{display_name} (custom)"

            custom[display_name] = (mod_path, func_name)
        except Exception as e:
            logger.debug(f"Skipping {fname}: {e}")

    return custom


def _get_session_uploaded_strategies():
    """
    從 session_state 取得回測頁面上傳但尚未儲存到磁碟的策略。
    回傳 dict: display_name -> module object (直接呼叫)
    """
    uploaded = {}
    mod = st.session_state.get("_custom_strategy_module")
    if mod and hasattr(mod, "run_strategy"):
        name = getattr(mod, "__name__", "uploaded_strategy")
        uploaded[f"⬆️ {name} (暫存)"] = mod
    return uploaded


def _build_strategy_registry():
    """
    合併內建 + custom 資料夾 + session 上傳策略，回傳完整 registry。
    """
    registry = dict(_BUILTIN_STRATEGIES)
    registry.update(_discover_custom_strategies())
    return registry


STRATEGY_REGISTRY = _build_strategy_registry()

# KPI definitions: (key_in_stats, display_name, format_str, higher_is_better)
KPI_DEFS = [
    ("cagr", "CAGR (年化報酬)", "{:.2%}", True),
    ("max_drawdown", "最大回撤 (MDD)", "{:.2%}", False),
    ("daily_sharpe", "Sharpe Ratio", "{:.2f}", True),
    ("win_ratio", "勝率", "{:.1%}", True),
    ("avg_period", "平均持倉天數", "{:.1f}", None),  # no winner
    ("trade_count", "交易次數", "{:,.0f}", None),
]


def render_comparison():
    """Render the strategy A/B comparison panel."""
    st.markdown("### :balance_scale: 策略 A/B 比較")
    st.caption("並排比較兩個策略的回測表現，找出最適合你的交易系統")

    # 每次渲染時重新掃描，確保新上傳的策略立即可見
    global STRATEGY_REGISTRY
    STRATEGY_REGISTRY = _build_strategy_registry()
    session_uploaded = _get_session_uploaded_strategies()

    strategy_names = list(STRATEGY_REGISTRY.keys()) + list(session_uploaded.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**策略 A** <span style='color:{CYAN}'>&#x25CF;</span>",
                    unsafe_allow_html=True)
        strat_a = st.selectbox("選擇策略 A", strategy_names, index=0, key="cmp_strat_a")
    with col_b:
        st.markdown(f"**策略 B** <span style='color:{AMBER}'>&#x25CF;</span>",
                    unsafe_allow_html=True)
        default_b = 1 if len(strategy_names) > 1 else 0
        strat_b = st.selectbox("選擇策略 B", strategy_names, index=default_b, key="cmp_strat_b")

    if strat_a == strat_b:
        st.warning("請選擇兩個不同的策略進行比較。")
        return

    finlab_token = st.session_state.get("finlab_token", "")
    if not finlab_token:
        st.error("請先在側邊欄設定 FinLab API Token。")
        return

    if st.button(":rocket: 執行 A/B 比較回測", type="primary", key="cmp_run"):
        _run_comparison(strat_a, strat_b, finlab_token)
    elif "cmp_results" in st.session_state:
        _render_comparison(st.session_state["cmp_results"])


def _run_comparison(name_a, name_b, api_token):
    """Run both strategies and render comparison."""
    result_a = _run_strategy(name_a, api_token, label="A")
    result_b = _run_strategy(name_b, api_token, label="B")

    if result_a is None or result_b is None:
        return

    payload = {
        "name_a": name_a,
        "name_b": name_b,
        "report_a": result_a,
        "report_b": result_b,
    }
    st.session_state["cmp_results"] = payload
    _render_comparison(payload)


def _run_strategy(name, api_token, label=""):
    """Run a single strategy and return (stats, trades, equity)."""
    session_uploaded = _get_session_uploaded_strategies()

    with st.spinner(f"策略 {label} ({name}) 回測中..."):
        try:
            if name in session_uploaded:
                # Session-uploaded module: call run_strategy directly
                mod = session_uploaded[name]
                report = mod.run_strategy(api_token)
            elif name in STRATEGY_REGISTRY:
                mod_path, func_name = STRATEGY_REGISTRY[name]
                mod = importlib.import_module(mod_path)
                importlib.reload(mod)  # 確保載入最新版本
                func = getattr(mod, func_name)
                report = func(api_token)
            else:
                st.error(f"找不到策略: {name}")
                return None
            stats = report.get_stats()
            trades = report.get_trades()
            equity = _extract_equity(report)
            # Compute avg_period and trade_count into stats for convenience
            if trades is not None and not trades.empty and "period" in trades.columns:
                stats["avg_period"] = trades["period"].mean()
            else:
                stats["avg_period"] = 0
            stats["trade_count"] = len(trades) if trades is not None else 0
            return {"stats": stats, "trades": trades, "equity": equity}
        except Exception as e:
            st.error(f"策略 {name} 回測失敗: {e}")
            logger.exception(f"Strategy {name} backtest failed")
            return None


def _extract_equity(report):
    """Try to extract equity curve from finlab report object."""
    try:
        eq = report.get_equity()
        if isinstance(eq, pd.Series):
            return eq
        if isinstance(eq, pd.DataFrame):
            # Use the first column
            return eq.iloc[:, 0]
    except Exception:
        pass
    # Fallback: try cumulative_returns
    try:
        cr = report.get_cumulative_returns()
        if isinstance(cr, pd.Series):
            return cr
    except Exception:
        pass
    return None


# ================================================================
# Rendering
# ================================================================

def _render_comparison(data):
    """Render all comparison views."""
    name_a = data["name_a"]
    name_b = data["name_b"]
    stats_a = data["report_a"]["stats"]
    stats_b = data["report_b"]["stats"]
    trades_a = data["report_a"]["trades"]
    trades_b = data["report_b"]["trades"]
    eq_a = data["report_a"]["equity"]
    eq_b = data["report_b"]["equity"]

    # 1. KPI comparison table
    _render_kpi_table(name_a, name_b, stats_a, stats_b)

    # 2. Equity curves overlay
    _render_equity_overlay(name_a, name_b, eq_a, eq_b)

    # 3. Drawdown comparison
    _render_drawdown_comparison(name_a, name_b, eq_a, eq_b)

    # 4. Monthly returns heatmaps
    _render_monthly_returns(name_a, name_b, eq_a, eq_b)

    # 5. Trade distribution comparison
    _render_trade_distributions(name_a, name_b, trades_a, trades_b)


def _render_kpi_table(name_a, name_b, stats_a, stats_b):
    """Side-by-side KPI metrics with winner highlighting."""
    st.markdown("#### 核心指標對照")

    rows_html = []
    for key, label, fmt, higher_better in KPI_DEFS:
        val_a = stats_a.get(key, 0) or 0
        val_b = stats_b.get(key, 0) or 0
        disp_a = fmt.format(val_a)
        disp_b = fmt.format(val_b)

        # Determine winner
        style_a = ""
        style_b = ""
        if higher_better is not None:
            if higher_better:
                if val_a > val_b:
                    style_a = f"color:{GREEN};font-weight:700;"
                elif val_b > val_a:
                    style_b = f"color:{GREEN};font-weight:700;"
            else:
                # Lower is better (e.g. MDD — less negative = better)
                if val_a > val_b:
                    style_a = f"color:{GREEN};font-weight:700;"
                elif val_b > val_a:
                    style_b = f"color:{GREEN};font-weight:700;"

        rows_html.append(
            f"<tr>"
            f"<td style='padding:8px 12px;{style_a}'>{disp_a}</td>"
            f"<td style='padding:8px 12px;text-align:center;color:{SLATE};font-weight:600;'>{label}</td>"
            f"<td style='padding:8px 12px;text-align:right;{style_b}'>{disp_b}</td>"
            f"</tr>"
        )

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
        <thead>
            <tr style="border-bottom:2px solid {CYAN}40;">
                <th style="padding:8px 12px;color:{CYAN};text-align:left;">{name_a}</th>
                <th style="padding:8px 12px;color:{SLATE};text-align:center;">指標</th>
                <th style="padding:8px 12px;color:{AMBER};text-align:right;">{name_b}</th>
            </tr>
        </thead>
        <tbody style="color:#e2e8f0;">
            {"".join(rows_html)}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("")


def _render_equity_overlay(name_a, name_b, eq_a, eq_b):
    """Overlay equity curves on one chart."""
    if eq_a is None and eq_b is None:
        return

    fig = go.Figure()

    if eq_a is not None:
        fig.add_trace(go.Scatter(
            x=eq_a.index, y=eq_a.values, mode="lines",
            line=dict(width=2, color=CYAN),
            name=name_a,
        ))
    if eq_b is not None:
        fig.add_trace(go.Scatter(
            x=eq_b.index, y=eq_b.values, mode="lines",
            line=dict(width=2, color=AMBER),
            name=name_b,
        ))

    _plotly_dark_layout(fig, height=400,
                        title=dict(text="資金曲線比較 (Equity Curve Overlay)",
                                   font=dict(size=14, color="#e2e8f0")),
                        yaxis_title="累積報酬",
                        yaxis=dict(tickformat=",.0f"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="center", x=0.5))

    st.plotly_chart(fig, use_container_width=True, key="cmp_equity")


def _render_drawdown_comparison(name_a, name_b, eq_a, eq_b):
    """Drawdown series comparison."""
    if eq_a is None and eq_b is None:
        return

    fig = go.Figure()

    for eq, name, color in [(eq_a, name_a, CYAN), (eq_b, name_b, AMBER)]:
        if eq is None:
            continue
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines",
            fill="tozeroy",
            line=dict(width=1, color=color),
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else f"{color}20",
            name=name,
        ))

    _plotly_dark_layout(fig, height=280,
                        title=dict(text="回撤比較 (Drawdown Comparison)",
                                   font=dict(size=13, color="#e2e8f0")),
                        yaxis_title="回撤 (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="center", x=0.5))

    st.plotly_chart(fig, use_container_width=True, key="cmp_drawdown")


def _render_monthly_returns(name_a, name_b, eq_a, eq_b):
    """Monthly returns heatmap side by side."""
    if eq_a is None and eq_b is None:
        return

    st.markdown("#### 月度報酬熱力圖")
    c1, c2 = st.columns(2)

    for col, eq, name, colorscale in [
        (c1, eq_a, name_a, [[0, RED], [0.5, "#0f172a"], [1, GREEN]]),
        (c2, eq_b, name_b, [[0, RED], [0.5, "#0f172a"], [1, GREEN]]),
    ]:
        if eq is None:
            with col:
                st.info(f"{name}: 無資金曲線資料")
            continue

        monthly = _compute_monthly_returns(eq)
        if monthly is None or monthly.empty:
            with col:
                st.info(f"{name}: 無法計算月度報酬")
            continue

        with col:
            fig = go.Figure(data=go.Heatmap(
                z=monthly.values,
                x=[f"{m}月" for m in monthly.columns],
                y=monthly.index.astype(str),
                colorscale=colorscale,
                zmid=0,
                text=np.where(np.isnan(monthly.values), "",
                              np.vectorize(lambda v: f"{v:.1f}%")(monthly.values)),
                texttemplate="%{text}",
                textfont=dict(size=9),
                hovertemplate="年:%{y}<br>月:%{x}<br>報酬:%{z:.2f}%<extra></extra>",
                colorbar=dict(title="%", ticksuffix="%"),
            ))

            _plotly_dark_layout(fig, height=max(200, len(monthly) * 28 + 80),
                                title=dict(text=name,
                                           font=dict(size=12, color="#e2e8f0")),
                                xaxis=dict(side="top"),
                                yaxis=dict(autorange="reversed"))

            st.plotly_chart(fig, use_container_width=True, key=f"cmp_hm_{name}")


def _compute_monthly_returns(equity):
    """Compute monthly returns pivot table from equity series."""
    if equity is None or len(equity) < 2:
        return None
    try:
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index)
        monthly = eq.resample("ME").last().pct_change() * 100
        monthly = monthly.dropna()
        if monthly.empty:
            return None
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        })
        pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
        return pivot
    except Exception as e:
        logger.debug(f"Monthly returns computation failed: {e}")
        return None


def _render_trade_distributions(name_a, name_b, trades_a, trades_b):
    """Compare holding period and return distributions."""
    st.markdown("#### 交易分佈比較")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["持倉天數分佈", "單筆報酬率分佈"],
                        horizontal_spacing=0.12)

    # Holding period distribution
    for trades, name, color in [(trades_a, name_a, CYAN), (trades_b, name_b, AMBER)]:
        if trades is None or trades.empty:
            continue
        if "period" in trades.columns:
            periods = trades["period"].dropna()
            fig.add_trace(go.Histogram(
                x=periods, nbinsx=40, opacity=0.6,
                marker_color=color, name=name,
                legendgroup=name, showlegend=True,
            ), row=1, col=1)

    # Return distribution
    for trades, name, color in [(trades_a, name_a, CYAN), (trades_b, name_b, AMBER)]:
        if trades is None or trades.empty:
            continue
        ret_col = _find_return_col(trades)
        if ret_col is not None:
            returns = trades[ret_col].dropna()
            fig.add_trace(go.Histogram(
                x=returns, nbinsx=50, opacity=0.6,
                marker_color=color, name=name,
                legendgroup=name, showlegend=False,
            ), row=1, col=2)

    _plotly_dark_layout(fig, height=340)
    fig.update_xaxes(title_text="天數", row=1, col=1)
    fig.update_xaxes(title_text="報酬率 (%)", row=1, col=2)
    fig.update_yaxes(title_text="次數", row=1, col=1)
    fig.update_yaxes(title_text="次數", row=1, col=2)
    fig.update_layout(
        barmode="overlay",
        margin=dict(t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="center", x=0.5),
    )

    st.plotly_chart(fig, use_container_width=True, key="cmp_trade_dist")


def _find_return_col(trades):
    """Find the return column in trades DataFrame."""
    for col in ["return", "profit_pct", "ret", "returns"]:
        if col in trades.columns:
            return col
    return None
