"""
Module: Quantitative Backtest System (量化回測系統)
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import os
import logging

from ui.theme import _plotly_dark_layout
from ui.components import cyber_spinner, cyber_kpi_strip, cyber_table


def render(_embedded=False):
    if not _embedded:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
            <div style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">量化策略回測實驗室</div>
            <span class="tag tag-ok" style="font-size:0.7rem;">Finlab Engine</span>
        </div>
        <div style="color:#64748b; font-size:0.8rem; margin-bottom:16px;">內建策略回測 | 自訂策略上傳 | 多策略 A/B 比較</div>
        """, unsafe_allow_html=True)

    # --- Strategy Config ---
    PRESET_STRATEGIES = {
        "純做多策略 (Long Only)": ("strategies.long_only", "run_long_strategy"),
        "多空策略 (Long + Short)": ("strategies.long_short", "run_long_short_strategy"),
        "VCP 波動收縮策略 (Minervini)": ("strategies.vcp", "run_vcp_strategy"),
        "Isaac 頂級多因子策略 (V3.7)": ("strategies.isaac", "run_isaac_strategy"),
    }

    # 從 session_state 取得 finlab_token（由 app.py 從 secrets.toml 載入）
    finlab_token = st.session_state.get('finlab_token', '')

    with st.expander("策略設定", expanded=True):
        all_options = list(PRESET_STRATEGIES.keys()) + ["📂 上傳自訂策略"]
        strategy_type = st.selectbox("選擇策略", all_options)

        # Custom strategy upload
        uploaded_file = None
        upload_valid = False
        if strategy_type == "📂 上傳自訂策略":
            u_c1, u_c2 = st.columns([3, 1])
            with u_c1:
                uploaded_file = st.file_uploader("上傳策略檔案 (.py)，需包含 `run_strategy(api_token)` 函式", type=["py"])
            with u_c2:
                try:
                    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "strategies", "template.py")
                    with open(template_path, "rb") as f:
                        st.download_button("📥 策略範本", f, file_name="template_strategy.py", mime="text/x-python", use_container_width=True)
                except Exception:
                    pass

            # --- Upload format check ---
            if uploaded_file is not None:
                import ast
                source_bytes = uploaded_file.getvalue()
                issues = []

                # 1. Encoding check
                try:
                    source_code = source_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        source_code = source_bytes.decode('utf-8-sig')
                    except Exception:
                        source_code = None
                        issues.append("❌ **編碼錯誤** — 檔案非 UTF-8 編碼，請確認儲存格式。")

                # 2. Syntax check
                tree = None
                if source_code is not None:
                    try:
                        tree = ast.parse(source_code)
                    except SyntaxError as e:
                        issues.append(f"❌ **語法錯誤** (第 {e.lineno} 行) — `{e.msg}`")

                # 3. Required function check
                if tree is not None:
                    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    if 'run_strategy' not in func_names:
                        issues.append("❌ **缺少入口函式** — 找不到 `def run_strategy(...):`，平台需要此函式作為回測入口。")
                    else:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == 'run_strategy':
                                args = [a.arg for a in node.args.args]
                                if len(args) == 0:
                                    issues.append("⚠️ **參數不足** — `run_strategy()` 需要至少一個參數 `api_token`，平台會自動傳入 Finlab Token。")
                                has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))
                                if not has_return:
                                    issues.append("⚠️ **缺少回傳值** — `run_strategy()` 應回傳 Finlab 回測報告物件 (report)。")
                                stress_params = {'stop_loss', 'take_profit'}
                                has_stress = stress_params.issubset(set(args))
                                has_kwargs = node.args.kwarg is not None
                                st.session_state['_upload_supports_stress'] = has_stress or has_kwargs
                                break

                # 4. Key component check
                if source_code is not None:
                    warnings = []
                    if 'finlab' not in source_code and 'backtest' not in source_code:
                        warnings.append("⚠️ **未使用 Finlab** — 未偵測到 `finlab` 或 `backtest` 相關程式碼，回測可能無法產生報告。")
                    if 'position' not in source_code.lower():
                        warnings.append("⚠️ **未建立部位** — 未偵測到 `position` 變數，策略需要建立持倉矩陣。")
                    if 'backtest.sim' not in source_code and 'safe_finlab_sim' not in source_code:
                        warnings.append("⚠️ **未執行回測** — 未偵測到 `backtest.sim()` 或 `safe_finlab_sim()` 呼叫。")
                    if 'from data_provider' in source_code:
                        warnings.append("⚠️ **import 路徑錯誤** — `from data_provider` 應改為 `from data.provider`。")
                    if "method='ffill'" in source_code and '.reindex(' in source_code:
                        warnings.append("⚠️ **已棄用 API** — `.reindex(method='ffill')` 在 Pandas 2.1+ 已移除，請改用 `.reindex(...).ffill()`。")
                    issues.extend(warnings)

                # 5. Show check results
                if not issues:
                    upload_valid = True
                    supports_stress = st.session_state.get('_upload_supports_stress', False)
                    stress_info = "✅ 支援壓力測試" if supports_stress else "ℹ️ 不支援壓力測試（如需支援，請加入 <code>stop_loss</code> 和 <code>take_profit</code> 參數）"
                    st.markdown(f"""
                    <div class="alert-card alert-ok">
                        <div class="alert-title">✅ 格式檢查通過</div>
                        <div class="alert-body">
                            檔案 <strong>{uploaded_file.name}</strong> 格式正確，包含 <code>run_strategy()</code> 入口函式，可以執行回測。<br>
                            <span style="font-size:0.8rem; color:#64748b;">{stress_info}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    has_critical = any(issue.startswith("❌") for issue in issues)
                    upload_valid = not has_critical
                    alert_type = "alert-danger" if has_critical else "alert-warn"
                    title = "格式檢查未通過" if has_critical else "格式檢查警告"
                    items_html = "".join(f"<div style='margin:4px 0;'>{issue}</div>" for issue in issues)
                    st.markdown(f"""
                    <div class="alert-card {alert_type}">
                        <div class="alert-title">{'❌' if has_critical else '⚠️'} {title} — {uploaded_file.name}</div>
                        <div class="alert-body">{items_html}</div>
                    </div>""", unsafe_allow_html=True)
                    if has_critical:
                        st.markdown("""<div style="padding:8px 12px; background:#1e293b; border-radius:6px; margin-top:8px; font-size:0.8rem; color:#64748b;">
                            💡 <strong>提示</strong>：請下載「策略範本」參考正確格式，確保包含 <code>def run_strategy(api_token):</code> 函式並回傳 Finlab 回測報告。
                        </div>""", unsafe_allow_html=True)

        run_btn = st.button("🔬 執行回測", use_container_width=True, type="primary")

    # --- Execute Backtest ---
    if run_btn:
        if not finlab_token:
            st.error("請輸入 Finlab API Token")
        elif strategy_type == "📂 上傳自訂策略" and uploaded_file is None:
            st.error("請上傳策略檔案")
        elif strategy_type == "📂 上傳自訂策略" and not upload_valid:
            st.error("策略檔案格式檢查未通過，請修正後重新上傳。")
        else:
            if strategy_type == "📂 上傳自訂策略":
                run_label = uploaded_file.name.replace('.py', '')
            else:
                run_label = strategy_type.split("(")[0].strip().split(" ")[0] if "(" in strategy_type else strategy_type[:10]

            with cyber_spinner("BACKTESTING", f"{strategy_type} 策略回測中..."):
                try:
                    import importlib
                    if strategy_type == "📂 上傳自訂策略":
                        import importlib.util, sys as _sys, tempfile as _tmpf
                        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        temp_fd = _tmpf.NamedTemporaryFile(suffix='.py', dir=base_dir, delete=False, prefix='_strat_')
                        temp_fn = temp_fd.name
                        try:
                            temp_fd.write(uploaded_file.getbuffer()); temp_fd.close()
                            spec = importlib.util.spec_from_file_location("custom_strategy", temp_fn)
                            module = importlib.util.module_from_spec(spec)
                            _sys.modules["custom_strategy"] = module
                            spec.loader.exec_module(module)
                            if not hasattr(module, 'run_strategy'):
                                st.error("找不到 `run_strategy(api_token)` 函式。"); st.stop()
                            report = module.run_strategy(finlab_token)
                            st.session_state['_custom_strategy_module'] = module
                        finally:
                            if os.path.exists(temp_fn): os.remove(temp_fn)
                    else:
                        mod_path, func_name = PRESET_STRATEGIES[strategy_type]
                        strat = importlib.import_module(mod_path)
                        importlib.reload(strat)
                        report = getattr(strat, func_name)(finlab_token)

                    equity = getattr(report, 'creturn', None)
                    benchmark = getattr(report, 'benchmark', None)
                    trades = report.get_trades() if hasattr(report, 'get_trades') else pd.DataFrame()
                    stats = report.get_stats() if hasattr(report, 'get_stats') else {}

                    st.session_state.backtest_report = report
                    st.session_state.current_strategy = strategy_type

                    st.session_state.backtest_results[run_label] = {
                        'strategy_type': strategy_type,
                        'report': report,
                        'equity': equity,
                        'benchmark': benchmark,
                        'trades': trades,
                        'stats': stats,
                        'timestamp': time.strftime('%H:%M:%S'),
                    }
                    st.success(f"✅ {run_label} 回測完成，已加入比較清單（共 {len(st.session_state.backtest_results)} 組）")

                except Exception as e:
                    st.error(f"回測錯誤: {e}")
                    import traceback; logging.error(traceback.format_exc())
                    st.caption("詳細錯誤已記錄至系統日誌。")

    # --- Results Display ---
    if st.session_state.backtest_report is not None:
        report = st.session_state.backtest_report
        equity = getattr(report, 'creturn', None)
        benchmark = getattr(report, 'benchmark', None)
        drawdown = equity / equity.cummax() - 1 if equity is not None and len(equity) > 0 else None
        trades = report.get_trades() if hasattr(report, 'get_trades') else pd.DataFrame()
        stats = report.get_stats()
        cagr = stats.get('cagr', 0); mdd = stats.get('max_drawdown', 0)
        win_rate = stats.get('win_ratio', 0)
        avg_win = trades[trades['return'] > 0]['return'].mean() if not trades.empty else 0
        avg_loss = abs(trades[trades['return'] <= 0]['return'].mean()) if not trades.empty else 0
        risk_reward = avg_win / avg_loss if avg_loss != 0 else 0
        avg_hold_win = trades[trades['return'] > 0]['period'].mean() if not trades.empty else 0
        avg_hold_loss = trades[trades['return'] <= 0]['period'].mean() if not trades.empty else 0

        has_comparison = len(st.session_state.backtest_results) >= 2

        if has_comparison:
            tab1, tab_cmp, tab2, tab3, tab_cost, tab_log, tab_mc = st.tabs(["📊 核心績效", "⚔️ 策略比較", "🛡️ 壓力測試", "📋 交易明細", "💰 交易成本", "📓 回測日誌", "🎲 統計驗證"])
        else:
            tab1, tab2, tab3, tab_cost, tab_log, tab_mc = st.tabs(["📊 核心績效", "🛡️ 壓力測試", "📋 交易明細", "💰 交易成本", "📓 回測日誌", "🎲 統計驗證"])

        # ------ TAB: Core Performance ------
        with tab1:
            st.markdown(f"""
            <div class="kpi-strip">
                <div class="kpi-item" style="border-left:3px solid #3b82f6">
                    <div class="kpi-label">CAGR</div><div class="kpi-value" style="color:#3b82f6">{cagr*100:.1f}%</div>
                </div>
                <div class="kpi-item" style="border-left:3px solid #ef4444">
                    <div class="kpi-label">MDD</div><div class="kpi-value" style="color:#ef4444">{mdd*100:.1f}%</div>
                </div>
                <div class="kpi-item">
                    <div class="kpi-label">勝率</div><div class="kpi-value">{win_rate*100:.0f}%</div>
                </div>
                <div class="kpi-item">
                    <div class="kpi-label">風報比</div><div class="kpi-value">{risk_reward:.2f}</div>
                </div>
                <div class="kpi-item">
                    <div class="kpi-label">持有天 (贏/輸)</div><div class="kpi-value">{avg_hold_win:.0f} / {avg_hold_loss:.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if equity is not None:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
                fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode='lines', name='策略', line=dict(color='#3b82f6', width=2)), row=1, col=1)
                if benchmark is not None:
                    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, mode='lines', name='大盤', line=dict(color='#64748b', width=1, dash='dot')), row=1, col=1)
                if drawdown is not None:
                    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', name='回撤', line=dict(color='#ef4444', width=1), fill='tozeroy', fillcolor='rgba(239,68,68,0.15)'), row=2, col=1)
                _plotly_dark_layout(fig, height=550, title_text=f'{st.session_state.current_strategy} 權益曲線')
                st.plotly_chart(fig, use_container_width=True)

        # ------ TAB: Strategy Comparison ------
        if has_comparison:
            with tab_cmp:
                results_dict = st.session_state.backtest_results

                st.markdown(f'<p class="sec-header">比較清單（{len(results_dict)} 組策略）</p>', unsafe_allow_html=True)
                del_cols = st.columns(min(len(results_dict), 6))
                to_delete = None
                for i, (label, _) in enumerate(results_dict.items()):
                    col_idx = i % min(len(results_dict), 6)
                    if del_cols[col_idx].button(f"❌ {label}", key=f"del_{label}", use_container_width=True):
                        to_delete = label
                if to_delete:
                    del st.session_state.backtest_results[to_delete]
                    st.rerun()

                # KPI comparison table
                st.markdown('<p class="sec-header">績效指標比較</p>', unsafe_allow_html=True)
                cmp_rows = ""
                STRATEGY_COLORS = ['#3b82f6', '#f59e0b', '#22c55e', '#ef4444', '#a855f7', '#ec4899', '#06b6d4', '#84cc16']
                for i, (label, data) in enumerate(results_dict.items()):
                    s = data['stats']
                    t = data['trades']
                    c = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                    s_cagr = s.get('cagr', 0)
                    s_mdd = s.get('max_drawdown', 0)
                    s_wr = s.get('win_ratio', 0)
                    s_sharpe = s.get('sharpe', 0)
                    s_avg_w = t[t['return'] > 0]['return'].mean() if not t.empty and len(t[t['return'] > 0]) > 0 else 0
                    s_avg_l = abs(t[t['return'] <= 0]['return'].mean()) if not t.empty and len(t[t['return'] <= 0]) > 0 else 0
                    s_rr = s_avg_w / s_avg_l if s_avg_l != 0 else 0
                    n_trades = len(t)
                    cmp_rows += f"""<tr>
                        <td style="text-align:left"><span style="color:{c}; font-weight:800;">●</span> <strong>{label}</strong></td>
                        <td style="color:{'#3b82f6' if s_cagr > 0 else '#ef4444'}">{s_cagr*100:.1f}%</td>
                        <td style="color:#ef4444">{s_mdd*100:.1f}%</td>
                        <td>{s_wr*100:.0f}%</td>
                        <td>{s_rr:.2f}</td>
                        <td>{s_sharpe:.2f}</td>
                        <td>{n_trades}</td>
                        <td style="font-size:0.75rem;color:#64748b">{data['timestamp']}</td>
                    </tr>"""
                st.markdown(f"""<table class="pro-table"><thead><tr>
                    <th style="text-align:left">策略</th><th>CAGR</th><th>MDD</th><th>勝率</th><th>風報比</th><th>Sharpe</th><th>交易數</th><th>時間</th>
                </tr></thead><tbody>{cmp_rows}</tbody></table>""", unsafe_allow_html=True)

                # Equity curve overlay
                st.markdown('<p class="sec-header">權益曲線疊圖</p>', unsafe_allow_html=True)
                fig_cmp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])

                for i, (label, data) in enumerate(results_dict.items()):
                    eq = data['equity']
                    c = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                    if eq is not None:
                        fig_cmp.add_trace(go.Scatter(
                            x=eq.index, y=eq.values, mode='lines', name=label,
                            line=dict(color=c, width=2),
                        ), row=1, col=1)
                        dd = eq / eq.cummax() - 1
                        fig_cmp.add_trace(go.Scatter(
                            x=dd.index, y=dd.values, mode='lines', name=f'{label} DD',
                            line=dict(color=c, width=1, dash='dot'), showlegend=False,
                        ), row=2, col=1)

                for label, data in results_dict.items():
                    bm = data.get('benchmark')
                    if bm is not None:
                        fig_cmp.add_trace(go.Scatter(
                            x=bm.index, y=bm.values, mode='lines', name='大盤',
                            line=dict(color='#475569', width=1, dash='dash'),
                        ), row=1, col=1)
                        break

                fig_cmp.update_yaxes(title_text="累積報酬", row=1, col=1)
                fig_cmp.update_yaxes(title_text="回撤", row=2, col=1)
                _plotly_dark_layout(fig_cmp, height=600, title_text='策略比較')
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Monthly return comparison
                st.markdown('<p class="sec-header">月報酬比較</p>', unsafe_allow_html=True)
                monthly_data = {}
                for label, data in results_dict.items():
                    eq = data['equity']
                    if eq is not None and len(eq) > 20:
                        monthly = eq.resample('ME').last().pct_change().dropna()
                        monthly_data[label] = monthly

                if monthly_data:
                    fig_monthly = go.Figure()
                    for i, (label, monthly) in enumerate(monthly_data.items()):
                        c = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                        fig_monthly.add_trace(go.Box(
                            y=monthly.values, name=label, marker_color=c,
                            boxmean=True,
                        ))
                    _plotly_dark_layout(fig_monthly, height=350, title_text='月報酬分佈')
                    fig_monthly.update_yaxes(tickformat='.1%')
                    st.plotly_chart(fig_monthly, use_container_width=True)

        # ------ TAB: Stress Test ------
        with tab2:
            current_strat = st.session_state.current_strategy or ""
            is_isaac = "Isaac" in current_strat
            is_custom = "上傳" in current_strat or "📂" in current_strat
            custom_supports_stress = st.session_state.get('_upload_supports_stress', False)
            can_stress = is_isaac or (is_custom and custom_supports_stress)

            st.markdown('<p class="sec-header">參數穩定性壓力測試</p>', unsafe_allow_html=True)

            if not can_stress:
                if is_custom and not custom_supports_stress:
                    st.markdown("""
                    <div class="alert-card alert-warn">
                        <div class="alert-title">⚠️ 此自訂策略不支援壓力測試</div>
                        <div class="alert-body">
                            您上傳的策略 <code>run_strategy()</code> 未包含 <code>stop_loss</code> 和 <code>take_profit</code> 參數。<br>
                            如需支援壓力測試，請修改函式簽名為：
                            <pre style="background:#0f172a; padding:8px; border-radius:4px; margin-top:8px; font-size:0.8rem;">def run_strategy(api_token, stop_loss=0.08, take_profit=0.20):</pre>
                            平台將自動傳入不同的停損/停利組合進行網格搜索。
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-card alert-info">
                        <div class="alert-body">壓力測試支援 Isaac 策略及包含 <code>stop_loss</code>/<code>take_profit</code> 參數的自訂策略。</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("對策略的停損/停利參數進行網格搜索，檢驗績效穩定度。")
                c_p1, c_p2 = st.columns(2)
                stop_loss_range = c_p1.slider("停損範圍 (%)", 5, 15, (8, 12))
                take_profit_range = c_p2.slider("停利範圍 (%)", 15, 40, (20, 30))
                if st.button("🔥 開始壓力測試"):
                    import importlib

                    if is_isaac:
                        import strategies.isaac as strategy_mod; importlib.reload(strategy_mod)
                        stress_func = lambda token, sl, tp: strategy_mod.run_isaac_strategy(token, stop_loss=sl, take_profit=tp)
                        stress_label = "Isaac V3.7"
                    else:
                        custom_mod = st.session_state.get('_custom_strategy_module')
                        if custom_mod and hasattr(custom_mod, 'run_strategy'):
                            stress_func = lambda token, sl, tp: custom_mod.run_strategy(token, stop_loss=sl, take_profit=tp)
                            stress_label = "自訂策略"
                        else:
                            st.error("請先重新上傳並執行自訂策略。"); st.stop()

                    results = []
                    sl_steps = list(range(stop_loss_range[0], stop_loss_range[1]+1, 2))
                    tp_steps = list(range(take_profit_range[0], take_profit_range[1]+1, 5))
                    total_steps = len(sl_steps) * len(tp_steps)
                    progress_bar = st.progress(0); step_count = 0
                    for sl in sl_steps:
                        for tp in tp_steps:
                            try:
                                rep = stress_func(finlab_token, sl/100, tp/100)
                                sg = rep.get_stats()
                                results.append({'停損 (%)': sl, '停利 (%)': tp, 'CAGR': sg.get('cagr', 0), 'Sharpe': sg.get('sharpe', 0)})
                            except Exception:
                                pass
                            step_count += 1
                            progress_bar.progress(min(step_count / max(total_steps, 1), 1.0))
                    df_grid = pd.DataFrame(results)
                    if not df_grid.empty:
                        pivot_table = df_grid.pivot(index='停損 (%)', columns='停利 (%)', values='CAGR')
                        fig_heat = px.imshow(pivot_table, labels=dict(x="停利 (%)", y="停損 (%)", color="CAGR"),
                                             color_continuous_scale='RdYlGn', text_auto='.1%')
                        _plotly_dark_layout(fig_heat, height=400, title_text=f'{stress_label} CAGR 參數熱力圖')
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.warning("所有參數組合均回測失敗，請確認策略邏輯。")

        # ------ TAB: Trade Details ------
        with tab3:
            if not trades.empty:
                rename_map = {"stock_id": "代碼", "entry_date": "進場", "exit_date": "出場",
                              "entry_price": "進場價", "exit_price": "出場價", "return": "報酬率",
                              "mae": "MAE", "mfe": "MFE", "period": "天數"}
                td = trades.copy(); td.rename(columns=rename_map, inplace=True)
                if '進場' in td.columns: td['進場'] = pd.to_datetime(td['進場'])
                if '出場' in td.columns: td['出場'] = pd.to_datetime(td['出場'], errors='coerce')

                csv = td.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下載交易明細 (.csv)", data=csv,
                                   file_name=f'trades_{time.strftime("%Y%m%d_%H%M")}.csv', mime='text/csv')

                def hl(val):
                    if pd.isna(val): return ''
                    if isinstance(val, (int, float)): return 'color: #ef4444' if val > 0 else 'color: #22c55e'
                    return ''
                cols_to_show = [c for c in ['代碼', '進場', '出場', '進場價', '出場價', '報酬率', '天數', 'MAE', 'MFE'] if c in td.columns]
                st.dataframe(
                    td[cols_to_show].sort_values("進場", ascending=False).head(500).style.format(
                        {'報酬率': '{:.2%}', 'MAE': '{:.2%}', 'MFE': '{:.2%}', '進場價': '{:.2f}', '出場價': '{:.2f}'}, na_rep="N/A"
                    ).map(hl, subset=['報酬率']),
                    use_container_width=True, height=600,
                )
            else:
                st.info("無交易紀錄")

        # ------ TAB: Trading Cost ------
        with tab_cost:
            try:
                from analysis.cost_analysis import analyze_trading_costs, render_cost_chart, render_cost_over_time

                cost_data = analyze_trading_costs(trades)

                # KPI strip
                cyber_kpi_strip([
                    {'label': '總交易數', 'value': f"{cost_data['total_trades']}", 'accent': '#3b82f6'},
                    {'label': '總成本 (NTD)', 'value': f"NT${cost_data['total_cost']:,.0f}", 'accent': '#ef4444', 'color': '#ef4444'},
                    {'label': '平均成本/筆 (%)', 'value': f"{cost_data['avg_cost_per_trade_pct']:.3f}%", 'accent': '#f59e0b'},
                    {'label': '年化成本拖累', 'value': f"{cost_data['cost_drag_annualized']:.3f}%", 'accent': '#a855f7', 'color': '#a855f7'},
                    {'label': '成本/毛利比', 'value': f"{cost_data['cost_ratio']:.2%}", 'accent': '#ec4899'},
                ])

                # Cost breakdown pie chart
                st.markdown('<p class="sec-header">成本結構分析</p>', unsafe_allow_html=True)
                fig_pie = render_cost_chart(cost_data)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("無成本資料可顯示")

                # Cumulative cost over time chart
                st.markdown('<p class="sec-header">累計成本趨勢</p>', unsafe_allow_html=True)
                fig_cum = render_cost_over_time(cost_data)
                if fig_cum:
                    st.plotly_chart(fig_cum, use_container_width=True)
                else:
                    st.info("無交易資料可繪製趨勢圖")

                # Per-trade cost table (top 20 most expensive)
                by_trade_list = cost_data.get('by_trade', [])
                if by_trade_list:
                    st.markdown('<p class="sec-header">單筆成本明細（前 20 高）</p>', unsafe_allow_html=True)
                    sorted_trades = sorted(by_trade_list, key=lambda x: x['total_cost'], reverse=True)[:20]
                    headers = ['代碼', '進場價', '出場價', '買入手續費', '賣出手續費', '證交稅', '滑價', '總成本', '成本比 (%)']
                    rows = []
                    for t in sorted_trades:
                        rows.append([
                            t.get('stock_id', 'N/A'),
                            f"{t['entry_price']:,.2f}",
                            f"{t['exit_price']:,.2f}",
                            f"${t['commission_buy']:,.0f}",
                            f"${t['commission_sell']:,.0f}",
                            f"${t['tax']:,.0f}",
                            f"${t['slippage']:,.0f}",
                            f"<strong style='color:#ef4444'>${t['total_cost']:,.0f}</strong>",
                            f"{t['cost_pct']*100:.3f}%",
                        ])
                    cyber_table(headers, rows)

            except Exception as e:
                st.error(f"交易成本分析失敗: {e}")
                import traceback; logging.error(traceback.format_exc())

        # ------ TAB: Backtest Log ------
        with tab_log:
            st.markdown('<p class="sec-header">回測決策日誌</p>', unsafe_allow_html=True)
            st.caption("回測日誌記錄進出場決策，供策略調整參考。可匯出 JSON 做進一步分析。")

            if not trades.empty:
                try:
                    from analysis.backtest_logger import BacktestLogger, render_backtest_log_summary

                    # 從交易明細重建回測日誌
                    strategy_name = st.session_state.current_strategy or "unknown"
                    bt_logger = BacktestLogger(strategy_name)

                    for _, row in trades.iterrows():
                        ticker = str(row.get('stock_id', ''))
                        entry_date = str(row.get('entry_date', ''))
                        exit_date = str(row.get('exit_date', ''))
                        entry_price = float(row.get('entry_price', 0)) if pd.notna(row.get('entry_price')) else 0
                        exit_price = float(row.get('exit_price', 0)) if pd.notna(row.get('exit_price')) else 0
                        ret = float(row.get('return', 0)) if pd.notna(row.get('return')) else 0
                        period = int(row.get('period', 0)) if pd.notna(row.get('period')) else 0

                        # 根據報酬率推斷信號類型
                        if ret > 0.15:
                            signal_type = "A"
                        elif ret > 0.05:
                            signal_type = "B"
                        elif ret > 0:
                            signal_type = "C"
                        else:
                            signal_type = "D"

                        # 推斷出場類型
                        if ret <= -0.08:
                            exit_type = "trail_stop"
                        elif period > 60:
                            exit_type = "time_stop"
                        elif ret > 0:
                            exit_type = "signal_d"
                        else:
                            exit_type = "ma_break"

                        bt_logger.log_entry(entry_date, ticker, signal_type, abs(ret) * 10, entry_price)
                        bt_logger.log_exit(exit_date, ticker, exit_type, exit_price, ret * 100, period)

                    # 顯示摘要
                    summary = bt_logger.get_summary()
                    md_text = render_backtest_log_summary(summary)
                    st.markdown(md_text)

                    # 匯出按鈕
                    col_save, col_dl = st.columns(2)
                    with col_save:
                        if st.button("💾 儲存回測日誌", key="save_bt_log"):
                            filepath = bt_logger.save()
                            st.success(f"已儲存: {filepath}")
                    with col_dl:
                        import json
                        log_json = json.dumps({
                            "strategy_name": summary.get("strategy_name", ""),
                            "summary": summary,
                            "entries": bt_logger.entries[:50],
                            "exits": bt_logger.exits[:50],
                        }, ensure_ascii=False, indent=2)
                        st.download_button(
                            "📥 下載日誌 (.json)",
                            data=log_json.encode('utf-8'),
                            file_name=f'backtest_log_{time.strftime("%Y%m%d_%H%M")}.json',
                            mime='application/json',
                        )

                except Exception as e:
                    st.error(f"回測日誌建置失敗: {e}")
                    import traceback; logging.error(traceback.format_exc())
            else:
                st.info("無交易紀錄，無法產生回測日誌")

        # ------ TAB: Monte Carlo Statistical Validation ------
        with tab_mc:
            st.markdown('<p class="sec-header">Monte Carlo 統計驗證</p>', unsafe_allow_html=True)
            st.caption("用歷史交易的報酬分佈進行 1000 次隨機模擬，驗證策略績效是否具統計顯著性。")

            if st.button("🎲 執行 Monte Carlo 模擬", key="mc_btn"):
                with cyber_spinner("MONTE CARLO", "1000 次路徑模擬中..."):
                    try:
                        from analysis.monte_carlo import run_monte_carlo as run_mc, render_monte_carlo_chart, render_monte_carlo_distribution
                        mc = run_mc(trades)
                        if mc:
                            kc1, kc2, kc3, kc4 = st.columns(4)
                            kc1.metric("獲利機率", f"{mc['prob_profit']*100:.1f}%")
                            kc2.metric("中位數報酬", f"{mc['median_return']:.1f}%")
                            kc3.metric("最差 5%", f"{mc['percentile_5']:.1f}%")
                            kc4.metric("中位數 MDD", f"{mc['median_mdd']:.1f}%")

                            fig1 = render_monte_carlo_chart(mc)
                            if fig1:
                                st.plotly_chart(fig1, use_container_width=True)

                            fig2 = render_monte_carlo_distribution(mc)
                            if fig2:
                                st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.warning("交易筆數不足（需至少 10 筆）")
                    except Exception as e:
                        st.error(f"Monte Carlo 模擬失敗: {e}")

    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#475569;">
            <div style="font-size:3rem; margin-bottom:12px;">🧬</div>
            <div style="font-size:1.1rem; font-weight:600; color:#64748b; margin-bottom:8px;">選擇策略並按下「執行回測」開始</div>
            <div style="font-size:0.8rem; color:#334155;">
                支援 4 種內建策略 + 自訂 .py 上傳<br>
                可執行多次回測，在「策略比較」分頁進行 A/B 對照
            </div>
        </div>
        """, unsafe_allow_html=True)
