"""
Page: Trading (交易執行) — 扁平化單層 tab 架構
一次初始化，直接呼叫各子模組的內部渲染函式
"""
import streamlit as st
import os

from ui.components import cyber_header


def render():
    cyber_header("交易執行", "下單管理 | 持倉監控 | 風險控制")

    # ── 一次初始化交易上下文 (避免 7 次重複建立) ──
    try:
        from ui.pages.auto_trading import _get_trading_context, _render_control_panel, \
            _render_position_charts, _render_order_book, _render_config
        trader, config, status, pt, pt_status, positions = _get_trading_context()
        auto_ok = True
    except Exception as e:
        auto_ok = False
        st.warning(f"自動交易模組載入失敗: {e}")

    # 一層扁平 tab
    tab_ctrl, tab_charts, tab_rec, tab_paper, tab_scan, tab_orders, tab_strat, tab_config = st.tabs([
        "🎮 控制面板",
        "📊 持倉線圖",
        "📋 每日精選",
        "💰 模擬交易",
        "📡 突破偵測",
        "📝 委託紀錄",
        "🧬 策略管理",
        "⚙️ 系統設定",
    ])

    # ── 控制面板 ──
    with tab_ctrl:
        if auto_ok:
            try:
                _render_control_panel(trader, config, status, pt, positions)
            except Exception as e:
                st.error(f"控制面板錯誤: {e}")
        else:
            st.info("請確認自動交易模組設定正確")

    # ── 持倉線圖 ──
    with tab_charts:
        if auto_ok:
            try:
                _render_position_charts(positions, pt_status)
            except Exception as e:
                st.error(f"持倉線圖錯誤: {e}")
        else:
            st.info("請確認自動交易模組設定正確")

    # ── 每日精選 ──
    with tab_rec:
        try:
            from ui.pages.live_monitor import render_daily_picks
            render_daily_picks()
        except Exception as e:
            st.error(f"每日精選錯誤: {e}")

    # ── 模擬交易 ──
    with tab_paper:
        try:
            from ui.pages.live_monitor import render_paper_trading
            render_paper_trading()
        except Exception as e:
            st.error(f"模擬交易錯誤: {e}")

    # ── 突破偵測 ──
    with tab_scan:
        try:
            from ui.pages.alerts import render as render_alerts
            render_alerts(_embedded=True)
        except Exception as e:
            st.error(f"突破偵測錯誤: {e}")

    # ── 委託紀錄 ──
    with tab_orders:
        if auto_ok:
            try:
                _render_order_book(trader)
            except Exception as e:
                st.error(f"委託紀錄錯誤: {e}")
        else:
            st.info("請確認自動交易模組設定正確")

    # ── 策略管理 ──
    with tab_strat:
        try:
            _render_strategy_manager()
        except Exception as e:
            st.error(f"策略管理錯誤: {e}")

    # ── 系統設定 ──
    with tab_config:
        if auto_ok:
            try:
                _render_config(trader, config)
            except Exception as e:
                st.error(f"系統設定錯誤: {e}")
        else:
            st.info("請確認自動交易模組設定正確")


def _render_strategy_manager():
    """策略管理 — 查看/上傳/刪除自訂策略。"""
    from ui.components import cyber_spinner
    from data.signal_format import AVAILABLE_STRATEGIES
    import glob

    st.markdown("#### 可用策略")
    for s in AVAILABLE_STRATEGIES:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:8px 14px;margin-bottom:4px;background:rgba(0,240,255,0.04);'
            f'border-radius:6px;border-left:3px solid #3b82f6">'
            f'<span style="font-weight:700;font-size:0.85rem;color:#e2e8f0">{s["label"]}</span>'
            f'<span style="font-size:0.7rem;color:#64748b;font-family:JetBrains Mono,monospace">{s["source_tag"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Upload custom strategy
    st.markdown("#### 上傳自訂策略")
    st.caption("上傳 .py 檔案，需包含 `run_strategy(api_token)` 函式，回傳 Finlab 回測報告。")

    uploaded = st.file_uploader("選擇策略檔案", type=["py"], key="strat_upload")
    if uploaded:
        import ast
        source = uploaded.read().decode('utf-8')
        issues = []

        # Validate
        try:
            tree = ast.parse(source)
            has_func = any(
                isinstance(n, ast.FunctionDef) and n.name == 'run_strategy'
                for n in ast.walk(tree)
            )
            if not has_func:
                issues.append("找不到 `run_strategy()` 函式")
        except SyntaxError as e:
            issues.append(f"語法錯誤: {e}")

        if issues:
            for iss in issues:
                st.error(f"❌ {iss}")
        else:
            custom_dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                           'strategies', 'custom')
            os.makedirs(custom_dest_dir, exist_ok=True)
            dest = os.path.join(custom_dest_dir, uploaded.name)
            if st.button(f"儲存到 strategies/custom/{uploaded.name}", type="primary"):
                with open(dest, 'w', encoding='utf-8') as f:
                    f.write(source)
                st.success(f"已儲存: strategies/custom/{uploaded.name}")

    # ── 內建策略清單（僅展示，不可刪除）──
    strat_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'strategies')
    all_files = sorted(glob.glob(os.path.join(strat_dir, '*.py')))

    # 不應顯示的工具/測試檔
    _SKIP = {'__init__.py', 'template.py', 'advanced_test.py', 'allocation_test.py',
             'decay_test.py', 'monte_carlo.py', 'opt_test.py', 'rotation_test.py',
             'round3_test.py', 'v37_validation.py'}

    builtin_files = [f for f in all_files if os.path.basename(f) not in _SKIP]
    if builtin_files:
        st.markdown("#### 內建策略檔案")
        st.caption("以下為系統內建策略，供回測及交易功能使用，無法刪除。")
        for fp in builtin_files:
            fname = os.path.basename(fp)
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:6px 14px;margin-bottom:3px;'
                f'background:rgba(0,240,255,0.03);border-radius:6px;border-left:3px solid #334155">'
                f'<span style="font-size:0.8rem;color:#94a3b8;font-family:JetBrains Mono,monospace">'
                f'strategies/{fname}</span>'
                f'<span class="tag" style="font-size:0.55rem;background:#1e293b;color:#64748b;margin-left:8px">內建</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── 自訂策略（可刪除）──
    custom_dir = os.path.join(strat_dir, 'custom')
    os.makedirs(custom_dir, exist_ok=True)
    custom_files = sorted(glob.glob(os.path.join(custom_dir, '*.py')))

    st.markdown("#### 自訂策略管理")
    if custom_files:
        backup_dir = os.path.join(strat_dir, '.backup')
        os.makedirs(backup_dir, exist_ok=True)

        for fp in custom_files:
            fname = os.path.basename(fp)
            c1, c2 = st.columns([6, 1])
            c1.markdown(
                f'<div style="padding:6px 14px;background:rgba(139,92,246,0.06);border-radius:6px;'
                f'border-left:3px solid #8b5cf6;font-size:0.8rem;font-family:JetBrains Mono,monospace;color:#e2e8f0">'
                f'custom/{fname}'
                f'<span class="tag tag-new" style="font-size:0.55rem;margin-left:8px">自訂</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if c2.button("🗑️", key=f"del_{fname}", help=f"刪除 {fname}"):
                st.session_state[f'_confirm_del_{fname}'] = True

            if st.session_state.get(f'_confirm_del_{fname}'):
                st.warning(f"確定要刪除 **{fname}** 嗎？檔案會備份到 `strategies/.backup/`，可隨時還原。")
                cc1, cc2 = st.columns(2)
                if cc1.button(f"確認刪除 {fname}", key=f"confirm_{fname}", type="primary"):
                    import shutil
                    from datetime import datetime as _dt
                    backup_name = f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{fname}"
                    shutil.copy2(fp, os.path.join(backup_dir, backup_name))
                    os.remove(fp)
                    st.session_state.pop(f'_confirm_del_{fname}', None)
                    st.success(f"已刪除 {fname}（備份: .backup/{backup_name}）")
                    st.rerun()
                if cc2.button("取消", key=f"cancel_{fname}"):
                    st.session_state.pop(f'_confirm_del_{fname}', None)
                    st.rerun()
    else:
        st.caption("目前無自訂策略。上傳 .py 後會存入 `strategies/custom/`。")

    # Show backups
    backup_dir = os.path.join(strat_dir, '.backup')
    os.makedirs(backup_dir, exist_ok=True)
    backup_files = sorted(glob.glob(os.path.join(backup_dir, '*.py')))
    if backup_files:
        with st.expander(f"🗄️ 備份區（{len(backup_files)} 個檔案）"):
            for bp in backup_files:
                bname = os.path.basename(bp)
                bc1, bc2 = st.columns([6, 1])
                bc1.markdown(f'<span style="font-size:0.75rem;color:#64748b">`{bname}`</span>', unsafe_allow_html=True)
                if bc2.button("還原", key=f"restore_{bname}"):
                    import shutil
                    restore_dir = os.path.join(strat_dir, 'custom')
                    os.makedirs(restore_dir, exist_ok=True)
                    original_name = '_'.join(bname.split('_')[2:])  # Remove timestamp prefix
                    shutil.copy2(bp, os.path.join(restore_dir, original_name))
                    st.success(f"已還原至 strategies/custom/{original_name}")
                    st.rerun()
