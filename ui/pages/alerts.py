"""智能警報 — 突破偵測 + 價格監控 + 技術指標。"""
import streamlit as st
import json
import os
from datetime import datetime

from ui.components import cyber_header, cyber_table, cyber_spinner
from data.alerts import AlertManager, ALERT_TYPES

SCAN_RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'scan_results.json')

THRESHOLD_LABELS = {
    "price_above": "目標價",
    "price_below": "停損價",
    "rsi_above": "RSI值",
    "rsi_below": "RSI值",
    "volume_spike": "爆量倍數",
}

SIGNAL_COLORS = {
    'critical': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
}


def _load_scan_results():
    """讀取排程器的掃描結果。"""
    path = os.path.normpath(SCAN_RESULTS_PATH)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _render_vcp_bar(r):
    """渲染 VCP 條件指標 (4 項條件的視覺化)。"""
    vcp_score = r.get('vcp_score', 0)
    conditions = r.get('vcp_conditions', {})
    details = r.get('vcp_details', {})

    if vcp_score == 0 and not conditions:
        return ""

    # Condition labels and status
    cond_items = [
        ('收斂', conditions.get('contraction', False), f"帶寬 {details.get('bandwidth_pctl', 1):.0%}"),
        ('縮量', conditions.get('volume_dry', False), f"{details.get('vol_dry_days', 0)} 天低量"),
        ('墊高', conditions.get('higher_low', False), f"低點 {details.get('low_change_pct', 0):+.1f}%"),
        ('趨勢', conditions.get('trend_healthy', False), "MA50>MA150"),
    ]

    vcp_color = '#a855f7' if vcp_score >= 3 else '#64748b'
    dots_html = ""
    for label, ok, detail in cond_items:
        c = '#a855f7' if ok else '#334155'
        icon = '●' if ok else '○'
        dots_html += (
            f'<span title="{label}: {detail}" style="color:{c};cursor:help;font-size:0.7rem">'
            f'{icon} {label}</span> '
        )

    vcp_label = ""
    if vcp_score >= 3:
        vcp_label = '<span style="color:#a855f7;font-weight:700;font-size:0.7rem"> VCP</span>'

    return (
        f'<div style="margin-top:6px;display:flex;align-items:center;gap:6px;'
        f'font-family:JetBrains Mono,monospace">'
        f'<span style="font-size:0.65rem;color:#64748b">VCP [{vcp_score}/4]</span>'
        f'{dots_html}{vcp_label}'
        f'</div>'
    )


def _render_breakout_card(r):
    """渲染單一突破偵測卡片。"""
    info = r.get('signal_info') or {}
    level = info.get('level', 'info')
    color = SIGNAL_COLORS.get(level, '#00f0ff')
    icon = info.get('icon', '📊')
    label = info.get('label', '—')

    price = r.get('price', 0)
    resistance = r.get('resistance')
    support = r.get('support')
    r_score = r.get('resistance_score', 0)
    s_score = r.get('support_score', 0)
    r_methods = r.get('resistance_methods', [])
    s_methods = r.get('support_methods', [])
    dist_r = r.get('distance_to_resistance_pct')
    dist_s = r.get('distance_to_support_pct')
    vol_ratio = r.get('volume_ratio', 0)
    daily_chg = r.get('daily_change_pct', 0)
    source = r.get('source', '')

    # Source badge
    source_badge = {
        'watchlist': '<span class="tag tag-new">自選股</span>',
        'portfolio': '<span class="tag tag-warn">持倉</span>',
        'both': '<span class="tag tag-new">自選股</span><span class="tag tag-warn">持倉</span>',
    }.get(source, '')

    # Volume bar
    vol_bar_width = min(vol_ratio / 3 * 100, 100)
    vol_color = '#ef4444' if vol_ratio >= 1.5 else '#f59e0b' if vol_ratio >= 1.2 else '#22c55e'
    vol_fire = '🔥' if vol_ratio >= 1.5 else ''

    # Resistance/Support info
    r_info = ""
    if resistance:
        score_dots = '●' * r_score + '○' * (3 - r_score)
        dist_text = f"{dist_r*100:+.1f}%" if dist_r is not None else ""
        r_info = (
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px">'
            f'<span style="color:#22c55e;font-size:0.72rem">▼ 壓力 {resistance:,.1f} {dist_text}</span>'
            f'<span style="font-size:0.65rem;color:#64748b" title="{", ".join(r_methods)}">'
            f'<span style="color:#22c55e">{score_dots}</span></span></div>'
        )

    s_info = ""
    if support:
        score_dots = '●' * s_score + '○' * (3 - s_score)
        dist_text = f"{dist_s*100:+.1f}%" if dist_s is not None else ""
        s_info = (
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:3px">'
            f'<span style="color:#ef4444;font-size:0.72rem">▲ 支撐 {support:,.1f} {dist_text}</span>'
            f'<span style="font-size:0.65rem;color:#64748b" title="{", ".join(s_methods)}">'
            f'<span style="color:#ef4444">{score_dots}</span></span></div>'
        )

    # Daily change color
    chg_color = '#ef4444' if daily_chg >= 0 else '#22c55e'
    chg_text = f"{daily_chg*100:+.2f}%"

    return f"""
    <div class="alert-card" style="border-color:{color};padding:12px 16px;margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="font-size:1.1rem;font-weight:900;color:#e2e8f0">
                    {icon} {r.get('name', r['ticker'])}
                </span>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#64748b;margin-left:8px">
                    {r['ticker']}
                </span>
                {source_badge}
            </div>
            <div style="text-align:right">
                <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:#e2e8f0">
                    {price:,.1f}
                </div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.75rem;color:{chg_color}">
                    {chg_text}
                </div>
            </div>
        </div>
        <div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,0.3);border-radius:6px;
                    border-left:3px solid {color}">
            <span style="font-weight:700;font-size:0.8rem;color:{color}">{label}</span>
        </div>
        {r_info}
        {s_info}
        <div style="margin-top:8px;display:flex;align-items:center;gap:8px">
            <span style="font-size:0.68rem;color:#64748b">量比</span>
            <div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden">
                <div style="width:{vol_bar_width}%;height:100%;background:{vol_color};border-radius:3px"></div>
            </div>
            <span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{vol_color}">
                {vol_ratio:.1f}x {vol_fire}
            </span>
        </div>
        {_render_vcp_bar(r)}
    </div>
    """


def render(_embedded=False):
    am = AlertManager()
    if not _embedded:
        cyber_header("智能警報", "突破偵測 | 價格監控 | 技術指標")

    tab_scan, tab_alerts = st.tabs(["📡 突破偵測", "🔔 自訂警報"])

    # ══════════════════════════════════════════
    # TAB 1: 突破偵測
    # ══════════════════════════════════════════
    with tab_scan:
        st.markdown(
            '<div style="font-size:0.75rem;color:#64748b;margin-bottom:12px">'
            '從「自選股」和「投資組合」持倉中掃描壓力/支撐突破信號。'
            '支援獨立排程器 (<code>python scheduler.py</code>) 背景執行。'
            '</div>',
            unsafe_allow_html=True,
        )

        # Controls
        c_scan, c_status = st.columns([1, 2])
        with c_scan:
            scan_clicked = st.button("📡 立即掃描", use_container_width=True, type="primary")

        # Show last scan time
        scan_data = _load_scan_results()
        with c_status:
            if scan_data:
                scan_time = scan_data.get('scan_time', '')
                try:
                    dt = datetime.fromisoformat(scan_time)
                    elapsed = (datetime.now() - dt).total_seconds() / 60
                    if elapsed < 1:
                        time_label = "剛剛"
                    elif elapsed < 60:
                        time_label = f"{int(elapsed)} 分鐘前"
                    elif elapsed < 1440:
                        time_label = f"{int(elapsed/60)} 小時前"
                    else:
                        time_label = f"{int(elapsed/1440)} 天前"
                except Exception:
                    time_label = scan_time[:16]

                st.markdown(
                    f'<div style="padding:10px;background:rgba(0,240,255,0.05);border-radius:8px;'
                    f'border:1px solid rgba(0,240,255,0.1);font-size:0.75rem;color:#94a3b8">'
                    f'上次掃描: <span style="color:var(--neon-cyan)">{time_label}</span> '
                    f'| 已掃 {scan_data.get("total_scanned", 0)} 檔 '
                    f'| {scan_data.get("signals_count", 0)} 檔有信號 '
                    f'| <span style="color:#ef4444">{scan_data.get("critical_count", 0)} 重大</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="padding:10px;background:rgba(0,0,0,0.2);border-radius:8px;'
                    'font-size:0.75rem;color:#64748b">'
                    '尚無掃描紀錄。點擊「立即掃描」或執行 <code>python scheduler.py --once --force</code>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # Execute scan
        if scan_clicked:
            with cyber_spinner("SCANNING", "突破信號偵測中..."):
                try:
                    from data.watchlist import WatchlistManager
                    from data.provider import get_data_provider
                    from analysis.breakout import scan_breakouts

                    # Collect tickers
                    wm = WatchlistManager()
                    watchlist = [
                        {'ticker': s['ticker'], 'name': s.get('name', s['ticker']), 'source': 'watchlist'}
                        for s in wm.get_all()
                    ]

                    portfolio = []
                    try:
                        from config.paths import PAPER_TRADE_PATH as _ptp
                        paper_path = _ptp
                    except ImportError:
                        paper_path = os.path.normpath(os.path.join(
                            os.path.dirname(__file__), '..', '..', 'data', 'paper_trade.json'))
                    if os.path.exists(paper_path):
                        with open(paper_path, 'r', encoding='utf-8') as f:
                            pt_data = json.load(f)
                        for p in pt_data.get('positions', []):
                            portfolio.append({
                                'ticker': p['ticker'],
                                'name': p.get('name', p['ticker']),
                                'source': 'portfolio',
                            })

                    # Merge and dedup
                    seen = set()
                    merged = []
                    for item in watchlist + portfolio:
                        if item['ticker'] not in seen:
                            seen.add(item['ticker'])
                            merged.append(item)

                    if not merged:
                        st.warning("自選股和持倉皆為空。請先到「自選股」或「投資組合」新增標的。")
                    else:
                        provider = get_data_provider("auto", market_type="TW")
                        results = scan_breakouts(merged, provider)

                        signals_count = len([r for r in results if r.get('signal')])
                        critical_count = len([
                            r for r in results
                            if (r.get('signal_info') or {}).get('level') == 'critical'
                        ])

                        # Save results
                        scan_output = {
                            'scan_time': datetime.now().isoformat(),
                            'total_scanned': len(merged),
                            'signals_count': signals_count,
                            'critical_count': critical_count,
                            'results': results,
                        }
                        result_path = os.path.normpath(SCAN_RESULTS_PATH)
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(scan_output, f, ensure_ascii=False, indent=2, default=str)

                        st.success(f"掃描完成！{len(merged)} 檔已分析，{signals_count} 檔有信號。")
                        st.rerun()

                except Exception as e:
                    st.error(f"掃描失敗: {e}")

        # Display results
        if scan_data and scan_data.get('results'):
            results = scan_data['results']

            # Signals first
            signals = [r for r in results if r.get('signal')]
            no_signals = [r for r in results if not r.get('signal')]

            if signals:
                st.markdown('<p class="sec-header">突破信號</p>', unsafe_allow_html=True)

                # Legend
                st.markdown(
                    '<div style="display:flex;gap:16px;margin-bottom:10px;font-size:0.68rem;'
                    'font-family:JetBrains Mono,monospace;color:#64748b;flex-wrap:wrap">'
                    '<span>🔥 VCP突破</span><span>🔴 重大信號</span>'
                    '<span>🟣 VCP成形</span><span>🟡 注意信號</span><span>🔵 參考信號</span>'
                    '<span>●●● S/R Score</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                for r in signals:
                    st.markdown(_render_breakout_card(r), unsafe_allow_html=True)

            # No signal stocks (collapsed)
            if no_signals:
                with st.expander(f"📊 無信號標的 ({len(no_signals)} 檔)", expanded=False):
                    headers = ["代碼", "名稱", "現價", "壓力", "支撐", "量比", "來源"]
                    rows = []
                    for r in no_signals:
                        price = r.get('price', 0)
                        res = r.get('resistance')
                        sup = r.get('support')
                        vr = r.get('volume_ratio', 0)
                        src = {'watchlist': '自選股', 'portfolio': '持倉', 'both': '兩者'}.get(
                            r.get('source', ''), '—')

                        rows.append([
                            f'<strong>{r["ticker"]}</strong>',
                            r.get('name', '—'),
                            f'{price:,.1f}',
                            f'{res:,.1f} [{r.get("resistance_score",0)}/3]' if res else '—',
                            f'{sup:,.1f} [{r.get("support_score",0)}/3]' if sup else '—',
                            f'{vr:.1f}x',
                            src,
                        ])
                    cyber_table(headers, rows)

    # ══════════════════════════════════════════
    # TAB 2: 自訂警報
    # ══════════════════════════════════════════
    with tab_alerts:
        # 新增警報
        st.markdown('<p class="sec-header">新增警報</p>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 2, 1.5, 1])
        with c1:
            ticker = st.text_input("股票代碼", placeholder="例: 2330", key="alert_ticker")
        with c2:
            name = st.text_input("股票名稱", placeholder="例: 台積電", key="alert_name")
        with c3:
            type_keys = list(ALERT_TYPES.keys())
            type_labels = [ALERT_TYPES[k] for k in type_keys]
            selected_label = st.selectbox("警報類型", type_labels, key="alert_type")
            selected_type = type_keys[type_labels.index(selected_label)]
        with c4:
            th_label = THRESHOLD_LABELS.get(selected_type, "門檻值")
            threshold = st.number_input(th_label, min_value=0.0, step=0.1, format="%.2f", key="alert_threshold")
        with c5:
            st.markdown("<br>", unsafe_allow_html=True)
            add_clicked = st.button("➕ 新增", use_container_width=True)

        if add_clicked:
            if not ticker.strip():
                st.error("請輸入股票代碼")
            elif threshold <= 0:
                st.error("請輸入有效門檻值")
            else:
                am.add_alert(
                    ticker=ticker.strip(),
                    name=name.strip(),
                    alert_type=selected_type,
                    threshold=threshold,
                )
                st.success("警報已新增！")
                st.rerun()

        # 警報清單
        st.markdown('<p class="sec-header">警報清單</p>', unsafe_allow_html=True)

        alerts = am.get_all()

        if not alerts:
            st.markdown(
                '<div class="alert-card alert-info">'
                '<div class="alert-title">尚無自訂警報</div>'
                '<div class="alert-body">新增你的第一個價格或技術指標警報。</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            headers = ["代碼", "名稱", "類型", "門檻", "狀態", "建立時間"]
            rows = []
            for a in alerts:
                if a.get("triggered"):
                    status_cell = (
                        f'<span class="tag tag-warn">已觸發</span>'
                        f'<br><span style="font-size:0.7rem;color:#94a3b8;">{a.get("triggered_at", "")}</span>'
                    )
                elif a.get("active"):
                    status_cell = '<span class="status-dot status-live"></span><span style="color:#6ee7b7;">監控中</span>'
                else:
                    status_cell = '<span class="status-dot status-off"></span><span style="color:#64748b;">已暫停</span>'

                rows.append([
                    f'<strong>{a["ticker"]}</strong>',
                    a.get("name", "—"),
                    ALERT_TYPES.get(a["type"], a["type"]),
                    f'{a["threshold"]}',
                    status_cell,
                    a.get("created_at", "—"),
                ])

            cyber_table(headers, rows)

            # Delete buttons
            st.markdown('<p class="sec-header">管理警報</p>', unsafe_allow_html=True)
            cols_per_row = 4
            for i in range(0, len(alerts), cols_per_row):
                chunk = alerts[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for j, a in enumerate(chunk):
                    with cols[j]:
                        label = f"🗑️ {a['ticker']} {ALERT_TYPES.get(a['type'], '')}"
                        if st.button(label, key=f"del_alert_{a['id']}"):
                            am.remove_alert(a["id"])
                            st.rerun()
