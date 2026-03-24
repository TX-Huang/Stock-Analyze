"""經濟行事曆 — 月曆視圖 + 重要日程。"""
import streamlit as st
import calendar
from datetime import datetime, timedelta
from collections import defaultdict

from ui.components import cyber_header


EVENTS_2026 = [
    {"date": "2026-01-28", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-03-18", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-05-06", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-06-17", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-07-29", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-09-16", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-11-04", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-12-16", "event": "FOMC 會議", "type": "macro", "icon": "🏛️"},
    {"date": "2026-01-01", "event": "元旦", "type": "holiday", "icon": "🔴"},
    {"date": "2026-01-26", "event": "除夕", "type": "holiday", "icon": "🔴"},
    {"date": "2026-01-27", "event": "春節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-01-28", "event": "春節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-01-29", "event": "春節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-01-30", "event": "春節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-02-28", "event": "和平紀念日", "type": "holiday", "icon": "🔴"},
    {"date": "2026-04-03", "event": "兒童節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-04-04", "event": "清明節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-05-25", "event": "端午節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-10-05", "event": "中秋節", "type": "holiday", "icon": "🔴"},
    {"date": "2026-10-10", "event": "國慶日", "type": "holiday", "icon": "🔴"},
    {"date": "2026-03-31", "event": "Q1 季報截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-06-30", "event": "Q2 季報截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-09-30", "event": "Q3 季報截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-12-31", "event": "Q4 季報截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-03-10", "event": "2月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-04-10", "event": "3月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-05-10", "event": "4月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-06-10", "event": "5月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-07-10", "event": "6月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-08-10", "event": "7月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-09-10", "event": "8月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-10-10", "event": "9月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-11-10", "event": "10月營收截止", "type": "earnings", "icon": "📊"},
    {"date": "2026-12-10", "event": "11月營收截止", "type": "earnings", "icon": "📊"},
    # 除權息 (常見高股息標的，預估日期)
    {"date": "2026-06-25", "event": "2881 富邦金 除息", "type": "dividend", "icon": "💰"},
    {"date": "2026-07-15", "event": "2330 台積電 除息", "type": "dividend", "icon": "💰"},
    {"date": "2026-07-20", "event": "2884 玉山金 除息", "type": "dividend", "icon": "💰"},
    {"date": "2026-08-01", "event": "0056 高股息ETF 除息", "type": "dividend", "icon": "💰"},
    {"date": "2026-08-15", "event": "2882 國泰金 除息", "type": "dividend", "icon": "💰"},
    {"date": "2026-08-20", "event": "2886 兆豐金 除息", "type": "dividend", "icon": "💰"},
]

TYPE_COLORS = {
    "holiday": "#ef4444",
    "macro": "#3b82f6",
    "earnings": "#f59e0b",
    "dividend": "#a855f7",
}

TYPE_DOT = {
    "holiday": "🔴",
    "macro": "🔵",
    "earnings": "🟡",
    "dividend": "🟣",
}

MONTH_NAMES = ["", "1月", "2月", "3月", "4月", "5月", "6月",
               "7月", "8月", "9月", "10月", "11月", "12月"]
WEEKDAY_HEADERS = ["一", "二", "三", "四", "五", "六", "日"]


def _build_event_map(year):
    """Build {(month, day): [events]} lookup."""
    emap = defaultdict(list)
    for ev in EVENTS_2026:
        try:
            dt = datetime.strptime(ev["date"], "%Y-%m-%d")
            if dt.year == year:
                emap[(dt.month, dt.day)].append(ev)
        except Exception:
            pass
    return emap


def _render_month_grid(year, month, event_map, today):
    """Render a single month as an HTML table grid."""
    cal = calendar.monthcalendar(year, month)
    is_current_month = (today.year == year and today.month == month)

    # Weekday headers
    header_cells = "".join(
        f'<th style="text-align:center;padding:6px 2px;color:{"#ef4444" if i >= 5 else "var(--neon-cyan)"};'
        f'font-size:0.68rem;font-family:JetBrains Mono,monospace;letter-spacing:0.05em;'
        f'border-bottom:1px solid rgba(0,240,255,0.1)">{d}</th>'
        for i, d in enumerate(WEEKDAY_HEADERS)
    )

    rows_html = ""
    for week in cal:
        cells = ""
        for i, day in enumerate(week):
            if day == 0:
                cells += '<td style="padding:4px;border:1px solid rgba(0,240,255,0.04);"></td>'
                continue

            is_today = is_current_month and day == today.day
            events_today = event_map.get((month, day), [])
            is_weekend = i >= 5

            # Cell background
            if is_today:
                bg = "rgba(0,240,255,0.12)"
                border = "1px solid rgba(0,240,255,0.4)"
            elif events_today:
                bg = "rgba(0,240,255,0.04)"
                border = "1px solid rgba(0,240,255,0.08)"
            else:
                bg = "transparent"
                border = "1px solid rgba(0,240,255,0.04)"

            # Day number color
            if is_today:
                num_style = "color:var(--neon-cyan);font-weight:900;text-shadow:0 0 8px rgba(0,240,255,0.4)"
            elif any(e["type"] == "holiday" for e in events_today) or is_weekend:
                num_style = "color:#ef4444;font-weight:600"
            else:
                num_style = "color:#94a3b8"

            # Event dots/labels
            event_html = ""
            for ev in events_today[:2]:  # max 2 events per cell
                c = TYPE_COLORS.get(ev["type"], "#00f0ff")
                label = ev["event"][:4]  # truncate to 4 chars
                event_html += (
                    f'<div style="font-size:0.55rem;color:{c};white-space:nowrap;'
                    f'overflow:hidden;text-overflow:ellipsis;max-width:100%;line-height:1.2">'
                    f'{ev["icon"]}{label}</div>'
                )
            if len(events_today) > 2:
                event_html += f'<div style="font-size:0.5rem;color:#64748b">+{len(events_today)-2}</div>'

            cells += (
                f'<td style="padding:3px 4px;background:{bg};border:{border};'
                f'vertical-align:top;min-width:70px;height:60px;border-radius:4px">'
                f'<div style="font-size:0.75rem;font-family:JetBrains Mono,monospace;{num_style}">{day}</div>'
                f'{event_html}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    return (
        f'<table style="width:100%;border-collapse:separate;border-spacing:2px;table-layout:fixed">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def render(_embedded=False):
    if not _embedded:
        cyber_header("經濟行事曆", "重要日程 | 除息日 | FOMC 會議")

    today = datetime.now().date()
    year = 2026
    event_map = _build_event_map(year)

    # Month selector
    if '_cal_month' not in st.session_state:
        st.session_state['_cal_month'] = today.month if today.year == year else 1

    col_prev, col_title, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀ 上月", use_container_width=True):
            st.session_state['_cal_month'] = max(1, st.session_state['_cal_month'] - 1)
    with col_next:
        if st.button("下月 ▶", use_container_width=True):
            st.session_state['_cal_month'] = min(12, st.session_state['_cal_month'] + 1)

    sel_month = st.session_state['_cal_month']

    with col_title:
        st.markdown(
            f'<div style="text-align:center;font-family:JetBrains Mono,monospace;'
            f'font-size:1.3rem;font-weight:700;color:var(--neon-cyan);'
            f'text-shadow:0 0 12px rgba(0,240,255,0.2);padding:6px 0">'
            f'{year} 年 {MONTH_NAMES[sel_month]}</div>',
            unsafe_allow_html=True,
        )

    # Render calendar grid
    grid_html = _render_month_grid(year, sel_month, event_map, today)
    st.markdown(grid_html, unsafe_allow_html=True)

    # Legend
    st.markdown(
        '<div style="display:flex;gap:20px;margin-top:12px;font-size:0.72rem;'
        'font-family:JetBrains Mono,monospace;color:#64748b">'
        '<span>🔴 休市</span><span>🔵 FOMC</span><span>🟡 財報/營收</span><span>🟣 除息</span>'
        '<span style="color:var(--neon-cyan)">■ 今天</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Upcoming events (below calendar)
    st.markdown('<p class="sec-header" style="margin-top:24px">本月事件</p>', unsafe_allow_html=True)

    month_events = [
        e for e in sorted(EVENTS_2026, key=lambda x: x["date"])
        if datetime.strptime(e["date"], "%Y-%m-%d").month == sel_month
    ]

    if month_events:
        for ev in month_events:
            evt_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            border_color = TYPE_COLORS.get(ev["type"], "#00f0ff")
            is_past = evt_date < today
            opacity = "opacity:0.5;" if is_past else ""

            st.markdown(
                f'<div class="alert-card" style="border-color:{border_color};{opacity}padding:10px 14px;margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-weight:700;font-size:0.82rem;color:{"#475569" if is_past else "#e2e8f0"}">'
                f'{ev["icon"]} {ev["event"]}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#64748b">{ev["date"]}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="alert-card alert-info">'
            '<div class="alert-body">本月無已排定事件。</div></div>',
            unsafe_allow_html=True,
        )
