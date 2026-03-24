"""
Shared UI theme utilities: color helpers, tag builders, plotly layout, CSS injection.
Cyberpunk Terminal Theme v2.0
"""
import streamlit as st
import os
import json


# ==========================================
# Color / Tag Helpers
# ==========================================
def _tw_color(val, fmt="+.2f"):
    """Taiwan convention: red=positive, green=negative"""
    try:
        v = float(val)
        if v > 0:
            return f'<span class="c-up">{v:{fmt}}</span>'
        elif v < 0:
            return f'<span class="c-down">{v:{fmt}}</span>'
        return f'<span class="c-flat">{v:{fmt}}</span>'
    except Exception:
        return f'<span class="c-flat">{val}</span>'


def _tw_color_pct(val):
    return _tw_color(val, fmt="+.2f") + '%'


def _tw_tag(val):
    try:
        v = float(val)
        if v > 0:
            return f'<span class="tag tag-bull">+{v:.1f}%</span>'
        elif v < 0:
            return f'<span class="tag tag-bear">{v:.1f}%</span>'
        return f'<span class="tag" style="background:#1e293b;color:#94a3b8">{v:.1f}%</span>'
    except Exception:
        return ''


def _plotly_dark_layout(fig, height=400, **kwargs):
    """Apply consistent cyber-dark terminal layout to plotly figures"""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(5,8,16,0.9)',
        font=dict(family='Noto Sans TC, Microsoft JhengHei, sans-serif', size=11, color='#94a3b8'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        xaxis=dict(gridcolor='rgba(0,240,255,0.06)', zerolinecolor='#1e293b'),
        yaxis=dict(gridcolor='rgba(0,240,255,0.06)', zerolinecolor='#1e293b'),
        **kwargs,
    )
    return fig


# ==========================================
# Data Loaders
# ==========================================
def _load_recommendation():
    """Cache-read daily_recommendation.json (once per rerun)"""
    if '_rec_cache' not in st.session_state:
        rec_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'daily_recommendation.json')
        try:
            with open(rec_path, 'r', encoding='utf-8') as f:
                st.session_state['_rec_cache'] = json.load(f)
        except Exception:
            st.session_state['_rec_cache'] = None
    return st.session_state.get('_rec_cache')


# ==========================================
# CSS Injection — Cyberpunk Terminal Theme
# ==========================================
def inject_cyber_theme():
    """Inject the full cyberpunk dark terminal theme CSS."""
    st.markdown("""
<style>
/* ===== Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ===== CSS Custom Properties ===== */
:root {
    --neon-cyan: #00f0ff;
    --neon-cyan-dim: rgba(0, 240, 255, 0.15);
    --neon-magenta: #ff00ff;
    --electric-blue: #3b82f6;
    --bg-deep: #050810;
    --bg-primary: #0a0e17;
    --bg-card: rgba(10, 14, 23, 0.7);
    --bg-card-hover: rgba(15, 23, 42, 0.9);
    --glass-border: rgba(0, 240, 255, 0.12);
    --glass-glow: rgba(0, 240, 255, 0.06);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --tw-up: #ef4444;
    --tw-down: #22c55e;
    --border-dim: #1e293b;
    --border-mid: #334155;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
.stMarkdown, .stButton, .stTextInput, .stSelectbox, .stNumberInput, .stCheckbox {
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', 'PingFang TC', sans-serif !important;
}

/* ===== Deep-space Background ===== */
[data-testid="stAppViewContainer"] > section > div {
    background: var(--bg-deep);
}

/* ===== Sidebar — Dark Glass ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050810 0%, #0a0e17 50%, #0f1320 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
    box-shadow: 2px 0 20px rgba(0, 240, 255, 0.03);
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] .stRadio label:hover { color: var(--neon-cyan) !important; text-shadow: 0 0 8px rgba(0,240,255,0.3); }
[data-testid="stSidebar"] hr { border-color: var(--glass-border) !important; }

/* ===== Metric Cards — Glassmorphism ===== */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    box-shadow: 0 0 15px var(--glass-glow), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0, 240, 255, 0.25) !important;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.1), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ===== Pro Table — Cyber Style ===== */
.pro-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.pro-table th {
    background: rgba(0, 240, 255, 0.04);
    color: var(--neon-cyan);
    padding: 10px 12px;
    text-align: right;
    border-bottom: 1px solid var(--glass-border);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}
.pro-table th:first-child { text-align: left; }
.pro-table td {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(0, 240, 255, 0.05);
    text-align: right;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', 'Noto Sans TC', monospace;
    font-size: 0.8rem;
}
.pro-table td:first-child { text-align: left; font-weight: 600; color: var(--text-primary); }
.pro-table tr:hover { background: rgba(0, 240, 255, 0.04); }

/* ===== Color Utilities (TW: red=up, green=down) ===== */
.c-up { color: var(--tw-up) !important; font-weight: 700; text-shadow: 0 0 6px rgba(239,68,68,0.3); }
.c-down { color: var(--tw-down) !important; font-weight: 700; text-shadow: 0 0 6px rgba(34,197,94,0.3); }
.c-flat { color: var(--text-secondary) !important; }
.c-warn { color: #f59e0b !important; font-weight: 700; }
.c-danger { color: #dc2626 !important; font-weight: 700; }

/* ===== Tag Badges — Neon Glow ===== */
.tag { display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; font-family: 'JetBrains Mono', monospace; }
.tag-bull { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.tag-bear { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }
.tag-new { background: rgba(0,240,255,0.1); color: var(--neon-cyan); border: 1px solid rgba(0,240,255,0.25); }
.tag-exit { background: rgba(245,158,11,0.15); color: #fde68a; border: 1px solid rgba(245,158,11,0.3); }
.tag-ok { background: rgba(34,197,94,0.1); color: #6ee7b7; border: 1px solid rgba(34,197,94,0.2); }
.tag-danger { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.tag-warn { background: rgba(245,158,11,0.1); color: #fde68a; border: 1px solid rgba(245,158,11,0.2); }

/* ===== Alert Cards — Neon Border ===== */
.alert-card {
    border-left: 3px solid;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    background: var(--bg-card);
    backdrop-filter: blur(8px);
    box-shadow: 0 0 12px var(--glass-glow);
}
.alert-danger { border-color: #dc2626; box-shadow: 0 0 12px rgba(220,38,38,0.08); }
.alert-warn { border-color: #f59e0b; box-shadow: 0 0 12px rgba(245,158,11,0.08); }
.alert-ok { border-color: var(--tw-down); box-shadow: 0 0 12px rgba(34,197,94,0.08); }
.alert-info { border-color: var(--neon-cyan); box-shadow: 0 0 12px rgba(0,240,255,0.08); }
.alert-card .alert-title { font-weight: 700; font-size: 0.85rem; margin-bottom: 4px; color: var(--text-primary); }
.alert-card .alert-body { font-size: 0.8rem; color: var(--text-secondary); }

/* ===== Status Pulse ===== */
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.status-live { background: var(--neon-cyan); box-shadow: 0 0 8px var(--neon-cyan), 0 0 20px rgba(0,240,255,0.3); animation: cyber-pulse 2s infinite; }
.status-sim { background: #f59e0b; box-shadow: 0 0 8px rgba(245,158,11,0.5); animation: cyber-pulse 3s infinite; }
.status-off { background: var(--text-muted); }
@keyframes cyber-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px currentColor, 0 0 20px currentColor; }
    50% { opacity: 0.6; box-shadow: 0 0 4px currentColor; }
}

/* ===== Order Book Row ===== */
.ob-row { display: flex; justify-content: space-between; padding: 6px 12px; border-bottom: 1px solid rgba(0,240,255,0.05); font-size: 0.82rem; }
.ob-row:hover { background: rgba(0, 240, 255, 0.04); }

/* ===== Score Bar ===== */
.score-bar { height: 6px; border-radius: 3px; background: rgba(0,240,255,0.08); overflow: hidden; }
.score-fill { height: 100%; border-radius: 3px; }

/* ===== Section Headers — Cyber Accent ===== */
.sec-header {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--neon-cyan);
    font-weight: 700;
    padding: 8px 0 6px;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 10px rgba(0,240,255,0.2);
}

/* ===== KPI Strip — Glass Cards ===== */
.kpi-strip { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.kpi-item {
    background: var(--bg-card);
    backdrop-filter: blur(8px);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    padding: 10px 16px;
    min-width: 125px;
    box-shadow: 0 0 10px var(--glass-glow);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.kpi-item:hover {
    border-color: rgba(0, 240, 255, 0.25);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.08);
}
.kpi-label {
    font-size: 0.62rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ===== Cyber Module Header ===== */
.cyber-title {
    font-size: 1.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--electric-blue) 50%, var(--neon-magenta) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 20px rgba(0,240,255,0.15));
}
.cyber-subtitle {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-top: -4px;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}

/* ===== Ticker Tape (Top Bar) ===== */
.ticker-tape {
    display: flex;
    gap: 24px;
    padding: 8px 16px;
    background: linear-gradient(90deg, rgba(0,240,255,0.03), rgba(5,8,16,0.9), rgba(0,240,255,0.03));
    border-bottom: 1px solid var(--glass-border);
    overflow-x: auto;
    white-space: nowrap;
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    scrollbar-width: none;
}
.ticker-tape::-webkit-scrollbar { display: none; }
.ticker-item { display: inline-flex; align-items: center; gap: 8px; color: var(--text-secondary); }
.ticker-name { color: var(--text-muted); font-size: 0.68rem; }
.ticker-price { color: var(--text-primary); font-weight: 700; }

/* ===== Plotly Chart Dark Bg ===== */
.js-plotly-plot .plotly { background: transparent !important; }

/* ===== Streamlit Overrides for Cyber Feel ===== */
[data-testid="stExpander"] {
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    background: var(--bg-card) !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(0, 240, 255, 0.2) !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid var(--glass-border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom-color: var(--neon-cyan) !important;
    text-shadow: 0 0 8px rgba(0,240,255,0.3);
}
button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,240,255,0.15), rgba(59,130,246,0.2)) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--neon-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.04em;
    transition: all 0.3s;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(0,240,255,0.25), rgba(59,130,246,0.3)) !important;
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0,240,255,0.15);
}
input, textarea, select, [data-baseweb="select"], [data-baseweb="input"] {
    background: rgba(5,8,16,0.8) !important;
    border-color: var(--glass-border) !important;
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)
