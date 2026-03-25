# ==========================================
# 全域量化終端 v200 — Cyber Trading Terminal
# 5-Page Architecture: Dashboard / Research / Trading / Lab / Review
# ==========================================
import streamlit as st
import pandas as pd
from google import genai
import os

from config.settings import GEMINI_MODEL, DEFAULT_CHART_SETTINGS
from state import init_session_state
from ui.theme import inject_cyber_theme

# --- Config ---
st.set_page_config(page_title="全域量化終端 v200", layout="wide", page_icon="💎")
pd.set_option("styler.render.max_elements", 1_000_000)

# --- Authentication (graceful degradation) ---
_auth_available = False
_auth_lib_installed = False
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    _auth_lib_installed = True
except ImportError:
    _auth_lib_installed = False

if _auth_lib_installed:
    _auth_config_path = os.path.join(os.path.dirname(__file__), 'config', 'auth_config.yaml')
    if not os.path.exists(_auth_config_path):
        st.error("認證設定檔 config/auth_config.yaml 不存在，無法啟動應用。")
        st.stop()
    else:
        try:
            with open(_auth_config_path, 'r', encoding='utf-8') as _f:
                _auth_config = yaml.load(_f, Loader=SafeLoader)
            if not _auth_config or 'credentials' not in _auth_config or 'cookie' not in _auth_config:
                st.error("認證設定檔 config/auth_config.yaml 格式錯誤，無法啟動應用。")
                st.stop()
            _authenticator = stauth.Authenticate(
                _auth_config['credentials'],
                _auth_config['cookie']['name'],
                _auth_config['cookie']['key'],
                _auth_config['cookie']['expiry_days'],
            )
            _auth_available = True
        except Exception as e:
            st.error(f"認證設定檔載入失敗: {e}")
            st.stop()
else:
    st.sidebar.warning("⚠️ 認證系統未啟用（開發模式）")

if _auth_available:
    with st.sidebar:
        try:
            _authenticator.login(
                location='sidebar',
                fields={
                    'Form name': '登入',
                    'Username': '使用者名稱',
                    'Password': '密碼',
                    'Login': '登入',
                },
            )
        except TypeError:
            # v0.4.x: login() returns None, status in session_state
            _authenticator.login()

    authentication_status = st.session_state.get('authentication_status')
    name = st.session_state.get('name', '')

    if authentication_status is False:
        st.sidebar.error('使用者名稱或密碼錯誤')
        st.stop()
    elif authentication_status is None:
        st.markdown(
            '<div style="text-align:center;margin-top:100px;color:#94a3b8;font-size:1.1rem">'
            '請在左側欄登入以使用全域量化終端</div>',
            unsafe_allow_html=True,
        )
        st.stop()
    else:
        with st.sidebar:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#22c55e;margin-bottom:4px">'
                f'👤 {name}</div>',
                unsafe_allow_html=True,
            )
            try:
                _authenticator.logout('登出', 'sidebar')
            except TypeError:
                _authenticator.logout()

# --- Theme + State ---
inject_cyber_theme()
init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="cyber-title" style="font-size:1.2rem;">全域量化終端</div>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-subtitle" style="font-size:0.7rem;">v200 CYBER TERMINAL</div>', unsafe_allow_html=True)

    app_mode = st.radio(
        "nav",
        [
            "📊 交易總覽",
            "🔍 研究分析",
            "⚡ 交易執行",
            "🧬 策略實驗室",
            "📝 覆盤紀錄",
        ],
        label_visibility="collapsed",
    )

    # --- API Keys from secrets.toml ---
    def _get_secret(key, fallback=""):
        try: return st.secrets.get(key, fallback)
        except Exception: return fallback

    api_key = _get_secret("GEMINI_API_KEY")
    finlab_token = _get_secret("FINLAB_API_KEY") or _get_secret("FINLAB_API_TOKEN")
    sinopac_key = _get_secret("SINOPAC_API_KEY")
    sinopac_secret = _get_secret("SINOPAC_SECRET_KEY")
    telegram_token = _get_secret("TELEGRAM_BOT_TOKEN")
    telegram_chat = _get_secret("TELEGRAM_CHAT_ID")

    if finlab_token: st.session_state['finlab_token'] = finlab_token
    if sinopac_key:
        st.session_state['sinopac_api_key'] = sinopac_key
        st.session_state['sinopac_secret_key'] = sinopac_secret
    if telegram_token:
        st.session_state['telegram_bot_token'] = telegram_token
        st.session_state['telegram_chat_id'] = telegram_chat

    client = None
    if api_key:
        try: client = genai.Client(api_key=api_key)
        except Exception: pass

    # API status
    connected = []
    if api_key: connected.append("Gemini")
    if finlab_token: connected.append("FinLab")
    if sinopac_key: connected.append("永豐金")
    if telegram_token: connected.append("Telegram")
    if connected:
        st.markdown(
            f'<div style="font-size:0.6rem;color:#22c55e;font-family:JetBrains Mono,monospace;'
            f'background:rgba(34,197,94,0.08);padding:3px 6px;border-radius:4px;margin-top:4px">'
            f'🔗 {" | ".join(connected)}</div>',
            unsafe_allow_html=True,
        )

    # Research sidebar controls
    if app_mode == "🔍 研究分析":
        st.divider()
        market_mode = st.radio("市場", ["🇹🇼 台股 (TW)", "🗽 美股 (US)"],
                               index=0 if "台股" in st.session_state.market_mode else 1,
                               label_visibility="collapsed")
        st.session_state.market_mode = market_mode
        timeframe = st.radio("時間框架", ["1d (日線)", "1wk (週線)"], index=0, label_visibility="collapsed")
        tf_code = "1wk" if "週線" in timeframe else "1d"
        is_weekly = (tf_code == "1wk")
        strategy_mode = st.radio("風格", ["🔥 順勢突破", "🛡️ 拉回抄底"], label_visibility="collapsed")
        st.divider()
        st.markdown('<p class="sec-header">線圖工具</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        st.session_state.chart_settings['trendline'] = c1.checkbox("支撐壓力", value=True)
        st.session_state.chart_settings['gaps'] = c2.checkbox("跳空缺口", value=True)
        c3, c4 = st.columns(2)
        st.session_state.chart_settings['ma'] = c3.checkbox("MA均線", value=True)
        st.session_state.chart_settings['bbands'] = c4.checkbox("BBands", value=False)
        st.session_state.chart_settings['candle_patterns'] = st.checkbox("K線型態", value=True)

    st.divider()
    st.caption("Isaac V3.7 | Shioaji | FinLab")

# --- Market Ticker Tape ---
from ui.components import render_market_tape
render_market_tape()

# --- First Run Guide ---
_has_any_key = bool(api_key or finlab_token or sinopac_key)
if not _has_any_key and app_mode == "📊 交易總覽":
    st.markdown("""
    <div style="max-width:700px;margin:60px auto;text-align:center">
        <div style="font-size:3rem;margin-bottom:16px">🔑</div>
        <div class="cyber-title" style="font-size:1.5rem;margin-bottom:8px">歡迎使用全域量化終端</div>
        <div style="color:#94a3b8;font-size:0.9rem;margin-bottom:32px">
            請先設定 API 金鑰，才能使用完整功能。
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📋 設定步驟")
    st.markdown("""
    1. 找到安裝目錄下的 `.streamlit/secrets.toml` 檔案
    2. 用記事本開啟，填入您的 API 金鑰
    3. 儲存後**重新整理瀏覽器** (F5)
    > 💡 至少需要 **FinLab API 金鑰** 才能使用回測功能
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("| 服務 | 用途 | 必要性 |\n|------|------|--------|\n| **FinLab** | 台股資料、回測 | ⭐ 必要 |\n| **Gemini** | AI 分析報告 | 選配 |")
    with col2:
        st.markdown("| 服務 | 用途 | 必要性 |\n|------|------|--------|\n| **永豐金 Shioaji** | 實盤交易 | 選配 |\n| **Telegram** | 推播通知 | 選配 |")
    st.stop()

# ==========================================
# Module Router (5 pages)
# ==========================================

if app_mode == "📊 交易總覽":
    from ui.pages.dashboard import render
    render()

elif app_mode == "🔍 研究分析":
    from ui.pages.research import render
    render(client=client, market_mode=market_mode, strategy_mode=strategy_mode,
           tf_code=tf_code, is_weekly=is_weekly)

elif app_mode == "⚡ 交易執行":
    from ui.pages.trading import render
    render()

elif app_mode == "🧬 策略實驗室":
    from ui.pages.lab import render
    render()

elif app_mode == "📝 覆盤紀錄":
    from ui.pages.review import render
    render()
