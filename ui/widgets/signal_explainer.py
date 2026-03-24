"""
Signal explainer widget (P3-2)
為每個技術信號提供白話解釋，幫助新手理解「為什麼是這個訊號」
"""
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Signal explanations database
SIGNAL_EXPLANATIONS = {
    # === Momentum / Trend Signals ===
    'rsi_oversold': {
        'name': 'RSI 超賣',
        'icon': '📉',
        'short': 'RSI 低於 30，表示短期內賣壓過大，可能即將反彈',
        'detail': (
            'RSI（相對強弱指標）衡量近期漲跌幅的相對強度。'
            '當 RSI < 30 時，代表股價在短時間內跌幅較大，'
            '歷史數據顯示此情況下反彈機率約 60-70%。'
            '但注意：強勢下跌趨勢中，RSI 可能長時間維持在低位。'
        ),
        'confirmation': ['成交量萎縮（賣壓減弱）', '出現看漲 K 線型態（如十字星、吞噬）', 'MACD 出現金叉'],
        'reliability': '中等（60-70% 在震盪市場，40-50% 在趨勢市場）',
    },
    'rsi_overbought': {
        'name': 'RSI 超買',
        'icon': '📈',
        'short': 'RSI 高於 70，表示短期漲幅過大，可能面臨回調',
        'detail': (
            '當 RSI > 70 時，代表股價在短時間內漲幅較大。'
            '這不一定代表「要跌了」，強勢股可以在超買區維持很久。'
            '重要的是觀察是否出現「頂背離」（價格創新高但 RSI 未創新高）。'
        ),
        'confirmation': ['量縮價不漲', 'RSI 與價格出現頂背離', '跌破短期均線'],
        'reliability': '中等（作為賣出信號時需搭配其他指標）',
    },
    'macd_golden_cross': {
        'name': 'MACD 金叉',
        'icon': '✨',
        'short': 'MACD 快線向上穿越慢線，代表短期動能轉強',
        'detail': (
            'MACD 金叉是最常用的買入信號之一。'
            '當 DIF 線（快線）從下方穿越 DEA 線（慢線）時形成金叉。'
            '零軸上方的金叉比零軸下方更可靠。'
            '搭配放量確認更佳。'
        ),
        'confirmation': ['金叉發生在零軸附近或上方', '同時出現放量', 'K 線收紅'],
        'reliability': '中高（零軸上方約 65-75%，零軸下方約 50-60%）',
    },
    'macd_death_cross': {
        'name': 'MACD 死叉',
        'icon': '💀',
        'short': 'MACD 快線向下穿越慢線，代表短期動能轉弱',
        'detail': (
            '當 DIF 線從上方穿越 DEA 線時形成死叉，通常視為賣出信號。'
            '零軸下方的死叉更具威脅性，代表中期趨勢已轉空。'
        ),
        'confirmation': ['死叉發生在零軸附近或下方', '同時出現放量', '跌破重要支撐'],
        'reliability': '中高（零軸下方約 65-70%）',
    },
    'kd_golden_cross': {
        'name': 'KD 金叉',
        'icon': '🔀',
        'short': 'K 線向上穿越 D 線，短期買入信號',
        'detail': (
            'KD 指標（隨機指標）的金叉通常出現在超賣區（20 以下）更有參考價值。'
            'K 線代表短期動能，D 線是平滑後的趨勢。'
            '在盤整區間內，KD 金叉的勝率較高；趨勢市場中容易產生雜訊。'
        ),
        'confirmation': ['金叉發生在 20 以下（超賣區）', '價格站上短期均線', 'MACD 也轉正'],
        'reliability': '中等（超賣區金叉約 60-65%）',
    },
    'volume_breakout': {
        'name': '量價突破',
        'icon': '🚀',
        'short': '股價突破壓力位且成交量顯著放大',
        'detail': (
            '量價突破是技術分析中最重要的信號之一。'
            '當股價突破前高/壓力線，且成交量是平均的 1.5 倍以上時，'
            '代表有大量買盤願意在更高的價位進場，突破的可靠性大增。'
            '「價漲量增」是健康上漲的基本條件。'
        ),
        'confirmation': ['突破後站穩 2-3 天不回跌', '5 日均量 > 20 日均量', '外資或投信同步買超'],
        'reliability': '高（搭配量能確認約 70-80%）',
    },
    'vcp_pattern': {
        'name': 'VCP 型態',
        'icon': '🔍',
        'short': '波動收縮型態 — 收縮幅度遞減，即將突破',
        'detail': (
            'VCP（Volatility Contraction Pattern）是 Mark Minervini 提出的經典型態。'
            '股價在整理期間，每次回調的幅度越來越小（如 15%→8%→4%），'
            '成交量也隨之萎縮，這代表賣壓逐步被吸收。'
            '當最後一次收縮結束並伴隨放量突破時，是最佳買點。'
        ),
        'confirmation': ['至少 3 次收縮且幅度遞減', '突破時量能放大 2 倍以上', '處於 Stage 2 上升趨勢'],
        'reliability': '高（符合完整條件約 70-75%）',
    },
    'supply_zone_danger': {
        'name': '接近供給區',
        'icon': '⚠️',
        'short': '股價接近前高套牢區，可能面臨賣壓',
        'detail': (
            '供給區是指過去有大量交易發生的價格區間。'
            '當股價從下方接近這個區域時，之前被套牢的投資人可能會趁解套出場，'
            '造成額外的賣壓。建議觀察量能是否足以消化這些賣壓。'
        ),
        'confirmation': ['突破供給區時量能充足', '回測供給區不破', '供給區已被多次測試'],
        'reliability': '高（作為風險提示約 75-80%）',
    },
    'trend_up': {
        'name': '多頭趨勢',
        'icon': '📈',
        'short': '價格在短中長期均線之上，趨勢向上',
        'detail': (
            '當股價 > MA20 > MA50 > MA120 時，稱為多頭排列。'
            '這是最適合做多的環境。順勢交易的勝率遠高於逆勢。'
            '建議在多頭趨勢中的回調找買點，而非追高。'
        ),
        'confirmation': ['MA20 > MA50 > MA120（多頭排列）', '回調不破 MA20', '回調後放量反彈'],
        'reliability': '高（趨勢延續性約 65-70%）',
    },
    'trend_down': {
        'name': '空頭趨勢',
        'icon': '📉',
        'short': '價格在均線之下，趨勢向下',
        'detail': (
            '當股價 < MA20 < MA50 < MA120 時，稱為空頭排列。'
            '在空頭趨勢中做多風險很高，即使有反彈也容易被再次打壓。'
            '建議等待趨勢反轉確認後再進場。'
        ),
        'confirmation': ['跌破所有均線', 'MA20 < MA50 < MA120', '反彈量縮受壓'],
        'reliability': '高（趨勢延續性約 65-70%）',
    },
    'bullish_engulfing': {
        'name': '看漲吞噬',
        'icon': '🟢',
        'short': '大紅 K 線完全包覆前一根綠 K 線，強烈反轉信號',
        'detail': (
            '看漲吞噬型態出現在下跌趨勢的底部時最有意義。'
            '紅 K 線的實體完全覆蓋前一根綠 K 線，代表多方強力反攻。'
            '搭配放量和重要支撐位置確認更佳。'
        ),
        'confirmation': ['出現在下跌趨勢底部或支撐位', '成交量明顯放大', '次日持續收紅'],
        'reliability': '中高（在支撐位約 65-70%）',
    },
    'doji': {
        'name': '十字星',
        'icon': '✝️',
        'short': '開盤價約等於收盤價，多空力量平衡，趨勢可能反轉',
        'detail': (
            '十字星代表當天多空雙方拉鋸後平手，通常暗示原有趨勢的動能減弱。'
            '在上漲頂部出現十字星（黃昏之星）是警告信號；'
            '在下跌底部出現（晨星）可能是反轉信號。'
            '十字星本身不構成交易信號，需要次日 K 線確認方向。'
        ),
        'confirmation': ['次日 K 線確認方向', '出現在趨勢末端', '搭配量能變化'],
        'reliability': '低中（需要確認，單獨出現約 50-55%）',
    },
}


def get_signal_explanation(signal_key: str) -> dict:
    """Get explanation for a specific signal.

    Args:
        signal_key: Key in SIGNAL_EXPLANATIONS
    Returns:
        Explanation dict or None if not found
    """
    return SIGNAL_EXPLANATIONS.get(signal_key)


def render_signal_explanation(signal_key: str, compact=False):
    """Render a signal explanation card.

    Args:
        signal_key: Key in SIGNAL_EXPLANATIONS
        compact: If True, show only the short version
    """
    info = SIGNAL_EXPLANATIONS.get(signal_key)
    if not info:
        return

    if compact:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#94a3b8;padding:4px 8px;'
            f'background:rgba(0,212,255,0.05);border-radius:4px;margin:2px 0">'
            f'{info["icon"]} <b>{info["name"]}</b> — {info["short"]}</div>',
            unsafe_allow_html=True,
        )
        return

    with st.expander(f'{info["icon"]} {info["name"]} — 為什麼是這個信號？', expanded=False):
        st.markdown(f'**{info["short"]}**')
        st.markdown(info['detail'])

        st.markdown("**✅ 確認條件（越多越可靠）：**")
        for c in info.get('confirmation', []):
            st.markdown(f"- {c}")

        reliability = info.get('reliability', '未知')
        if '高' in reliability:
            r_color = '#22c55e'
        elif '中' in reliability:
            r_color = '#eab308'
        else:
            r_color = '#ef4444'

        st.markdown(
            f'<div style="margin-top:8px;padding:6px 10px;background:rgba(0,0,0,0.2);'
            f'border-radius:4px;font-size:0.8rem">'
            f'📊 歷史可靠度：<span style="color:{r_color};font-weight:700">{reliability}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_signals_summary(signals: list):
    """Render a summary of multiple signal explanations.

    Args:
        signals: List of signal_key strings
    """
    if not signals:
        return

    st.markdown("#### 💡 信號解讀")
    for sig_key in signals:
        if sig_key in SIGNAL_EXPLANATIONS:
            render_signal_explanation(sig_key, compact=False)
