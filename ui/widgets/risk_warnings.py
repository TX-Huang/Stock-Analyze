"""
Contextual risk warning widgets for AI Invest HQ.

Each ``check_*`` function evaluates a single risk condition and returns either
a warning dict ``{level, title, message, icon}`` or ``None``.

``generate_stock_warnings`` aggregates all checks for a given stock.
``render_risk_warnings`` turns a list of warnings into styled HTML cards that
match the cyberpunk terminal theme.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Warning levels
# ---------------------------------------------------------------------------
_LEVEL_STYLES: dict[str, dict[str, str]] = {
    'danger': {
        'border': '#ef4444',
        'bg': 'rgba(239,68,68,0.08)',
        'title_color': '#ef4444',
        'text_color': '#fca5a5',
        'glow': 'rgba(239,68,68,0.15)',
    },
    'warning': {
        'border': '#f59e0b',
        'bg': 'rgba(245,158,11,0.08)',
        'title_color': '#f59e0b',
        'text_color': '#fcd34d',
        'glow': 'rgba(245,158,11,0.12)',
    },
    'info': {
        'border': '#00f0ff',
        'bg': 'rgba(0,240,255,0.05)',
        'title_color': '#00f0ff',
        'text_color': '#94a3b8',
        'glow': 'rgba(0,240,255,0.08)',
    },
}


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_drawdown_warning(
    current_price: float,
    peak_price: float,
    threshold: float = 0.15,
) -> dict[str, str] | None:
    """Check if stock has dropped significantly from its peak.

    Args:
        current_price: Current (or latest close) price.
        peak_price: Recent peak price (e.g. 52-week high).
        threshold: Fractional drop to trigger the warning (default 15%).

    Returns:
        Warning dict or None.
    """
    if peak_price <= 0:
        return None
    drawdown = (current_price - peak_price) / peak_price
    if drawdown > -threshold:
        return None

    pct = abs(drawdown) * 100
    level = 'danger' if pct >= 30 else 'warning'
    return {
        'level': level,
        'title': '\u26A0\uFE0F \u5927\u5E45\u56DE\u6A94\u8B66\u544A',  # 大幅回檔警告
        'message': (
            f'\u7576\u524D\u50F9\u683C\u8DDD\u8FD1\u671F\u9AD8\u9EDE\u5DF2\u4E0B\u8DCC '  # 當前價格距近期高點已下跌
            f'{pct:.1f}%\u3002'
            + ('\u5DF2\u9054\u7A7A\u982D\u5E02\u5834\u6A19\u6E96\uFF08-20%\uFF09\uFF0C\u8ACB\u5BE9\u614E\u8A55\u4F30\u98A8\u96AA\u3002'  # 已達空頭市場標準
               if pct >= 20 else
               '\u5EFA\u8B70\u6AA2\u67E5\u57FA\u672C\u9762\u662F\u5426\u8B8A\u5316\u3002')  # 建議檢查基本面是否變化
        ),
        'icon': '\U0001F4C9',  # chart decreasing
    }


def check_volume_spike(
    current_volume: float,
    avg_volume: float,
    threshold: float = 3.0,
) -> dict[str, str] | None:
    """Check for abnormal volume spike.

    Args:
        current_volume: Today's trading volume.
        avg_volume: Average volume (e.g. 20-day average).
        threshold: Multiple of avg that triggers the warning.

    Returns:
        Warning dict or None.
    """
    if avg_volume <= 0:
        return None
    ratio = current_volume / avg_volume
    if ratio < threshold:
        return None

    return {
        'level': 'warning',
        'title': '\U0001F4A5 \u6210\u4EA4\u91CF\u7570\u5E38\u653E\u5927',  # 成交量異常放大
        'message': (
            f'\u4ECA\u65E5\u6210\u4EA4\u91CF\u70BA\u5747\u91CF\u7684 {ratio:.1f} \u500D\u3002'  # 今日成交量為均量的 X 倍
            '\u91CF\u80FD\u7570\u5E38\u653E\u5927\u53EF\u80FD\u4EE3\u8868\u91CD\u5927\u6D88\u606F\u3001\u4E3B\u529B\u9032\u51FA\u6216\u7C4C\u78BC\u9B06\u52D5\uFF0C'  # 量能異常放大可能代表重大消息、主力進出或籌碼鬆動
            '\u8ACB\u642D\u914D\u50F9\u683C\u8D70\u52E2\u5224\u65B7\u3002'  # 請搭配價格走勢判斷
        ),
        'icon': '\U0001F4CA',
    }


def check_low_liquidity(
    avg_volume: float,
    min_volume: float = 500,
) -> dict[str, str] | None:
    """Check for low liquidity (avg daily volume in thousands of shares).

    Args:
        avg_volume: Average daily volume in *thousands* of shares (e.g. 500 = 50 萬股).
        min_volume: Minimum acceptable volume in thousands.

    Returns:
        Warning dict or None.
    """
    if avg_volume >= min_volume:
        return None

    return {
        'level': 'warning',
        'title': '\U0001F6B0 \u6D41\u52D5\u6027\u4E0D\u8DB3',  # 流動性不足
        'message': (
            f'\u5E73\u5747\u65E5\u6210\u4EA4\u91CF\u50C5 {avg_volume:.0f} \u5343\u80A1'  # 平均日成交量僅 X 千股
            f'\uFF08\u9580\u6ABB {min_volume:.0f} \u5343\u80A1\uFF09\u3002'  # (門檻 X 千股)
            '\u4F4E\u6D41\u52D5\u6027\u80A1\u7968\u53EF\u80FD\u9762\u81E8\u8F03\u5927\u6ED1\u50F9\u548C\u51FA\u5834\u56F0\u96E3\u3002'  # 低流動性股票可能面臨較大滑價和出場困難
        ),
        'icon': '\U0001F4A7',  # droplet
    }


def check_sector_concentration(
    portfolio_weights: dict[str, float],
    max_sector_pct: float = 0.40,
) -> dict[str, str] | None:
    """Check if portfolio is over-concentrated in one sector.

    Args:
        portfolio_weights: Mapping of sector_name -> weight (0~1).
        max_sector_pct: Maximum allowed weight for a single sector.

    Returns:
        Warning dict or None.
    """
    if not portfolio_weights:
        return None

    top_sector = max(portfolio_weights, key=portfolio_weights.get)  # type: ignore[arg-type]
    top_weight = portfolio_weights[top_sector]

    if top_weight <= max_sector_pct:
        return None

    return {
        'level': 'danger' if top_weight >= 0.60 else 'warning',
        'title': '\U0001F3AF \u7522\u696D\u904E\u5EA6\u96C6\u4E2D',  # 產業過度集中
        'message': (
            f'\u300C{top_sector}\u300D\u4F54\u6BD4\u9054 {top_weight*100:.1f}%'  # 「XX」佔比達 X%
            f'\uFF08\u4E0A\u9650 {max_sector_pct*100:.0f}%\uFF09\u3002'  # (上限 X%)
            '\u55AE\u4E00\u7522\u696D\u904E\u5EA6\u96C6\u4E2D\u6703\u653E\u5927\u7522\u696D\u98A8\u96AA\uFF0C'  # 單一產業過度集中會放大產業風險
            '\u5EFA\u8B70\u5206\u6563\u81F3\u4E0D\u540C\u985E\u80A1\u3002'  # 建議分散至不同類股
        ),
        'icon': '\U0001F4CD',  # pin
    }


def check_rsi_extreme(
    rsi_value: float | None,
    overbought: float = 70,
    oversold: float = 30,
) -> dict[str, str] | None:
    """Check for RSI extreme readings.

    Args:
        rsi_value: Current RSI value (0-100).
        overbought: Upper threshold.
        oversold: Lower threshold.

    Returns:
        Warning dict or None.
    """
    if rsi_value is None:
        return None

    if rsi_value >= overbought:
        return {
            'level': 'warning',
            'title': '\U0001F525 RSI \u8D85\u8CB7',  # RSI 超買
            'message': (
                f'RSI = {rsi_value:.1f}\uFF08\u8D85\u8CB7\u9580\u6ABB {overbought}\uFF09\u3002'  # RSI = X (超買門檻 X)
                '\u8D85\u8CB7\u4E0D\u4EE3\u8868\u5FC5\u7136\u4E0B\u8DCC\uFF0C\u4F46\u77ED\u7DDA\u8FFD\u9AD8\u98A8\u96AA\u8F03\u5927\u3002'  # 超買不代表必然下跌，但短線追高風險較大
                '\u53EF\u89C0\u5BDF\u662F\u5426\u51FA\u73FE\u9802\u80CC\u96E2\u3002'  # 可觀察是否出現頂背離
            ),
            'icon': '\U0001F4C8',
        }

    if rsi_value <= oversold:
        return {
            'level': 'info',
            'title': '\u2744\uFE0F RSI \u8D85\u8CE3',  # RSI 超賣
            'message': (
                f'RSI = {rsi_value:.1f}\uFF08\u8D85\u8CE3\u9580\u6ABB {oversold}\uFF09\u3002'  # RSI = X (超賣門檻 X)
                '\u8D85\u8CE3\u53EF\u80FD\u662F\u53CD\u5F48\u6A5F\u6703\uFF0C\u4F46\u4E5F\u53EF\u80FD\u662F\u8D8B\u52E2\u5F37\u52C1\u4E0B\u8DCC\u3002'  # 超賣可能是反彈機會，但也可能是趨勢強勁下跌
                '\u5EFA\u8B70\u7B49\u5F85\u78BA\u8A8D\u8A0A\u865F\u518D\u9032\u5834\u3002'  # 建議等待確認訊號再進場
            ),
            'icon': '\U0001F4C9',
        }

    return None


def check_position_size(
    position_pct: float,
    max_pct: float = 0.15,
) -> dict[str, str] | None:
    """Check if a single position is too large.

    Args:
        position_pct: Position weight as a fraction (0~1).
        max_pct: Maximum allowed weight.

    Returns:
        Warning dict or None.
    """
    if position_pct <= max_pct:
        return None

    return {
        'level': 'danger' if position_pct >= 0.25 else 'warning',
        'title': '\U0001F4B0 \u55AE\u4E00\u90E8\u4F4D\u904E\u5927',  # 單一部位過大
        'message': (
            f'\u6B64\u90E8\u4F4D\u4F54\u7E3D\u8CC7\u7522 {position_pct*100:.1f}%'  # 此部位佔總資產 X%
            f'\uFF08\u4E0A\u9650 {max_pct*100:.0f}%\uFF09\u3002'  # (上限 X%)
            '\u55AE\u4E00\u90E8\u4F4D\u904E\u5927\u6703\u5C0E\u81F4\u500B\u80A1\u98A8\u96AA\u96C6\u4E2D\uFF0C'  # 單一部位過大會導致個股風險集中
            '\u5EFA\u8B70\u63A7\u5236\u5728\u8CC7\u7522 10~15% \u4EE5\u5167\u3002'  # 建議控制在資產 10~15% 以內
        ),
        'icon': '\u26A0\uFE0F',
    }


def check_high_volatility(
    hv: float | None,
    threshold: float = 0.50,
) -> dict[str, str] | None:
    """Check if historical volatility is extremely high.

    Args:
        hv: Annualized historical volatility (e.g. 0.50 = 50%).
        threshold: Level above which we warn.

    Returns:
        Warning dict or None.
    """
    if hv is None or hv <= threshold:
        return None

    return {
        'level': 'warning',
        'title': '\U0001F32A\uFE0F \u6CE2\u52D5\u7387\u904E\u9AD8',  # 波動率過高
        'message': (
            f'\u5E74\u5316\u6CE2\u52D5\u7387 {hv*100:.1f}%'  # 年化波動率 X%
            f'\uFF08\u8B66\u6212\u9580\u6ABB {threshold*100:.0f}%\uFF09\u3002'  # (警戒門檻 X%)
            '\u9AD8\u6CE2\u52D5\u80A1\u7968\u7684\u505C\u640D\u8DDD\u96E2\u61C9\u8A2D\u5F97\u66F4\u5BEC\uFF0C'  # 高波動股票的停損距離應設得更寬
            '\u4E26\u964D\u4F4E\u90E8\u4F4D\u5927\u5C0F\u4EE5\u63A7\u5236\u98A8\u96AA\u3002'  # 並降低部位大小以控制風險
        ),
        'icon': '\U0001F32A\uFE0F',
    }


def check_price_near_resistance(
    current_price: float,
    resistance_price: float,
    threshold: float = 0.03,
) -> dict[str, str] | None:
    """Check if price is very close to a known resistance level.

    Args:
        current_price: Latest price.
        resistance_price: Nearest resistance level.
        threshold: Fractional distance that triggers warning.

    Returns:
        Warning dict or None.
    """
    if resistance_price <= 0:
        return None
    distance = (resistance_price - current_price) / current_price
    if distance < 0 or distance > threshold:
        return None

    return {
        'level': 'info',
        'title': '\U0001F6A7 \u63A5\u8FD1\u58D3\u529B\u5340',  # 接近壓力區
        'message': (
            f'\u7576\u524D\u50F9\u683C\u8DDD\u58D3\u529B\u4F4D {resistance_price:.2f} '  # 當前價格距壓力位 X
            f'\u50C5 {distance*100:.1f}%\u3002'
            '\u63A5\u8FD1\u58D3\u529B\u5340\u6642\u8FFD\u9AD8\u98A8\u96AA\u8F03\u5927\uFF0C'  # 接近壓力區時追高風險較大
            '\u53EF\u7B49\u5F85\u7A81\u7834\u78BA\u8A8D\u5F8C\u518D\u9032\u5834\u3002'  # 可等待突破確認後再進場
        ),
        'icon': '\U0001F6A7',
    }


# ---------------------------------------------------------------------------
# Render warnings
# ---------------------------------------------------------------------------

def _inject_warning_css():
    """Inject warning card CSS once per Streamlit rerun."""
    st.markdown("""<style>
.risk-card {
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    transition: box-shadow 0.2s ease;
}
.risk-card:hover {
    box-shadow: 0 0 12px var(--glow-color, rgba(0,240,255,0.1));
}
.risk-card-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.risk-card-body {
    flex: 1;
    min-width: 0;
}
.risk-card-title {
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}
.risk-card-msg {
    font-size: 0.8rem;
    line-height: 1.55;
}
</style>""", unsafe_allow_html=True)


def render_risk_warnings(warnings_list: list[dict[str, str]]) -> None:
    """Render all risk warnings as styled HTML alert cards.

    Args:
        warnings_list: List of warning dicts with keys
            ``level``, ``title``, ``message``, ``icon``.
    """
    if not warnings_list:
        return

    _inject_warning_css()

    for w in warnings_list:
        level = w.get('level', 'info')
        style = _LEVEL_STYLES.get(level, _LEVEL_STYLES['info'])
        icon = w.get('icon', '')
        title = w.get('title', '')
        message = w.get('message', '')

        st.markdown(
            f'<div class="risk-card" style="'
            f'background:{style["bg"]}; '
            f'border:1px solid {style["border"]}33; '
            f'border-left:3px solid {style["border"]}; '
            f'--glow-color:{style["glow"]};">'
            f'<div class="risk-card-icon">{icon}</div>'
            f'<div class="risk-card-body">'
            f'<div class="risk-card-title" style="color:{style["title_color"]};">{title}</div>'
            f'<div class="risk-card-msg" style="color:{style["text_color"]};">{message}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Aggregate warning generator
# ---------------------------------------------------------------------------

def generate_stock_warnings(
    df: pd.DataFrame,
    ticker: str,
    portfolio: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate all applicable warnings for a stock.

    Args:
        df: OHLCV DataFrame with columns like Close, Volume, RSI, etc.
            Must be sorted by date ascending with the latest row last.
        ticker: Stock ticker string (for logging).
        portfolio: Optional dict with keys like ``position_pct``,
            ``sector_weights`` for portfolio-level checks.

    Returns:
        List of warning dicts (may be empty).
    """
    warnings: list[dict[str, str]] = []

    if df is None or df.empty:
        logger.debug('generate_stock_warnings: empty df for %s', ticker)
        return warnings

    try:
        latest = df.iloc[-1]
        close = float(latest.get('Close', 0))

        # --- Drawdown from peak ---
        if 'Close' in df.columns and len(df) >= 5:
            peak = float(df['Close'].max())
            w = check_drawdown_warning(close, peak, threshold=0.15)
            if w:
                warnings.append(w)

        # --- Volume spike ---
        if 'Volume' in df.columns and len(df) >= 20:
            vol_now = float(latest['Volume'])
            vol_avg = float(df['Volume'].iloc[-21:-1].mean())
            if vol_avg > 0:
                w = check_volume_spike(vol_now, vol_avg, threshold=3.0)
                if w:
                    warnings.append(w)

        # --- Low liquidity ---
        if 'Volume' in df.columns and len(df) >= 20:
            avg_vol_k = float(df['Volume'].iloc[-20:].mean()) / 1000
            w = check_low_liquidity(avg_vol_k, min_volume=500)
            if w:
                warnings.append(w)

        # --- RSI extreme ---
        rsi_val = None
        if 'RSI' in df.columns:
            rsi_val = latest.get('RSI')
        elif 'rsi' in df.columns:
            rsi_val = latest.get('rsi')
        if rsi_val is not None:
            try:
                rsi_val = float(rsi_val)
                w = check_rsi_extreme(rsi_val)
                if w:
                    warnings.append(w)
            except (ValueError, TypeError):
                pass

        # --- High volatility ---
        if 'Close' in df.columns and len(df) >= 60:
            try:
                returns = df['Close'].pct_change().dropna()
                if len(returns) >= 20:
                    hv = float(returns.iloc[-20:].std() * (252 ** 0.5))
                    w = check_high_volatility(hv, threshold=0.50)
                    if w:
                        warnings.append(w)
            except Exception:
                pass

        # --- Portfolio-level checks ---
        if portfolio:
            pos_pct = portfolio.get('position_pct')
            if pos_pct is not None:
                w = check_position_size(float(pos_pct), max_pct=0.15)
                if w:
                    warnings.append(w)

            sector_weights = portfolio.get('sector_weights')
            if sector_weights:
                w = check_sector_concentration(sector_weights, max_sector_pct=0.40)
                if w:
                    warnings.append(w)

    except Exception as exc:
        logger.warning('generate_stock_warnings failed for %s: %s', ticker, exc)

    # Sort: danger first, then warning, then info
    level_order = {'danger': 0, 'warning': 1, 'info': 2}
    warnings.sort(key=lambda w: level_order.get(w.get('level', 'info'), 9))

    return warnings
