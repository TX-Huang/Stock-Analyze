"""
突破偵測引擎 — 複合壓力/支撐分析
三方法投票: 趨勢線 + Swing High/Low + 成交量密集區
"""
import numpy as np
import pandas as pd
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


# ─── Method 1: Trendline (argrelextrema) ───

def _trendline_levels(df, order=10):
    """用 argrelextrema 找波峰波谷，連線延伸到最新一根 K 棒的價位。"""
    close = df['Close'].values
    n = len(close)
    if n < order * 3:
        return None, None

    peak_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    trough_idx = argrelextrema(close, np.less_equal, order=order)[0]

    resistance = None
    support = None

    # 壓力線: 最近兩個波峰連線延伸
    if len(peak_idx) >= 2:
        p1, p2 = peak_idx[-2], peak_idx[-1]
        if p2 != p1:
            slope = (close[p2] - close[p1]) / (p2 - p1)
            resistance = close[p2] + slope * (n - 1 - p2)

    # 支撐線: 最近兩個波谷連線延伸
    if len(trough_idx) >= 2:
        t1, t2 = trough_idx[-2], trough_idx[-1]
        if t2 != t1:
            slope = (close[t2] - close[t1]) / (t2 - t1)
            support = close[t2] + slope * (n - 1 - t2)

    if resistance is not None and resistance <= 0:
        resistance = None
    if support is not None and support <= 0:
        support = None

    return resistance, support


# ─── Method 2: Swing High/Low ───

def _swing_levels(df, lookback=60):
    """近 N 天的 swing high/low 群聚偵測。"""
    recent = df.tail(lookback)
    if len(recent) < 10:
        return None, None

    high_vals = recent['High'].values
    low_vals = recent['Low'].values

    # 找近期的顯著高點 (top 5 highs 的中位數)
    top_n = min(5, len(high_vals))
    sorted_highs = np.sort(high_vals)[-top_n:]
    resistance = float(np.median(sorted_highs))

    # 找近期的顯著低點 (bottom 5 lows 的中位數)
    sorted_lows = np.sort(low_vals)[:top_n]
    support = float(np.median(sorted_lows))

    return resistance, support


# ─── Method 3: Volume Profile ───

def _volume_levels(df, bins=20, lookback=120):
    """成交量密集區 — 哪些價位交易量最大。"""
    recent = df.tail(lookback)
    if len(recent) < 20 or 'Volume' not in recent.columns:
        return None, None

    prices = recent['Close'].values
    volumes = recent['Volume'].values

    # 價格分箱
    price_min, price_max = prices.min(), prices.max()
    if price_max == price_min:
        return None, None

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)

    for i in range(len(prices)):
        if np.isnan(prices[i]) or np.isnan(volumes[i]):
            continue
        bin_idx = np.searchsorted(bin_edges[1:], prices[i])
        bin_idx = min(bin_idx, bins - 1)
        bin_volumes[bin_idx] += volumes[i]

    current_price = prices[-1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 找 current_price 上方成交量最大的區間 → 壓力
    above_mask = bin_centers > current_price
    resistance = None
    if above_mask.any():
        above_vols = bin_volumes.copy()
        above_vols[~above_mask] = 0
        if above_vols.max() > 0:
            resistance = float(bin_centers[np.argmax(above_vols)])

    # 找 current_price 下方成交量最大的區間 → 支撐
    below_mask = bin_centers < current_price
    support = None
    if below_mask.any():
        below_vols = bin_volumes.copy()
        below_vols[~below_mask] = 0
        if below_vols.max() > 0:
            support = float(bin_centers[np.argmax(below_vols)])

    return resistance, support


# ─── Composite Level Detection ───

def detect_levels(df, tolerance_pct=0.02):
    """
    複合壓力/支撐偵測。三方法投票，回傳最終 level + score (1-3)。

    Returns:
        {
            'resistance': float or None,
            'resistance_score': int (0-3),
            'resistance_methods': list,
            'support': float or None,
            'support_score': int (0-3),
            'support_methods': list,
        }
    """
    r1, s1 = _trendline_levels(df)
    r2, s2 = _swing_levels(df)
    r3, s3 = _volume_levels(df)

    def _consensus(values, method_names):
        """找共識值: 在 tolerance_pct 範圍內的值取平均。"""
        valid = [(v, m) for v, m in zip(values, method_names) if v is not None and np.isfinite(v)]
        if not valid:
            return None, 0, []

        if len(valid) == 1:
            return valid[0][0], 1, [valid[0][1]]

        # 以第一個值為基準，找在容忍範圍內的
        vals = [v for v, _ in valid]
        methods = [m for _, m in valid]
        base = np.median(vals)

        agreed_vals = []
        agreed_methods = []
        for v, m in zip(vals, methods):
            if abs(v - base) / max(abs(base), 1e-6) <= tolerance_pct * 2:
                agreed_vals.append(v)
                agreed_methods.append(m)

        if agreed_vals:
            return float(np.mean(agreed_vals)), len(agreed_vals), agreed_methods
        else:
            # 沒有共識，取中位數，score=1
            return float(np.median(vals)), 1, [methods[0]]

    r_val, r_score, r_methods = _consensus(
        [r1, r2, r3], ['趨勢線', 'Swing', '量能']
    )
    s_val, s_score, s_methods = _consensus(
        [s1, s2, s3], ['趨勢線', 'Swing', '量能']
    )

    return {
        'resistance': r_val,
        'resistance_score': r_score,
        'resistance_methods': r_methods,
        'support': s_val,
        'support_score': s_score,
        'support_methods': s_methods,
    }


# ─── Method 4: VCP (Volatility Contraction Pattern) ───

def detect_vcp(df, bb_lookback=20, bb_quantile_window=60, vol_dry_lookback=10,
               vol_dry_ratio=0.5, higher_low_recent=20, higher_low_past=60):
    """
    VCP 偵測 — 波動收縮 + 量能萎縮 + 底部墊高 + 趨勢健康。
    從 strategies/vcp.py (Minervini) 的核心邏輯抽出，適用於單支個股即時偵測。

    Returns:
        {
            'is_vcp': bool,              # 是否符合 VCP
            'vcp_score': int (0-4),      # 符合幾項條件
            'conditions': {
                'contraction': bool,     # 波動收縮
                'volume_dry': bool,      # 量能萎縮
                'higher_low': bool,      # 底部墊高
                'trend_healthy': bool,   # 趨勢健康 (MA50 > MA150)
            },
            'details': {
                'bandwidth_pctl': float, # BB帶寬百分位 (越低=越收斂)
                'vol_dry_days': int,     # 近N天中有幾天量能萎縮
                'low_change_pct': float, # 近期低點 vs 前期低點 變化%
            }
        }
    """
    n = len(df)
    if n < 200:
        return {'is_vcp': False, 'vcp_score': 0,
                'conditions': {'contraction': False, 'volume_dry': False,
                               'higher_low': False, 'trend_healthy': False},
                'details': {'bandwidth_pctl': 1.0, 'vol_dry_days': 0, 'low_change_pct': 0}}

    close = df['Close'].values
    low = df['Low'].values
    volume = df['Volume'].values if 'Volume' in df.columns else np.ones(n)

    # ── 1. 波動收縮 (Bollinger Band 帶寬在低位) ──
    close_s = pd.Series(close, index=df.index)
    ma_bb = close_s.rolling(bb_lookback).mean()
    std_bb = close_s.rolling(bb_lookback).std()
    bandwidth = (2 * std_bb) / ma_bb.where(ma_bb > 0, np.nan)
    bw_pctl = bandwidth.rolling(bb_quantile_window).rank(pct=True)

    bw_current = float(bw_pctl.iloc[-1]) if pd.notna(bw_pctl.iloc[-1]) else 1.0
    contraction = bw_current <= 0.25  # 帶寬在近60天最低25%

    # ── 2. 量能萎縮 (近10天有出現低量日) ──
    vol_s = pd.Series(volume, index=df.index)
    vol_ma50 = vol_s.rolling(50).mean()
    is_dry_day = vol_s < vol_ma50 * vol_dry_ratio  # 成交量 < MA50 × 0.5
    dry_days = int(is_dry_day.tail(vol_dry_lookback).sum())
    volume_dry = dry_days >= 2  # 近10天至少2天量能萎縮

    # ── 3. 底部墊高 (Higher Lows) ──
    low_s = pd.Series(low, index=df.index)
    low_recent = float(low_s.tail(higher_low_recent).min())
    low_past = float(low_s.iloc[-(higher_low_recent + higher_low_past):-higher_low_recent].min()) \
        if n >= higher_low_recent + higher_low_past else low_recent
    low_change = (low_recent - low_past) / max(abs(low_past), 1e-6)
    higher_low = low_recent > low_past  # 近期低點 > 前期低點

    # ── 4. 趨勢健康 (Minervini 模板簡化版) ──
    ma50 = close_s.rolling(50).mean()
    ma150 = close_s.rolling(150).mean()
    trend_ok = bool(
        close[-1] > ma50.iloc[-1] and
        ma50.iloc[-1] > ma150.iloc[-1] and
        pd.notna(ma50.iloc[-1]) and pd.notna(ma150.iloc[-1])
    )

    # ── Score ──
    conditions = {
        'contraction': contraction,
        'volume_dry': volume_dry,
        'higher_low': higher_low,
        'trend_healthy': trend_ok,
    }
    score = sum(conditions.values())

    return {
        'is_vcp': score >= 3,  # 至少3/4才算VCP成形
        'vcp_score': score,
        'conditions': conditions,
        'details': {
            'bandwidth_pctl': round(bw_current, 3),
            'vol_dry_days': dry_days,
            'low_change_pct': round(low_change * 100, 2),
        }
    }


# ─── Signal Detection ───

SIGNAL_TYPES = {
    'volume_breakout':    {'label': '帶量突破壓力', 'level': 'critical', 'icon': '🔴', 'color': '#ef4444'},
    'vcp_breakout':       {'label': 'VCP 帶量突破', 'level': 'critical', 'icon': '🔥', 'color': '#ef4444'},
    'breakout':           {'label': '突破壓力',     'level': 'warning',  'icon': '🟡', 'color': '#f59e0b'},
    'vcp_ready':          {'label': 'VCP 成形中',   'level': 'warning',  'icon': '🟣', 'color': '#a855f7'},
    'near_resistance':    {'label': '即將觸壓',     'level': 'warning',  'icon': '🟡', 'color': '#f59e0b'},
    'support_bounce':     {'label': '回測支撐反彈', 'level': 'info',     'icon': '🔵', 'color': '#3b82f6'},
    'break_support':      {'label': '跌破支撐',     'level': 'critical', 'icon': '🔴', 'color': '#ef4444'},
    'volume_break_support': {'label': '帶量跌破支撐', 'level': 'critical', 'icon': '🔴', 'color': '#ef4444'},
}


def detect_signal(df, levels, near_pct=0.02, vol_ratio_threshold=1.5):
    """
    根據當前價格、壓力/支撐、成交量、VCP 型態，判斷信號。

    信號優先順序:
        1. VCP 帶量突破 (VCP 成形 + 突破壓力 + 爆量)
        2. 帶量突破壓力
        3. 突破壓力 (量能不足)
        4. VCP 成形中 (尚未突破，但型態到位)
        5. 即將觸壓
        6. 帶量跌破支撐
        7. 跌破支撐
        8. 回測支撐反彈
    """
    if len(df) < 2:
        return None

    price = float(df['Close'].iloc[-1])
    if np.isnan(price) or price == 0:
        return None
    prev_close = float(df['Close'].iloc[-2])
    if prev_close == 0 or np.isnan(prev_close):
        return None
    daily_change = (price - prev_close) / prev_close

    # Volume ratio (exclude current day from MA to avoid lookahead)
    vol_now = float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0
    vol_ma20 = float(df['Volume'].iloc[-21:-1].mean()) if len(df) >= 21 else float(df['Volume'].iloc[:-1].mean())
    vol_ratio = vol_now / max(vol_ma20, 1)

    resistance = levels.get('resistance')
    support = levels.get('support')

    dist_r = (resistance - price) / price if resistance and price > 0 else None
    dist_s = (price - support) / price if support and price > 0 else None

    # VCP 偵測: 用「不含最後一根 K 棒」的資料判斷
    # 因為突破當天的大陽線會破壞收縮條件，應以突破前的型態為準
    vcp = detect_vcp(df.iloc[:-1]) if len(df) > 50 else detect_vcp(df)
    # 也對最新資料跑一次，取較高的 score
    vcp_current = detect_vcp(df)
    if vcp_current['vcp_score'] > vcp['vcp_score']:
        vcp = vcp_current

    signal = None

    # 1. VCP 帶量突破 (最強信號 — VCP 成形 + 突破 + 爆量)
    if vcp['is_vcp'] and resistance and price > resistance and vol_ratio >= vol_ratio_threshold:
        signal = 'vcp_breakout'
    # 2. 帶量突破壓力
    elif resistance and price > resistance and vol_ratio >= vol_ratio_threshold:
        signal = 'volume_breakout'
    # 3. 突破壓力 (量能不足)
    elif resistance and price > resistance:
        signal = 'breakout'
    # 4. VCP 成形中 (型態到位，還沒突破 — 這就是「可能突破」提醒)
    elif vcp['is_vcp']:
        signal = 'vcp_ready'
    # 5. 即將觸壓
    elif resistance and dist_r is not None and 0 < dist_r <= near_pct:
        signal = 'near_resistance'
    # 6. 帶量跌破支撐
    elif support and price < support and vol_ratio >= vol_ratio_threshold:
        signal = 'volume_break_support'
    # 7. 跌破支撐
    elif support and price < support:
        signal = 'break_support'
    # 8. 回測支撐反彈 (觸及支撐 + 今日反彈)
    elif support and dist_s is not None and dist_s <= near_pct * 1.5 and daily_change > 0:
        signal = 'support_bounce'

    result = {
        'signal': signal,
        'signal_info': SIGNAL_TYPES.get(signal) if signal else None,
        'price': price,
        'distance_to_resistance_pct': dist_r,
        'distance_to_support_pct': dist_s,
        'volume_ratio': vol_ratio,
        'daily_change_pct': daily_change,
        'vcp': vcp,
    }
    return result


# ─── Batch Scanner ───

def scan_breakouts(ticker_list, provider, period="6mo"):
    """
    批次掃描多檔股票的突破信號。

    Args:
        ticker_list: [{'ticker': '2330', 'name': '台積電', 'source': 'watchlist'}, ...]
        provider: DataProvider instance
        period: 歷史資料期間

    Returns:
        list of scan results, sorted by signal severity
    """
    results = []
    severity_order = {'critical': 0, 'warning': 1, 'info': 2}

    for item in ticker_list:
        ticker = item['ticker']
        name = item.get('name', ticker)
        source = item.get('source', 'unknown')

        try:
            df = provider.get_historical_data(ticker, period=period, interval="1d")
            if df is None or df.empty or len(df) < 30:
                continue

            levels = detect_levels(df)
            signal_result = detect_signal(df, levels)

            if signal_result is None:
                continue

            vcp = signal_result.get('vcp', {})
            results.append({
                'ticker': ticker,
                'name': name,
                'source': source,
                'price': signal_result['price'],
                'resistance': levels['resistance'],
                'resistance_score': levels['resistance_score'],
                'resistance_methods': levels['resistance_methods'],
                'support': levels['support'],
                'support_score': levels['support_score'],
                'support_methods': levels['support_methods'],
                'signal': signal_result['signal'],
                'signal_info': signal_result['signal_info'],
                'distance_to_resistance_pct': signal_result['distance_to_resistance_pct'],
                'distance_to_support_pct': signal_result['distance_to_support_pct'],
                'volume_ratio': signal_result['volume_ratio'],
                'daily_change_pct': signal_result['daily_change_pct'],
                'vcp_score': vcp.get('vcp_score', 0),
                'vcp_conditions': vcp.get('conditions', {}),
                'vcp_details': vcp.get('details', {}),
            })
        except Exception as e:
            logger.warning(f"掃描 {ticker} 失敗: {e}")
            continue

    # 有信號的排前面，按嚴重度排序
    def sort_key(r):
        sig_info = r.get('signal_info')
        if r.get('signal') is None or sig_info is None:
            return (3, 0)
        return (severity_order.get(sig_info.get('level', ''), 9), -abs(r.get('volume_ratio', 0)))

    results.sort(key=sort_key)
    return results


def format_scan_results_for_telegram(results):
    """將掃描結果格式化為 Telegram 訊息。"""
    if not results:
        return "📡 掃描完成：無突破信號。"

    signals_only = [r for r in results if r['signal'] is not None]
    if not signals_only:
        return "📡 掃描完成：無突破信號。"

    lines = [f"📡 突破偵測報告 ({len(signals_only)} 檔有信號)\n"]

    for r in signals_only:
        info = r.get('signal_info') or {}
        if not info:
            continue
        dist_r = r.get('distance_to_resistance_pct')
        dist_s = r.get('distance_to_support_pct')

        line = f"{info['icon']} {r['name']} ({r['ticker']})"
        line += f"\n   {info['label']} | 現價: {r['price']:,.1f}"

        if r.get('resistance') is not None:
            line += f"\n   壓力: {r['resistance']:,.1f}"
            if dist_r is not None:
                line += f" (距{dist_r*100:+.1f}%)"
            line += f" [Score {r['resistance_score']}/3]"

        if r.get('support') is not None:
            line += f"\n   支撐: {r['support']:,.1f}"
            if dist_s is not None:
                line += f" (距{dist_s*100:+.1f}%)"

        if r['volume_ratio'] > 1.2:
            line += f"\n   量比: {r['volume_ratio']:.1f}x {'🔥' if r['volume_ratio'] > 1.5 else ''}"

        vcp_score = r.get('vcp_score', 0)
        if vcp_score >= 3:
            conds = r.get('vcp_conditions', {})
            checks = ''.join(['✅' if v else '❌' for v in conds.values()])
            line += f"\n   VCP [{vcp_score}/4] {checks} 收斂|縮量|墊高|趨勢"

        lines.append(line)

    return "\n\n".join(lines)
