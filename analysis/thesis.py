"""
交易論述引擎 — Trade Thesis Generator
第一層：規則引擎（確定性評分）
第二層：AI 解讀（Gemini，選配）
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ==========================================
# Layer 1: Rule-Based Scoring Engine
# ==========================================

def generate_thesis(ticker, df, chip_data=None, signal_data=None,
                    levels=None, position=None, portfolio_tickers=None):
    """
    生成交易論述（規則引擎）。

    Args:
        ticker: 股票代碼
        df: 歷史 K 線 DataFrame
        chip_data: analyze_chip_for_ticker() 的結果
        signal_data: detect_signal() 的結果
        levels: detect_levels() 的結果
        position: 持倉資訊 {'entry_price', 'shares', 'days_held'} or None
        portfolio_tickers: 目前持倉的其他股票代碼列表 (用於相關性計算)

    Returns:
        dict: {
            'composite_score': float (0-10),
            'verdict': str,
            'technical': {...},
            'chip': {...},
            'risk': {...},
            'action': {...},
        }
    """
    if df is None or df.empty or len(df) < 20:
        return _empty_thesis()

    price = float(df['Close'].iloc[-1])

    # ── Technical Score (0-10) ──
    tech = _score_technical(df, signal_data, levels)

    # ── Chip Score (0-10) ──
    chip = _score_chip(chip_data)

    # ── Risk Score (0-10, higher = MORE risk) ──
    risk = _score_risk(df, position, portfolio_tickers)

    # ── Action Plan ──
    action = _compute_action(df, price, levels, position)

    # ── Composite Score ──
    # 技術 40% + 籌碼 30% + (10 - 風險) 30%
    composite = tech['score'] * 0.4 + chip['score'] * 0.3 + (10 - risk['score']) * 0.3
    composite = round(max(0, min(10, composite)), 1)

    # Verdict
    if composite >= 8:
        verdict = "強力看多"
    elif composite >= 6.5:
        verdict = "偏多"
    elif composite >= 5:
        verdict = "中性觀望"
    elif composite >= 3.5:
        verdict = "偏空"
    else:
        verdict = "強力看空"

    return {
        'ticker': ticker,
        'price': price,
        'composite_score': composite,
        'verdict': verdict,
        'technical': tech,
        'chip': chip,
        'risk': risk,
        'action': action,
        'generated_at': datetime.now().isoformat(),
    }


def _score_technical(df, signal_data, levels):
    """技術面評分 (0-10)"""
    score = 5.0  # baseline
    details = []

    close = df['Close'].values
    n = len(close)

    # 1. Trend (MA alignment)
    if n >= 60:
        ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        ma60 = pd.Series(close).rolling(60).mean().iloc[-1]
        if pd.notna(ma20) and pd.notna(ma60):
            if close[-1] > ma20 > ma60:
                score += 1.5
                details.append("均線多頭排列 (價>MA20>MA60)")
            elif close[-1] < ma20 < ma60:
                score -= 1.5
                details.append("均線空頭排列 (價<MA20<MA60)")
            elif close[-1] > ma20:
                score += 0.5
                details.append("站上月線")
            else:
                score -= 0.5
                details.append("跌破月線")

    # 2. VCP
    if signal_data:
        vcp = signal_data.get('vcp', {})
        vcp_score = vcp.get('vcp_score', 0)
        if vcp_score >= 3:
            score += 2.0
            details.append(f"VCP 成形 ({vcp_score}/4)")
        elif vcp_score >= 2:
            score += 1.0
            details.append(f"VCP 部分成形 ({vcp_score}/4)")

    # 3. Distance to resistance
    if signal_data:
        dist_r = signal_data.get('distance_to_resistance_pct')
        if dist_r is not None:
            if dist_r < 0:  # Already above resistance
                score += 1.5
                details.append("已突破壓力線")
            elif dist_r < 0.02:
                score += 0.5
                details.append(f"距壓力 {dist_r*100:.1f}% (即將觸壓)")
            elif dist_r > 0.10:
                score -= 0.5
                details.append(f"距壓力 {dist_r*100:.1f}% (空間充足)")

    # 4. Volume ratio
    if signal_data:
        vol_ratio = signal_data.get('volume_ratio', 1.0)
        if vol_ratio >= 2.0:
            score += 1.0
            details.append(f"爆量 (量比 {vol_ratio:.1f}x)")
        elif vol_ratio >= 1.5:
            score += 0.5
            details.append(f"放量 (量比 {vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            details.append(f"量能萎縮 (量比 {vol_ratio:.1f}x)")

    # 5. Signal type bonus
    if signal_data and signal_data.get('signal'):
        sig = signal_data['signal']
        if sig in ('vcp_breakout', 'volume_breakout'):
            score += 1.0
            details.append("帶量突破確認")
        elif sig == 'break_support':
            score -= 2.0
            details.append("跌破支撐線")
        elif sig == 'volume_break_support':
            score -= 2.5
            details.append("帶量跌破支撐")

    score = round(max(0, min(10, score)), 1)
    return {'score': score, 'details': details}


def _score_chip(chip_data):
    """籌碼面評分 (0-10)"""
    score = 5.0
    details = []

    if not chip_data:
        return {'score': 5.0, 'details': ["無籌碼資料"]}

    # Foreign investor
    foreign_streak = chip_data.get('foreign_streak', 0)
    if foreign_streak >= 5:
        score += 2.0
        details.append(f"外資連 {foreign_streak} 天買超")
    elif foreign_streak >= 3:
        score += 1.0
        details.append(f"外資連 {foreign_streak} 天買超")
    elif foreign_streak <= -5:
        score -= 2.0
        details.append(f"外資連 {abs(foreign_streak)} 天賣超")
    elif foreign_streak <= -3:
        score -= 1.0
        details.append(f"外資連 {abs(foreign_streak)} 天賣超")

    # Trust (投信)
    trust_streak = chip_data.get('trust_streak', 0)
    if trust_streak >= 3:
        score += 1.5
        details.append(f"投信連 {trust_streak} 天買超")
    elif trust_streak <= -3:
        score -= 1.5
        details.append(f"投信連 {abs(trust_streak)} 天賣超")

    # Overall chip score
    chip_score = chip_data.get('chip_score', 0)
    if chip_score >= 4:
        score += 1.0
        details.append(f"籌碼綜合分 {chip_score}/6 (偏多)")
    elif chip_score <= -3:
        score -= 1.0
        details.append(f"籌碼綜合分 {chip_score}/6 (偏空)")

    # Foreign + Trust aligned
    if foreign_streak > 0 and trust_streak > 0:
        score += 0.5
        details.append("外資投信同步買超")
    elif foreign_streak < 0 and trust_streak < 0:
        score -= 0.5
        details.append("外資投信同步賣超")

    score = round(max(0, min(10, score)), 1)
    return {'score': score, 'details': details}


def _score_risk(df, position, portfolio_tickers):
    """風險評分 (0-10, 越高=越危險)"""
    score = 3.0  # baseline medium-low risk
    details = []

    close = df['Close'].values

    # 1. Volatility (20-day)
    if len(close) >= 20:
        returns = np.diff(close[-21:]) / close[-21:-1]
        vol = float(np.std(returns)) * np.sqrt(252) * 100
        if vol > 50:
            score += 2.0
            details.append(f"年化波動 {vol:.0f}% (極高)")
        elif vol > 35:
            score += 1.0
            details.append(f"年化波動 {vol:.0f}% (偏高)")
        elif vol < 15:
            score -= 0.5
            details.append(f"年化波動 {vol:.0f}% (穩定)")

    # 2. Drawdown from recent high
    if len(close) >= 60:
        high_60 = float(np.max(close[-60:]))
        dd = (close[-1] - high_60) / high_60 * 100
        if dd < -20:
            score += 2.0
            details.append(f"近 60 日回撤 {dd:.1f}%")
        elif dd < -10:
            score += 1.0
            details.append(f"近 60 日回撤 {dd:.1f}%")

    # 3. VIX check (passed in via signal_data or external caller)
    # Removed dependency on ui.components; VIX data should be provided externally
    # via signal_data['vix'] if available
    if signal_data:
        vix = signal_data.get('vix')
        if vix is not None:
            try:
                vix = float(vix)
                if vix > 30:
                    score += 1.5
                    details.append(f"VIX {vix:.1f} (恐慌)")
                elif vix > 25:
                    score += 0.5
                    details.append(f"VIX {vix:.1f} (偏高)")
                elif vix < 15:
                    score -= 0.5
                    details.append(f"VIX {vix:.1f} (平穩)")
            except (ValueError, TypeError):
                pass

    # 4. Position concentration risk
    if position and position.get('shares', 0) > 0:
        try:
            from utils.helpers import safe_json_read
            from config.paths import PAPER_TRADE_PATH
            paper = safe_json_read(PAPER_TRADE_PATH, {})
            equity = paper.get('cash', 0) + sum(
                p.get('current_price', p.get('entry_price', 0)) * p.get('shares', 0)
                for p in paper.get('positions', [])
            )
            if equity > 0:
                pos_value = position['entry_price'] * position['shares']
                weight = pos_value / equity * 100
                if weight > 20:
                    score += 1.5
                    details.append(f"單一持倉占比 {weight:.0f}% (過度集中)")
                elif weight > 10:
                    score += 0.5
                    details.append(f"單一持倉占比 {weight:.0f}%")
        except (ImportError, FileNotFoundError, KeyError, TypeError) as e:
            logger.debug(f"持倉集中度計算失敗: {e}")

    score = round(max(0, min(10, score)), 1)
    return {'score': score, 'details': details}


def _compute_action(df, price, levels, position):
    """計算建議操作：進場/停損/停利/建議股數"""
    action = {
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'suggested_shares': 0,
        'risk_reward': None,
        'action_type': 'hold',  # buy / sell / hold
        'reason': '',
    }

    resistance = levels.get('resistance') if levels else None
    support = levels.get('support') if levels else None

    # ATR for stop calculation
    atr = _calc_atr(df, period=14)

    # Entry: use resistance as breakout entry, or current price for dip buy
    if resistance and price < resistance:
        action['entry'] = round(resistance, 1)
        action['reason'] = f"突破 {resistance:.0f} 時進場"
    else:
        action['entry'] = round(price, 1)
        action['reason'] = "可於現價附近進場"

    # Stop loss: support line or ATR-based
    if support and support < price:
        action['stop_loss'] = round(support * 0.99, 1)  # 1% below support
    elif atr > 0:
        action['stop_loss'] = round(price - 2 * atr, 1)
    else:
        action['stop_loss'] = round(price * 0.92, 1)  # Default 8%

    # Take profit: 2x the risk distance
    if action['stop_loss'] and action['entry']:
        risk_dist = action['entry'] - action['stop_loss']
        if risk_dist > 0:
            action['take_profit'] = round(action['entry'] + risk_dist * 2, 1)

    # Risk/reward ratio
    if action['entry'] and action['stop_loss'] and action['take_profit']:
        risk = action['entry'] - action['stop_loss']
        reward = action['take_profit'] - action['entry']
        if risk > 0:
            action['risk_reward'] = round(reward / risk, 1)

    # Suggested shares (fixed 2% risk per trade)
    try:
        from utils.helpers import safe_json_read
        from config.paths import PAPER_TRADE_PATH
        paper = safe_json_read(PAPER_TRADE_PATH, {})
        equity = paper.get('cash', 0) + sum(
            p.get('current_price', p.get('entry_price', 0)) * p.get('shares', 0)
            for p in paper.get('positions', [])
        )
        if equity > 0 and action['entry'] and action['stop_loss']:
            risk_per_share = action['entry'] - action['stop_loss']
            if risk_per_share > 0:
                max_risk = equity * 0.02  # 2% of portfolio
                shares = int(max_risk / risk_per_share)
                # Round to lot (台股零股也可以，但建議整數)
                action['suggested_shares'] = max(1, shares)
    except (ImportError, FileNotFoundError, KeyError, TypeError) as e:
        logger.debug(f"建議股數計算失敗: {e}")

    # Action type
    if position:
        entry_price = position.get('entry_price', 0)
        if entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price
            if action['stop_loss'] and price < action['stop_loss']:
                action['action_type'] = 'sell'
                action['reason'] = "已跌破停損線，建議出場"
            elif pnl_pct > 0.15:
                action['action_type'] = 'hold'
                action['reason'] = f"獲利 {pnl_pct*100:.1f}%，持有並上移停損"
            else:
                action['action_type'] = 'hold'
    else:
        action['action_type'] = 'buy'

    return action


def _calc_atr(df, period=14):
    """計算 ATR"""
    try:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        n = len(close)
        if n < period + 1:
            return 0
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(abs(high[1:] - close[:-1]),
                                   abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-period:]))
        return atr
    except (KeyError, IndexError, ValueError, TypeError) as e:
        logger.debug(f"ATR 計算失敗: {e}")
        return 0


def _empty_thesis():
    return {
        'ticker': '',
        'price': 0,
        'composite_score': 0,
        'verdict': '資料不足',
        'technical': {'score': 0, 'details': ['資料不足']},
        'chip': {'score': 0, 'details': ['資料不足']},
        'risk': {'score': 0, 'details': ['資料不足']},
        'action': {
            'entry': None, 'stop_loss': None, 'take_profit': None,
            'suggested_shares': 0, 'risk_reward': None,
            'action_type': 'hold', 'reason': '資料不足',
        },
        'generated_at': datetime.now().isoformat(),
    }


# ==========================================
# Layer 2: AI Narrator (Gemini, Optional)
# ==========================================

def generate_ai_narrative(ticker, thesis_json, _client=None):
    """
    用 Gemini 生成自然語言交易解讀（快取 30 分鐘）。

    Args:
        ticker: 股票代碼
        thesis_json: generate_thesis() 的結果（dict）
        _client: google.genai.Client instance

    Returns:
        str: AI 生成的解讀文字，或 None
    """
    if _client is None:
        return None

    try:
        tech = thesis_json.get('technical', {})
        chip = thesis_json.get('chip', {})
        risk = thesis_json.get('risk', {})
        action = thesis_json.get('action', {})

        prompt = f"""你是專業台股分析師。以下是 {ticker} 的量化分析結果，請用繁體中文、3-5 句話給出精簡的交易建議。

## 量化分析摘要
- 綜合評分: {thesis_json.get('composite_score', 0)}/10 ({thesis_json.get('verdict', '')})
- 現價: {thesis_json.get('price', 0):,.1f}

## 技術面 ({tech.get('score', 0)}/10)
{chr(10).join('- ' + d for d in tech.get('details', []))}

## 籌碼面 ({chip.get('score', 0)}/10)
{chr(10).join('- ' + d for d in chip.get('details', []))}

## 風險 ({risk.get('score', 0)}/10)
{chr(10).join('- ' + d for d in risk.get('details', []))}

## 建議操作
- 進場: {action.get('entry', '--')}
- 停損: {action.get('stop_loss', '--')}
- 停利: {action.get('take_profit', '--')}
- 風報比: {action.get('risk_reward', '--')}

要求：
1. 只基於以上數據分析，不要編造數據
2. 說明目前型態和主要驅動因素
3. 指出最大風險
4. 給出明確的操作時機建議
5. 用簡潔專業的語氣，每句話要有資訊量"""

        from config.settings import GEMINI_MODEL
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    except Exception as e:
        logger.warning(f"AI narrative generation failed for {ticker}: {e}")
        return None
