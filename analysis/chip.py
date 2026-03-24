"""
法人籌碼分析模組
- 外資/投信/自營商買賣超
- 融資融券餘額變化
- 籌碼集中度指標
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_institutional_data(api_token=None):
    """
    從 FinLab 取得法人買賣超資料。
    Returns dict of DataFrames.
    """
    try:
        import finlab
        if api_token:
            finlab.login(api_token)
        from finlab import data

        result = {}

        # 三大法人買賣超 (股數)
        try:
            result['foreign_buy'] = data.get('institutional_investors:外資買賣超股數')
        except Exception:
            result['foreign_buy'] = None

        try:
            result['trust_buy'] = data.get('institutional_investors:投信買賣超股數')
        except Exception:
            result['trust_buy'] = None

        try:
            result['dealer_buy'] = data.get('institutional_investors:自營商買賣超股數')
        except Exception:
            result['dealer_buy'] = None

        # 融資融券
        try:
            result['margin_buy'] = data.get('margin_trading:融資買進')
            result['margin_sell'] = data.get('margin_trading:融資賣出')
            result['margin_balance'] = data.get('margin_trading:融資餘額')
        except Exception:
            result['margin_buy'] = None
            result['margin_sell'] = None
            result['margin_balance'] = None

        try:
            result['short_sell'] = data.get('margin_trading:融券賣出')
            result['short_cover'] = data.get('margin_trading:融券買進')
            result['short_balance'] = data.get('margin_trading:融券餘額')
        except Exception:
            result['short_sell'] = None
            result['short_cover'] = None
            result['short_balance'] = None

        return result

    except Exception as e:
        logger.warning(f"無法取得法人資料: {e}")
        return {}


def analyze_chip_for_ticker(ticker, chip_data, lookback=20):
    """
    分析單一個股的籌碼面。

    Returns:
        {
            'foreign_net_5d': float,      # 外資近5日累計淨買(萬股)
            'foreign_net_20d': float,     # 外資近20日累計淨買
            'foreign_streak': int,        # 外資連續買超天數 (負=連續賣超)
            'trust_net_5d': float,
            'trust_net_20d': float,
            'trust_streak': int,
            'dealer_net_5d': float,
            'total_inst_net_20d': float,  # 三大法人合計20日淨買
            'margin_change_5d': float,    # 融資餘額5日變化%
            'margin_change_20d': float,
            'short_change_5d': float,     # 融券餘額5日變化%
            'chip_score': int,            # 籌碼面綜合評分 (-5 to +5)
            'chip_signals': list,         # 籌碼面信號列表
        }
    """
    result = {
        'foreign_net_5d': 0, 'foreign_net_20d': 0, 'foreign_streak': 0,
        'trust_net_5d': 0, 'trust_net_20d': 0, 'trust_streak': 0,
        'dealer_net_5d': 0, 'total_inst_net_20d': 0,
        'margin_change_5d': 0, 'margin_change_20d': 0,
        'short_change_5d': 0,
        'chip_score': 0, 'chip_signals': [],
    }

    ticker_str = str(ticker)
    score = 0
    signals = []

    # ── 外資 ──
    fb = chip_data.get('foreign_buy')
    if fb is not None and ticker_str in fb.columns:
        s = fb[ticker_str].dropna()
        if len(s) >= lookback:
            result['foreign_net_5d'] = float(s.tail(5).sum()) / 10000  # 萬股
            result['foreign_net_20d'] = float(s.tail(20).sum()) / 10000

            # 計算連續買超天數
            streak = 0
            for v in reversed(s.values):
                if streak >= 0 and v > 0:
                    streak += 1
                elif streak <= 0 and v < 0:
                    streak -= 1
                else:
                    break
            result['foreign_streak'] = streak

            if result['foreign_net_20d'] > 0:
                score += 1
                if streak >= 5:
                    score += 1
                    signals.append(f"外資連續買超{streak}天")
            elif result['foreign_net_20d'] < 0:
                score -= 1
                if streak <= -5:
                    score -= 1
                    signals.append(f"外資連續賣超{abs(streak)}天")

    # ── 投信 ──
    tb = chip_data.get('trust_buy')
    if tb is not None and ticker_str in tb.columns:
        s = tb[ticker_str].dropna()
        if len(s) >= lookback:
            result['trust_net_5d'] = float(s.tail(5).sum()) / 10000
            result['trust_net_20d'] = float(s.tail(20).sum()) / 10000

            streak = 0
            for v in reversed(s.values):
                if streak >= 0 and v > 0:
                    streak += 1
                elif streak <= 0 and v < 0:
                    streak -= 1
                else:
                    break
            result['trust_streak'] = streak

            if result['trust_net_20d'] > 0:
                score += 1
                if streak >= 3:
                    signals.append(f"投信連續買超{streak}天")
            elif result['trust_net_20d'] < 0:
                score -= 1

    # ── 自營商 ──
    db = chip_data.get('dealer_buy')
    if db is not None and ticker_str in db.columns:
        s = db[ticker_str].dropna()
        if len(s) >= 5:
            result['dealer_net_5d'] = float(s.tail(5).sum()) / 10000

    # ── 三大法人合計 ──
    result['total_inst_net_20d'] = (result['foreign_net_20d'] +
                                     result['trust_net_20d'] +
                                     result['dealer_net_5d'] * 4)  # 估算20日

    # ── 融資 ──
    mb = chip_data.get('margin_balance')
    if mb is not None and ticker_str in mb.columns:
        s = mb[ticker_str].dropna()
        if len(s) >= lookback:
            current = float(s.iloc[-1])
            if current > 0:
                d5_ago = float(s.iloc[-6]) if len(s) >= 6 else current
                d20_ago = float(s.iloc[-21]) if len(s) >= 21 else current
                result['margin_change_5d'] = (current - d5_ago) / d5_ago * 100 if d5_ago > 0 else 0
                result['margin_change_20d'] = (current - d20_ago) / d20_ago * 100 if d20_ago > 0 else 0

                if result['margin_change_20d'] > 20:
                    score -= 1  # 融資暴增 = 散戶過熱 = 偏空
                    signals.append(f"融資20日增{result['margin_change_20d']:.1f}%")
                elif result['margin_change_20d'] < -15:
                    score += 1  # 融資大減 = 籌碼沉澱 = 偏多
                    signals.append(f"融資20日減{abs(result['margin_change_20d']):.1f}%")

    # ── 融券 ──
    sb = chip_data.get('short_balance')
    if sb is not None and ticker_str in sb.columns:
        s = sb[ticker_str].dropna()
        if len(s) >= 6:
            current = float(s.iloc[-1])
            d5_ago = float(s.iloc[-6]) if len(s) >= 6 else current
            result['short_change_5d'] = (current - d5_ago) / d5_ago * 100 if d5_ago > 0 else 0

            if result['short_change_5d'] > 30:
                signals.append(f"融券5日增{result['short_change_5d']:.1f}% (潛在軋空)")

    result['chip_score'] = max(-5, min(5, score))
    result['chip_signals'] = signals
    return result


def chip_score_color(score):
    """根據籌碼評分回傳顏色。"""
    if score >= 3:
        return '#ef4444'   # 強多 (紅)
    if score >= 1:
        return '#f59e0b'   # 偏多 (黃)
    if score <= -3:
        return '#22c55e'   # 強空 (綠)
    if score <= -1:
        return '#3b82f6'   # 偏空 (藍)
    return '#64748b'       # 中性 (灰)
