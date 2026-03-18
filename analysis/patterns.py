import numpy as np
import pandas as pd


def detect_candlestick_patterns(df):
    patterns = []
    if len(df) < 20:
        return patterns

    df['Range'] = df['High'] - df['Low']
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgRange'] = df['Range'].rolling(14).mean()
    df['AvgBody'] = df['Body'].rolling(14).mean()

    df['LocalHigh'] = df['High'].rolling(10).max()
    df['LocalLow'] = df['Low'].rolling(10).min()
    df['MA5'] = df['Close'].rolling(5).mean()

    df['Position'] = (df['Close'] - df['LocalLow']) / (df['LocalHigh'] - df['LocalLow'] + 1e-9)

    scan_df = df.tail(60)
    if len(scan_df) < 3:
        return patterns

    for i in range(2, len(scan_df)):
        idx0 = scan_df.index[i-2]
        idx1 = scan_df.index[i-1]
        idx2 = scan_df.index[i]

        row0 = scan_df.loc[idx0]
        row1 = scan_df.loc[idx1]
        row2 = scan_df.loc[idx2]

        if row2['Range'] < row2['AvgRange'] * 0.2:
            continue

        found = False

        is_downtrend = row1['Position'] <= 0.25
        is_uptrend = row1['Position'] >= 0.75

        def is_bullish(row):
            return row['Close'] > row['Open']

        def is_bearish(row):
            return row['Close'] < row['Open']

        def upper_shadow(row):
            return row['High'] - max(row['Open'], row['Close'])

        def lower_shadow(row):
            return min(row['Open'], row['Close']) - row['Low']

        # Priority 1: Multi-Candle
        if is_bearish(row0) and row0['Body'] > row0['AvgBody'] and \
           is_bullish(row2) and row2['Body'] > row2['AvgBody'] and \
           row2['Close'] > (row0['Open'] + row0['Close']) / 2 and \
           row1['Body'] <= row1['AvgBody'] * 0.5 and \
           max(row1['Open'], row1['Close']) < row0['Close'] and is_downtrend:
            patterns.append({"name": "晨星 (Morning Star)", "date": idx2, "type": "Bullish", "points": [idx0, idx1, idx2]})
            found = True

        elif is_bullish(row0) and row0['Body'] > row0['AvgBody'] and \
             is_bearish(row2) and row2['Body'] > row2['AvgBody'] and \
             row2['Close'] < (row0['Open'] + row0['Close']) / 2 and \
             row1['Body'] <= row1['AvgBody'] * 0.5 and \
             min(row1['Open'], row1['Close']) > row0['Close'] and is_uptrend:
            patterns.append({"name": "暮星 (Evening Star)", "date": idx2, "type": "Bearish", "points": [idx0, idx1, idx2]})
            found = True

        # Priority 2: Two-Candle
        if not found:
            if is_bearish(row1) and is_bullish(row2) and \
               row2['Open'] < row1['Close'] and row2['Close'] > row1['Open'] and is_downtrend:
                patterns.append({"name": "多頭吞噬", "date": idx2, "type": "Bullish", "points": [idx1, idx2]})
                found = True

            elif is_bullish(row1) and is_bearish(row2) and \
                 row2['Open'] > row1['Close'] and row2['Close'] < row1['Open'] and is_uptrend:
                patterns.append({"name": "空頭吞噬", "date": idx2, "type": "Bearish", "points": [idx1, idx2]})
                found = True

            elif is_bullish(row1) and row1['Body'] > row1['AvgBody'] and \
                 is_bearish(row2) and row2['Body'] > row2['AvgBody'] and \
                 row2['Open'] > row1['High'] and \
                 row2['Close'] < (row1['Open'] + row1['Close']) / 2 and \
                 row2['Close'] > row1['Open'] and is_uptrend:
                patterns.append({"name": "烏雲罩頂", "date": idx2, "type": "Bearish", "points": [idx1, idx2]})
                found = True

            elif is_bearish(row1) and row1['Body'] > row1['AvgBody'] and \
                 is_bullish(row2) and row2['Body'] > row2['AvgBody'] and \
                 row2['Open'] < row1['Low'] and \
                 row2['Close'] > (row1['Open'] + row1['Close']) / 2 and \
                 row2['Close'] < row1['Open'] and is_downtrend:
                patterns.append({"name": "貫穿線", "date": idx2, "type": "Bullish", "points": [idx1, idx2]})
                found = True

        # Priority 3: Single-Candle
        if not found:
            is_marubozu = row2['Body'] >= row2['AvgBody'] * 1.5 and upper_shadow(row2) < row2['AvgBody'] * 0.1 and lower_shadow(row2) < row2['AvgBody'] * 0.1
            is_doji = row2['Body'] <= row2['AvgBody'] * 0.1

            if is_marubozu and (is_downtrend or is_uptrend):
                if is_bullish(row2):
                    patterns.append({"name": "大長紅", "date": idx2, "type": "Bullish", "points": [idx2]})
                else:
                    patterns.append({"name": "大長黑", "date": idx2, "type": "Bearish", "points": [idx2]})
            elif is_doji and (is_downtrend or is_uptrend):
                patterns.append({"name": "十字星", "date": idx2, "type": "Neutral", "points": [idx2]})
            else:
                if lower_shadow(row2) >= 2 * row2['Body'] and upper_shadow(row2) <= row2['Body'] * 0.2:
                    if is_downtrend:
                        patterns.append({"name": "錘子", "date": idx2, "type": "Bullish", "points": [idx2]})
                    elif is_uptrend:
                        patterns.append({"name": "吊人", "date": idx2, "type": "Bearish", "points": [idx2]})
                elif upper_shadow(row2) >= 2 * row2['Body'] and lower_shadow(row2) <= row2['Body'] * 0.2:
                    if is_downtrend:
                        patterns.append({"name": "倒錘", "date": idx2, "type": "Bullish", "points": [idx2]})
                    elif is_uptrend:
                        if row2['Close'] >= df.loc[idx2, 'MA5']:
                            patterns.append({"name": "倒錘", "date": idx2, "type": "Bullish", "points": [idx2]})
                        else:
                            patterns.append({"name": "流星", "date": idx2, "type": "Bearish", "points": [idx2]})

    return patterns


def detect_complex_patterns(df, peaks, troughs):
    patterns = []
    if df.empty or len(peaks) < 2:
        return patterns

    if len(peaks) >= 3 and len(troughs) >= 3:
        p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
        t1, t2, t3 = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
        if (p2 > p1 and p2 > p3) and (t2 < t1 and t2 < t3):
            patterns.append({"name": "鑽石頂", "points": [peaks.index[-3], peaks.index[-1]], "type": "Bearish", "is_broadening": False})

    if len(peaks) >= 2 and len(troughs) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]
        t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]
        if p2 > p1 and t2 < t1:
            patterns.append({
                "name": "擴散", "points": [p1_idx, p2_idx], "type": "Volatility", "is_broadening": True,
                "p_coords": [(p1_idx, p1), (p2_idx, p2)], "t_coords": [(t1_idx, t1), (t2_idx, t2)]
            })

    if len(peaks) >= 3:
        p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
        p1_idx, p2_idx, p3_idx = peaks.index[-3], peaks.index[-2], peaks.index[-1]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.15:
            patterns.append({"name": "頭肩頂", "points": [p1_idx, p2_idx, p3_idx], "type": "Bearish", "is_broadening": False})

    if len(peaks) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        p1_idx, p2_idx = peaks.index[-2], peaks.index[-1]
        if abs(p1 - p2) / p1 < 0.03 and (p2_idx - p1_idx).days > 10:
            patterns.append({"name": "M頭", "points": [p1_idx, p2_idx], "type": "Bearish", "is_broadening": False})

    if len(troughs) >= 2:
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        t1_idx, t2_idx = troughs.index[-2], troughs.index[-1]
        if abs(t1 - t2) / t1 < 0.03 and (t2_idx - t1_idx).days > 10:
            patterns.append({"name": "W底", "points": [t1_idx, t2_idx], "type": "Bullish", "is_broadening": False})

    return patterns
