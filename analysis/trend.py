import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress


def calculate_structural_lines(df, lookback=100):
    """
    計算長期結構線：
    1. 線性回歸通道 (Linear Regression Channel)
    2. 主要支撐/壓力位 (Major S/R Levels)
    """
    structure = {"channel": None, "levels": []}
    if len(df) < 30:
        return structure

    data = df.tail(lookback).copy()
    if data.empty:
        return structure

    # 1. 通道計算 (High+Low)/2
    data['Mid'] = (data['High'] + data['Low']) / 2
    x = np.arange(len(data))

    try:
        slope, intercept, r_value, p_value, std_err = linregress(x, data['Mid'])

        line = slope * x + intercept
        residuals = data['Mid'] - line
        std_resid = np.std(residuals)

        upper_line = line + 2 * std_resid
        lower_line = line - 2 * std_resid

        structure["channel"] = {
            "slope": slope,
            "intercept": intercept,
            "std": std_resid,
            "x_start": data.index[0],
            "x_end": data.index[-1],
            "y_start_mid": line[0],
            "y_end_mid": line[-1],
            "y_start_upper": upper_line[0],
            "y_end_upper": upper_line[-1],
            "y_start_lower": lower_line[0],
            "y_end_lower": lower_line[-1]
        }
    except Exception:
        pass

    # 2. 主要支撐壓力 (Pivot Clustering)
    n = 20
    data['peaks'] = data.iloc[argrelextrema(data.High.values, np.greater_equal, order=n)[0]]['High']
    data['troughs'] = data.iloc[argrelextrema(data.Low.values, np.less_equal, order=n)[0]]['Low']

    pivots = pd.concat([data['peaks'].dropna(), data['troughs'].dropna()])
    if not pivots.empty:
        pivots = pivots.sort_values()
        clusters = []
        if len(pivots) > 0:
            current_cluster = [pivots.iloc[0]]
            for p in pivots.iloc[1:]:
                if (p - current_cluster[-1]) / current_cluster[-1] < 0.02:
                    current_cluster.append(p)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [p]
            clusters.append(current_cluster)

        for c in clusters:
            if len(c) >= 2:
                avg_price = np.mean(c)
                strength = len(c)
                structure["levels"].append({"price": avg_price, "strength": strength})

    return structure


def calculate_pattern_convergence(df, peaks, troughs):
    """
    計算型態收斂與結構轉折區間 (2/3 ~ 3/4)
    Apex Convergence & Reversal Zone
    """
    p_idx = peaks.dropna().index
    t_idx = troughs.dropna().index

    if len(p_idx) < 2 or len(t_idx) < 2:
        return None

    p1_idx, p2_idx = p_idx[-2], p_idx[-1]
    t1_idx, t2_idx = t_idx[-2], t_idx[-1]

    p1_val, p2_val = peaks[p1_idx], peaks[p2_idx]
    t1_val, t2_val = troughs[t1_idx], troughs[t2_idx]

    x_p1, x_p2 = df.index.get_loc(p1_idx), df.index.get_loc(p2_idx)
    x_t1, x_t2 = df.index.get_loc(t1_idx), df.index.get_loc(t2_idx)

    if x_p2 == x_p1 or x_t2 == x_t1:
        return None

    m_p = (p2_val - p1_val) / (x_p2 - x_p1)
    c_p = p1_val - m_p * x_p1

    m_t = (t2_val - t1_val) / (x_t2 - x_t1)
    c_t = t1_val - m_t * x_t1

    if abs(m_p - m_t) < 1e-4:
        return None

    x_int = (c_t - c_p) / (m_p - m_t)

    x_start = min(x_p1, x_t1)
    length = x_int - x_start

    if length <= 10:
        return None

    x_zone_start = x_start + length * 0.66
    x_zone_end = x_start + length * 0.75

    return {
        "x_int": x_int,
        "y_int": m_p * x_int + c_p,
        "x_start": x_start,
        "x_zone_start": x_zone_start,
        "x_zone_end": x_zone_end,
        "m_p": m_p, "c_p": c_p,
        "m_t": m_t, "c_t": c_t
    }


def calculate_trend_logic(df, n=10, is_weekly=False):
    verdict = {"trend": "盤整/不明", "signal": "觀望", "color": "gray", "details": [], "is_box": False}
    if df.empty:
        return verdict

    df['peaks'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']
    df['troughs'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']
    peaks, troughs = df['peaks'].dropna(), df['troughs'].dropna()

    volatility = df['Close'].rolling(5).std() / df['Close']
    if volatility.iloc[-1] < 0.005:
        verdict["trend"] = "線圈狀態"
        verdict["color"] = "orange"
        verdict["details"].append("波動率極度壓縮")

    recent = df.tail(40)
    r_max, r_min = recent['High'].max(), recent['Low'].min()
    if (r_max - r_min) / r_min < 0.10:
        verdict["trend"] = "矩形整理"
        verdict["color"] = "blue"
        verdict["is_box"] = True
        if df['Close'].iloc[-1] > r_max * 1.01:
            verdict["signal"] = "箱型突破"
            verdict["color"] = "green"
        return verdict

    if len(peaks) >= 2 and len(troughs) >= 2:
        p_last, p_prev = peaks.iloc[-1], peaks.iloc[-2]
        t_last, t_prev = troughs.iloc[-1], troughs.iloc[-2]

        x_p1, x_p2 = df.index.get_loc(peaks.index[-2]), df.index.get_loc(peaks.index[-1])
        x_t1, x_t2 = df.index.get_loc(troughs.index[-2]), df.index.get_loc(troughs.index[-1])

        if x_p2 != x_p1 and x_t2 != x_t1:
            m_peak = (p_last - p_prev) / (x_p2 - x_p1)
            m_trough = (t_last - t_prev) / (x_t2 - x_t1)

            if p_last > p_prev and t_last > t_prev:
                if m_trough > m_peak * 1.2:
                    verdict["trend"] = "上升楔形"
                    verdict["color"] = "green"
                else:
                    verdict["trend"] = "多頭趨勢"
                    verdict["color"] = "red"
            elif p_last < p_prev and t_last < t_prev:
                if m_peak < m_trough * 1.2:
                    verdict["trend"] = "下降楔形"
                    verdict["color"] = "red"
                else:
                    verdict["trend"] = "空頭趨勢"
                    verdict["color"] = "green"
            elif p_last < p_prev and t_last > t_prev:
                verdict["trend"] = "收斂整理"
                verdict["color"] = "orange"
            elif p_last > p_prev and t_last < t_prev:
                verdict["trend"] = "擴散型態"
                verdict["color"] = "orange"

    return verdict
