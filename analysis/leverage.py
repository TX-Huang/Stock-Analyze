"""
槓桿ETF評估模組 (Leveraged ETF Evaluation Module)

提供波動率衰減計算、HV 政權分類、進場信號評分等功能，
幫助投資人量化評估槓桿 ETF 的進場時機與風險。
"""
import pandas as pd
import numpy as np
from math import sqrt


# ==========================================
# 常用槓桿 ETF 預設清單
# ==========================================
LEVERAGED_ETF_PRESETS = {
    "00631L": {"name": "元大台灣50正2", "underlying": "0050", "leverage": 2.0, "market": "TW"},
    "00632R": {"name": "元大台灣50反1", "underlying": "0050", "leverage": -1.0, "market": "TW"},
    "00663L": {"name": "國泰臺灣加權正2", "underlying": "^TWII", "leverage": 2.0, "market": "TW"},
    "00685L": {"name": "群益臺灣加權正2", "underlying": "^TWII", "leverage": 2.0, "market": "TW"},
    "TQQQ":   {"name": "ProShares UltraPro QQQ", "underlying": "QQQ", "leverage": 3.0, "market": "US"},
    "SOXL":   {"name": "Direxion Semiconductor 3x", "underlying": "SOXX", "leverage": 3.0, "market": "US"},
    "UPRO":   {"name": "ProShares UltraPro S&P500", "underlying": "SPY", "leverage": 3.0, "market": "US"},
    "SSO":    {"name": "ProShares Ultra S&P500", "underlying": "SPY", "leverage": 2.0, "market": "US"},
}


# ==========================================
# 1. Historical Volatility (HV)
# ==========================================
def compute_hv(price_series, look_back_period=50):
    """
    使用 log return 的滾動標準差計算歷史波動率。
    獨立版本，不依賴 FinLab。
    """
    log_return = np.log(price_series / price_series.shift(1))
    return log_return.rolling(look_back_period).std()


# ==========================================
# 2. Volatility Decay (波動率衰減)
# ==========================================
def calculate_volatility_decay(underlying_returns, leverage, windows=None):
    """
    計算槓桿 ETF 在不同持有天數下的波動率衰減。

    原理: 比較「每日重新平衡的槓桿複利」與「標的簡單倍數」的差異。
      leveraged_compound = prod(1 + L * r_i) - 1
      theoretical_linear = L * (prod(1 + r_i) - 1)
      decay = leveraged_compound - theoretical_linear

    使用 log-sum 避免數值溢位。

    Returns: DataFrame with columns decay_Xd for each window.
    """
    if windows is None:
        windows = [7, 14, 30, 60]

    r = underlying_returns.dropna()
    result = pd.DataFrame(index=r.index)

    for w in windows:
        # 槓桿複利 (每日重新平衡)
        log_lev = np.log(1 + leverage * r).replace([np.inf, -np.inf], 0).fillna(0)
        cum_lev = log_lev.rolling(w).sum()
        lev_return = np.exp(cum_lev) - 1

        # 標的原始複利 × 槓桿倍數 (理論值，無衰減)
        log_und = np.log(1 + r).replace([np.inf, -np.inf], 0).fillna(0)
        cum_und = log_und.rolling(w).sum()
        und_return = leverage * (np.exp(cum_und) - 1)

        # 衰減 = 實際槓桿報酬 - 理論線性報酬
        result[f'decay_{w}d'] = lev_return - und_return

    return result


# ==========================================
# 3. HV Regime Classification (波動率政權分類)
# ==========================================
def classify_hv_regime(hv_series, lookback=120):
    """
    將 HV 分類為 Low / Normal / High 三種政權。

    Returns: DataFrame with columns:
      - hv: 原始 HV
      - hv_percentile: 滾動百分位 (0-100)
      - hv_regime: 'Low' (<P20), 'Normal' (P20-P80), 'High' (>P80)
    """
    df = pd.DataFrame(index=hv_series.index)
    df['hv'] = hv_series

    hv_p20 = hv_series.rolling(lookback).quantile(0.2)
    hv_p80 = hv_series.rolling(lookback).quantile(0.8)

    # 百分位排名
    df['hv_percentile'] = hv_series.rolling(lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # 政權分類
    conditions = [
        hv_series < hv_p20,
        hv_series > hv_p80,
    ]
    choices = ['Low', 'High']
    df['hv_regime'] = np.select(conditions, choices, default='Normal')

    df['hv_p20'] = hv_p20
    df['hv_p80'] = hv_p80

    return df


# ==========================================
# 4. Daily Decay Cost (每日衰減成本)
# ==========================================
def calculate_decay_cost_per_day(hv, leverage):
    """
    根據連續時間近似公式估算每日衰減成本 (basis points)。

    公式: E[leveraged return] = L*mu - 0.5*L*(L-1)*sigma^2
    其中 -0.5*L*(L-1)*sigma^2 就是波動率拖累。

    Parameters:
        hv: 年化歷史波動率
        leverage: 槓桿倍數
    Returns:
        daily_decay_bps: 每日衰減成本 (bps, 負值表示損失)
    """
    if hv is None or np.isnan(hv) or hv == 0:
        return 0.0
    daily_variance = (hv / sqrt(252)) ** 2
    daily_decay = -0.5 * leverage * (leverage - 1) * daily_variance
    return daily_decay * 10000  # 轉換為 bps


# ==========================================
# 5. Breakeven Move (損平所需漲幅)
# ==========================================
def calculate_breakeven_move(accumulated_decay_pct, leverage):
    """
    計算需要多少標的方向性漲幅才能抵銷累積的衰減。

    Parameters:
        accumulated_decay_pct: 累積衰減百分比 (負值)
        leverage: 槓桿倍數
    Returns:
        breakeven_move_pct: 標的需要的漲幅百分比
    """
    if leverage == 0:
        return 0.0
    return abs(accumulated_decay_pct) / abs(leverage)


# ==========================================
# 6. Actual vs Theoretical Performance (實際 vs 理論績效)
# ==========================================
def compare_actual_vs_theoretical(etf_df, underlying_df, leverage):
    """
    比較實際 ETF 績效 vs 理論槓桿績效。

    Parameters:
        etf_df: ETF 的 OHLCV DataFrame
        underlying_df: 標的的 OHLCV DataFrame
        leverage: 槓桿倍數
    Returns:
        DataFrame with columns:
          - actual_etf: 實際 ETF 累積報酬
          - theoretical_leveraged: 理論每日重平衡累積報酬
          - underlying_linear: 標的報酬 × 槓桿 (線性)
          - tracking_error: 實際 - 理論 的差異
    """
    # 對齊日期
    etf_close = etf_df['Close']
    und_close = underlying_df['Close']

    combined = pd.DataFrame({
        'etf': etf_close,
        'underlying': und_close
    }).dropna()

    if combined.empty:
        return pd.DataFrame()

    # 實際 ETF 累積報酬
    combined['actual_etf'] = combined['etf'] / combined['etf'].iloc[0]

    # 標的每日報酬
    und_daily_ret = combined['underlying'].pct_change().fillna(0)

    # 理論每日重平衡槓桿累積報酬
    combined['theoretical_leveraged'] = (1 + leverage * und_daily_ret).cumprod()

    # 標的線性 × 槓桿
    combined['underlying_linear'] = 1 + leverage * (combined['underlying'] / combined['underlying'].iloc[0] - 1)

    # 追蹤誤差
    combined['tracking_error'] = combined['actual_etf'] - combined['theoretical_leveraged']

    return combined[['actual_etf', 'theoretical_leveraged', 'underlying_linear', 'tracking_error']]


# ==========================================
# 7. Decay Heatmap Data (衰減熱力圖)
# ==========================================
def generate_decay_heatmap_data(underlying_returns, leverage, windows=None, lookback_days=120):
    """
    產生衰減熱力圖的 2D 矩陣。

    Returns: DataFrame
      - Index: 日期 (最近 lookback_days 個交易日)
      - Columns: 持有天數
      - Values: 衰減百分比
    """
    if windows is None:
        windows = [7, 14, 21, 30, 45, 60]

    decay_df = calculate_volatility_decay(underlying_returns, leverage, windows)

    # 只取最近 lookback_days
    decay_df = decay_df.tail(lookback_days)

    # 重命名欄位為中文
    rename_map = {f'decay_{w}d': f'{w}天' for w in windows}
    decay_df = decay_df.rename(columns=rename_map)

    # 轉換為百分比
    decay_df = decay_df * 100

    return decay_df


# ==========================================
# 8. Entry Signal Score (進場信號評分)
# ==========================================
def calculate_entry_signal_score(df, candle_patterns, hv_regime, trend_verdict):
    """
    綜合評分系統 (0-100)，結合 K 線型態、成交量、HV 政權、趨勢方向。

    Parameters:
        df: OHLCV DataFrame
        candle_patterns: detect_candlestick_patterns() 的結果
        hv_regime: 'Low', 'Normal', 'High'
        trend_verdict: calculate_trend_logic() 的結果
    Returns:
        dict: {score, recommendation, factors[]}
    """
    score = 0
    factors = []

    # === K 線型態評分 (0-40) ===
    pattern_score = 0
    bullish_patterns = [p for p in candle_patterns if p.get('type') == 'Bullish']
    bearish_patterns = [p for p in candle_patterns if p.get('type') == 'Bearish']

    # 只看最近 5 根 K 線內的型態
    if df is not None and len(df) > 0:
        recent_idx = df.index[-5:] if len(df) >= 5 else df.index
        recent_bullish = [p for p in bullish_patterns
                         if p.get('date') is not None and p['date'] in recent_idx]
        recent_bearish = [p for p in bearish_patterns
                         if p.get('date') is not None and p['date'] in recent_idx]
    else:
        recent_bullish = []
        recent_bearish = []

    high_value_patterns = {'晨星', '多頭吞噬', '貫穿線'}
    medium_value_patterns = {'錘子', '倒錘'}

    for p in recent_bullish:
        name = p.get('name', '')
        if any(hv in name for hv in high_value_patterns):
            pattern_score = max(pattern_score, 40)
            factors.append(f"強力反轉型態: {name}")
        elif any(mv in name for mv in medium_value_patterns):
            pattern_score = max(pattern_score, 30)
            factors.append(f"反轉型態: {name}")
        else:
            pattern_score = max(pattern_score, 20)
            factors.append(f"多頭型態: {name}")

    if recent_bearish:
        pattern_score = max(0, pattern_score - 20)
        factors.append(f"注意: 偵測到 {len(recent_bearish)} 個空頭型態")

    score += pattern_score

    # === 成交量評分 (0-20) ===
    vol_score = 0
    if df is not None and 'Volume' in df.columns and len(df) >= 20:
        recent_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].iloc[-20:].mean()
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            if vol_ratio > 1.5:
                vol_score = 20
                factors.append(f"成交量爆增 ({vol_ratio:.1f}x 均量)")
            elif vol_ratio > 1.2:
                vol_score = 10
                factors.append(f"成交量放大 ({vol_ratio:.1f}x 均量)")
            else:
                factors.append(f"成交量正常 ({vol_ratio:.1f}x 均量)")
    score += vol_score

    # === HV 政權評分 (0-20) ===
    hv_score = 0
    if hv_regime == 'Low':
        hv_score = 20
        factors.append("HV 低檔: 波動收縮，衰減成本低")
    elif hv_regime == 'Normal':
        hv_score = 10
        factors.append("HV 正常: 衰減成本中等")
    elif hv_regime == 'High':
        hv_score = -10
        factors.append("HV 高檔: 衰減成本高，不利槓桿持有")
    score += hv_score

    # === 趨勢評分 (0-20) ===
    trend_score = 0
    trend_str = trend_verdict.get('trend', '') if trend_verdict else ''

    if any(kw in trend_str for kw in ['多頭', '上升']):
        trend_score = 20
        factors.append(f"趨勢: {trend_str} (槓桿正向複利)")
    elif any(kw in trend_str for kw in ['整理', '收斂', '線圈', '矩形']):
        trend_score = 5
        factors.append(f"趨勢: {trend_str} (盤整消耗衰減)")
    elif any(kw in trend_str for kw in ['空頭', '下降']):
        trend_score = -10
        factors.append(f"趨勢: {trend_str} (槓桿加速虧損)")
    else:
        trend_score = 5
        factors.append(f"趨勢: {trend_str}")
    score += trend_score

    # 確保 score 在 0-100 範圍
    score = max(0, min(100, score))

    # 建議等級
    if score >= 70:
        recommendation = "Strong Buy"
        rec_zh = "強力買進"
    elif score >= 50:
        recommendation = "Buy"
        rec_zh = "可考慮買進"
    elif score >= 30:
        recommendation = "Neutral"
        rec_zh = "觀望"
    elif score >= 15:
        recommendation = "Avoid"
        rec_zh = "建議避開"
    else:
        recommendation = "Strong Avoid"
        rec_zh = "強烈不建議"

    return {
        'score': score,
        'recommendation': recommendation,
        'recommendation_zh': rec_zh,
        'factors': factors,
    }


# ==========================================
# 9. Optimal Leverage Calculator (最佳槓桿計算)
# ==========================================
def calculate_optimal_leverage(hv, expected_daily_return):
    """
    根據 Kelly Criterion 近似計算當前 HV 下的最佳槓桿倍數。

    公式: L* = mu / sigma^2 (連續時間 Kelly)

    Parameters:
        hv: 年化歷史波動率
        expected_daily_return: 預期日報酬率
    Returns:
        dict: {optimal_leverage, leverage_curve}
    """
    if hv is None or np.isnan(hv) or hv == 0:
        return {'optimal_leverage': 0, 'leverage_curve': []}

    daily_var = (hv / sqrt(252)) ** 2

    if daily_var == 0:
        return {'optimal_leverage': 0, 'leverage_curve': []}

    optimal_l = expected_daily_return / daily_var

    # 限制在合理範圍
    optimal_l = max(-5, min(5, optimal_l))

    # 生成不同槓桿倍數的預期報酬曲線
    leverage_range = np.arange(-3, 5.1, 0.5)
    curve = []
    for l_val in leverage_range:
        expected = l_val * expected_daily_return - 0.5 * l_val * (l_val - 1) * daily_var
        curve.append({'leverage': float(l_val), 'expected_daily_return': float(expected * 10000)})

    return {
        'optimal_leverage': round(float(optimal_l), 2),
        'leverage_curve': curve,
    }
