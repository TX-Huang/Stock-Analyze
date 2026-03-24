from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab
import logging

logger = logging.getLogger(__name__)

# ==========================================
# Will's VCP 策略 (Weekly Trend + MA Alignment + Volume Breakout)
# 原始作者: Will (strategy_will_0320.py)
# 整合日期: 2026-03-24
# ==========================================


def run_will_vcp_strategy(api_token, params=None):
    """
    Will 的 VCP 突破策略 — 結合週線趨勢、均線排列、量能突破。

    核心邏輯：
    1. Universe: 台股電子七大產業 + 200日均線 + 流動性 + 價格門檻
    2. Weekly Trend: 週線 MA30 > MA40（前一完整週，避免資訊滲入）
    3. MA Alignment: close > MA50 > MA150 > MA200
    4. Volume Breakout: 量 > 60日均量 × 2 + 紅K
    5. Price Breakout: 收盤突破前20日最高 + 收盤接近當日高點
    6. Exit: 收盤跌破 MA50

    Args:
        api_token: FinLab API token
        params: 可選參數覆蓋 dict

    Returns:
        FinLab backtest report
    """
    from data.provider import sanitize_dataframe, safe_finlab_sim

    # =========================================================
    # 0. Parameters
    # =========================================================
    p = {
        'volume_multiplier': 2.0,     # 量能放大倍率
        'start_date': '2008-01-01',
        'end_date': '2026-12-31',
        'max_positions': 10,
        'trail_stop': 0.15,           # 追蹤停損 15%
        'weekly_ma_fast': 30,         # 週線快均
        'weekly_ma_slow': 40,         # 週線慢均
        'breakout_window': 20,        # 突破回顧天數
        'close_near_high_pct': 0.95,  # 收盤接近高點比例
        'vol_avg_window': 60,         # 量能均線天數
        'liquidity_threshold': 500_000,  # 流動性門檻（股）
        'min_price': 20,              # 最低股價
    }
    if params:
        p.update(params)

    if api_token:
        finlab.login(api_token)

    # =========================================================
    # 1. Universe Setting — 電子七大產業
    # =========================================================
    logger.info("Will VCP: 開始載入資料...")

    with data.universe(
        market='TSE_OTC',
        category=[
            '半導體', '電子零組件', '電腦及週邊設備',
            '通信網路業', '光電業', '其他電子業', '電機機械'
        ]
    ):
        close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
        high  = sanitize_dataframe(data.get('price:最高價'), "FinLab_High")
        low   = sanitize_dataframe(data.get('price:最低價'), "FinLab_Low")
        vol   = sanitize_dataframe(data.get('price:成交股數'), "FinLab_Vol")
        open_ = sanitize_dataframe(data.get('price:開盤價'), "FinLab_Open")

    # =========================================================
    # 2. Basic Cleanup — 對齊欄位與時間區間
    #    注意：不要直接覆寫 FinlabDataFrame 的 index，會破壞內部機制
    #    價量資料 index 已是 DatetimeIndex，只需確保 columns 為 str
    # =========================================================
    for df in [close, high, low, vol, open_]:
        df.columns = df.columns.astype(str)

    close, high, low, vol, open_ = [
        df.sort_index() for df in [close, high, low, vol, open_]
    ]

    # 取共同欄位
    common_cols = (close.columns
                   .intersection(high.columns)
                   .intersection(low.columns)
                   .intersection(vol.columns)
                   .intersection(open_.columns))

    close = close[common_cols]
    high  = high[common_cols]
    low   = low[common_cols]
    vol   = vol[common_cols]
    open_ = open_[common_cols]

    # 回測區間
    s, e = p['start_date'], p['end_date']
    close = close.loc[s:e]
    high  = high.loc[s:e]
    low   = low.loc[s:e]
    vol   = vol.loc[s:e]
    open_ = open_.loc[s:e]

    logger.info(f"Will VCP: 資料載入完成 — {len(close)} 交易日, {len(common_cols)} 檔股票")

    # =========================================================
    # 3. 技術指標
    # =========================================================
    ma20  = close.rolling(20, min_periods=20).mean()
    ma50  = close.rolling(50, min_periods=50).mean()
    ma150 = close.rolling(150, min_periods=150).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    vol_avg = vol.rolling(p['vol_avg_window'], min_periods=p['vol_avg_window']).mean()

    # =========================================================
    # 4. Universe Filter
    # =========================================================
    listed_200 = ma200.notna()
    liquid = vol_avg >= p['liquidity_threshold']
    price_ok = close >= p['min_price']
    universe_mask = listed_200 & liquid & price_ok

    # =========================================================
    # 5. Weekly Trend Filter
    #    使用上一個完整週，避免未完成週資訊提前滲入
    # =========================================================
    weekly_close = close.resample('W-FRI').last().ffill()
    weekly_ma_fast = weekly_close.rolling(p['weekly_ma_fast'], min_periods=p['weekly_ma_fast']).mean()
    weekly_ma_slow = weekly_close.rolling(p['weekly_ma_slow'], min_periods=p['weekly_ma_slow']).mean()

    weekly_valid = weekly_ma_fast.notna() & weekly_ma_slow.notna()
    weekly_trend = (weekly_ma_fast > weekly_ma_slow) & weekly_valid

    # shift(1) 避免前瞻偏差
    weekly_trend_daily = weekly_trend.shift(1).reindex(close.index).ffill().fillna(False)

    trend_mask = universe_mask & weekly_trend_daily

    # =========================================================
    # 6. Daily MA Alignment: close > MA50 > MA150 > MA200
    # =========================================================
    ma_alignment = (
        (close > ma50) & (ma50 > ma150) & (ma150 > ma200) &
        ma50.notna() & ma150.notna() & ma200.notna()
    )

    core_mask = trend_mask & ma_alignment

    # =========================================================
    # 7. Volume Expansion — 量 > N倍均量 + 紅K
    # =========================================================
    bullish_k = (close > open_).fillna(False)
    vol_big = ((vol > (vol_avg * p['volume_multiplier'])) & bullish_k).fillna(False)

    # =========================================================
    # 8. Price Breakout — 突破前 N 日最高 + 收盤接近當日高點
    # =========================================================
    bw = p['breakout_window']
    prev_high = high.rolling(bw, min_periods=bw).max().shift(1)
    breakout = (close > prev_high).fillna(False)
    close_near_high = (close >= (high * p['close_near_high_pct'])).fillna(False)

    # =========================================================
    # 9. Entry / Exit Signal
    #    T日信號 → T+1 開盤進場（shift(1) 避免前瞻偏差）
    # =========================================================
    signal = (core_mask & vol_big & breakout & close_near_high).fillna(False)
    entries = signal.shift(1).fillna(False)

    exit_signal = (close < ma50)
    exits = exit_signal.shift(1).fillna(False)

    # =========================================================
    # 9-1. Portfolio Ranking（強者恆強）
    # =========================================================
    vol_score = (vol / vol_avg).replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=8)
    break_score = (close / prev_high).replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=1.15)
    score = (vol_score * 0.6 + break_score * 0.4).fillna(0)

    position = entries.hold_until(
        exits,
        nstocks_limit=p['max_positions'],
        rank=score
    )

    # =========================================================
    # 10. Debug Logging
    # =========================================================
    latest = close.index[-1]
    debug_info = {
        'date': latest.strftime('%Y-%m-%d'),
        'universe': int(universe_mask.iloc[-1].sum()),
        'trend': int(trend_mask.iloc[-1].sum()),
        'core': int(core_mask.iloc[-1].sum()),
        'vol_big': int(vol_big.iloc[-1].sum()),
        'signal': int(signal.iloc[-1].sum()),
        'entries': int(entries.iloc[-1].sum()),
        'exits': int(exits.iloc[-1].sum()),
    }
    logger.info(f"Will VCP Pipeline: {debug_info}")

    # 印出最新信號股票
    signal_stocks = signal.iloc[-1]
    signal_list = signal_stocks[signal_stocks].index.tolist()[:20]
    if signal_list:
        logger.info(f"Will VCP 最新信號: {signal_list}")

    # =========================================================
    # 11. Backtest — 使用 safe_finlab_sim（CategoricalIndex 安全）
    # =========================================================
    sim_kwargs = {
        'name': 'Will VCP 突破策略',
        'upload': False,
        'trail_stop': p['trail_stop'],
        'position_limit': 1.0 / p['max_positions'],
        'trade_at_price': 'open',
    }

    logger.info(f"Will VCP: 開始回測, sim_kwargs={sim_kwargs}")

    try:
        report = safe_finlab_sim(position, **sim_kwargs)
        logger.info("Will VCP: 回測完成")
        return report
    except Exception as e:
        logger.error(f"Will VCP 回測失敗: {type(e).__name__}: {e}", exc_info=True)
        raise


# ==========================================
# 平台標準入口函式
# ==========================================
def run_strategy(api_token, **kwargs):
    """平台標準入口 — 供回測系統自動呼叫。"""
    return run_will_vcp_strategy(api_token, params=kwargs if kwargs else None)
