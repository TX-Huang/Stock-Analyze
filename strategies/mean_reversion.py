from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab
import logging

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)

# ── 策略 Metadata ──
STRATEGY_NAME = "均值回歸策略"
STRATEGY_DESCRIPTION = "RSI 超賣 + 布林帶下軌 + 偏離均線 → 反彈回歸"

# ── PARAM_SCHEMA — UI 動態表單定義 ──
PARAM_SCHEMA = {
    'rsi_oversold': {
        'label': 'RSI 超賣門檻',
        'type': 'int',
        'min': 15,
        'max': 45,
        'default': 30,
        'help': 'RSI 低於此值觸發超賣信號',
    },
    'bb_std': {
        'label': '布林帶標準差',
        'type': 'float',
        'min': 1.0,
        'max': 3.0,
        'default': 2.0,
        'step': 0.1,
        'help': '布林帶寬度 (幾個標準差)',
    },
    'deviation_threshold': {
        'label': '均線偏離門檻',
        'type': 'float',
        'min': -0.20,
        'max': -0.02,
        'default': -0.08,
        'step': 0.01,
        'help': '價格偏離均線的比例門檻 (負數)',
    },
    'stop_loss': {
        'label': '固定停損',
        'type': 'float',
        'min': 0.03,
        'max': 0.20,
        'default': 0.08,
        'step': 0.01,
        'help': '虧損超過此比例自動賣出',
    },
    'mode': {
        'label': '策略模式',
        'type': 'select',
        'options': ['classic', 'connors', 'momentum_pullback'],
        'default': 'classic',
        'help': 'classic=經典反彈, connors=RSI2短線, momentum_pullback=動能回調',
    },
}


def run_mean_reversion_strategy(api_token, params=None):
    """
    均值回歸策略 — 買入超跌股票，等待回歸均線。

    核心邏輯：
    1. Universe: 全市場 TSE_OTC + 200日上市 + 流動性 + 價格門檻
    2. RSI 超賣: RSI(14) < 30
    3. 布林帶下軌: close < BB_lower (20日, 2σ)
    4. 偏離均線: (close - MA60) / MA60 < -10%
    5. 量能確認: vol > vol_avg_20 × 1.2 (恐慌量)
    6. 紅K確認: close > open
    7. Exit: close > MA20 或 RSI > 70 或停損 8%

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
        'rsi_period': 14,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_period': 20,
        'bb_std': 2.0,
        'ma_mean': 60,
        'deviation_threshold': -0.08,
        'exit_ma': 10,
        'stop_loss': 0.08,
        'trail_stop': None,
        'take_profit': None,
        'uptrend_filter': True,
        'require_bullish_k': True,
        'mode': 'classic',
        'max_positions': 10,
        'min_price': 15,
        'liquidity_threshold': 500_000,
        'vol_multiplier': 1.2,
        'start_date': '2008-01-01',
        'end_date': '2026-12-31',
    }
    if params:
        p.update(params)

    if api_token:
        finlab.login(api_token)

    # =========================================================
    # 1. 取得數據
    # =========================================================
    logger.info("均值回歸: 開始載入資料...")

    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    open_ = sanitize_dataframe(data.get('price:開盤價'), "FinLab_Open")
    high = sanitize_dataframe(data.get('price:最高價'), "FinLab_High")
    low = sanitize_dataframe(data.get('price:最低價'), "FinLab_Low")
    vol = sanitize_dataframe(data.get('price:成交股數'), "FinLab_Vol")

    # =========================================================
    # 2. 對齊欄位與時間區間
    # =========================================================
    for df in [close, open_, high, low, vol]:
        df.columns = df.columns.astype(str)

    common_cols = (close.columns
                   .intersection(open_.columns)
                   .intersection(high.columns)
                   .intersection(low.columns)
                   .intersection(vol.columns))

    close = close[common_cols]
    open_ = open_[common_cols]
    high = high[common_cols]
    low = low[common_cols]
    vol = vol[common_cols]

    s, e = p['start_date'], p['end_date']
    close = close.loc[s:e]
    open_ = open_.loc[s:e]
    high = high.loc[s:e]
    low = low.loc[s:e]
    vol = vol.loc[s:e]

    logger.info(f"均值回歸: 資料載入完成 — {len(close)} 交易日, {len(common_cols)} 檔股票")

    # =========================================================
    # 3. 技術指標
    # =========================================================
    # MA
    ma_mean = close.rolling(p['ma_mean'], min_periods=p['ma_mean']).mean()
    ma_exit = close.rolling(p['exit_ma'], min_periods=p['exit_ma']).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    # Volume average
    vol_avg = vol.rolling(20, min_periods=20).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(p['rsi_period'], min_periods=p['rsi_period']).mean()
    avg_loss = loss.rolling(p['rsi_period'], min_periods=p['rsi_period']).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_ma = close.rolling(p['bb_period'], min_periods=p['bb_period']).mean()
    bb_std = close.rolling(p['bb_period'], min_periods=p['bb_period']).std()
    bb_lower = bb_ma - p['bb_std'] * bb_std

    # Deviation from MA
    deviation = (close - ma_mean) / ma_mean

    # MA50 for trend filter
    ma50 = close.rolling(50, min_periods=50).mean()

    # =========================================================
    # 4. Universe Filter
    # =========================================================
    listed_200 = ma200.notna()
    liquid = vol_avg >= p['liquidity_threshold']
    price_ok = close >= p['min_price']
    universe_mask = listed_200 & liquid & price_ok

    # 趨勢保護: 只在上升趨勢中買超跌 (避免接住落刀)
    if p.get('uptrend_filter', True):
        uptrend = (ma50 > ma200) & ma50.notna() & ma200.notna()
        universe_mask = universe_mask & uptrend

    # =========================================================
    # 5. Entry Signal
    # =========================================================
    mode = p.get('mode', 'classic')

    if mode == 'connors':
        # Connors RSI(2) 模式: 極短期超賣反彈
        # 條件: close > MA200 (長期上升趨勢)
        #        + RSI(2) < 10 (極度超賣)
        #        + close 連跌 2 天 (確認是回調)
        # 出場: close > MA5 (短期回歸)
        consecutive_down = ((close < close.shift(1)) &
                           (close.shift(1) < close.shift(2))).fillna(False)
        rsi_extreme = (rsi < p['rsi_oversold']).fillna(False)

        signal = (universe_mask & rsi_extreme & consecutive_down).fillna(False)

    elif mode == 'momentum_pullback':
        ma20 = close.rolling(20, min_periods=20).mean()
        was_above_ma20 = (close > ma20).rolling(5, min_periods=1).max().fillna(0) > 0
        pullback = (close < ma20).fillna(False)
        rsi_moderate = (rsi < p['rsi_oversold']).fillna(False)
        bullish_k = (close > open_).fillna(False)
        vol_confirm = (vol > vol_avg * 1.0).fillna(False)

        signal = (universe_mask & pullback & was_above_ma20
                  & rsi_moderate & bullish_k & vol_confirm).fillna(False)
    else:
        # 經典均值回歸模式
        rsi_oversold = (rsi < p['rsi_oversold']).fillna(False)
        below_bb = (close < bb_lower).fillna(False)
        deviated = (deviation < p['deviation_threshold']).fillna(False)
        vol_panic = (vol > vol_avg * p['vol_multiplier']).fillna(False)
        bullish_k = (close > open_).fillna(False)

        if p.get('require_bullish_k', True):
            signal = (universe_mask & rsi_oversold & below_bb & deviated
                      & vol_panic & bullish_k).fillna(False)
        else:
            signal = (universe_mask & rsi_oversold & below_bb & deviated
                      & vol_panic).fillna(False)

    # T日信號 → T+1 開盤進場
    entries = signal.shift(1).fillna(False)

    # =========================================================
    # 6. Exit Signal
    # =========================================================
    above_exit_ma = (close > ma_exit).fillna(False)
    rsi_overbought = (rsi > p['rsi_overbought']).fillna(False)

    exit_signal = (above_exit_ma | rsi_overbought).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    # =========================================================
    # 7. Ranking — 偏離幅度最大的優先
    # =========================================================
    score = deviation.abs().replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=0.5)

    position = entries.hold_until(
        exits,
        nstocks_limit=p['max_positions'],
        rank=score
    )

    # =========================================================
    # 8. Debug Logging
    # =========================================================
    latest = close.index[-1]
    debug_info = {
        'date': latest.strftime('%Y-%m-%d'),
        'universe': int(universe_mask.iloc[-1].sum()),
        'signal': int(signal.iloc[-1].sum()),
        'entries': int(entries.iloc[-1].sum()),
    }
    logger.info(f"均值回歸 Pipeline: {debug_info}")

    signal_stocks = signal.iloc[-1]
    signal_list = signal_stocks[signal_stocks].index.tolist()[:20]
    if signal_list:
        logger.info(f"均值回歸 最新信號: {signal_list}")

    # =========================================================
    # 9. Backtest
    # =========================================================
    sim_kwargs = {
        'name': '均值回歸策略',
        'upload': False,
        'stop_loss': p['stop_loss'],
        'position_limit': 1.0 / p['max_positions'],
        'trade_at_price': 'open',
    }
    if p.get('trail_stop'):
        sim_kwargs['trail_stop'] = p['trail_stop']
    if p.get('take_profit'):
        sim_kwargs['take_profit'] = p['take_profit']

    logger.info(f"均值回歸: 開始回測, sim_kwargs={sim_kwargs}")

    try:
        report = safe_finlab_sim(position, **sim_kwargs)
        logger.info("均值回歸: 回測完成")
        return report
    except Exception as e:
        logger.error(f"均值回歸 回測失敗: {type(e).__name__}: {e}", exc_info=True)
        raise


# ==========================================
# 平台標準入口函式
# ==========================================
def run_strategy(api_token, **kwargs):
    """平台標準入口 — 供回測系統自動呼叫。"""
    return run_mean_reversion_strategy(api_token, params=kwargs if kwargs else None)
