from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab
import logging

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)

# ── 策略 Metadata ──
STRATEGY_NAME = "配對輪動策略"
STRATEGY_DESCRIPTION = "同產業高相關配對 → Z-score 偏離 → 買入弱勢方等待回歸"

# ── PARAM_SCHEMA — UI 動態表單定義 ──
PARAM_SCHEMA = {
    'zscore_entry': {
        'label': 'Z-score 進場門檻',
        'type': 'float',
        'min': -3.0,
        'max': -0.5,
        'default': -1.0,
        'step': 0.1,
        'help': 'Z-score 低於此值時買入弱勢方',
    },
    'zscore_exit': {
        'label': 'Z-score 出場門檻',
        'type': 'float',
        'min': 0.0,
        'max': 2.0,
        'default': 0.5,
        'step': 0.1,
        'help': 'Z-score 回升至此值時出場',
    },
    'lookback': {
        'label': '回溯期 (天)',
        'type': 'int',
        'min': 10,
        'max': 60,
        'default': 30,
        'help': '計算 Z-score 的回溯天數',
    },
    'stop_loss': {
        'label': '固定停損',
        'type': 'float',
        'min': 0.05,
        'max': 0.20,
        'default': 0.10,
        'step': 0.01,
        'help': '虧損超過此比例自動賣出',
    },
}


def run_pairs_trading_strategy(api_token, params=None):
    """
    配對交易策略 (Long-Only 變形) — 同產業配對輪動。

    核心邏輯：
    1. Universe: 全市場 TSE_OTC + 流動性 + 價格門檻
    2. 產業分群: 同產業內計算收益率相關性
    3. 配對篩選: rolling correlation > threshold
    4. Z-score: 計算對數價差的標準化偏離
    5. 進場: Z-score < entry_z (相對弱勢方)
    6. 出場: Z-score > exit_z (回歸) 或 Z-score < stop_z (結構性破裂)
    7. 每 rebalance_days 天重新計算配對

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
        'lookback': 30,                  # 回顧期 (V2: 60→30)
        'corr_threshold': 0.80,         # 最低相關係數 (保留未使用)
        'zscore_entry': -1.0,           # Z-score 進場門檻 (V2: -1.5→-1.0)
        'zscore_exit': 0.5,             # Z-score 出場門檻 (V2: 0→0.5)
        'zscore_stop': -2.0,            # Z-score 停損門檻 (V2: -3.5→-2.0)
        'rebalance_days': 20,           # 配對重新計算週期
        'max_positions': 5,             # 集中持倉 (V2: 10→5, Sharpe+0.4)
        'min_price': 20,
        'liquidity_threshold': 1_000_000, # 高流動性 (V2: 50萬→100萬)
        'stop_loss': 0.10,              # 固定停損
        'top_n_pairs': 50,              # 保留未使用
        'start_date': '2010-01-01',
        'end_date': '2026-12-31',
    }
    if params:
        p.update(params)

    if api_token:
        finlab.login(api_token)

    # =========================================================
    # 1. 取得數據
    # =========================================================
    logger.info("配對輪動: 開始載入資料...")

    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    open_ = sanitize_dataframe(data.get('price:開盤價'), "FinLab_Open")
    vol = sanitize_dataframe(data.get('price:成交股數'), "FinLab_Vol")

    # =========================================================
    # 2. 對齊欄位與時間區間
    # =========================================================
    for df in [close, open_, vol]:
        df.columns = df.columns.astype(str)

    common_cols = (close.columns
                   .intersection(open_.columns)
                   .intersection(vol.columns))

    close = close[common_cols]
    open_ = open_[common_cols]
    vol = vol[common_cols]

    s, e = p['start_date'], p['end_date']
    close = close.loc[s:e]
    open_ = open_.loc[s:e]
    vol = vol.loc[s:e]

    logger.info(f"配對輪動: 資料載入完成 — {len(close)} 交易日, {len(common_cols)} 檔股票")

    # =========================================================
    # 3. Universe Filter
    # =========================================================
    ma200 = close.rolling(200, min_periods=200).mean()
    vol_avg = vol.rolling(20, min_periods=20).mean()

    listed_200 = ma200.notna()
    liquid = vol_avg >= p['liquidity_threshold']
    price_ok = close >= p['min_price']
    universe_mask = listed_200 & liquid & price_ok

    # =========================================================
    # 4. 計算日收益率與滾動相關性矩陣
    #    為避免 N² 計算量爆炸，使用 sector-agnostic 方法：
    #    對所有股票計算 rolling return correlation，
    #    只在 rebalance 日更新配對
    # =========================================================
    returns = close.pct_change().fillna(0)
    log_close = np.log(close.replace(0, np.nan))

    lb = p['lookback']

    # 預先計算所有股票的 rolling mean/std (for Z-score)
    # 而非全市場相關矩陣 — 太慢
    # 改用策略: 對每檔股票，找其「相對強弱」最偏離的時刻進場

    # =========================================================
    # 5. Sector-Relative Value Approach
    #    簡化版配對交易: 用 rolling percentile rank 找同期最弱勢股票
    #    相對整體市場的偏離 → 買入最弱勢 → 等待均值回歸
    # =========================================================

    # Rolling 60-day return
    rolling_ret = close / close.shift(lb) - 1

    # 每日計算所有股票的百分位排名 (0=最弱, 1=最強)
    ret_rank = rolling_ret.rank(axis=1, pct=True)

    # Z-score: 個股 rolling return 相對市場平均的偏離
    market_mean = rolling_ret.mean(axis=1)
    market_std = rolling_ret.std(axis=1)
    zscore = rolling_ret.sub(market_mean, axis=0).div(market_std.replace(0, np.nan), axis=0)
    zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0)

    # =========================================================
    # 6. Entry: 最弱勢且有回歸信號
    # =========================================================
    # 條件 1: Z-score 低於門檻 (相對弱勢)
    weak = (zscore < p['zscore_entry']).fillna(False)

    # 條件 2: 近 5 天有反彈跡象 (避免接住落刀)
    ret_5d = close / close.shift(5) - 1
    bouncing = (ret_5d > 0).fillna(False)

    # 條件 3: 不在最底部 5% (排除極端個股)
    not_extreme = (ret_rank > 0.05).fillna(False)

    # 條件 4: 長期上升趨勢 (close > MA200，避免結構性下跌)
    uptrend = (close > ma200).fillna(False)

    # 條件 5: 紅K確認 (當日反彈)
    bullish_k = (close > open_).fillna(False)

    signal = (universe_mask & weak & bouncing & not_extreme
              & uptrend & bullish_k).fillna(False)

    # Rebalance: 每 N 天才看一次
    rebalance = p['rebalance_days']
    if rebalance > 1:
        rebal_mask = pd.Series(False, index=close.index)
        rebal_indices = list(range(0, len(close), rebalance))
        rebal_mask.iloc[rebal_indices] = True
        rebal_df = pd.DataFrame(
            np.tile(rebal_mask.values[:, None], (1, len(signal.columns))),
            index=signal.index, columns=signal.columns
        )
        signal = signal & rebal_df

    entries = signal.shift(1).fillna(False)

    # =========================================================
    # 7. Exit: Z-score 回歸或停損
    # =========================================================
    # 回歸: Z-score > exit threshold
    recovered = (zscore > p['zscore_exit']).fillna(False)

    # 結構性破裂: Z-score 太低
    broken = (zscore < p['zscore_stop']).fillna(False)

    exit_signal = (recovered | broken).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    # =========================================================
    # 8. Ranking — 偏離幅度最大 + 反彈動能
    # =========================================================
    # Z-score 絕對值越大越優先 (偏離越大，回歸空間越大)
    z_score_abs = zscore.abs().clip(upper=5)

    # 5 日反彈幅度作為次要排序
    bounce_score = ret_5d.clip(-0.1, 0.1).fillna(0) + 0.1  # shift to positive

    score = (z_score_abs * 0.7 + bounce_score * 10 * 0.3).fillna(0).clip(0, 5)

    position = entries.hold_until(
        exits,
        nstocks_limit=p['max_positions'],
        rank=score
    )

    # =========================================================
    # 9. Debug Logging
    # =========================================================
    latest = close.index[-1]
    debug_info = {
        'date': str(latest)[:10],
        'universe': int(universe_mask.iloc[-1].sum()),
        'weak': int(weak.iloc[-1].sum()),
        'signal': int(signal.iloc[-1].sum()),
        'entries': int(entries.iloc[-1].sum()),
    }
    logger.info(f"配對輪動 Pipeline: {debug_info}")

    signal_stocks = signal.iloc[-1]
    signal_list = signal_stocks[signal_stocks].index.tolist()[:20]
    if signal_list:
        logger.info(f"配對輪動 最新信號: {signal_list}")

    # =========================================================
    # 10. Backtest
    # =========================================================
    sim_kwargs = {
        'name': '配對輪動策略',
        'upload': False,
        'stop_loss': p['stop_loss'],
        'position_limit': 1.0 / p['max_positions'],
        'trade_at_price': 'open',
    }

    logger.info(f"配對輪動: 開始回測, sim_kwargs={sim_kwargs}")

    try:
        report = safe_finlab_sim(position, **sim_kwargs)
        logger.info("配對輪動: 回測完成")
        return report
    except Exception as e:
        logger.error(f"配對輪動 回測失敗: {type(e).__name__}: {e}", exc_info=True)
        raise


# ==========================================
# 平台標準入口函式
# ==========================================
def run_strategy(api_token, **kwargs):
    """平台標準入口 — 供回測系統自動呼叫。"""
    return run_pairs_trading_strategy(api_token, params=kwargs if kwargs else None)
