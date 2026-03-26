from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab
import logging

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)

# ── 策略 Metadata ──
STRATEGY_NAME = "價值股息策略"
STRATEGY_DESCRIPTION = "低 PE + 高殖利率 + 營收成長 + MA200 保護 → 價值型長期持有"

# ── PARAM_SCHEMA — UI 動態表單定義 ──
PARAM_SCHEMA = {
    'pe_max': {
        'label': 'PE 上限',
        'type': 'int',
        'min': 5,
        'max': 30,
        'default': 15,
        'help': '本益比上限，超過此值不買入',
    },
    'dividend_yield_min': {
        'label': '最低殖利率 (%)',
        'type': 'float',
        'min': 2.0,
        'max': 12.0,
        'default': 6.0,
        'step': 0.5,
        'help': '最低現金股息殖利率',
    },
    'stop_loss': {
        'label': '固定停損',
        'type': 'float',
        'min': 0.05,
        'max': 0.25,
        'default': 0.12,
        'step': 0.01,
        'help': '虧損超過此比例自動賣出',
    },
    'max_positions': {
        'label': '最大持股數',
        'type': 'int',
        'min': 5,
        'max': 30,
        'default': 15,
        'help': '同時持有的最大股票數',
    },
}


def run_value_dividend_strategy(api_token, params=None):
    """
    價值+股息策略 — 買入基本面被低估且有穩定股息的股票。

    核心邏輯：
    1. Universe: 全市場 TSE_OTC + 200日上市 + 流動性 + 價格門檻
    2. 低 PE: 3 < PE < 12
    3. 高殖利率: dividend_yield > 4%
    4. EPS 正值: 最近一季 EPS > 0
    5. 營收不衰退: YoY 營收成長 >= -5%
    6. 技術面保護: close > MA200
    7. Exit: close < MA200 或 PE > pe_exit_max
    8. 每 rebalance_days 天重新平衡

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
        'pe_max': 15,                    # PE 上限 (V2: 12→15, CAGR+5%)
        'pe_min': 3,
        'dividend_yield_min': 6.0,       # 殖利率 % (V2: 4→6, Sharpe+0.4)
        'eps_positive': True,            # 最近一季 EPS > 0
        'revenue_growth_min': -5.0,      # 營收 YoY >= -5%
        'op_margin_min': 0.0,            # 營業利益率 > 0%
        'ma_protect': 200,               # 技術面保護均線
        'pe_exit_max': 25,               # PE 超過此值出場 (V2: 20→25, CAGR+1%)
        'rebalance_days': 20,            # 重新平衡週期 (V2: 40→20, CAGR+3%)
        'max_positions': 15,
        'min_price': 15,
        'liquidity_threshold': 300_000,
        'stop_loss': 0.12,               # 固定停損 12%
        'start_date': '2010-01-01',      # 殖利率數據從 2010 開始
        'end_date': '2026-12-31',
    }
    if params:
        p.update(params)

    if api_token:
        finlab.login(api_token)

    # =========================================================
    # 1. 取得數據
    # =========================================================
    logger.info("價值股息: 開始載入資料...")

    close = sanitize_dataframe(data.get('price:收盤價'), "FinLab_Close")
    open_ = sanitize_dataframe(data.get('price:開盤價'), "FinLab_Open")
    vol = sanitize_dataframe(data.get('price:成交股數'), "FinLab_Vol")

    # 基本面數據 (日頻)
    pe = data.get('price_earning_ratio:本益比')
    dy = data.get('price_earning_ratio:殖利率(%)')
    pb = data.get('price_earning_ratio:股價淨值比')

    # 基本面數據 (季頻/月頻) — 需要 reindex 到日頻
    eps = data.get('financial_statement:每股盈餘')
    op_margin = data.get('fundamental_features:營業利益率')
    rev_growth = data.get('monthly_revenue:去年同月增減(%)')

    # =========================================================
    # 2. 對齊欄位與時間區間
    # =========================================================
    for df in [close, open_, vol]:
        df.columns = df.columns.astype(str)

    # 日頻基本面數據欄位也轉 str
    for df in [pe, dy, pb]:
        df.columns = df.columns.astype(str)

    # 季/月頻數據: 用 FinLab 的 deadline() 轉為日頻 (避免前視偏差)
    eps_daily = eps.deadline().reindex(close.index, method='ffill')
    eps_daily.columns = eps_daily.columns.astype(str)

    op_margin_daily = op_margin.deadline().reindex(close.index, method='ffill')
    op_margin_daily.columns = op_margin_daily.columns.astype(str)

    rev_daily = rev_growth.deadline().reindex(close.index, method='ffill')
    rev_daily.columns = rev_daily.columns.astype(str)

    # 找共同欄位
    common_cols = (close.columns
                   .intersection(open_.columns)
                   .intersection(vol.columns)
                   .intersection(pe.columns)
                   .intersection(dy.columns))

    close = close[common_cols]
    open_ = open_[common_cols]
    vol = vol[common_cols]
    pe = pe[common_cols]
    dy = dy[common_cols]
    pb = pb[pb.columns.intersection(common_cols)]

    # 季/月頻只取交集部分
    eps_daily = eps_daily[eps_daily.columns.intersection(common_cols)]
    op_margin_daily = op_margin_daily[op_margin_daily.columns.intersection(common_cols)]
    rev_daily = rev_daily[rev_daily.columns.intersection(common_cols)]

    # 回測區間
    s, e = p['start_date'], p['end_date']
    close = close.loc[s:e]
    open_ = open_.loc[s:e]
    vol = vol.loc[s:e]
    pe = pe.loc[s:e]
    dy = dy.loc[s:e]
    pb = pb.loc[s:e] if len(pb.columns) > 0 else pb
    eps_daily = eps_daily.loc[s:e]
    op_margin_daily = op_margin_daily.loc[s:e]
    rev_daily = rev_daily.loc[s:e]

    logger.info(f"價值股息: 資料載入完成 — {len(close)} 交易日, {len(common_cols)} 檔股票")

    # =========================================================
    # 3. 技術指標
    # =========================================================
    ma200 = close.rolling(200, min_periods=200).mean()
    vol_avg = vol.rolling(20, min_periods=20).mean()

    # =========================================================
    # 4. Universe Filter
    # =========================================================
    listed_200 = ma200.notna()
    liquid = vol_avg >= p['liquidity_threshold']
    price_ok = close >= p['min_price']
    universe_mask = listed_200 & liquid & price_ok

    # =========================================================
    # 5. Value + Dividend Conditions
    # =========================================================
    # 低 PE
    pe_valid = pe.notna() & (pe > p['pe_min']) & (pe < p['pe_max'])

    # 高殖利率
    dy_high = dy.notna() & (dy >= p['dividend_yield_min'])

    # EPS 正值
    if p.get('eps_positive', True):
        eps_ok = eps_daily.reindex(columns=common_cols).fillna(0) > 0
    else:
        eps_ok = pd.DataFrame(True, index=close.index, columns=common_cols)

    # 營收成長
    rev_ok = rev_daily.reindex(columns=common_cols).fillna(0) >= p['revenue_growth_min']

    # 營業利益率
    if p.get('op_margin_min', 0) > 0:
        opm_ok = op_margin_daily.reindex(columns=common_cols).fillna(0) > p['op_margin_min']
    else:
        opm_ok = op_margin_daily.reindex(columns=common_cols).fillna(0) >= 0

    # 技術面保護: close > MA200
    above_ma = (close > ma200).fillna(False)

    # =========================================================
    # 6. Entry Signal + Rebalance
    # =========================================================
    signal = (universe_mask & pe_valid & dy_high & eps_ok
              & rev_ok & opm_ok & above_ma).fillna(False)

    # 重新平衡: 每 N 天才看一次信號
    rebalance = p['rebalance_days']
    if rebalance > 1:
        rebal_mask = pd.Series(False, index=close.index)
        rebal_indices = list(range(0, len(close), rebalance))
        rebal_mask.iloc[rebal_indices] = True
        # 廣播到所有欄位
        rebal_df = pd.DataFrame(
            np.tile(rebal_mask.values[:, None], (1, len(signal.columns))),
            index=signal.index, columns=signal.columns
        )
        signal = signal & rebal_df

    # T日信號 → T+1 開盤進場
    entries = signal.shift(1).fillna(False)

    # =========================================================
    # 7. Exit Signal
    # =========================================================
    # 跌破 MA200
    below_ma = (close < ma200).fillna(False)

    # PE 過高 (估值修復完成)
    pe_too_high = (pe > p['pe_exit_max']).fillna(False)

    exit_signal = (below_ma | pe_too_high).fillna(False)
    exits = exit_signal.shift(1).fillna(False)

    # =========================================================
    # 8. Ranking — 複合估值分數
    # =========================================================
    # 殖利率越高越好 (正規化到 0-1)
    dy_score = dy.clip(0, 15) / 15.0

    # PE 越低越好 (反向正規化)
    pe_score = 1.0 - (pe.clip(p['pe_min'], p['pe_max']) - p['pe_min']) / (p['pe_max'] - p['pe_min'])

    # PB 越低越好
    if len(pb.columns) > 0:
        pb_score = (1.0 - pb.clip(0, 3) / 3.0).reindex(columns=common_cols, fill_value=0)
    else:
        pb_score = pd.DataFrame(0, index=close.index, columns=common_cols)

    # 營收成長越高越好
    rev_score = rev_daily.reindex(columns=common_cols).clip(-30, 50).fillna(0) / 50.0

    # 複合分數
    score = (dy_score.reindex(columns=common_cols).fillna(0) * 0.35
             + pe_score.reindex(columns=common_cols).fillna(0) * 0.30
             + pb_score.fillna(0) * 0.15
             + rev_score.fillna(0) * 0.20)

    score = score.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

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
        'pe_valid': int(pe_valid.iloc[-1].sum()),
        'dy_high': int(dy_high.iloc[-1].sum()),
        'signal': int(signal.iloc[-1].sum()),
        'entries': int(entries.iloc[-1].sum()),
    }
    logger.info(f"價值股息 Pipeline: {debug_info}")

    signal_stocks = signal.iloc[-1]
    signal_list = signal_stocks[signal_stocks].index.tolist()[:20]
    if signal_list:
        logger.info(f"價值股息 最新信號: {signal_list}")

    # =========================================================
    # 10. Backtest
    # =========================================================
    sim_kwargs = {
        'name': '價值股息策略',
        'upload': False,
        'stop_loss': p['stop_loss'],
        'position_limit': 1.0 / p['max_positions'],
        'trade_at_price': 'open',
    }

    logger.info(f"價值股息: 開始回測, sim_kwargs={sim_kwargs}")

    try:
        report = safe_finlab_sim(position, **sim_kwargs)
        logger.info("價值股息: 回測完成")
        return report
    except Exception as e:
        logger.error(f"價值股息 回測失敗: {type(e).__name__}: {e}", exc_info=True)
        raise


# ==========================================
# 平台標準入口函式
# ==========================================
def run_strategy(api_token, **kwargs):
    """平台標準入口 — 供回測系統自動呼叫。"""
    return run_value_dividend_strategy(api_token, params=kwargs if kwargs else None)
