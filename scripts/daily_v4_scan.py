"""
每日 V4 策略信號掃描器

功能:
1. 偵測當前市場 Regime
2. 取得 V4.0 / V4.1 / V4.2 的 Regime 配置
3. 跑各子策略取得今日持倉 & 新進場股票
4. 儲存 JSON + 發送 Telegram 通知

使用:
    ./python_embed/python.exe scripts/daily_v4_scan.py
"""
import sys
import os
import json
import logging
import importlib
from datetime import datetime, timezone, timedelta

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import V4_SIGNALS_PATH

logger = logging.getLogger('daily_v4_scan')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)

TW_TZ = timezone(timedelta(hours=8))

# V4 variants config
V4_VARIANTS = {
    'V4.0': {
        'module': 'strategies.isaac_v4',
        'alloc_func': 'get_current_regime_allocation',
        'label': 'V4.0 Balanced',
    },
    'V4.1': {
        'module': 'strategies.isaac_v4_razor',
        'alloc_func': 'get_current_regime_allocation',
        'label': 'V4.1 Razor',
    },
    'V4.2': {
        'module': 'strategies.isaac_v4_turbo',
        'alloc_func': 'get_current_regime_allocation',
        'label': 'V4.2 Turbo',
    },
}

# Sub-strategy → how to extract positions
SUBSTRATEGY_RUNNERS = {
    'Isaac V3.9': {
        'module': 'strategies.isaac',
        'func': 'run_isaac_strategy',
        'raw_mode': True,  # returns dict with 'final_pos'
    },
    'Will VCP V2.0': {
        'module': 'strategies.will_vcp',
        'func': 'run_strategy',
    },
    'Mean Reversion': {
        'module': 'strategies.mean_reversion',
        'func': 'run_strategy',
        'kwargs': {'mode': 'classic', 'uptrend_filter': True,
                   'deviation_threshold': -0.08, 'exit_ma': 10},
    },
    'Value Dividend': {
        'module': 'strategies.value_dividend',
        'func': 'run_strategy',
    },
    'Pairs Trading': {
        'module': 'strategies.pairs_trading',
        'func': 'run_strategy',
    },
}


def _load_api_token():
    """從 secrets.toml 取得 FinLab API key。"""
    import toml
    secrets = toml.load('.streamlit/secrets.toml')
    token = secrets.get('FINLAB_API_KEY', secrets.get('FINLAB_API_TOKEN', ''))
    if not token:
        raise ValueError("缺少 FINLAB_API_KEY，請設定 .streamlit/secrets.toml")
    return token


def _get_stock_names(tickers):
    """取得股票名稱對照表。"""
    try:
        from finlab import data as fdata
        info = fdata.get('company_basic_info')
        if 'symbol' in info.columns:
            info = info.set_index('symbol')
        # 第一個欄位通常是公司簡稱
        name_col = info.columns[0] if len(info.columns) > 0 else None
        if name_col:
            return {t: info[name_col].get(t, '') for t in tickers}
    except Exception:
        pass
    return {t: '' for t in tickers}


def _get_stock_prices(tickers, close_df):
    """從收盤價 DataFrame 取得最新價格。"""
    prices = {}
    if close_df is not None:
        last_row = close_df.iloc[-1]
        for t in tickers:
            if t in last_row.index:
                p = last_row[t]
                if not (p != p):  # not NaN
                    prices[t] = round(float(p), 2)
    return prices


def _run_substrategy(name, api_token):
    """
    執行子策略，回傳 position DataFrame。

    Returns:
        pd.DataFrame or None — index=dates, columns=tickers, values=scores
    """
    config = SUBSTRATEGY_RUNNERS.get(name)
    if not config:
        logger.warning(f"未知子策略: {name}")
        return None

    try:
        mod = importlib.import_module(config['module'])
        func = getattr(mod, config['func'])
        kwargs = config.get('kwargs', {})

        if config.get('raw_mode'):
            result = func(api_token, raw_mode=True, **kwargs)
            return result.get('final_pos')
        else:
            report = func(api_token, **kwargs)
            return getattr(report, 'position', None)

    except Exception as e:
        logger.error(f"子策略 {name} 執行失敗: {e}")
        return None


def _extract_signals(position_df, name):
    """
    從 position DataFrame 萃取今日持倉 & 新進場。

    Returns:
        dict: {holdings: [{ticker, score}], entered: [{ticker, score}], exited: [ticker]}
    """
    if position_df is None or len(position_df) < 2:
        return {'holdings': [], 'entered': [], 'exited': []}

    today = position_df.iloc[-1]
    yesterday = position_df.iloc[-2]

    today_set = set(today[today > 0].index)
    yesterday_set = set(yesterday[yesterday > 0].index)

    entered = today_set - yesterday_set
    exited = yesterday_set - today_set

    holdings = []
    for t in sorted(today_set):
        holdings.append({'ticker': t, 'score': round(float(today[t]), 1)})

    entered_list = []
    for t in sorted(entered):
        entered_list.append({'ticker': t, 'score': round(float(today[t]), 1)})

    return {
        'holdings': holdings,
        'entered': entered_list,
        'exited': sorted(exited),
        'n_holdings': len(today_set),
        'n_entered': len(entered),
        'n_exited': len(exited),
    }


def run_daily_scan():
    """主掃描流程。"""
    logger.info("=== V4 每日信號掃描 開始 ===")
    api_token = _load_api_token()

    # 1. 取得各 V4 variant 的 regime allocation
    allocations = {}
    current_regime = None
    for variant_key, cfg in V4_VARIANTS.items():
        try:
            mod = importlib.import_module(cfg['module'])
            alloc_func = getattr(mod, cfg['alloc_func'])
            alloc = alloc_func(api_token)
            allocations[variant_key] = alloc
            if current_regime is None:
                current_regime = alloc.get('regime', 'unknown')
            logger.info(f"{variant_key}: regime={alloc['regime']}, weights={alloc['weights']}")
        except Exception as e:
            logger.error(f"{variant_key} allocation 取得失敗: {e}")
            allocations[variant_key] = {'regime': 'error', 'weights': {}, 'error': str(e)}

    # 2. 收集所有需要跑的子策略（去重）
    needed_strategies = set()
    for alloc in allocations.values():
        for strat_name, weight in alloc.get('weights', {}).items():
            if weight > 0.01:
                needed_strategies.add(strat_name)
    logger.info(f"需要跑的子策略: {needed_strategies}")

    # 3. 取得收盤價（for stock prices）
    close_df = None
    try:
        import finlab
        from data.provider import sanitize_dataframe
        finlab.login(api_token)
        from finlab import data as fdata
        close_df = sanitize_dataframe(fdata.get('price:收盤價'), "close")
        close_df.columns = close_df.columns.astype(str)
    except Exception as e:
        logger.warning(f"收盤價載入失敗: {e}")

    # 4. 跑各子策略，取得信號
    strategy_signals = {}
    for strat_name in needed_strategies:
        logger.info(f"執行子策略: {strat_name}")
        position = _run_substrategy(strat_name, api_token)
        signals = _extract_signals(position, strat_name)
        strategy_signals[strat_name] = signals
        logger.info(
            f"  {strat_name}: {signals['n_holdings']} 持倉, "
            f"{signals['n_entered']} 新進場, {signals['n_exited']} 出場"
        )

    # 5. 組合各 V4 variant 的推薦股票
    all_entered_tickers = set()
    all_holding_tickers = set()
    variant_picks = {}

    for variant_key, alloc in allocations.items():
        picks = []
        weights = alloc.get('weights', {})
        for strat_name, weight in weights.items():
            if weight < 0.01:
                continue
            sig = strategy_signals.get(strat_name, {})
            for stock in sig.get('entered', []):
                picks.append({
                    'ticker': stock['ticker'],
                    'score': stock['score'],
                    'strategy': strat_name,
                    'weight': round(weight * 100),
                })
                all_entered_tickers.add(stock['ticker'])
            for stock in sig.get('holdings', []):
                all_holding_tickers.add(stock['ticker'])

        variant_picks[variant_key] = picks

    # 6. Enrich with names and prices
    all_tickers = all_entered_tickers | all_holding_tickers
    name_map = _get_stock_names(list(all_tickers)) if all_tickers else {}
    price_map = _get_stock_prices(list(all_tickers), close_df) if all_tickers else {}

    for variant_key, picks in variant_picks.items():
        for p in picks:
            p['name'] = name_map.get(p['ticker'], '')
            p['price'] = price_map.get(p['ticker'], 0)

    for strat_name, sig in strategy_signals.items():
        for stock in sig.get('holdings', []):
            stock['name'] = name_map.get(stock['ticker'], '')
            stock['price'] = price_map.get(stock['ticker'], 0)
        for stock in sig.get('entered', []):
            stock['name'] = name_map.get(stock['ticker'], '')
            stock['price'] = price_map.get(stock['ticker'], 0)

    # 7. 組裝最終結果
    now = datetime.now(TW_TZ)
    result = {
        'scan_time': now.isoformat(),
        'date': now.strftime('%Y-%m-%d'),
        'regime': current_regime,
        'allocations': allocations,
        'strategy_signals': {
            name: {
                'n_holdings': sig['n_holdings'],
                'n_entered': sig['n_entered'],
                'n_exited': sig['n_exited'],
                'entered': sig['entered'],
                'exited': sig['exited'],
                'holdings': sig['holdings'],
            }
            for name, sig in strategy_signals.items()
        },
        'variant_new_entries': variant_picks,
    }

    # 8. 儲存
    os.makedirs(os.path.dirname(V4_SIGNALS_PATH), exist_ok=True)
    with open(V4_SIGNALS_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"結果已儲存: {V4_SIGNALS_PATH}")

    # 9. 發送 Telegram
    _send_notification(result)

    logger.info("=== V4 每日信號掃描 完成 ===")
    return result


def _send_notification(result):
    """發送 Telegram 通知。"""
    try:
        from utils.notify import send_telegram, format_v4_daily_signals
        msg = format_v4_daily_signals(result)
        send_telegram(msg)
        logger.info("Telegram 通知已發送")
    except Exception as e:
        logger.error(f"Telegram 通知失敗: {e}")


if __name__ == '__main__':
    run_daily_scan()
