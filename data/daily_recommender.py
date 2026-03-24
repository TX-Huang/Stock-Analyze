"""
每日推薦引擎 — Isaac V3.7
跑策略 → 產出今日推薦持股 + 即時報價 + 風控狀態

使用方式:
    python data/daily_recommender.py          # CLI 直接執行
    from data.daily_recommender import get_daily_recommendation  # 模組呼叫
"""
import sys
import os
import json
import logging
from datetime import datetime, date, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import RECOMMENDATION_PATH

TW_TZ = timezone(timedelta(hours=8))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


_name_cache = {}

def _lookup_stock_name(ticker):
    """Lookup stock name from FinLab or yfinance. Cached in-memory."""
    if ticker in _name_cache:
        return _name_cache[ticker]
    name = ''
    # Method 1: FinLab security_categories
    try:
        from finlab import data
        cats = data.get('security_categories')
        if cats is not None and 'stock_id' in cats.columns:
            cats_idx = cats.set_index('stock_id')
            if ticker in cats_idx.index:
                name = str(cats_idx.loc[ticker, 'name']) if 'name' in cats_idx.columns else ''
    except Exception:
        pass
    # Method 2: yfinance fallback
    if not name:
        try:
            from data.provider import get_data_provider
            provider = get_data_provider("auto", market_type="TW")
            info = provider.get_stock_info(ticker)
            name = info.get('name', '')
            if name == ticker:
                name = ''
        except Exception:
            pass
    _name_cache[ticker] = name or ticker
    return _name_cache[ticker]


def _get_sinopac_provider():
    """建立永豐金連線"""
    try:
        import toml
        secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
        from data.provider import SinoPacProvider
        # Reset to avoid stale connection
        SinoPacProvider._api_instance = None
        SinoPacProvider._logged_in = False
        provider = SinoPacProvider(
            api_key=secrets.get('SINOPAC_API_KEY', ''),
            secret_key=secrets.get('SINOPAC_SECRET_KEY', ''),
            simulation=False,
        )
        if SinoPacProvider._logged_in:
            return provider
    except Exception as e:
        logging.warning(f"永豐金連線失敗: {e}")
    return None


def get_daily_recommendation(api_token=None, with_quotes=True):
    """
    執行 Isaac V3.7 策略，產出今日推薦

    Returns:
        dict: {
            'date': '2026-03-21',
            'strategy': 'Isaac V3.7',
            'market_regime': '多頭/空頭/盤整',
            'exposure': 1.0/0.6/0.3,
            'recommendations': [
                {'rank': 1, 'ticker': '2330', 'name': '台積電', 'score': 8,
                 'signal': 'A+E', 'close': 1840, 'change_rate': -0.54, ...},
            ],
            'exits': [...],  # 今日出場信號
            'risk_summary': {...},
            'stats': {...},
        }
    """
    import toml
    import pandas as pd
    import numpy as np

    if not api_token:
        secrets = toml.load(os.path.join(PROJECT_ROOT, '.streamlit', 'secrets.toml'))
        api_token = secrets.get('FINLAB_API_KEY', '')

    logging.info("執行 Isaac V3.7 策略...")
    from strategies.isaac import run_isaac_strategy

    # 1. 取得 raw position matrix
    raw = run_isaac_strategy(api_token, raw_mode=True)
    final_pos = raw['final_pos']
    close = raw['close']
    etf_close = raw['etf_close']
    trail_stop = raw['trail_stop']
    max_concurrent = raw['max_concurrent']

    # 2. 取最後交易日的持倉
    last_date = final_pos.index[-1]
    last_pos = final_pos.iloc[-1]
    prev_pos = final_pos.iloc[-2] if len(final_pos) > 1 else pd.Series(0, index=final_pos.columns)

    # 多頭推薦 (score > 0)
    long_picks = last_pos[last_pos > 0].sort_values(ascending=False)
    # 空頭避險信號
    short_signals = last_pos[last_pos < 0]

    # 3. 判斷市場環境 (Dynamic Exposure)
    etf_s = etf_close.reindex(final_pos.index).ffill()
    ma60 = etf_s.rolling(60).mean()
    ma120 = etf_s.rolling(120).mean()
    last_etf = etf_s.iloc[-1]
    last_ma60 = ma60.iloc[-1]
    last_ma120 = ma120.iloc[-1]

    if last_etf > last_ma60:
        market_regime = '多頭'
        exposure = 1.0
    elif last_etf > last_ma120:
        market_regime = '盤整偏空'
        exposure = 0.6
    else:
        market_regime = '空頭'
        exposure = 0.3

    # 4. 偵測今日新進場 / 新出場
    new_entries = []
    new_exits = []
    for col in final_pos.columns:
        curr = last_pos.get(col, 0)
        prev = prev_pos.get(col, 0)
        if curr > 0 and prev <= 0:
            new_entries.append(col)
        elif curr <= 0 and prev > 0:
            new_exits.append(col)

    # 5. 即時報價 (永豐金 API)
    sinopac = None
    quote_map = {}
    if with_quotes:
        sinopac = _get_sinopac_provider()
        if sinopac:
            all_tickers = list(long_picks.index[:20]) + new_exits[:10] + ['0050']
            all_tickers = list(set(all_tickers))
            logging.info(f"取得 {len(all_tickers)} 檔即時報價...")
            snaps = sinopac.get_snapshots(all_tickers)
            for s in snaps:
                quote_map[s['code']] = s

    # 6. 組裝推薦清單
    recommendations = []
    for rank, (ticker, score) in enumerate(long_picks.head(max_concurrent).items(), 1):
        rec = {
            'rank': rank,
            'ticker': ticker,
            'score': round(float(score), 1),
            'is_new': ticker in new_entries,
        }
        # 即時報價
        if ticker in quote_map:
            q = quote_map[ticker]
            rec['name'] = q.get('name', '')
            rec['close'] = q.get('close', 0)
            rec['change_rate'] = q.get('change_rate', 0)
            rec['volume'] = q.get('total_volume', 0)
            rec['buy_price'] = q.get('buy_price', 0)
            rec['sell_price'] = q.get('sell_price', 0)
        else:
            # 用 FinLab close 作為備用
            if ticker in close.columns:
                last_close = float(close[ticker].iloc[-1])
                rec['close'] = round(last_close, 2)
                # 用前一日收盤計算漲跌
                if len(close[ticker].dropna()) >= 2:
                    prev_close = float(close[ticker].dropna().iloc[-2])
                    if prev_close > 0:
                        rec['change_rate'] = round((last_close - prev_close) / prev_close * 100, 2)
                    else:
                        rec['change_rate'] = 0
                else:
                    rec['change_rate'] = 0
            else:
                rec['change_rate'] = 0
            rec['name'] = ''
        # Fallback name lookup: try FinLab security_categories or yfinance
        if not rec.get('name'):
            rec['name'] = _lookup_stock_name(ticker)
        recommendations.append(rec)

    # 7. 出場清單
    exits = []
    for ticker in new_exits:
        ex = {'ticker': ticker}
        if ticker in quote_map:
            q = quote_map[ticker]
            ex['name'] = q.get('name', '')
            ex['close'] = q.get('close', 0)
            ex['change_rate'] = q.get('change_rate', 0)
        elif ticker in close.columns:
            ex['close'] = round(float(close[ticker].iloc[-1]), 2)
            ex['name'] = ''
        if not ex.get('name'):
            ex['name'] = _lookup_stock_name(ticker)
        exits.append(ex)

    # 8. 大盤報價
    etf_quote = quote_map.get('0050', {})

    # 9. 空頭避險狀態
    n_short_signals = len(short_signals)
    if n_short_signals >= 2:
        hedge_status = f'強避險 (hedge 60%, {n_short_signals} 空頭信號)'
    elif n_short_signals >= 1:
        hedge_status = f'輕避險 (hedge 30%, {n_short_signals} 空頭信號)'
    else:
        hedge_status = '無避險信號'

    # 10. 組裝結果
    result = {
        'date': str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
        'generated_at': datetime.now(TW_TZ).isoformat(),
        'strategy': 'Isaac V3.7',
        'market_regime': market_regime,
        'exposure': exposure,
        'hedge_status': hedge_status,
        'etf_0050': {
            'close': etf_quote.get('close', round(float(last_etf), 2)),
            'change_rate': etf_quote.get('change_rate', 0),
            'ma60': round(float(last_ma60), 2),
            'ma120': round(float(last_ma120), 2),
        },
        'recommendations': recommendations,
        'new_entries': new_entries,
        'exits': exits,
        'n_short_signals': n_short_signals,
        'summary': {
            'total_picks': len(recommendations),
            'new_entries': len(new_entries),
            'new_exits': len(new_exits),
            'trail_stop': trail_stop,
            'max_concurrent': max_concurrent,
        },
    }

    # 儲存到 JSON（原子寫入）
    from utils.helpers import safe_json_write
    safe_json_write(RECOMMENDATION_PATH, result, default=str)
    logging.info(f"推薦結果已儲存至 {RECOMMENDATION_PATH}")

    # 清理連線
    if sinopac:
        from data.provider import SinoPacProvider
        SinoPacProvider.logout()

    return result


def format_recommendation_text(result: dict) -> str:
    """將推薦結果格式化為文字 (供 Telegram / CLI 使用)"""
    lines = []
    lines.append(f"{'='*40}")
    lines.append(f"  Isaac V3.7 每日推薦")
    lines.append(f"  {result['date']}")
    lines.append(f"{'='*40}")

    # 大盤環境
    etf = result.get('etf_0050', {})
    lines.append(f"\n  [大盤環境]")
    lines.append(f"  0050: {etf.get('close', 'N/A')} ({etf.get('change_rate', 0):+.2f}%)")
    lines.append(f"  MA60: {etf.get('ma60', 'N/A')} | MA120: {etf.get('ma120', 'N/A')}")
    lines.append(f"  市場狀態: {result['market_regime']} | 曝險: {result['exposure']*100:.0f}%")
    lines.append(f"  避險: {result['hedge_status']}")

    # 推薦持股
    recs = result.get('recommendations', [])
    lines.append(f"\n  [推薦持股] Top-{len(recs)}")
    lines.append(f"  {'Rank':<4} {'代碼':<6} {'名稱':<8} {'Score':>5} {'現價':>8} {'漲跌%':>7} {'新進場':>4}")
    lines.append(f"  {'-'*50}")
    for r in recs:
        new_tag = 'NEW' if r.get('is_new') else ''
        name = r.get('name', '')[:6]
        lines.append(
            f"  {r['rank']:<4} {r['ticker']:<6} {name:<8} {r['score']:>5.0f} "
            f"{r.get('close', 0):>8.1f} {r.get('change_rate', 0):>+6.2f}% {new_tag:>4}"
        )

    # 今日出場
    exits = result.get('exits', [])
    if exits:
        lines.append(f"\n  [今日出場信號]")
        for ex in exits:
            lines.append(f"    {ex['ticker']} {ex.get('name', '')} | {ex.get('close', 'N/A')}")

    # 新進場
    new_entries = result.get('new_entries', [])
    if new_entries:
        lines.append(f"\n  [今日新進場]")
        for t in new_entries:
            rec = next((r for r in recs if r['ticker'] == t), None)
            if rec:
                lines.append(f"    {t} {rec.get('name', '')} | score={rec['score']:.0f} | {rec.get('close', 'N/A')}")

    # 摘要
    summary = result.get('summary', {})
    lines.append(f"\n  [參數]")
    lines.append(f"  trail_stop: {summary.get('trail_stop', 'N/A')} | max_concurrent: {summary.get('max_concurrent', 'N/A')}")
    lines.append(f"  產生時間: {result.get('generated_at', '')[:19]}")
    lines.append(f"{'='*40}")

    return '\n'.join(lines)


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == '__main__':
    result = get_daily_recommendation()
    print(format_recommendation_text(result))
