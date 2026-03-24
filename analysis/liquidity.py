"""流動性風險評估 — 檢查部位大小 vs 日均成交量。"""
import logging

logger = logging.getLogger(__name__)


def check_liquidity(ticker, shares, provider, period="1mo", warn_threshold=0.10, danger_threshold=0.25):
    """
    檢查持倉股數是否超過日均成交量的安全比例。

    Args:
        ticker: 股票代碼
        shares: 持有股數
        provider: DataProvider instance
        period: 計算日均量的期間
        warn_threshold: 警告門檻 (持股/日均量 > 10%)
        danger_threshold: 危險門檻 (持股/日均量 > 25%)

    Returns:
        {
            'ticker': str,
            'shares': int,
            'avg_daily_volume': float,
            'position_ratio': float,  # 持股佔日均量比例
            'days_to_exit': float,    # 以日均量10%估算需幾天才能全部出場
            'risk_level': str,        # 'safe', 'warning', 'danger'
            'message': str,
        }
    """
    try:
        df = provider.get_historical_data(str(ticker), period=period, interval="1d")
        if df is None or df.empty or 'Volume' not in df.columns:
            return {
                'ticker': ticker, 'shares': shares, 'avg_daily_volume': 0,
                'position_ratio': 0, 'days_to_exit': 0,
                'risk_level': 'unknown', 'message': '無法取得成交量資料',
            }

        avg_vol = float(df['Volume'].mean())
        if avg_vol <= 0:
            return {
                'ticker': ticker, 'shares': shares, 'avg_daily_volume': 0,
                'position_ratio': 0, 'days_to_exit': 0,
                'risk_level': 'danger', 'message': '日均成交量為零',
            }

        ratio = shares / avg_vol
        # Estimate days to exit at 10% of daily volume participation rate
        days_to_exit = shares / (avg_vol * 0.10) if avg_vol > 0 else 9999

        if ratio >= danger_threshold:
            risk = 'danger'
            msg = f'持股佔日均量 {ratio*100:.1f}% — 流動性極差，出場可能需 {days_to_exit:.1f} 天'
        elif ratio >= warn_threshold:
            risk = 'warning'
            msg = f'持股佔日均量 {ratio*100:.1f}% — 流動性偏低，注意出場衝擊'
        else:
            risk = 'safe'
            msg = f'持股佔日均量 {ratio*100:.1f}% — 流動性充足'

        return {
            'ticker': ticker,
            'shares': shares,
            'avg_daily_volume': avg_vol,
            'position_ratio': ratio,
            'days_to_exit': days_to_exit,
            'risk_level': risk,
            'message': msg,
        }
    except Exception as e:
        logger.warning(f"流動性檢查失敗 {ticker}: {e}")
        return {
            'ticker': ticker, 'shares': shares, 'avg_daily_volume': 0,
            'position_ratio': 0, 'days_to_exit': 0,
            'risk_level': 'unknown', 'message': f'檢查失敗: {e}',
        }


def batch_liquidity_check(positions, provider):
    """批次檢查所有持倉的流動性。"""
    results = []
    for pos in positions:
        ticker = pos.get('ticker', '')
        shares = pos.get('shares', 0)
        if ticker and shares > 0:
            result = check_liquidity(ticker, shares, provider)
            result['name'] = pos.get('name', ticker)
            results.append(result)

    # Sort by risk level (danger first)
    risk_order = {'danger': 0, 'warning': 1, 'unknown': 2, 'safe': 3}
    results.sort(key=lambda r: risk_order.get(r['risk_level'], 9))
    return results
