from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

# ── 策略 Metadata（系統自動讀取，顯示在策略選擇器中）──
STRATEGY_NAME = "我的自訂策略"            # 策略顯示名稱
STRATEGY_DESCRIPTION = "基於均線突破的多頭策略"  # 策略簡短描述


def run_strategy(api_token, stop_loss=0.08, take_profit=0.20):
    """
    策略範本 — 支援回測 + 壓力測試

    參數:
        api_token (str): Finlab API 金鑰 (由平台自動帶入)
        stop_loss (float): 停損比例，預設 8%（壓力測試時由平台傳入不同值）
        take_profit (float): 停利比例，預設 20%（壓力測試時由平台傳入不同值）

    回傳:
        report: Finlab 回測報告物件
    """
    if api_token:
        finlab.login(api_token)

    # 1. 取得數據 (Fetch Data)
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')

    # 2. 計算指標 (Indicators)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vol_ma20 = vol.rolling(20).mean()

    # 3. 進場條件 (Entry Signal)
    # 範例：股價站上月線 (20MA) 且站上季線 (60MA) 且爆量
    cond1 = close > ma20
    cond2 = close > ma60
    cond3 = vol > vol_ma20 * 1.5

    buy_signal = cond1 & cond2 & cond3

    # 4. 出場條件 (Exit Signal)
    # 使用 stop_loss / take_profit 參數，讓壓力測試可以測不同組合
    entry_price = close.where(buy_signal).ffill()
    change_pct = (close - entry_price) / entry_price

    exit_stop = change_pct < -stop_loss          # 停損出場
    exit_profit = change_pct > take_profit        # 停利出場
    exit_ma = close < ma20                        # 跌破均線出場

    sell_signal = exit_stop | exit_profit | exit_ma

    # 5. 建立部位 (Position)
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[sell_signal] = 0
    position = position.ffill().fillna(0)

    # 6. 回測 (Backtest)
    report = backtest.sim(position, resample='D', name='我的自訂策略', upload=False)

    return report
