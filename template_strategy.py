from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_strategy(api_token):
    """
    這是一個標準的策略範本。
    您可以在這裡修改選股邏輯，然後上傳到平台進行回測。

    參數:
    api_token (str): Finlab API 金鑰 (由平台自動帶入)

    回傳:
    report: Finlab 回測報告物件
    """
    if api_token:
        finlab.login(api_token)

    # 1. 取得數據 (Fetch Data)
    # 常用數據：收盤價, 成交股數, 營收, EPS...
    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')

    # 2. 計算指標 (Indicators)
    # 例如：均線、RSI、成交量平均
    ma20 = close.average(20)
    ma60 = close.average(60)
    vol_ma20 = vol.average(20)

    # 3. 進場條件 (Entry Signal)
    # 範例策略：股價站上月線 (20MA) 且 站上季線 (60MA) 且 爆量
    cond1 = close > ma20
    cond2 = close > ma60
    cond3 = vol > vol_ma20 * 1.5

    buy_signal = cond1 & cond2 & cond3

    # 4. 出場條件 (Exit Signal)
    # 範例：跌破月線出場
    sell_signal = close < ma20

    # 5. 建立部位 (Position)
    # 1 代表持有，0 代表空手
    position = pd.DataFrame(np.nan, index=buy_signal.index, columns=buy_signal.columns)
    position[buy_signal] = 1
    position[sell_signal] = 0

    # 填補訊號 (Hold)
    position = position.ffill().fillna(0)

    # 6. 回測 (Backtest)
    # upload=False 代表不自動上傳到 Finlab 官網
    report = backtest.sim(position, resample='D', name='我的自訂策略', upload=False)

    return report
