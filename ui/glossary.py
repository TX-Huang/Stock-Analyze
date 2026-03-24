"""
Financial glossary and tooltip system for AI Invest HQ.

Provides:
1. GLOSSARY — 50+ financial terms in Traditional Chinese with formulas
2. tooltip() — hover-to-reveal HTML wrapper
3. render_glossary_sidebar() — searchable sidebar component
4. render_glossary_page() — full glossary page with categories and search
"""
import streamlit as st

# ---------------------------------------------------------------------------
# Glossary database
# ---------------------------------------------------------------------------
# Each entry: {zh, en, short, detail, formula (optional), category}
# Categories: 'performance', 'risk', 'technical', 'fundamental',
#             'trading', 'strategy', 'market'

GLOSSARY = {
    # ── Performance metrics ──────────────────────────────────────────
    'cagr': {
        'zh': '年化報酬率',
        'en': 'CAGR (Compound Annual Growth Rate)',
        'short': '投資每年平均成長的百分比',
        'detail': '年化報酬率衡量投資在一段時間內的平均年度成長率，考慮複利效果。'
                  'CAGR = (終值/初值)^(1/年數) - 1。'
                  '例如 3 年內從 100 漲到 200，CAGR = 26.0%。',
        'formula': 'CAGR = (V_final / V_initial)^(1/n) - 1',
        'category': 'performance',
    },
    'total_return': {
        'zh': '總報酬率',
        'en': 'Total Return',
        'short': '投資期間內的累計報酬',
        'detail': '總報酬率包含資本利得及股利收入。100% 表示資產翻倍。',
        'formula': 'Total Return = (V_final - V_initial + Dividends) / V_initial',
        'category': 'performance',
    },
    'win_ratio': {
        'zh': '勝率',
        'en': 'Win Ratio',
        'short': '獲利交易佔總交易的百分比',
        'detail': '勝率 = 獲利交易數 / 總交易數。勝率高不代表策略好，需搭配盈虧比觀察。'
                  '45% 勝率搭配 2:1 盈虧比仍可獲利。',
        'formula': 'Win% = Winning Trades / Total Trades',
        'category': 'performance',
    },
    'profit_factor': {
        'zh': '獲利因子',
        'en': 'Profit Factor',
        'short': '總獲利除以總虧損的倍數',
        'detail': '獲利因子 > 1 表示整體獲利。> 1.5 算良好，> 2.0 算優秀。'
                  '獲利因子 = 總毛利 / 總毛損。',
        'formula': 'PF = Gross Profit / Gross Loss',
        'category': 'performance',
    },
    'payoff_ratio': {
        'zh': '盈虧比',
        'en': 'Payoff Ratio (Reward-to-Risk)',
        'short': '平均獲利交易的獲利 / 平均虧損交易的虧損',
        'detail': '盈虧比 2:1 表示平均每次獲利是每次虧損的 2 倍。'
                  '配合勝率可計算期望值。',
        'formula': 'Payoff = Avg Win / Avg Loss',
        'category': 'performance',
    },
    'expectancy': {
        'zh': '期望值',
        'en': 'Expectancy',
        'short': '每筆交易預期獲利金額',
        'detail': '期望值 = (勝率 x 平均獲利) - (敗率 x 平均虧損)。'
                  '正值表示長期會賺錢。',
        'formula': 'E = Win% x AvgWin - Loss% x AvgLoss',
        'category': 'performance',
    },

    # ── Risk metrics ─────────────────────────────────────────────────
    'sharpe': {
        'zh': '夏普比率',
        'en': 'Sharpe Ratio',
        'short': '每承擔一單位風險能獲得多少超額報酬',
        'detail': '夏普比率衡量投資的風險調整後報酬。數值越高表示單位風險獲得的報酬越多。'
                  '> 1.0 算好，> 2.0 算優秀。',
        'formula': 'Sharpe = (R_p - R_f) / sigma_p',
        'category': 'risk',
    },
    'sortino': {
        'zh': '索提諾比率',
        'en': 'Sortino Ratio',
        'short': '只考慮下行風險的報酬比率',
        'detail': '索提諾比率與夏普類似，但分母只用下行偏差（虧損的波動），'
                  '不懲罰上行波動。更適合衡量非對稱報酬分布的策略。',
        'formula': 'Sortino = (R_p - R_f) / sigma_downside',
        'category': 'risk',
    },
    'mdd': {
        'zh': '最大回撤',
        'en': 'Maximum Drawdown (MDD)',
        'short': '從高點到低點的最大跌幅',
        'detail': '最大回撤是投資組合從歷史高點到後續低點的最大跌幅百分比。'
                  '-20% 表示曾經從高點下跌 20%。投資人心理承受力通常在 -30% 左右。',
        'formula': 'MDD = (Trough - Peak) / Peak',
        'category': 'risk',
    },
    'var': {
        'zh': '風險值',
        'en': 'Value at Risk (VaR)',
        'short': '在特定信心水準下的最大可能損失',
        'detail': '95% VaR = -5% 表示有 95% 的信心，明天的損失不會超過 5%。'
                  '但仍有 5% 的機率損失更多。',
        'formula': 'VaR_alpha = -Quantile(returns, 1-alpha)',
        'category': 'risk',
    },
    'cvar': {
        'zh': '條件風險值',
        'en': 'Conditional VaR (CVaR / Expected Shortfall)',
        'short': '超過 VaR 時的平均損失',
        'detail': 'CVaR 衡量當損失超過 VaR 門檻時的平均損失大小。'
                  '比 VaR 更能捕捉尾端風險（黑天鵝事件）。',
        'formula': 'CVaR_alpha = E[Loss | Loss > VaR_alpha]',
        'category': 'risk',
    },
    'volatility': {
        'zh': '波動率',
        'en': 'Volatility (Historical Volatility)',
        'short': '報酬率的標準差，衡量價格波動程度',
        'detail': '年化波動率 30% 表示股價一年內的漲跌幅大約在 +-30% 範圍。'
                  '波動率越高，風險越大但潛在報酬也越高。',
        'formula': 'sigma = std(daily returns) x sqrt(252)',
        'category': 'risk',
    },
    'beta': {
        'zh': '貝塔係數',
        'en': 'Beta',
        'short': '相對大盤的系統性風險',
        'detail': 'Beta = 1 表示與大盤同步波動。> 1 表示波動比大盤大，< 1 表示較穩定。'
                  '負 Beta 表示與大盤反向。',
        'formula': 'Beta = Cov(R_i, R_m) / Var(R_m)',
        'category': 'risk',
    },
    'calmar': {
        'zh': '卡瑪比率',
        'en': 'Calmar Ratio',
        'short': 'CAGR 除以最大回撤',
        'detail': '卡瑪比率 = 年化報酬 / |最大回撤|。衡量每承受一單位回撤能獲得多少報酬。'
                  '> 1 表示報酬大於最大回撤。',
        'formula': 'Calmar = CAGR / |MDD|',
        'category': 'risk',
    },
    'correlation': {
        'zh': '相關性',
        'en': 'Correlation',
        'short': '兩項資產報酬的線性關聯度',
        'detail': '相關係數介於 -1 到 +1。+1 完全正相關，-1 完全負相關，0 無關。'
                  '分散投資應選低相關性資產。',
        'formula': 'rho = Cov(X,Y) / (sigma_X x sigma_Y)',
        'category': 'risk',
    },

    # ── Technical indicators ─────────────────────────────────────────
    'rsi': {
        'zh': '相對強弱指標',
        'en': 'RSI (Relative Strength Index)',
        'short': '衡量近期漲跌強度的動能指標',
        'detail': 'RSI 介於 0~100。> 70 為超買（可能回檔），< 30 為超賣（可能反彈）。'
                  '常用 14 日 RSI。背離訊號更具參考價值。',
        'formula': 'RSI = 100 - 100 / (1 + RS), RS = AvgGain / AvgLoss',
        'category': 'technical',
    },
    'macd': {
        'zh': 'MACD 指標',
        'en': 'MACD (Moving Average Convergence Divergence)',
        'short': '快慢均線差值的趨勢動能指標',
        'detail': 'MACD = EMA12 - EMA26，Signal = EMA9(MACD)。'
                  'MACD 穿越 Signal 為黃金交叉（買入訊號），跌破為死亡交叉。'
                  '柱狀圖 (Histogram) 代表兩者差距。',
        'formula': 'MACD = EMA(12) - EMA(26); Signal = EMA(9, MACD)',
        'category': 'technical',
    },
    'kd': {
        'zh': 'KD 隨機指標',
        'en': 'Stochastic Oscillator (KD)',
        'short': '比較收盤價在近期高低範圍中的位置',
        'detail': 'K 值 = (收盤 - 最低) / (最高 - 最低) x 100，D 值 = K 的移動平均。'
                  'K > 80 超買，K < 20 超賣。KD 黃金交叉在低檔更有效。',
        'formula': 'K = (Close - Low_n) / (High_n - Low_n) x 100',
        'category': 'technical',
    },
    'bollinger': {
        'zh': '布林通道',
        'en': 'Bollinger Bands',
        'short': '以均線為中心加減標準差的通道',
        'detail': '上軌 = MA20 + 2sigma，下軌 = MA20 - 2sigma。'
                  '價格觸及上軌可能回落，觸及下軌可能反彈。'
                  '通道收窄 (squeeze) 代表即將出現大行情。',
        'formula': 'Upper = MA + k x sigma; Lower = MA - k x sigma',
        'category': 'technical',
    },
    'atr': {
        'zh': '平均真實範圍',
        'en': 'ATR (Average True Range)',
        'short': '衡量每日價格波動幅度的指標',
        'detail': 'ATR 計算近 N 日的真實範圍平均值。True Range = max(H-L, |H-C_prev|, |L-C_prev|)。'
                  '常用於設定停損距離和衡量市場波動程度。',
        'formula': 'ATR = SMA(TrueRange, n)',
        'category': 'technical',
    },
    'adx': {
        'zh': '平均方向性指標',
        'en': 'ADX (Average Directional Index)',
        'short': '衡量趨勢強度（不分方向）',
        'detail': 'ADX > 25 表示有趨勢（強趨勢），< 20 表示盤整。'
                  'ADX 只衡量趨勢強度，不判斷方向。搭配 +DI / -DI 判斷多空。',
        'formula': 'ADX = SMA(|+DI - -DI| / (+DI + -DI), 14)',
        'category': 'technical',
    },
    'obv': {
        'zh': '能量潮',
        'en': 'OBV (On-Balance Volume)',
        'short': '以成交量驗證價格趨勢的指標',
        'detail': '價漲日加成交量，價跌日減成交量。'
                  'OBV 上升而股價不動 = 籌碼集中（正面訊號）。'
                  'OBV 與價格背離常預示反轉。',
        'formula': 'OBV_t = OBV_{t-1} +/- Volume_t',
        'category': 'technical',
    },
    'cci': {
        'zh': '商品通道指標',
        'en': 'CCI (Commodity Channel Index)',
        'short': '衡量價格偏離統計平均的程度',
        'detail': 'CCI > +100 為超買，< -100 為超賣。'
                  '用於判斷價格是否偏離正常範圍過多，可能出現均值回歸。',
        'formula': 'CCI = (TP - SMA(TP)) / (0.015 x MeanDev)',
        'category': 'technical',
    },
    'mfi': {
        'zh': '資金流量指標',
        'en': 'MFI (Money Flow Index)',
        'short': '結合量能的 RSI 變形',
        'detail': 'MFI 類似 RSI 但加入成交量權重。> 80 超買，< 20 超賣。'
                  '比 RSI 更能反映實際資金進出。',
        'formula': 'MFI = 100 - 100 / (1 + MF_ratio)',
        'category': 'technical',
    },
    'ma': {
        'zh': '移動平均線',
        'en': 'MA (Moving Average)',
        'short': '過去 N 日收盤價的平均值',
        'detail': '常用天數：5/10/20/60/120/240 日。短均穿越長均向上 = 黃金交叉。'
                  '均線排列多頭（短 > 中 > 長）代表強勢趨勢。',
        'formula': 'SMA = sum(Close, n) / n; EMA = alpha x Close + (1-alpha) x EMA_prev',
        'category': 'technical',
    },
    'vwap': {
        'zh': '成交量加權平均價',
        'en': 'VWAP (Volume Weighted Average Price)',
        'short': '以成交量加權的日內平均成交價',
        'detail': 'VWAP 是法人常用的基準價格。股價在 VWAP 之上表示多數買方獲利。'
                  '常作為日內交易的支撐/壓力參考。',
        'formula': 'VWAP = sum(Price x Volume) / sum(Volume)',
        'category': 'technical',
    },

    # ── Trading concepts ─────────────────────────────────────────────
    'stop_loss': {
        'zh': '停損',
        'en': 'Stop Loss',
        'short': '預設虧損上限，到達時自動賣出',
        'detail': '停損是風險管理的基本工具。常見設定為 -5% 到 -10%。'
                  '嚴格執行停損可保護資金，避免小虧變成大虧。',
        'formula': None,
        'category': 'trading',
    },
    'take_profit': {
        'zh': '停利',
        'en': 'Take Profit',
        'short': '預設獲利目標，到達時賣出鎖定利潤',
        'detail': '停利讓獲利落袋為安。但過早停利會錯過大行情。'
                  '可搭配追蹤停損讓獲利奔跑。',
        'formula': None,
        'category': 'trading',
    },
    'trail_stop': {
        'zh': '追蹤停損',
        'en': 'Trailing Stop',
        'short': '隨價格上漲自動上移的動態停損',
        'detail': '追蹤停損會跟隨最高價上移，但不會下移。'
                  '例如 15% 追蹤停損：股價從 100 漲到 150 時，停損自動升到 127.5。'
                  '可讓獲利奔跑同時保護部分利潤。',
        'formula': 'Stop = Peak x (1 - trail%)',
        'category': 'trading',
    },
    'position_sizing': {
        'zh': '部位大小',
        'en': 'Position Sizing',
        'short': '決定每筆交易投入多少資金',
        'detail': '常見方法：固定比例（每檔 10%）、波動率調整（ATR）、Kelly 公式。'
                  '單一部位不宜超過總資金 15%，避免集中風險。',
        'formula': None,
        'category': 'trading',
    },
    'slippage': {
        'zh': '滑價',
        'en': 'Slippage',
        'short': '實際成交價與預期價格的差異',
        'detail': '流動性差的股票滑價較大。回測時加入滑價假設（如 0.1~0.5%）'
                  '可讓結果更貼近實際交易。',
        'formula': 'Slippage = Execution Price - Expected Price',
        'category': 'trading',
    },
    'liquidity': {
        'zh': '流動性',
        'en': 'Liquidity',
        'short': '資產能多快、多容易地以合理價格買賣',
        'detail': '高流動性 = 成交量大、買賣價差小、容易進出。'
                  '流動性不足的股票容易出現大幅滑價和無法出場的風險。',
        'formula': None,
        'category': 'trading',
    },
    'bid_ask_spread': {
        'zh': '買賣價差',
        'en': 'Bid-Ask Spread',
        'short': '買方最高出價與賣方最低要價的差距',
        'detail': '價差越小表示流動性越好。價差是隱性交易成本。'
                  '台股有升降單位制度（tick size），影響最小價差。',
        'formula': 'Spread = Ask - Bid',
        'category': 'trading',
    },
    'leverage': {
        'zh': '槓桿',
        'en': 'Leverage',
        'short': '用借來的資金放大投資部位',
        'detail': '2 倍槓桿 = 用 1 元資金控制 2 元部位。獲利和虧損都放大。'
                  '槓桿 ETF（如 00631L）每日重新平衡，長期持有會有複利侵蝕。',
        'formula': 'Leverage Ratio = Total Exposure / Equity',
        'category': 'trading',
    },

    # ── Fundamental analysis ─────────────────────────────────────────
    'pe_ratio': {
        'zh': '本益比',
        'en': 'P/E Ratio (Price-to-Earnings)',
        'short': '股價相對於每股盈餘的倍數',
        'detail': 'P/E = 20 表示投資人願意用 20 倍的盈餘買這檔股票。'
                  '低 P/E 可能代表低估或成長性差。高 P/E 可能代表高估或高成長預期。'
                  '需與同產業比較。',
        'formula': 'P/E = Price / EPS',
        'category': 'fundamental',
    },
    'eps': {
        'zh': '每股盈餘',
        'en': 'EPS (Earnings Per Share)',
        'short': '公司每一股賺多少錢',
        'detail': 'EPS = 淨利 / 流通股數。EPS 連續成長是好現象。'
                  '注意稀釋 EPS（考慮可轉債、認股權等）更保守。',
        'formula': 'EPS = Net Income / Shares Outstanding',
        'category': 'fundamental',
    },
    'roe': {
        'zh': '股東權益報酬率',
        'en': 'ROE (Return on Equity)',
        'short': '公司用股東的錢創造多少報酬',
        'detail': 'ROE > 15% 通常代表不錯的經營效率。巴菲特偏好長期 ROE > 15% 的公司。'
                  '但高負債也會拉高 ROE，需搭配負債比觀察。',
        'formula': 'ROE = Net Income / Shareholders Equity',
        'category': 'fundamental',
    },
    'roa': {
        'zh': '資產報酬率',
        'en': 'ROA (Return on Assets)',
        'short': '公司用全部資產創造多少報酬',
        'detail': 'ROA 衡量公司運用所有資源的效率。'
                  '不受槓桿影響，比 ROE 更能看出真實營運效率。',
        'formula': 'ROA = Net Income / Total Assets',
        'category': 'fundamental',
    },
    'dividend_yield': {
        'zh': '殖利率',
        'en': 'Dividend Yield',
        'short': '每年配息金額佔股價的百分比',
        'detail': '殖利率 5% 表示每投資 100 元，每年可領 5 元股利。'
                  '台股高殖利率標準通常 > 4%。但需注意配息是否來自盈餘而非資本。',
        'formula': 'Yield = Annual Dividend / Price',
        'category': 'fundamental',
    },
    'revenue_growth': {
        'zh': '營收成長率',
        'en': 'Revenue Growth Rate',
        'short': '營業收入相比去年同期的成長幅度',
        'detail': '營收年增率（YoY）> 20% 通常被視為高成長。'
                  '月增率（MoM）受季節性影響大。台股每月 10 日前公布上月營收。',
        'formula': 'Growth = (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}',
        'category': 'fundamental',
    },
    'pb_ratio': {
        'zh': '股價淨值比',
        'en': 'P/B Ratio (Price-to-Book)',
        'short': '股價相對於每股淨值的倍數',
        'detail': 'P/B < 1 表示股價低於公司清算價值（可能低估）。'
                  '科技股 P/B 通常較高，金融股較低。',
        'formula': 'P/B = Price / Book Value per Share',
        'category': 'fundamental',
    },
    'operating_margin': {
        'zh': '營業利益率',
        'en': 'Operating Margin',
        'short': '每一元營收中來自本業的獲利比例',
        'detail': '營業利益率 = 營業利益 / 營收。衡量公司本業的獲利能力。'
                  '排除業外損益的干擾，更能反映核心競爭力。',
        'formula': 'OPM = Operating Income / Revenue',
        'category': 'fundamental',
    },

    # ── Strategy & backtesting ───────────────────────────────────────
    'backtest': {
        'zh': '回測',
        'en': 'Backtest',
        'short': '用歷史數據模擬策略表現',
        'detail': '回測讓你看到策略在過去的表現，但不保證未來結果。'
                  '回測陷阱包括：過擬合、前視偏差、倖存者偏差。',
        'formula': None,
        'category': 'strategy',
    },
    'overfitting': {
        'zh': '過擬合',
        'en': 'Overfitting',
        'short': '策略過度適應歷史數據，失去預測能力',
        'detail': '過擬合的策略在回測中表現完美，但實盤交易時失效。'
                  '參數越多、條件越複雜，越容易過擬合。'
                  '對抗方法：樣本外測試、Walk-forward、減少參數。',
        'formula': None,
        'category': 'strategy',
    },
    'walk_forward': {
        'zh': '滾動前進分析',
        'en': 'Walk-Forward Analysis',
        'short': '分段優化 + 樣本外驗證的回測方法',
        'detail': '將歷史資料分成多段，每段先用前半優化、後半驗證。'
                  '比單一回測更能檢測策略的穩健性和過擬合程度。',
        'formula': None,
        'category': 'strategy',
    },
    'monte_carlo': {
        'zh': '蒙地卡羅模擬',
        'en': 'Monte Carlo Simulation',
        'short': '透過隨機模擬評估策略的可能結果分布',
        'detail': '對交易結果做隨機重排或抽樣，產生數千種可能情境。'
                  '可估計最大回撤、報酬的信賴區間，了解運氣成分有多大。',
        'formula': None,
        'category': 'strategy',
    },
    'kelly': {
        'zh': '乖乖公式',
        'en': 'Kelly Criterion',
        'short': '計算最佳下注比例以最大化長期成長',
        'detail': 'Kelly% = W - (1-W)/R，W=勝率，R=盈虧比。'
                  '完整 Kelly 太激進，實務上常用半 Kelly（Kelly/2）。',
        'formula': 'f* = W - (1-W) / R',
        'category': 'strategy',
    },
    'lookback_bias': {
        'zh': '前視偏差',
        'en': 'Look-Ahead Bias',
        'short': '在回測中不當使用未來才知道的資訊',
        'detail': '例如用「年報」數據做交易決策，但年報在隔年 3 月才公布。'
                  '前視偏差會讓回測結果虛假好看，實盤無法複製。',
        'formula': None,
        'category': 'strategy',
    },
    'survivorship_bias': {
        'zh': '倖存者偏差',
        'en': 'Survivorship Bias',
        'short': '只分析存活下來的股票，忽略下市的',
        'detail': '如果回測只用「現在還在上市的股票」，會忽略已下市（破產）的公司，'
                  '導致績效被高估。正確做法是使用包含下市股的完整資料。',
        'formula': None,
        'category': 'strategy',
    },

    # ── Market terminology ───────────────────────────────────────────
    'bull_market': {
        'zh': '多頭市場',
        'en': 'Bull Market',
        'short': '股市持續上漲的期間',
        'detail': '通常定義為從低點上漲超過 20%。多頭市場中大多數股票都會上漲，'
                  '策略更容易獲利。要注意在多頭末期過度樂觀。',
        'formula': None,
        'category': 'market',
    },
    'bear_market': {
        'zh': '空頭市場',
        'en': 'Bear Market',
        'short': '股市持續下跌的期間',
        'detail': '通常定義為從高點下跌超過 20%。空頭市場中防禦和風控最重要。'
                  '歷史上空頭平均持續 9-16 個月。',
        'formula': None,
        'category': 'market',
    },
    'consolidation': {
        'zh': '盤整',
        'en': 'Consolidation / Sideways',
        'short': '股價在一定範圍內橫向波動',
        'detail': '盤整代表多空力道平衡。盤整後的突破（上或下）通常帶來較大行情。'
                  'VCP 策略專門捕捉盤整收斂後的突破。',
        'formula': None,
        'category': 'market',
    },
    'market_breadth': {
        'zh': '市場寬度',
        'en': 'Market Breadth',
        'short': '衡量多少股票參與上漲或下跌',
        'detail': '漲跌家數比、新高新低比等都是寬度指標。'
                  '大盤上漲但寬度收窄（只有少數股票在漲）是危險訊號。',
        'formula': None,
        'category': 'market',
    },
    'sector_rotation': {
        'zh': '類股輪動',
        'en': 'Sector Rotation',
        'short': '資金在不同產業間輪流流動的現象',
        'detail': '景氣循環不同階段有不同的強勢類股。例如復甦期科技股領漲，'
                  '衰退期防禦股（公用事業、必需消費）較抗跌。',
        'formula': None,
        'category': 'market',
    },
    'vcp': {
        'zh': 'VCP 型態',
        'en': 'VCP (Volatility Contraction Pattern)',
        'short': '波動逐漸收縮的整理型態',
        'detail': '由 Mark Minervini 推廣。股價在整理過程中，每次回撤幅度遞減（如 -20%、-12%、-6%），'
                  '成交量也遞減。突破時量增代表主力進場。',
        'formula': None,
        'category': 'market',
    },
    'breakout': {
        'zh': '突破',
        'en': 'Breakout',
        'short': '股價衝破關鍵壓力線或盤整區間',
        'detail': '有效突破通常伴隨成交量放大（1.5 倍以上）。'
                  '假突破（突破後迅速跌回）是常見陷阱，可用 2 日確認法過濾。',
        'formula': None,
        'category': 'market',
    },
    'support_resistance': {
        'zh': '支撐與壓力',
        'en': 'Support & Resistance',
        'short': '股價下方的支撐區和上方的壓力區',
        'detail': '支撐是買盤集中的價位（股價不容易跌破），壓力是賣盤集中的價位。'
                  '支撐被跌破後常變成壓力（角色互換）。',
        'formula': None,
        'category': 'market',
    },
    'gap': {
        'zh': '跳空缺口',
        'en': 'Gap',
        'short': '開盤價與前日收盤之間的價格空白',
        'detail': '突破缺口（伴隨大量）通常不會被回補，代表新趨勢開始。'
                  '衰竭缺口出現在趨勢末端，容易被回補。',
        'formula': None,
        'category': 'market',
    },
    'margin_trading': {
        'zh': '融資融券',
        'en': 'Margin Trading',
        'short': '借錢買股（融資）或借股來賣（融券）',
        'detail': '融資餘額增加代表散戶看多。融資使用率過高（> 60%）是危險訊號。'
                  '融券餘額增加代表有人放空，回補時可能推升股價。',
        'formula': None,
        'category': 'market',
    },
}

# ---------------------------------------------------------------------------
# Category labels (Traditional Chinese)
# ---------------------------------------------------------------------------
CATEGORY_LABELS = {
    'performance': '績效指標',
    'risk': '風險指標',
    'technical': '技術指標',
    'fundamental': '基本面',
    'trading': '交易概念',
    'strategy': '策略與回測',
    'market': '市場術語',
}

CATEGORY_ICONS = {
    'performance': '\U0001F4C8',   # chart increasing
    'risk': '\U0001F6E1\uFE0F',    # shield
    'technical': '\U0001F4CA',      # bar chart
    'fundamental': '\U0001F4D1',    # clipboard
    'trading': '\U0001F4B1',        # currency exchange
    'strategy': '\U0001F9EA',       # test tube
    'market': '\U0001F30D',         # globe
}


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------
_TOOLTIP_CSS_INJECTED = '_glossary_tooltip_css'

def _inject_tooltip_css():
    """Inject tooltip CSS once per Streamlit rerun."""
    if _TOOLTIP_CSS_INJECTED not in st.session_state:
        st.session_state[_TOOLTIP_CSS_INJECTED] = True
    st.markdown("""<style>
.glossary-tip {
    position: relative;
    display: inline;
    border-bottom: 1px dashed rgba(0,240,255,0.5);
    color: #00f0ff;
    cursor: help;
    font-weight: 500;
}
.glossary-tip .glossary-tip-box {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    min-width: 280px;
    max-width: 380px;
    padding: 12px 14px;
    background: rgba(10,15,30,0.97);
    border: 1px solid rgba(0,240,255,0.25);
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,240,255,0.12), 0 0 1px rgba(0,240,255,0.3);
    color: #cbd5e1;
    font-size: 0.78rem;
    line-height: 1.5;
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    font-weight: 400;
    z-index: 9999;
    transition: opacity 0.2s ease, visibility 0.2s ease;
    text-align: left;
    white-space: normal;
}
.glossary-tip:hover .glossary-tip-box {
    visibility: visible;
    opacity: 1;
}
.glossary-tip-box .tip-title {
    color: #00f0ff;
    font-weight: 700;
    font-size: 0.82rem;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.glossary-tip-box .tip-en {
    color: #64748b;
    font-size: 0.7rem;
    margin-bottom: 6px;
}
.glossary-tip-box .tip-formula {
    background: rgba(0,240,255,0.06);
    border-left: 2px solid rgba(0,240,255,0.3);
    padding: 4px 8px;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8b5cf6;
}
</style>""", unsafe_allow_html=True)


def tooltip(term_key: str, display_text: str | None = None) -> str:
    """Return an HTML snippet with a hover tooltip for *term_key*.

    Args:
        term_key: Key in GLOSSARY (e.g. 'sharpe', 'mdd').
        display_text: Text shown inline. Defaults to the Chinese term name.

    Returns:
        HTML string safe for ``st.markdown(..., unsafe_allow_html=True)``.
        If *term_key* is not found, returns *display_text* as plain text.
    """
    entry = GLOSSARY.get(term_key)
    if entry is None:
        return display_text or term_key

    label = display_text or entry['zh']
    formula_html = ''
    if entry.get('formula'):
        formula_html = f'<div class="tip-formula">{entry["formula"]}</div>'

    return (
        f'<span class="glossary-tip">{label}'
        f'<span class="glossary-tip-box">'
        f'<div class="tip-title">{entry["zh"]}</div>'
        f'<div class="tip-en">{entry["en"]}</div>'
        f'<div>{entry["detail"]}</div>'
        f'{formula_html}'
        f'</span></span>'
    )


def render_tooltip_css():
    """Inject tooltip CSS. Call once at the top of any page that uses tooltip()."""
    _inject_tooltip_css()


# ---------------------------------------------------------------------------
# Sidebar glossary component
# ---------------------------------------------------------------------------
def render_glossary_sidebar():
    """Render a searchable glossary in the Streamlit sidebar."""
    _inject_tooltip_css()

    with st.sidebar:
        st.markdown(
            '<div style="font-family:JetBrains Mono,monospace; color:#00f0ff; '
            'font-size:0.85rem; letter-spacing:0.1em; margin-bottom:8px;">'
            '\U0001F4D6 GLOSSARY</div>',
            unsafe_allow_html=True,
        )
        query = st.text_input(
            'Search',
            placeholder='搜尋術語 (中/英文)...',
            key='_glossary_sidebar_search',
            label_visibility='collapsed',
        )

        q = query.strip().lower()
        matches = []
        for key, entry in GLOSSARY.items():
            if q and not any(q in field.lower() for field in [
                key, entry['zh'], entry['en'], entry['short'],
            ]):
                continue
            matches.append((key, entry))

        if not matches:
            st.caption('找不到符合的術語')
            return

        # Show at most 15 results in sidebar to avoid clutter
        for key, entry in matches[:15]:
            with st.expander(f"{entry['zh']}  {entry['en']}", expanded=False):
                st.markdown(
                    f"<div style='color:#94a3b8; font-size:0.8rem;'>{entry['short']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='color:#cbd5e1; font-size:0.78rem; margin-top:4px;'>"
                    f"{entry['detail']}</div>",
                    unsafe_allow_html=True,
                )
                if entry.get('formula'):
                    st.code(entry['formula'], language='text')

        if len(matches) > 15:
            st.caption(f'... 還有 {len(matches) - 15} 個結果')


# ---------------------------------------------------------------------------
# Full glossary page
# ---------------------------------------------------------------------------
def render_glossary_page():
    """Render a full glossary page with search bar and category filters."""
    _inject_tooltip_css()

    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace; font-size:1.3rem; '
        'color:#00f0ff; letter-spacing:0.12em; margin-bottom:4px;">'
        '\U0001F4D6 FINANCIAL GLOSSARY</div>'
        '<div style="color:#64748b; font-size:0.82rem; margin-bottom:16px;">'
        '量化交易與投資術語辭典</div>',
        unsafe_allow_html=True,
    )

    # -- Search + category filter row --
    col_search, col_cat = st.columns([3, 2])
    with col_search:
        query = st.text_input(
            'Search',
            placeholder='搜尋術語... (例: RSI, 停損, Sharpe)',
            key='_glossary_page_search',
            label_visibility='collapsed',
        )
    with col_cat:
        all_cats = ['全部'] + [CATEGORY_LABELS[c] for c in CATEGORY_LABELS]
        selected_label = st.selectbox(
            'Category',
            all_cats,
            key='_glossary_page_cat',
            label_visibility='collapsed',
        )

    # Reverse-map label to key
    selected_cat = None
    if selected_label != '全部':
        for k, v in CATEGORY_LABELS.items():
            if v == selected_label:
                selected_cat = k
                break

    q = query.strip().lower()

    # Collect matches grouped by category
    grouped: dict[str, list[tuple[str, dict]]] = {}
    total = 0
    for key, entry in GLOSSARY.items():
        # Category filter
        if selected_cat and entry.get('category') != selected_cat:
            continue
        # Text search
        if q and not any(q in field.lower() for field in [
            key, entry['zh'], entry['en'], entry['short'], entry.get('detail', ''),
        ]):
            continue
        cat = entry.get('category', 'other')
        grouped.setdefault(cat, []).append((key, entry))
        total += 1

    if total == 0:
        st.info('找不到符合條件的術語，請嘗試其他搜尋關鍵字。')
        return

    st.caption(f'共 {total} 個術語')

    # Render by category
    cat_order = list(CATEGORY_LABELS.keys())
    for cat in cat_order:
        items = grouped.get(cat)
        if not items:
            continue
        icon = CATEGORY_ICONS.get(cat, '')
        label = CATEGORY_LABELS.get(cat, cat)
        st.markdown(
            f'<div style="margin-top:20px; margin-bottom:8px; font-family:JetBrains Mono,monospace; '
            f'font-size:0.95rem; color:#00f0ff; letter-spacing:0.08em;">'
            f'{icon} {label} ({len(items)})</div>',
            unsafe_allow_html=True,
        )

        for key, entry in items:
            with st.expander(f"{entry['zh']}  |  {entry['en']}", expanded=False):
                # Short description
                st.markdown(
                    f"<div style='color:#f1f5f9; font-size:0.85rem; font-weight:600; "
                    f"margin-bottom:6px;'>{entry['short']}</div>",
                    unsafe_allow_html=True,
                )
                # Detailed explanation
                st.markdown(
                    f"<div style='color:#cbd5e1; font-size:0.8rem; line-height:1.6;'>"
                    f"{entry['detail']}</div>",
                    unsafe_allow_html=True,
                )
                # Formula
                if entry.get('formula'):
                    st.markdown(
                        f"<div style='margin-top:8px; padding:6px 10px; "
                        f"background:rgba(0,240,255,0.04); border-left:3px solid rgba(0,240,255,0.3); "
                        f"font-family:JetBrains Mono,monospace; font-size:0.78rem; color:#8b5cf6;'>"
                        f"{entry['formula']}</div>",
                        unsafe_allow_html=True,
                    )
