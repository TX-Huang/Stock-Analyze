# Alpha Global 量化回測平台

整合 FinLab 台股回測、AI 股市分析、Telegram 即時通知的 Streamlit 應用程式。
本機執行，使用內嵌 Python 3.11 環境。

## 快速啟動

### 環境需求
- Python 3.11（建議使用專案內嵌的 `python_embed/`）
- Windows 10/11

### 啟動方式

```bash
# Windows — 使用內嵌 Python
./python_embed/python.exe -m streamlit run app.py

# 或雙擊 run_windows.bat
```

### API 金鑰設定

在 `.streamlit/secrets.toml` 中設定：

```toml
FINLAB_API_KEY = "your-key"        # FinLab 台股資料
SINOPAC_API_KEY = "your-key"       # 永豐金 Shioaji 交易
SINOPAC_SECRET_KEY = "your-key"
GEMINI_API_KEY = "your-key"        # Google Gemini AI
TELEGRAM_BOT_TOKEN = "your-token"  # Telegram 通知
TELEGRAM_CHAT_ID = "your-chat-id"
```

## 系統架構

```
app.py                          # Streamlit 主入口（5 頁面架構）
├── ui/
│   ├── pages/                  # 15+ 頁面模組
│   │   ├── dashboard.py        #   主儀表板
│   │   ├── war_room.py         #   股市戰情室
│   │   ├── backtest.py         #   回測執行
│   │   ├── research.py         #   研究工具
│   │   ├── trading.py          #   交易執行
│   │   ├── lab.py              #   策略實驗室
│   │   ├── review.py           #   績效覆盤
│   │   ├── portfolio.py        #   投資組合分析
│   │   ├── live_monitor.py     #   即時持倉監控
│   │   ├── comparison.py       #   策略 A/B 比較
│   │   ├── leverage.py         #   槓桿 ETF 評估
│   │   ├── monte_carlo_panel.py#   蒙地卡羅模擬
│   │   ├── journal.py          #   交易日誌
│   │   ├── heatmap.py          #   市場熱力圖
│   │   └── alerts.py           #   警報管理
│   ├── backtest_dashboard.py   # 回測儀表板（5 tab）
│   ├── charts.py               # Plotly 圖表渲染
│   ├── components.py           # Cyber 主題組件庫
│   ├── theme.py                # 主題樣式
│   ├── stock_profile.py        # 個股分析頁
│   └── widgets/                # 互動小工具
│       ├── position_calculator.py
│       ├── risk_warnings.py
│       └── signal_explainer.py
│
├── strategies/                 # 交易策略
│   ├── isaac.py                # Isaac V3.7（主力策略）
│   ├── vcp.py                  # VCP 波動收縮
│   ├── will_vcp.py             # Will VCP 變體
│   ├── minervini.py            # Minervini 趨勢模板
│   ├── elder.py                # Elder 三重濾網
│   ├── edwards_magee.py        # Edwards & Magee 經典型態
│   ├── momentum.py             # 動量策略
│   ├── long_short.py           # 多空對沖
│   ├── candlestick.py          # K 線型態策略
│   └── wfo.py                  # Walk-Forward 最佳化
│
├── analysis/                   # 分析引擎
│   ├── ai_core.py              # Gemini AI 信號分析
│   ├── indicators.py           # 技術指標（MA/RSI/MACD/KD/布林）
│   ├── patterns.py             # 型態辨識（K 線 / 幾何型態）
│   ├── trend.py                # 趨勢判斷
│   ├── breakout.py             # 突破偵測
│   ├── risk_calc.py            # 風險計算（VaR/Beta/Sharpe/MDD）
│   ├── cost_analysis.py        # 交易成本分析
│   ├── leverage.py             # 槓桿 ETF 評估
│   ├── monte_carlo.py          # 蒙地卡羅模擬
│   ├── correlation.py          # 相關性分析
│   ├── sensitivity.py          # 參數敏感度分析
│   ├── attribution.py          # 績效歸因
│   ├── decay.py                # 時間衰減分析
│   ├── liquidity.py            # 流動性分析
│   ├── chip.py                 # 籌碼分析
│   └── stock_report.py         # 個股報告
│
├── data/                       # 資料層
│   ├── provider.py             # 資料源抽象（FinLab/SinoPac/YFinance）
│   ├── scanner.py              # 市場掃描器
│   ├── auto_trader.py          # 自動交易執行
│   ├── paper_trader.py         # 模擬交易
│   ├── risk_monitor.py         # 即時風控監控
│   ├── daily_recommender.py    # 每日推薦
│   ├── journal.py              # 交易日誌紀錄
│   ├── watchlist.py            # 自選股管理
│   └── alerts.py               # 警報系統
│
├── bot/                        # Telegram Bot
│   ├── telegram_bot.py         # Bot 主程式
│   └── handlers.py             # 指令處理器
│
├── config/                     # 設定
│   ├── settings.py             # 全域常數
│   ├── i18n.py                 # 多語言切換
│   └── locales/                # 翻譯檔（zh_TW/zh_CN/en）
│
├── utils/                      # 工具
│   ├── helpers.py              # 通用輔助函數
│   ├── notify.py               # Telegram 通知
│   ├── retry.py                # API 重試邏輯
│   ├── sandbox.py              # 安全沙箱執行
│   └── validators.py           # 資料驗證
│
├── tests/                      # 測試（pytest）
│   ├── conftest.py             # 共用 fixtures
│   └── test_*.py               # 11 個測試模組
│
└── scripts/                    # 研究腳本
    ├── backtest_comparison.py  # 策略 A/B 比較
    ├── wfo.py                  # Walk-Forward 最佳化
    └── experiments/            # 實驗性測試
```

## 核心功能

### 量化回測
- **Isaac V3.7** — 五大信號整合策略，CAGR 19.4%、MDD -30.9%、Sharpe 1.85
- **VCP / Minervini / Elder** — 經典策略書籍回測實作
- **回測儀表板** — 戰情室、資金曲線、逐年績效、持倉異動、交易明細

### AI 分析
- **Gemini API** — 個股 AI 診斷、信號解讀
- **型態辨識** — K 線型態 + 幾何型態自動偵測
- **蒙地卡羅模擬** — 風險情境分析

### 交易工具
- **股市戰情室** — 即時市場掃描 + 全球指數 ticker tape
- **模擬交易** — 紙上交易紀錄與追蹤
- **自動交易** — 永豐金 Shioaji API 串接
- **Telegram Bot** — 即時持倉異動通知、策略報告推送

### 多語言支援
- 繁體中文 / 簡體中文 / English

## 技術棧

| 技術 | 用途 |
|------|------|
| Streamlit | Web UI 框架 |
| FinLab | 台股歷史資料 + 回測引擎 |
| Plotly | 互動式圖表 |
| Pandas | 資料處理 |
| Google Gemini | AI 分析 |
| Shioaji | 永豐金證券 API |
| YFinance | 全球市場資料 |
| Telegram Bot API | 即時通知 |

## 測試

```bash
./python_embed/python.exe -m pytest tests/ -v
```

## 版本

- **v1.0.0** — 完整量化回測平台 + AI 分析 + Telegram 通知 + Cyber 主題 UI
