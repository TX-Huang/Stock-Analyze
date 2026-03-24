# AI Invest HQ - Claude 自主開發指引

## 專案概覽
Alpha Global 量化回測平台 — 整合 FinLab 台股回測 + AI 股市分析的 Streamlit 應用。

## 技術架構

### 核心檔案
| 路徑 | 用途 |
|------|------|
| `app.py` | Streamlit 主入口，4 大模組 UI |
| `strategies/isaac.py` | Isaac V3 策略（主力策略） |
| `strategies/vcp.py` | VCP 策略 |
| `data/provider.py` | 資料源抽象層（YFinance / SinoPac / FinLab） |
| `data/scanner.py` | 股市戰情室掃描器 |
| `analysis/leverage.py` | 槓桿 ETF 評估系統 |
| `analysis/indicators.py` | 技術指標計算 |
| `analysis/trend.py` | 趨勢判斷邏輯 |
| `analysis/patterns.py` | 型態辨識（K 線 / 幾何型態） |
| `analysis/ai_core.py` | AI 信號分析（Gemini API） |
| `ui/charts.py` | Plotly 圖表渲染 |
| `ui/components.py` | 自定義 Streamlit 組件 |
| `ui/backtest_dashboard.py` | 回測結果儀表板 |
| `config/settings.py` | 全域設定 |
| `utils/helpers.py` | 輔助工具函數 |

### API 金鑰（`.streamlit/secrets.toml`）
- `FINLAB_API_KEY` — FinLab 台股資料 API
- `SINOPAC_API_KEY` / `SINOPAC_SECRET_KEY` — 永豐金 Shioaji API
- `GEMINI_API_KEY` — Google Gemini AI
- `TELEGRAM_BOT_TOKEN` — Telegram Bot（需手動新增）
- `TELEGRAM_CHAT_ID` — Telegram 聊天 ID（需手動新增）

### Python 環境
- 使用專案內嵌的 `python_embed/python.exe`（Python 3.11）
- 執行指令前綴: `./python_embed/python.exe`
- 套件安裝: `./python_embed/python.exe -m pip install <pkg>`

## 回測系統使用指南

### 執行 Isaac 策略回測
```bash
cd "C:/Users/Dodo/Documents/AI Invest HQ"
./python_embed/python.exe -c "
import sys; sys.path.insert(0, '.')
import toml
secrets = toml.load('.streamlit/secrets.toml')
from strategies.isaac import run_isaac_strategy
report = run_isaac_strategy(secrets['FINLAB_API_KEY'])
trades = report.get_trades()
stats = report.get_stats()
print(f'Trades: {len(trades)}')
print(f'CAGR: {stats[\"cagr\"]*100:.2f}%')
print(f'Max DD: {stats[\"max_drawdown\"]*100:.2f}%')
print(f'Sharpe: {stats[\"daily_sharpe\"]:.2f}')
print(f'Win: {stats[\"win_ratio\"]*100:.1f}%')
print(f'Period: mean={trades[\"period\"].mean():.1f}, median={trades[\"period\"].median():.1f}')
"
```

### 關鍵回測參數（`strategies/isaac.py` 的 `sim_kwargs`）
```python
sim_kwargs = {
    # 不要設定 resample='D'! 會導致所有交易持有天數=1
    'name': 'Isaac V3',
    'upload': False,
    'trail_stop': 0.15,        # 追蹤停損 15%
    'position_limit': 0.10,    # 每檔上限 10%（= 1/MAX_CONCURRENT）
    'touched_exit': False,     # 使用收盤價判斷停損（非日內價）
    # 可選:
    # 'stop_loss': 0.08,       # 固定停損 8%
    # 'take_profit': 0.30,     # 停利 30%
}
```

### 已知陷阱（歷史 Bug）
1. **`resample='D'` 導致 period=1** — 絕對不要加 `resample='D'`，FinLab 會把每天當成獨立交易
2. **CategoricalIndex 對齊** — `safe_finlab_sim()` 會自動處理，不要手動轉 `.astype(str)`
3. **FinLab login 方式** — 使用 `finlab.login(api_token)`，不是 `data.login()`
4. **ETF 黑名單正則** — `r'^\d{4,6}[A-Z]?$'` 才能匹配 00631L 等代碼

## 策略自主優化流程

### 優化循環步驟
1. **執行回測** → 記錄基準指標（CAGR, Max DD, Sharpe, Win Ratio, Period）
2. **分析弱點** → 檢查 `finlab_debug.log` 的信號觸發統計
3. **提出假設** → 例如「放寬 RSI 門檻可增加交易次數」
4. **修改策略** → 編輯 `strategies/isaac.py` 的對應條件
5. **重新回測** → 比較前後指標
6. **通知結果** → 透過 Telegram 發送摘要報告
7. **回到步驟 2** → 如果指標改善，保留修改；否則回退

### 評估指標優先順序
| 優先序 | 指標 | 目標 | 說明 |
|--------|------|------|------|
| 1 | Max Drawdown | > -30% | 風控最重要 |
| 2 | CAGR | > 15% | 年化報酬 |
| 3 | Sharpe Ratio | > 0.8 | 風險調整後報酬 |
| 4 | Win Ratio | > 45% | 勝率 |
| 5 | Avg Period | 5-30 天 | 避免過短或過長 |
| 6 | Trade Count | 500-5000 | 統計有效性 |

### 可調參數清單
| 參數 | 位置 | 當前值 | 可調範圍 |
|------|------|--------|----------|
| `trail_stop` | sim_kwargs | 0.15 | 0.05 ~ 0.25 |
| `stop_loss` | sim_kwargs | None | 0.05 ~ 0.15 |
| `take_profit` | sim_kwargs | None | 0.15 ~ 0.50 |
| `MAX_CONCURRENT` | 策略內 | 10 | 5 ~ 20 |
| RSI 超賣門檻 | `c_rsi_panic` | < 30 | 20 ~ 40 |
| 量能放大倍率 | `c_breakout` | > 1.5x | 1.2x ~ 2.5x |
| MA 均線天數 | ma20/50/60/120 | 20/50/60/120 | 各 ±30% |
| HV 波動區間 | `hv_q80/q20` | P80/P20 | P70~P90 / P10~P30 |
| 供給區安全距離 | `c_supply_danger` | 0.95 | 0.90 ~ 0.98 |
| 流動性門檻 | `v_liq` | 50萬股 | 20萬 ~ 200萬 |

### 禁止修改的部分
- 不要修改 `data/provider.py` 的 `safe_finlab_sim` CategoricalIndex 對齊邏輯
- 不要加回 `resample='D'`
- 不要修改 `.streamlit/secrets.toml` 中的 API 金鑰
- 不要刪除 `finlab_debug.log` 的日誌輸出

## Telegram 通知系統

### 設定步驟
1. 在 Telegram 找 @BotFather，建立新 Bot 取得 `BOT_TOKEN`
2. 取得自己的 Chat ID（找 @userinfobot 或 @RawDataBot）
3. 將以下內容加入 `.streamlit/secrets.toml`:
```toml
TELEGRAM_BOT_TOKEN = "你的Bot Token"
TELEGRAM_CHAT_ID = "你的Chat ID"
```

### 使用通知工具
```python
from utils.notify import send_telegram, format_backtest_report
# 發送文字訊息
send_telegram("策略優化完成！")
# 發送回測報告
msg = format_backtest_report(stats, trades, version="V3.3")
send_telegram(msg)
```

## 開發慣例
- 所有文件使用 UTF-8 編碼
- UI 文字使用繁體中文
- 變數名使用英文
- 註解中英混合（技術術語用英文）
- 每次策略修改需增加版本號（V3 → V3.1 → V3.2 ...）
- 重大修改需在 `finlab_debug.log` 記錄

## 啟動應用
```bash
cd "C:/Users/Dodo/Documents/AI Invest HQ"
./python_embed/python.exe -m streamlit run app.py
```
