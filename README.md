# 💎 Alpha Global 量化回測平台

歡迎使用 Alpha Global！這是一個整合了 **Finlab 量化回測** 與 **AI 股市分析** 的強大平台。
此版本設計為 **本機執行 (Local Execution)**，讓您可以利用自己的電腦效能進行大規模策略運算。

## 🚀 快速啟動 (Quick Start)

### ⚠️ 環境需求 (Prerequisites)
- **Python 版本**：強烈建議使用 **Python 3.11** 以確保 Finlab 套件的最佳穩定性。
- **作業系統**：Windows 10/11, macOS, 或 Linux。

### Windows 使用者
**方案 A：懶人免安裝 (推薦)**
1. 下載 [Python 3.11 Embeddable Package](https://www.python.org/downloads/release/python-3110/)。
2. 解壓縮，將資料夾改名為 `python_embed`，放入本專案目錄。
3. 雙擊 **`run_windows.bat`** 即可直接執行 (完全不影響您電腦的環境)。

**方案 B：使用系統 Python**
1. 確保電腦已安裝 Python 3.11。
2. 雙擊 **`run_windows.bat`**。
3. 程式將自動安裝套件並開啟瀏覽器。

### Mac / Linux 使用者
1. 打開終端機 (Terminal)。
2. 執行指令：
   ```bash
   chmod +x run_mac_linux.sh
   ./run_mac_linux.sh
   ```

---

## 📂 功能介紹

### 1. 📈 股市戰情室
- **個股分析**：輸入代碼 (如 2330) 查看趨勢、型態與 AI 診斷。
- **產業鏈搜尋**：輸入關鍵字 (如 伺服器) 尋找上下游供應鏈。

### 2. 🧬 量化回測系統
- **Isaac 頂級策略**：多空雙向全天候策略 (年化目標 >50%)。
- **VCP 波動收縮**：Minervini 經典與 SMC 結構融合。
- **自訂參數**：支援停損/停利壓力測試。

### 3. 📂 自訂策略實驗室
- **上傳策略**：將您的 Python 策略檔上傳，直接在本機回測。
- **範本下載**：提供 `template_strategy.py` 讓您快速上手。

---

## 🔧 常見問題

**Q: 為什麼第一次執行比較慢？**
A: 第一次需要下載 Finlab 的歷史數據，視網路速度可能需要幾分鐘。之後會有快取 (Cache)，速度會飛快。

**Q: 如何更新策略？**
A: 本平台支援熱重載 (Hot Reload)，您修改程式碼後，網頁重新整理即可生效。

---
*Happy Trading!*
