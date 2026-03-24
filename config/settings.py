"""Centralized configuration constants for AI Invest HQ."""

# === Gemini AI ===
GEMINI_MODEL = 'gemini-3.1-pro-preview'

DEFAULT_CHART_SETTINGS = {
    "trendline": True, "support": True, "gaps": True, "log_scale": False,
    "zigzag": False, "obv": True, "rectangle": True,
    "patterns": True, "ghost_lines": True,
    "volume_strict": True,
    "rounding": True, "fan": True, "wedge": True, "broadening": True,
    "diamond": True, "bbands": False, "macd": True, "kd": True,
    "structure": True,
    "ma": True, "candle_patterns": True,
}

# === Strategy Defaults (Isaac V3.7) ===
# Source: strategies/isaac.py run_isaac_strategy() default params
ISAAC_DEFAULTS = {
    'trail_stop': 0.18,             # 追蹤停損 18%
    'position_limit': 0.10,         # 每檔上限 10% (= 1/MAX_CONCURRENT_TOTAL)
    'max_concurrent_total': 10,     # 總部位上限
    'max_concurrent_long': 8,       # 多頭最多 8 檔
    'max_concurrent_short': 2,      # 空頭最多 2 檔
    'rsi_threshold': 28,            # Signal B RSI 超賣門檻
    'volume_mult': 1.5,             # 突破量能倍率
    'supply_danger_pct': 0.97,      # 供給區安全距離
    'liq_min': 500_000,             # 流動性門檻（均量股數）
    'min_score': 4,                 # 最低 score 門檻
}

# === Risk Defaults ===
# Source: data/risk_monitor.py _default_risk_config()
RISK_DEFAULTS = {
    'max_position_pct': 0.15,       # 單檔最大權重 15%
    'max_portfolio_dd_pct': 0.20,   # 組合最大回撤 20%
    'trail_stop_pct': 0.18,         # 追蹤停損 18%
    'max_loss_per_stock_pct': 0.15, # 個股最大虧損 15%
    'max_positions': 10,            # 最大持倉數
    'var_confidence': 0.95,         # VaR 信心水準 (analysis/risk_calc.py)
    'min_history_days': 30,         # Beta/VaR 最少歷史天數 (risk_calc.py)
    'daily_loss_alert_pct': 0.03,   # 單日虧損警報 3%
}

# === Trading Cost Defaults ===
# Source: analysis/cost_analysis.py, data/paper_trader.py, data/auto_trader.py
TRADING_DEFAULTS = {
    'slippage_per_side': 0.001,     # 估計滑價 0.1% per side
    'commission_rate': 0.001425,    # 法定手續費率 0.1425%
    'commission_discount': 0.35,    # 線上券商折扣 (3.5 折)
    'tax_rate_stock': 0.003,        # 證交稅 0.3% (一般股票，賣方)
    'tax_rate_etf_daytrade': 0.001, # 證交稅 0.1% (ETF 當沖)
}

# === Cache Settings ===
# Source: state.py, data/provider.py, ui/ various ttl= values
CACHE_TTL_COMPONENTS = 300          # 5 minutes  (ui/components.py, watchlist.py)
CACHE_TTL_SCANNER = 3600            # 1 hour     (data/scanner.py)
CACHE_TTL_THESIS = 1800             # 30 minutes (analysis/thesis.py)
CACHE_TTL_PROVIDER = 3600           # 1 hour     (data/provider.py SinoPacProvider._cache_ttl)
MAX_DATA_CACHE_SIZE = 10            # state.py MAX_CACHE_SIZE
SINOPAC_CACHE_MAX = 200             # data/provider.py SinoPacProvider._cache_max

# === API Settings ===
# Source: data/provider.py
SINOPAC_RECONNECT_INTERVAL = 4 * 3600  # 4 hours (SinoPacProvider._RECONNECT_INTERVAL)
PROVIDER_CACHE_MAX = 5                  # _PROVIDER_CACHE_MAX in data/provider.py

# === UI Settings ===
# Source: ui/charts.py
CHART_HEIGHT_COMPACT = 260          # render_position_chart default height
CHART_HEIGHT_FULL = 600             # render_trend_chart default height
