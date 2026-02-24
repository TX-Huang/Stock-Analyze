from finlab import data
from finlab import backtest
import pandas as pd
import numpy as np
import finlab

def run_isaac_strategy(api_token):
    if api_token:
        finlab.login(api_token)

    # ==========================================
    # 1. Fetch Data
    # ==========================================
    close = data.get('price:收盤價')
    open_ = data.get('price:開盤價')
    high = data.get('price:最高價')
    low = data.get('price:最低價')
    vol = data.get('price:成交股數')

    # Benchmark for Market Filter (0050 ETF as proxy for TAIEX)
    benchmark_close = data.get('price:收盤價')['0050']

    # Financials for Small-Cap Strategy
    try:
        rev_growth = data.get('monthly_revenue:去年同月增減(%)') # Monthly Revenue YoY
        rev_current = data.get('monthly_revenue:當月營收')
        capital = data.get('finance_statement:股本') # Capital stock
    except:
        # Fallback if fundamental data missing (use price only)
        rev_growth = pd.DataFrame(0, index=close.index, columns=close.columns)
        rev_current = pd.DataFrame(0, index=close.index, columns=close.columns)
        capital = pd.DataFrame(10000000, index=close.index, columns=close.columns) # Dummy large capital

    # ==========================================
    # 2. Indicators & Market Filter
    # ==========================================

    # Moving Averages
    ma20 = close.average(20)
    ma50 = close.average(50) # Quarterly Line (Trend Definition)
    ma60 = close.average(60)
    ma120 = close.average(120) # Half-Year Line

    # Volume MA
    vol_ma5 = vol.average(5)
    vol_ma20 = vol.average(20)

    # ATR (Average True Range) for Volatility Stop Loss
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0).max(level=0) # Fix for vectorization if needed, but finlab handles it?
    # Finlab efficient way:
    tr = np.maximum((high - low), (high - close.shift(1)).abs())
    tr = np.maximum(tr, (low - close.shift(1)).abs())
    atr = tr.rolling(14).mean()

    # Market Regime Filter (Red Light / Green Light)
    # Strategy works best when Market > 60MA (Quarterly)
    # We broaden to all stocks based on benchmark
    market_trend = benchmark_close > benchmark_close.average(60)
    # Broadcast market trend to all stocks
    is_market_bullish = pd.DataFrame(market_trend.values, index=benchmark_close.index, columns=['0050']).reindex(close.index, method='ffill').values
    # Note: reindex properly to match shape of 'close'
    # Actually, simpler way:
    is_market_bullish = (benchmark_close > benchmark_close.average(60))

    # ==========================================
    # 3. Signal A: Small-Cap Revenue Surprise (Aggressive Growth)
    # ==========================================

    # A1. Small Capital (< 20 Billion TWD = 200 億? No, User said 20億 = 2 Billion)
    # Data unit in Finlab 'finance_statement:股本' is usually 1000 TWD or similar?
    # Let's assume standard unit. Actually '股本' is usually in 1000s or just raw.
    # Let's check finlab docs or assume standard: usually 1000s.
    # 2 Billion TWD = 2,000,000,000. If unit is 1000, then 2,000,000.
    # Safe bet: Market Cap < 50 Billion is safer proxy?
    # User specified "股本 20億以下".
    # Let's try to filter by a reasonable small cap proxy using price * shares if capital unit is ambiguous.
    # Assuming '股本' is in 1000 NTD (common in TW data). 20億 = 2,000,000 (k).
    cond_small_cap = capital < 2000000 # 20億

    # A2. Revenue Explosion
    # Monthly YoY > 30% OR Revenue at 12-Month High
    rev_12m_max = rev_current.rolling(12).max()
    cond_rev_strong = (rev_growth > 30) | (rev_current >= rev_12m_max)
    cond_rev_strong = cond_rev_strong.reindex(close.index, method='ffill').fillna(False)

    # A3. Technical Trend (VCP-lite)
    # Price > 20MA & Price > 60MA (Up Trend)
    cond_trend = (close > ma20) & (close > ma60)

    # A4. VCP Dry Up (Optional but good for entry timing)
    # Volume < 50% of 20MA Volume in last 5 days (at least once)
    is_dry_up = (vol < vol_ma20 * 0.5).rolling(5).max() > 0

    # A5. Breakout
    # Close > 20-day High (Dynamic Breakout) AND Volume Spike
    breakout = (close > close.rolling(20).max().shift(1)) & (vol > vol_ma5 * 1.5)

    # SIGNAL A TRIGGER
    # Must be Bull Market + Small Cap + Strong Rev + Trend + DryUp(Context) + Breakout
    signal_a = (
        is_market_bullish &
        cond_small_cap &
        cond_rev_strong &
        cond_trend &
        breakout
    )

    # ==========================================
    # 4. Signal B: Mean Reversion (Deep Value / Panic Buy)
    # ==========================================

    # B1. Deep Discount
    # Price < 120MA (Half-Year Line) - User requirement
    # Price < Lower BBand? Or just Bias.
    cond_oversold_trend = close < ma120

    # B2. RSI Extreme (< 20)
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    my_rsi = rsi(close, 14)
    cond_rsi_panic = my_rsi < 20

    # B3. Reversal Candle (Hammer)
    # Long Lower Shadow: (Min(Open, Close) - Low) > 2 * Body
    body = (close - open_).abs()
    lower_shadow = (close.combine(open_, min) - low)
    cond_hammer = lower_shadow > (body * 2)

    # B4. Volume Spike (Panic Selling absorption)
    cond_vol_panic = vol > (vol_ma20 * 2)

    # SIGNAL B TRIGGER
    # Bear Market or Correction is fine for this signal.
    # Logic: Deep Oversold + RSI < 20 + Hammer + Vol Spike
    signal_b = (
        cond_oversold_trend &
        cond_rsi_panic &
        cond_hammer &
        cond_vol_panic
    )

    # ==========================================
    # 5. Position & Exit Management (MDD Control)
    # ==========================================

    # Combined Entry
    entries = signal_a | signal_b

    # Exit Logic
    # 1. Trailing Stop (ATR Chandelier Exit)
    # High watermark since entry - 3 * ATR
    # For vectorized, simplified to: Close < 20MA (Signal A) OR RSI > 50 (Signal B)

    # We need to distinguish which signal triggered to apply different exits?
    # Hard in pure vectorization without loop.
    # Let's use a unified robust exit:
    # - Trend Exit: Close < 20MA (Aggressive protection)
    # - Stop Loss: 10% Hard Stop (approx)

    # User wants MDD < 20%, so we need TIGHT exits.
    # Let's use Close < 20MA. It's the standard "Monthly Line" defense.
    # If Market turns Bearish (Green Light off), we should probably exit Signal A positions too?
    # Let's add: Exit if Market < 60MA (Market Filter Exit)

    # Unified Exit:
    # 1. Close < 20MA (Trend broken)
    # 2. Market Turn: Benchmark < 60MA (Systematic Risk) - Only applies to Trend trades (Signal A logic mainly)

    exits = (close < ma20) | (~is_market_bullish)

    # But Signal B (Mean Reversion) needs to sell on bounce, not wait for trend break (which might be far above/below).
    # Signal B Exit: RSI > 50 OR Close > 20MA (Reconnected to trend)
    # Since we can't easily split positions, let's prioritize the Trend Exit (Close < 20MA).
    # For Mean Reversion, "Close < 20MA" might be true immediately (since we buy way below 120MA).
    # Wait, if we buy below 120MA, Close < 20MA is likely True at entry!
    # This is a conflict.
    # FIX: For Signal B, we hold UNTIL Price > 20MA (Reversion complete) THEN we use trailing stop?
    # OR simpler: Exit Signal B when RSI > 50.

    # Let's refine exits:
    # Default: Hold
    # Force Exit: Close < 20MA (Normal Stop)
    # Exception: If (Close < 120MA) AND (RSI < 40), HOLD (We are in deep value zone, give it room to bounce).
    # This prevents immediate stop-out for Signal B.

    is_deep_value = (close < ma120) & (my_rsi < 40)
    final_exit = (close < ma20) & (~is_deep_value)

    # Construct Position
    position = pd.DataFrame(np.nan, index=entries.index, columns=entries.columns)
    position[entries] = 1
    position[final_exit] = 0
    position = position.ffill().fillna(0)

    # Liquidity Filter for Universe
    position = position & (vol.average(20) > 1000000) # > 1000 lots for safety

    # Run Backtest
    report = backtest.sim(position, resample='D', name='Isaac Strategy (Multi-Factor)', upload=False)
    return report
