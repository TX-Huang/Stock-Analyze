"""進階風險計算 — ATR 停損、Beta、VaR、壓力測試。"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─── ATR-Based Stop Loss ───

def calculate_atr(df, period=14):
    """計算 Average True Range。"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def atr_stop_price(df, multiplier=2.0, period=14):
    """
    計算 ATR-based 動態停損價。

    Args:
        df: OHLCV DataFrame
        multiplier: ATR 倍數 (預設 2x)
        period: ATR 計算週期

    Returns:
        {
            'current_price': float,
            'atr': float,
            'atr_pct': float,           # ATR 佔現價比例
            'stop_price': float,         # 動態停損價 = 現價 - ATR * multiplier
            'stop_distance_pct': float,  # 停損距離 %
        }
    """
    if df is None or df.empty or len(df) < period + 1:
        return None

    atr = calculate_atr(df, period)
    current_atr = float(atr.iloc[-1])
    current_price = float(df['Close'].iloc[-1])

    if current_price <= 0 or np.isnan(current_atr):
        return None

    stop = current_price - current_atr * multiplier

    return {
        'current_price': current_price,
        'atr': current_atr,
        'atr_pct': current_atr / current_price * 100,
        'stop_price': stop,
        'stop_distance_pct': (current_price - stop) / current_price * 100,
    }


# ─── Portfolio Beta ───

def calculate_portfolio_beta(positions, provider, benchmark_ticker='^TWII', period='1y'):
    """
    計算組合 Beta (相對加權指數)。

    Returns:
        {
            'portfolio_beta': float,
            'individual_betas': dict,  # {ticker: beta}
            'interpretation': str,
        }
    """
    try:
        # Benchmark returns
        bench_df = provider.get_historical_data(benchmark_ticker, period=period, interval='1d')
        if bench_df is None or bench_df.empty:
            return None
        bench_ret = bench_df['Close'].pct_change().dropna()
    except Exception:
        return None

    betas = {}
    weights = {}
    total_value = 0

    for pos in positions:
        ticker = str(pos.get('ticker', ''))
        shares = pos.get('shares', 0)
        entry_price = pos.get('entry_price', 0)
        value = shares * entry_price
        total_value += value

        try:
            df = provider.get_historical_data(ticker, period=period, interval='1d')
            if df is None or df.empty or len(df) < 30:
                continue

            stock_ret = df['Close'].pct_change().dropna()

            # Align dates
            aligned = pd.DataFrame({'stock': stock_ret, 'bench': bench_ret}).dropna()
            if len(aligned) < 20:
                continue

            cov = np.cov(aligned['stock'].values, aligned['bench'].values)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

            betas[ticker] = round(float(beta), 3)
            weights[ticker] = value
        except Exception:
            continue

    if not betas or total_value <= 0:
        return None

    # Weighted portfolio beta
    port_beta = sum(betas[t] * weights[t] / total_value for t in betas if t in weights)

    if port_beta > 1.3:
        interp = '高 Beta — 組合波動大於大盤，上漲多賺但下跌也多虧'
    elif port_beta > 0.8:
        interp = '中性 Beta — 組合波動接近大盤'
    elif port_beta > 0.5:
        interp = '低 Beta — 組合較防守，波動小於大盤'
    else:
        interp = '極低 Beta — 組合與大盤關聯性低'

    return {
        'portfolio_beta': round(float(port_beta), 3),
        'individual_betas': betas,
        'interpretation': interp,
    }


# ─── Value at Risk (VaR) ───

def calculate_var(positions, provider, confidence=0.95, period='6mo', horizon_days=1):
    """
    歷史模擬法 VaR (Value at Risk)。

    Args:
        positions: list of position dicts
        provider: DataProvider
        confidence: 信心水準 (預設 95%)
        period: 歷史資料期間
        horizon_days: 持有天數

    Returns:
        {
            'var_amount': float,       # VaR 金額 (NTD)
            'var_pct': float,          # VaR 佔組合比例 %
            'cvar_amount': float,      # Conditional VaR (Expected Shortfall)
            'cvar_pct': float,
            'portfolio_value': float,
            'interpretation': str,
        }
    """
    returns_list = []
    total_value = 0

    for pos in positions:
        ticker = str(pos.get('ticker', ''))
        shares = pos.get('shares', 0)
        entry_price = pos.get('entry_price', 0)

        try:
            df = provider.get_historical_data(ticker, period=period, interval='1d')
            if df is None or df.empty or len(df) < 30:
                continue

            current_price = float(df['Close'].iloc[-1])
            position_value = current_price * shares
            total_value += position_value

            daily_ret = df['Close'].pct_change().dropna().values
            # Scale returns by position value
            position_returns = daily_ret * position_value

            if len(returns_list) == 0:
                returns_list = [np.zeros(len(daily_ret))]

            # Align length
            min_len = min(len(returns_list[0]), len(position_returns))
            if min_len > 0:
                returns_list[0] = returns_list[0][-min_len:]
                returns_list.append(position_returns[-min_len:])
        except Exception:
            continue

    if not returns_list or total_value <= 0:
        return None

    # Sum position-level returns for portfolio returns
    min_len = min(len(r) for r in returns_list)
    portfolio_returns = sum(r[-min_len:] for r in returns_list)

    # Multi-day horizon
    if horizon_days > 1:
        portfolio_returns = portfolio_returns * np.sqrt(horizon_days)

    # VaR = percentile of losses
    var_amount = float(np.percentile(portfolio_returns, (1 - confidence) * 100))
    var_pct = var_amount / total_value * 100

    # CVaR (Expected Shortfall) = average of losses beyond VaR
    tail_losses = portfolio_returns[portfolio_returns <= var_amount]
    cvar_amount = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_amount
    cvar_pct = cvar_amount / total_value * 100

    if var_amount >= 0:
        interp = f'在 {confidence*100:.0f}% 信心水準下，即使最差情境仍預期獲利 {var_amount:,.0f} 元'
    else:
        interp = f'在 {confidence*100:.0f}% 信心水準下，單日最大預期虧損為 {abs(var_amount):,.0f} 元 ({abs(var_pct):.2f}%)'

    return {
        'var_amount': var_amount,
        'var_pct': var_pct,
        'cvar_amount': cvar_amount,
        'cvar_pct': cvar_pct,
        'portfolio_value': total_value,
        'interpretation': interp,
    }


# ─── Scenario Stress Test ───

def stress_test(positions, provider, scenarios=None):
    """
    情境壓力測試 — 模擬大盤不同跌幅對組合的影響。

    Args:
        positions: list of position dicts
        provider: DataProvider
        scenarios: list of market drop percentages (default: [-3, -5, -10, -15, -20])

    Returns:
        list of {scenario_pct, estimated_portfolio_loss, estimated_loss_pct}
    """
    if scenarios is None:
        scenarios = [-3, -5, -10, -15, -20]

    # Get individual betas
    beta_result = calculate_portfolio_beta(positions, provider)
    if beta_result is None:
        # Assume beta=1 for all
        individual_betas = {str(p.get('ticker', '')): 1.0 for p in positions}
    else:
        individual_betas = beta_result['individual_betas']

    total_value = 0
    pos_details = []
    for pos in positions:
        ticker = str(pos.get('ticker', ''))
        shares = pos.get('shares', 0)
        try:
            df = provider.get_historical_data(ticker, period='5d', interval='1d')
            price = float(df['Close'].iloc[-1]) if df is not None and not df.empty else pos.get('entry_price', 0)
        except Exception:
            price = pos.get('entry_price', 0)
        value = price * shares
        total_value += value
        beta = individual_betas.get(ticker, 1.0)
        pos_details.append({'ticker': ticker, 'value': value, 'beta': beta})

    results = []
    for scenario_pct in scenarios:
        total_loss = 0
        for p in pos_details:
            # Stock expected move = beta * market move
            stock_move = p['beta'] * (scenario_pct / 100)
            position_loss = p['value'] * stock_move
            total_loss += position_loss

        loss_pct = total_loss / total_value * 100 if total_value > 0 else 0
        results.append({
            'scenario': f'大盤 {scenario_pct}%',
            'scenario_pct': scenario_pct,
            'estimated_loss': total_loss,
            'estimated_loss_pct': loss_pct,
        })

    return results
