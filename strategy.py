from finlab import data
from finlab import backtest
import pandas as pd
import finlab

def run_strategy(api_token):
    if api_token:
        finlab.login(api_token)

    # 1. Fetch Data
    pe = data.get('price_earning_ratio:本益比')
    rev_growth = data.get('monthly_revenue:去年同月增減(%)')
    rev_mom = data.get('monthly_revenue:上月比較增減(%)')

    # 2. Define Strategy Conditions
    # P/E Ratio < 15 (Undervalued)
    cond1 = pe < 15

    # Monthly Revenue Growth (YoY) > 20% (High Growth)
    cond2 = rev_growth > 20

    # Monthly Revenue Growth (MoM) > 0 (Momentum)
    cond3 = rev_mom > 0

    # Combine conditions
    position = cond1 & cond2 & cond3

    # 3. Refine Selection (Optional: Limit number of stocks, e.g., Top 20 by something if needed)
    # For now, equal weight among all qualifying stocks is default behavior of backtest.sim

    # 4. Run Backtest
    # resample='M' means we rebalance monthly
    report = backtest.sim(position, resample='M', name='Optimized Growth Strategy', upload=False)

    return report

if __name__ == "__main__":
    print("Strategy module loaded. Call run_strategy(api_token) to execute.")
