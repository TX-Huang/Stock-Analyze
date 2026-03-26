"""Tests for analysis.cost_analysis — fee parameter forwarding."""

import pandas as pd
import pytest
from analysis.cost_analysis import (
    COMMISSION_RATE_DISCOUNTED,
    TAX_RATE_STOCK,
    SLIPPAGE_PER_SIDE,
    _estimate_trade_cost,
    analyze_trading_costs,
)


# ─── _estimate_trade_cost ───

class TestEstimateTradeCost:

    def test_default_rates(self):
        cost = _estimate_trade_cost(100.0, 110.0)
        assert cost['total_cost'] > 0
        assert cost['cost_pct'] > 0

    def test_custom_commission(self):
        default = _estimate_trade_cost(100.0, 110.0)
        custom = _estimate_trade_cost(100.0, 110.0, commission_rate=0.001)
        # Higher commission → higher total cost
        assert custom['commission_buy'] > default['commission_buy']

    def test_custom_tax(self):
        default = _estimate_trade_cost(100.0, 110.0)
        zero_tax = _estimate_trade_cost(100.0, 110.0, tax_rate=0.0)
        assert zero_tax['tax'] == 0.0
        assert zero_tax['total_cost'] < default['total_cost']

    def test_custom_slippage(self):
        zero_slip = _estimate_trade_cost(100.0, 110.0, slippage_rate=0.0)
        assert zero_slip['slippage'] == 0.0

    def test_zero_price_returns_zero(self):
        cost = _estimate_trade_cost(0, 100.0)
        assert cost['total_cost'] == 0.0

    def test_negative_price_returns_zero(self):
        cost = _estimate_trade_cost(-10, 100.0)
        assert cost['total_cost'] == 0.0


# ─── analyze_trading_costs — fee forwarding ───

class TestAnalyzeTradingCostsForwarding:

    @pytest.fixture
    def sample_trades(self):
        return pd.DataFrame({
            'entry_price': [100.0, 200.0],
            'exit_price': [110.0, 190.0],
            'stock_id': ['2330', '2317'],
            'return': [0.10, -0.05],
            'period': [20, 15],
        })

    def test_default_params(self, sample_trades):
        result = analyze_trading_costs(sample_trades)
        assert result['total_trades'] == 2
        assert result['total_cost'] > 0

    def test_custom_commission_forwarded(self, sample_trades):
        default = analyze_trading_costs(sample_trades)
        high_comm = analyze_trading_costs(
            sample_trades, commission_rate=0.001425
        )
        # Full-price commission should cost more than default discounted
        assert high_comm['total_commission'] > default['total_commission']

    def test_zero_tax_forwarded(self, sample_trades):
        result = analyze_trading_costs(sample_trades, tax_rate=0.0)
        assert result['total_tax'] == 0.0

    def test_zero_slippage_forwarded(self, sample_trades):
        result = analyze_trading_costs(sample_trades, slippage_rate=0.0)
        assert result['total_slippage'] == 0.0

    def test_all_custom_rates(self, sample_trades):
        result = analyze_trading_costs(
            sample_trades,
            commission_rate=0.001,
            tax_rate=0.005,
            slippage_rate=0.002,
        )
        assert result['total_trades'] == 2
        assert result['total_cost'] > 0

    def test_empty_df_returns_zeros(self):
        result = analyze_trading_costs(pd.DataFrame())
        assert result['total_trades'] == 0
        assert result['total_cost'] == 0.0

    def test_none_df_returns_zeros(self):
        result = analyze_trading_costs(None)
        assert result['total_trades'] == 0

    def test_backward_compat_no_optional_params(self, sample_trades):
        """Old callers without new params should still work."""
        result = analyze_trading_costs(sample_trades, capital=500_000)
        assert result['total_trades'] == 2
