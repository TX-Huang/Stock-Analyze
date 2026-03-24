"""Tests for analysis/risk_calc.py."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from analysis.risk_calc import (
    calculate_atr,
    atr_stop_price,
    atr_position_size,
    calculate_var,
)


# ===================================================================
# calculate_atr (risk_calc version)
# ===================================================================

class TestRiskCalcATR:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_atr(sample_ohlcv_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_non_negative(self, sample_ohlcv_df):
        atr = calculate_atr(sample_ohlcv_df)
        valid = atr.dropna()
        assert (valid >= 0).all()

    def test_constant_price_atr_zero(self):
        """For constant OHLC, true range = 0 so ATR = 0."""
        n = 30
        df = pd.DataFrame({
            "Open": [100.0] * n,
            "High": [100.0] * n,
            "Low": [100.0] * n,
            "Close": [100.0] * n,
            "Volume": [1_000_000] * n,
        })
        atr = calculate_atr(df, period=14)
        # After warm-up, ATR should be zero for constant prices
        assert atr.iloc[-1] == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# atr_stop_price
# ===================================================================

class TestATRStopPrice:
    def test_returns_dict(self, sample_ohlcv_df):
        result = atr_stop_price(sample_ohlcv_df)
        assert isinstance(result, dict)
        assert "stop_price" in result
        assert "current_price" in result
        assert "atr" in result

    def test_stop_below_current_price(self, sample_ohlcv_df):
        result = atr_stop_price(sample_ohlcv_df, multiplier=2.0)
        assert result["stop_price"] < result["current_price"]

    def test_higher_multiplier_wider_stop(self, sample_ohlcv_df):
        r1 = atr_stop_price(sample_ohlcv_df, multiplier=1.0)
        r2 = atr_stop_price(sample_ohlcv_df, multiplier=3.0)
        assert r2["stop_distance_pct"] > r1["stop_distance_pct"]

    def test_none_on_empty(self, sample_empty_df):
        assert atr_stop_price(sample_empty_df) is None

    def test_none_on_none(self):
        assert atr_stop_price(None) is None

    def test_none_on_short_df(self, sample_short_df):
        # 5 rows < period (14) + 1
        assert atr_stop_price(sample_short_df, period=14) is None


# ===================================================================
# atr_position_size
# ===================================================================

class TestATRPositionSize:
    def test_returns_dict(self, sample_ohlcv_df):
        result = atr_position_size(sample_ohlcv_df, total_capital=1_000_000)
        assert isinstance(result, dict)
        assert "shares" in result
        assert "lots" in result
        assert "weight_pct" in result

    def test_higher_risk_more_shares(self, sample_ohlcv_df):
        r1 = atr_position_size(sample_ohlcv_df, total_capital=1_000_000, risk_per_trade=0.01)
        r2 = atr_position_size(sample_ohlcv_df, total_capital=1_000_000, risk_per_trade=0.04)
        assert r2["shares"] > r1["shares"]

    def test_none_on_zero_capital(self, sample_ohlcv_df):
        assert atr_position_size(sample_ohlcv_df, total_capital=0) is None

    def test_none_on_empty(self, sample_empty_df):
        assert atr_position_size(sample_empty_df, total_capital=1_000_000) is None

    def test_lots_is_shares_div_1000(self, sample_ohlcv_df):
        result = atr_position_size(sample_ohlcv_df, total_capital=10_000_000)
        assert result["lots"] == result["shares"] // 1000


# ===================================================================
# calculate_var
# ===================================================================

class TestCalculateVaR:
    def _make_provider(self, df):
        provider = MagicMock()
        provider.get_historical_data.return_value = df
        return provider

    def test_returns_dict(self, sample_ohlcv_df):
        positions = [{"ticker": "2330", "shares": 1000, "entry_price": 100}]
        provider = self._make_provider(sample_ohlcv_df)
        result = calculate_var(positions, provider)
        assert isinstance(result, dict)
        assert "var_amount" in result
        assert "cvar_amount" in result
        assert "portfolio_value" in result

    def test_var_is_negative_for_typical_portfolio(self, sample_ohlcv_df):
        """For a typical portfolio, 95% VaR should be negative (a loss)."""
        positions = [{"ticker": "2330", "shares": 1000, "entry_price": 100}]
        provider = self._make_provider(sample_ohlcv_df)
        result = calculate_var(positions, provider, confidence=0.95)
        # VaR represents a loss at the given confidence level
        assert result["var_amount"] < 0

    def test_cvar_lte_var(self, sample_ohlcv_df):
        """CVaR (expected shortfall) should be <= VaR (more negative)."""
        positions = [{"ticker": "2330", "shares": 1000, "entry_price": 100}]
        provider = self._make_provider(sample_ohlcv_df)
        result = calculate_var(positions, provider, confidence=0.95)
        assert result["cvar_amount"] <= result["var_amount"]

    def test_none_on_empty_positions(self, sample_ohlcv_df):
        provider = self._make_provider(sample_ohlcv_df)
        result = calculate_var([], provider)
        assert result is None

    def test_none_on_no_data(self):
        provider = MagicMock()
        provider.get_historical_data.return_value = None
        positions = [{"ticker": "2330", "shares": 1000, "entry_price": 100}]
        result = calculate_var(positions, provider)
        assert result is None

    def test_higher_confidence_larger_var(self, sample_ohlcv_df):
        """99% VaR should be more extreme (more negative) than 90% VaR."""
        positions = [{"ticker": "2330", "shares": 1000, "entry_price": 100}]
        provider = self._make_provider(sample_ohlcv_df)
        var_90 = calculate_var(positions, provider, confidence=0.90)
        var_99 = calculate_var(positions, provider, confidence=0.99)
        assert var_99["var_amount"] <= var_90["var_amount"]
