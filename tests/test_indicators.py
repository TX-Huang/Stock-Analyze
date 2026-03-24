"""Tests for analysis/indicators.py -- all 17 indicator functions."""
import numpy as np
import pandas as pd
import pytest

from analysis.indicators import (
    calculate_ema,
    calculate_tema,
    calculate_rsi,
    calculate_macd,
    calculate_bbands,
    calculate_stoch,
    calculate_obv,
    calculate_cci,
    calculate_atr,
    calculate_williams_r,
    calculate_mfi,
    calculate_vwap,
    calculate_adx,
    calculate_ichimoku,
    calculate_supertrend,
    calculate_donchian,
    calculate_keltner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_series(value=100.0, n=100):
    """Return a constant-price Series for known-value testing."""
    return pd.Series([value] * n, dtype=float)


def _single_col_df(close_values, high=None, low=None, volume=None):
    """Build a minimal DataFrame from lists."""
    n = len(close_values)
    df = pd.DataFrame({
        "Open": close_values,
        "High": high if high is not None else close_values,
        "Low": low if low is not None else close_values,
        "Close": close_values,
        "Volume": volume if volume is not None else [1_000_000] * n,
    })
    return df


# ===================================================================
# EMA
# ===================================================================

class TestEMA:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_ema(sample_ohlcv_df["Close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_df)

    def test_flat_price_equals_price(self):
        s = _flat_series(50.0, 50)
        ema = calculate_ema(s, period=10)
        # For constant input, EMA should converge to the constant value
        assert abs(ema.iloc[-1] - 50.0) < 1e-6

    def test_single_column_df_accepted(self):
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
        result = calculate_ema(df, period=2)
        assert isinstance(result, pd.Series)

    def test_rejects_multi_column_df(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError):
            calculate_ema(df)


# ===================================================================
# TEMA
# ===================================================================

class TestTEMA:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_tema(sample_ohlcv_df["Close"])
        assert isinstance(result, pd.Series)

    def test_flat_price(self):
        s = _flat_series(80.0, 60)
        tema = calculate_tema(s, period=10)
        assert abs(tema.iloc[-1] - 80.0) < 1e-4


# ===================================================================
# RSI
# ===================================================================

class TestRSI:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_rsi(sample_ohlcv_df["Close"])
        assert isinstance(result, pd.Series)

    def test_range_0_100(self, sample_ohlcv_df):
        rsi = calculate_rsi(sample_ohlcv_df["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_flat_price_around_50(self):
        s = _flat_series(100.0, 50)
        rsi = calculate_rsi(s, period=14)
        # No gains or losses -> RSI should be 50
        assert abs(rsi.iloc[-1] - 50.0) < 1e-6

    def test_monotonic_up_near_100(self):
        s = pd.Series(range(1, 51), dtype=float)
        rsi = calculate_rsi(s, period=14)
        assert rsi.iloc[-1] > 90


# ===================================================================
# MACD
# ===================================================================

class TestMACD:
    def test_returns_three_series(self, sample_ohlcv_df):
        macd, signal, hist = calculate_macd(sample_ohlcv_df)
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_ohlcv_df)

    def test_histogram_equals_diff(self, sample_ohlcv_df):
        macd, signal, hist = calculate_macd(sample_ohlcv_df)
        np.testing.assert_allclose(hist.values, (macd - signal).values, atol=1e-10)

    def test_missing_close_column(self):
        df = pd.DataFrame({"Open": [1, 2, 3]})
        with pytest.raises(ValueError):
            calculate_macd(df)


# ===================================================================
# Bollinger Bands
# ===================================================================

class TestBBands:
    def test_returns_three_series(self, sample_ohlcv_df):
        upper, mid, lower = calculate_bbands(sample_ohlcv_df)
        assert isinstance(upper, pd.Series)
        assert isinstance(mid, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_upper_gte_lower(self, sample_ohlcv_df):
        upper, mid, lower = calculate_bbands(sample_ohlcv_df)
        valid = upper.dropna() >= lower.dropna()
        assert valid.all()

    def test_flat_price_bands_converge(self):
        df = _single_col_df([100.0] * 30)
        upper, mid, lower = calculate_bbands(df, length=10)
        # std = 0 for constant -> upper == mid == lower
        assert abs(upper.iloc[-1] - mid.iloc[-1]) < 1e-6


# ===================================================================
# Stochastic Oscillator
# ===================================================================

class TestStoch:
    def test_returns_two_series(self, sample_ohlcv_df):
        k, d = calculate_stoch(sample_ohlcv_df)
        assert isinstance(k, pd.Series) and isinstance(d, pd.Series)

    def test_range_0_100(self, sample_ohlcv_df):
        k, d = calculate_stoch(sample_ohlcv_df)
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Close": [1, 2]})
        with pytest.raises(ValueError):
            calculate_stoch(df)


# ===================================================================
# OBV
# ===================================================================

class TestOBV:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_obv(sample_ohlcv_df)
        assert isinstance(result, pd.Series)

    def test_monotonic_up_price_positive_obv(self):
        df = _single_col_df(
            list(range(10, 20)),
            volume=[1000] * 10,
        )
        obv = calculate_obv(df)
        # All up days -> OBV should be increasing
        assert obv.iloc[-1] > 0


# ===================================================================
# CCI
# ===================================================================

class TestCCI:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_cci(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(result, pd.Series)

    def test_flat_price_near_zero(self):
        n = 30
        s = _flat_series(100.0, n)
        cci = calculate_cci(s, s, s, period=10)
        # Constant TP -> CCI undefined (0/0), but should not raise
        assert len(cci) == n


# ===================================================================
# ATR
# ===================================================================

class TestATR:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(result, pd.Series)

    def test_first_value_nan(self, sample_ohlcv_df):
        atr = calculate_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert pd.isna(atr.iloc[0])

    def test_non_negative(self, sample_ohlcv_df):
        atr = calculate_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        valid = atr.dropna()
        assert (valid >= 0).all()


# ===================================================================
# Williams %R
# ===================================================================

class TestWilliamsR:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_williams_r(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(result, pd.Series)

    def test_range_neg100_to_0(self, sample_ohlcv_df):
        wr = calculate_williams_r(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        valid = wr.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()


# ===================================================================
# MFI
# ===================================================================

class TestMFI:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_mfi(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
            sample_ohlcv_df["Volume"],
        )
        assert isinstance(result, pd.Series)

    def test_range_0_100(self, sample_ohlcv_df):
        mfi = calculate_mfi(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
            sample_ohlcv_df["Volume"],
        )
        valid = mfi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ===================================================================
# VWAP
# ===================================================================

class TestVWAP:
    def test_returns_series(self, sample_ohlcv_df):
        result = calculate_vwap(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
            sample_ohlcv_df["Volume"],
        )
        assert isinstance(result, pd.Series)

    def test_flat_price_equals_price(self):
        n = 20
        s = _flat_series(50.0, n)
        vol = pd.Series([1000.0] * n)
        vwap = calculate_vwap(s, s, s, vol)
        assert abs(vwap.iloc[-1] - 50.0) < 1e-6


# ===================================================================
# ADX
# ===================================================================

class TestADX:
    def test_returns_three_series(self, sample_ohlcv_df):
        adx, plus_di, minus_di = calculate_adx(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(adx, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)

    def test_adx_non_negative(self, sample_ohlcv_df):
        adx, _, _ = calculate_adx(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        valid = adx.dropna()
        assert (valid >= 0).all()


# ===================================================================
# Ichimoku
# ===================================================================

class TestIchimoku:
    def test_returns_dict_with_keys(self, sample_ohlcv_df):
        result = calculate_ichimoku(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(result, dict)
        expected_keys = {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou"}
        assert set(result.keys()) == expected_keys

    def test_all_values_are_series(self, sample_ohlcv_df):
        result = calculate_ichimoku(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        for v in result.values():
            assert isinstance(v, pd.Series)


# ===================================================================
# SuperTrend
# ===================================================================

class TestSuperTrend:
    def test_returns_two_series(self, sample_ohlcv_df):
        st, dirn = calculate_supertrend(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(st, pd.Series)
        assert isinstance(dirn, pd.Series)

    def test_direction_values(self, sample_ohlcv_df):
        _, dirn = calculate_supertrend(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert set(dirn.unique()).issubset({1, -1})


# ===================================================================
# Donchian
# ===================================================================

class TestDonchian:
    def test_returns_three_series(self, sample_ohlcv_df):
        upper, lower, mid = calculate_donchian(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
        )
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        assert isinstance(mid, pd.Series)

    def test_upper_gte_lower(self, sample_ohlcv_df):
        upper, lower, _ = calculate_donchian(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
        )
        valid = (upper.dropna() >= lower.dropna())
        assert valid.all()


# ===================================================================
# Keltner Channel
# ===================================================================

class TestKeltner:
    def test_returns_three_series(self, sample_ohlcv_df):
        upper, lower, mid = calculate_keltner(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        assert isinstance(mid, pd.Series)

    def test_upper_gte_lower(self, sample_ohlcv_df):
        upper, lower, _ = calculate_keltner(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        valid_idx = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper.loc[valid_idx] >= lower.loc[valid_idx]).all()


# ===================================================================
# Edge cases applied across indicators
# ===================================================================

class TestEdgeCases:
    """Edge-case tests that apply to multiple indicators."""

    def test_empty_series_ema(self, sample_empty_df):
        result = calculate_ema(pd.Series([], dtype=float))
        assert len(result) == 0

    def test_single_row_rsi(self):
        s = pd.Series([100.0])
        rsi = calculate_rsi(s)
        # Single value -> no diff -> should still return a series of length 1
        assert len(rsi) == 1

    def test_all_nan_ema(self):
        s = pd.Series([np.nan] * 20)
        result = calculate_ema(s, period=5)
        assert result.isna().all()

    def test_type_error_on_non_series(self):
        with pytest.raises(TypeError):
            calculate_ema("not a series")

    def test_type_error_on_non_df_macd(self):
        with pytest.raises(TypeError):
            calculate_macd("not a df")
