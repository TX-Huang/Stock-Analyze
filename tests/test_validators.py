"""Tests for utils/validators.py."""
import math

import pandas as pd
import pytest

from utils.validators import (
    validate_ticker,
    validate_period,
    validate_positive_float,
    validate_dataframe,
    validate_json_schema,
)


# ===================================================================
# validate_ticker
# ===================================================================

class TestValidateTicker:
    @pytest.mark.parametrize("ticker,market,expected", [
        ("2330", "TW", "2330"),
        ("00631L", "TW", "00631L"),
        ("2330.TW", "TW", "2330"),
        ("2330.TWO", "TW", "2330"),
        ("  2408  ", "TW", "2408"),
    ])
    def test_valid_tw(self, ticker, market, expected):
        assert validate_ticker(ticker, market) == expected

    @pytest.mark.parametrize("ticker,market,expected", [
        ("AAPL", "US", "AAPL"),
        ("aapl", "US", "AAPL"),
        ("BRK.B", "US", "BRK.B"),
        ("A", "US", "A"),
    ])
    def test_valid_us(self, ticker, market, expected):
        assert validate_ticker(ticker, market) == expected

    @pytest.mark.parametrize("ticker,market", [
        ("", "TW"),
        ("  ", "TW"),
        ("ABC", "TW"),
        ("123", "TW"),
        ("1234567", "TW"),
    ])
    def test_invalid_tw_raises(self, ticker, market):
        with pytest.raises(ValueError):
            validate_ticker(ticker, market)

    @pytest.mark.parametrize("ticker,market", [
        ("12345", "US"),
        ("TOOLONGX", "US"),
        ("", "US"),
    ])
    def test_invalid_us_raises(self, ticker, market):
        with pytest.raises(ValueError):
            validate_ticker(ticker, market)

    def test_unknown_market_raises(self):
        with pytest.raises(ValueError, match="Unknown market"):
            validate_ticker("AAPL", "JP")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            validate_ticker(None, "TW")

    def test_integer_raises(self):
        with pytest.raises(ValueError):
            validate_ticker(2330, "TW")


# ===================================================================
# validate_period
# ===================================================================

class TestValidatePeriod:
    @pytest.mark.parametrize("period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"])
    def test_valid_periods(self, period):
        assert validate_period(period) == period

    def test_case_insensitive(self):
        assert validate_period("1Y") == "1y"

    def test_strips_whitespace(self):
        assert validate_period("  6mo  ") == "6mo"

    @pytest.mark.parametrize("period", ["1d", "1w", "3y", "", "abc"])
    def test_invalid_raises(self, period):
        with pytest.raises(ValueError):
            validate_period(period)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            validate_period(None)


# ===================================================================
# validate_positive_float
# ===================================================================

class TestValidatePositiveFloat:
    def test_normal_value(self):
        assert validate_positive_float(3.14, "pi") == 3.14

    def test_zero_allowed_by_default(self):
        assert validate_positive_float(0, "zero") == 0.0

    def test_string_number(self):
        assert validate_positive_float("2.5", "x") == 2.5

    def test_integer_cast(self):
        assert validate_positive_float(10, "n") == 10.0

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_positive_float(float("nan"), "x")

    def test_below_min_raises(self):
        with pytest.raises(ValueError):
            validate_positive_float(-1, "x", min_val=0.0)

    def test_above_max_raises(self):
        with pytest.raises(ValueError):
            validate_positive_float(200, "x", max_val=100)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_positive_float("abc", "x")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            validate_positive_float(None, "x")

    def test_boundary_values(self):
        assert validate_positive_float(5.0, "x", min_val=5.0, max_val=5.0) == 5.0


# ===================================================================
# validate_dataframe
# ===================================================================

class TestValidateDataframe:
    def test_valid_df(self, sample_ohlcv_df):
        result = validate_dataframe(sample_ohlcv_df, ["Open", "Close"])
        assert result is sample_ohlcv_df

    def test_missing_column_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(sample_ohlcv_df, ["Open", "NonExistent"])

    def test_too_few_rows_raises(self, sample_short_df):
        with pytest.raises(ValueError, match="at least"):
            validate_dataframe(sample_short_df, ["Close"], min_rows=100)

    def test_not_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            validate_dataframe([1, 2, 3], ["a"])

    def test_empty_df_raises(self, sample_empty_df):
        with pytest.raises(ValueError):
            validate_dataframe(sample_empty_df, ["Close"], min_rows=1)

    def test_min_rows_zero_ok(self, sample_empty_df):
        result = validate_dataframe(sample_empty_df, ["Close"], min_rows=0)
        assert len(result) == 0


# ===================================================================
# validate_json_schema
# ===================================================================

class TestValidateJsonSchema:
    def test_valid_dict(self):
        d = {"name": "test", "value": 42}
        assert validate_json_schema(d, ["name", "value"]) is d

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_json_schema({"a": 1}, ["a", "b"])

    def test_not_dict_raises(self):
        with pytest.raises(TypeError, match="dict"):
            validate_json_schema([1, 2], ["a"])

    def test_empty_required_ok(self):
        assert validate_json_schema({"x": 1}, []) == {"x": 1}

    def test_extra_keys_ok(self):
        d = {"a": 1, "b": 2, "c": 3}
        assert validate_json_schema(d, ["a"]) is d
