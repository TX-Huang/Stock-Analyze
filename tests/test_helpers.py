"""Tests for utils/helpers.py."""
import json
import os
import tempfile

import pytest

from utils.helpers import safe_json_write, safe_json_read, robust_json_extract, validate_ticker


# ===================================================================
# safe_json_write / safe_json_read
# ===================================================================

class TestSafeJsonIO:
    def test_roundtrip_dict(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        safe_json_write(path, data)
        result = safe_json_read(path)
        assert result == data

    def test_roundtrip_list(self, tmp_path):
        path = str(tmp_path / "list.json")
        data = [1, "two", None, True]
        safe_json_write(path, data)
        assert safe_json_read(path) == data

    def test_unicode(self, tmp_path):
        path = str(tmp_path / "unicode.json")
        data = {"name": "台積電", "symbol": "2330"}
        safe_json_write(path, data)
        assert safe_json_read(path) == data

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "data.json")
        safe_json_write(path, {"ok": True})
        assert safe_json_read(path) == {"ok": True}

    def test_read_missing_returns_default(self, tmp_path):
        path = str(tmp_path / "nope.json")
        assert safe_json_read(path) is None
        assert safe_json_read(path, default=[]) == []

    def test_read_corrupt_returns_default(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        assert safe_json_read(path, default="fallback") == "fallback"

    def test_write_with_default_kwarg(self, tmp_path):
        """Test passing extra kwargs like default=str to json.dump."""
        from datetime import date
        path = str(tmp_path / "date.json")
        safe_json_write(path, {"d": date(2024, 1, 1)}, default=str)
        result = safe_json_read(path)
        assert result["d"] == "2024-01-01"


# ===================================================================
# robust_json_extract
# ===================================================================

class TestRobustJsonExtract:
    def test_valid_json_string(self):
        assert robust_json_extract('{"a": 1}') == {"a": 1}

    def test_valid_json_list(self):
        assert robust_json_extract('[1, 2, 3]') == [1, 2, 3]

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"score": 85, "grade": "A"} end.'
        result = robust_json_extract(text)
        assert result == {"score": 85, "grade": "A"}

    def test_no_json_returns_none(self):
        assert robust_json_extract("no json here") is None

    def test_empty_string(self):
        assert robust_json_extract("") is None

    def test_malformed_json(self):
        assert robust_json_extract("{'bad': 'single quotes'}") is None

    def test_nested_json(self):
        text = '```json\n{"outer": {"inner": [1, 2]}}\n```'
        result = robust_json_extract(text)
        assert result == {"outer": {"inner": [1, 2]}}


# ===================================================================
# validate_ticker (helpers.py version)
# ===================================================================

class TestHelperValidateTicker:
    @pytest.mark.parametrize("ticker,expected", [
        ("2330", True),
        ("00631L", True),
        ("2408", True),
        ("00878", True),
    ])
    def test_valid_tw_tickers(self, ticker, expected):
        assert validate_ticker(ticker, "台股") == expected

    @pytest.mark.parametrize("ticker,expected", [
        ("2330.TW", True),
        ("2330.TWO", True),
    ])
    def test_tw_with_suffix(self, ticker, expected):
        assert validate_ticker(ticker, "台股") == expected

    @pytest.mark.parametrize("ticker,expected", [
        ("AAPL", True),
        ("MSFT", True),
        ("A", True),
    ])
    def test_valid_us_tickers(self, ticker, expected):
        assert validate_ticker(ticker, "US") == expected

    @pytest.mark.parametrize("ticker,expected", [
        ("abc", False),   # lowercase (will be uppercased, but only letters for US)
        ("", False),
        ("12345678", False),
    ])
    def test_invalid_tw_tickers(self, ticker, expected):
        assert validate_ticker(ticker, "台股") == expected

    def test_invalid_us_ticker_numeric(self):
        assert validate_ticker("12345", "US") is False
