"""Tests for analysis/stock_report.py -- AI 戰情室核心聚合服務."""
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from analysis.stock_report import (
    generate_stock_report,
    format_report_summary,
    _report_cache,
    _CACHE_TTL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_provider(df, stock_info=None):
    """Build a mock provider returning *df* for historical data."""
    provider = MagicMock()
    provider.get_historical_data.return_value = df
    provider.get_stock_info.return_value = stock_info or {
        "name": "TestCo", "pe": 15.0, "eps": 5.0, "yield": 2.1,
    }
    return provider


def _enter_patches(df, stock_info=None):
    """Start all patches and return (started_mocks, patch_objects) for cleanup."""
    provider = _make_mock_provider(df, stock_info)

    patch_defs = [
        patch("data.provider.get_data_provider", return_value=provider),
        patch("analysis.ai_core.analyze_signals", return_value={"overall": "neutral"}),
        patch("analysis.breakout.detect_levels", return_value={}),
        patch("analysis.breakout.detect_signal", return_value={"signal": None}),
        patch("analysis.thesis.generate_thesis", return_value={"composite_score": 7.0}),
        patch("ui.widgets.risk_warnings.generate_stock_warnings",
              return_value=[{"title": "RSI 偏高", "level": "warn"}]),
        patch("analysis.breakout.detect_vcp", return_value={"is_vcp": False}),
        patch("analysis.patterns.detect_candlestick_patterns", return_value=[]),
        patch("analysis.breakout.SIGNAL_TYPES", {}),
    ]

    started = [p.start() for p in patch_defs]
    return started, patch_defs


def _stop_patches(patch_objects):
    for p in patch_objects:
        p.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateStockReport:
    """Tests for generate_stock_report()."""

    def setup_method(self):
        """Clear cache before each test."""
        _report_cache.clear()

    # 1
    def test_generate_report_returns_all_sections(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            report = generate_stock_report("2330", market_type="TW")
        finally:
            _stop_patches(patches)

        expected_keys = {
            "ticker", "name", "market", "timestamp",
            "price_info", "technical", "fundamental",
            "ai_signals", "ai_analysis", "thesis",
            "risk_warnings", "strategy_signals",
        }
        assert expected_keys.issubset(report.keys())

    # 2
    def test_generate_report_invalid_ticker(self):
        """When the provider raises on get_historical_data, report should still return."""
        provider = MagicMock()
        provider.get_historical_data.side_effect = Exception("ticker not found")
        provider.get_stock_info.side_effect = Exception("ticker not found")

        with patch("data.provider.get_data_provider", return_value=provider):
            report = generate_stock_report("INVALID_TICKER_XYZ")

        assert report["ticker"] == "INVALID_TICKER_XYZ"
        # Sections that depend on data should be None
        assert report["price_info"] is None
        assert report["technical"] is None

    # 3
    def test_generate_report_price_info(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            report = generate_stock_report("2330")
        finally:
            _stop_patches(patches)

        pi = report["price_info"]
        assert pi is not None
        for field in ("price", "prev_close", "change", "change_pct", "volume"):
            assert field in pi
        assert isinstance(pi["price"], float)

    # 4
    def test_generate_report_technical_section(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            report = generate_stock_report("2330")
        finally:
            _stop_patches(patches)

        tech = report["technical"]
        assert tech is not None
        for key in ("rsi", "macd", "adx"):
            assert key in tech

    # 5
    def test_generate_report_without_gemini(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            report = generate_stock_report("2330", gemini_client=None)
        finally:
            _stop_patches(patches)

        assert report["ai_analysis"] is None

    # 6
    def test_report_cache_hit(self, sample_ohlcv_df):
        provider = _make_mock_provider(sample_ohlcv_df)
        p = patch("data.provider.get_data_provider", return_value=provider)
        mock_gdp = p.start()
        mocks, extra_patches = _enter_patches(sample_ohlcv_df)
        # The first _enter_patches also patched get_data_provider, so stop it
        # and use our explicit one. Simpler: just use _enter_patches directly.
        _stop_patches(extra_patches)
        p.stop()

        # Redo cleanly
        _report_cache.clear()
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            r1 = generate_stock_report("CACHE_TEST")
            r2 = generate_stock_report("CACHE_TEST")
        finally:
            _stop_patches(patches)

        # mocks[0] is the get_data_provider mock
        assert mocks[0].call_count == 1
        assert r1 is r2

    # 7
    def test_report_cache_miss_after_expiry(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            r1 = generate_stock_report("EXPIRY_TEST")

            # Simulate cache expiry
            for key in list(_report_cache.keys()):
                _report_cache[key]["_cached_at"] = time.time() - _CACHE_TTL - 10

            r2 = generate_stock_report("EXPIRY_TEST")
        finally:
            _stop_patches(patches)

        # Provider called twice: once original, once after expiry
        assert mocks[0].call_count == 2

    # 8
    def test_format_report_summary(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            report = generate_stock_report("2330")
        finally:
            _stop_patches(patches)

        summary = format_report_summary(report)
        assert isinstance(summary, str)
        assert "2330" in summary
        assert "RSI" in summary
        assert "MACD" in summary

    # 9
    def test_generate_report_risk_warnings(self, sample_ohlcv_df):
        mocks, patches = _enter_patches(sample_ohlcv_df)
        try:
            report = generate_stock_report("2330")
        finally:
            _stop_patches(patches)

        assert isinstance(report["risk_warnings"], list)

    # 10
    def test_generate_report_empty_df(self):
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        mocks, patches = _enter_patches(empty_df)
        try:
            report = generate_stock_report("9999")
        finally:
            _stop_patches(patches)

        assert report["ticker"] == "9999"
        assert report["price_info"] is None
        assert report["technical"] is None
