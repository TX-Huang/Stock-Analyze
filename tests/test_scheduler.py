"""Tests for scheduler.py -- run_daily_market_scan and argument parsing."""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from scheduler import run_daily_market_scan, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_scan_result():
    """Return a realistic scan_single_stock_deep result dict."""
    return {
        "signal": "breakout",
        "score": 7.5,
        "indicators": {"rsi": 62, "macd": "bullish"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunDailyMarketScan:
    """Tests for run_daily_market_scan()."""

    # 1
    def test_run_daily_market_scan_writes_json(self, tmp_path):
        """Verify scan writes results to the configured SCAN_RESULTS_PATH."""
        output_path = str(tmp_path / "scan_results.json")

        with patch("scheduler._get_watchlist_tickers", return_value=[
            {"ticker": "2330", "name": "TSMC", "source": "watchlist"},
        ]), patch("scheduler._get_portfolio_tickers", return_value=[]), \
             patch("data.scanner.scan_single_stock_deep", return_value=_mock_scan_result()), \
             patch("scheduler.SCAN_RESULTS_PATH", output_path):
            run_daily_market_scan()

        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "results" in data
        assert data["results_count"] == 1

    # 2
    def test_run_daily_market_scan_handles_error(self, tmp_path):
        """Verify graceful error handling when scan_single_stock_deep raises."""
        output_path = str(tmp_path / "scan_results.json")

        with patch("scheduler._get_watchlist_tickers", return_value=[
            {"ticker": "FAIL", "name": "FailCo", "source": "watchlist"},
        ]), patch("scheduler._get_portfolio_tickers", return_value=[]), \
             patch("data.scanner.scan_single_stock_deep", side_effect=Exception("API down")), \
             patch("scheduler.SCAN_RESULTS_PATH", output_path):
            run_daily_market_scan()

        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["results_count"] == 0

    # 3
    def test_scheduler_standalone_mode(self):
        """Verify --once --force arguments are correctly parsed and passed."""
        with patch("scheduler.run_scan") as mock_run_scan, \
             patch("sys.argv", ["scheduler.py", "--once", "--force"]):
            main()
        mock_run_scan.assert_called_once_with(notify=False, force=True)

    # 4
    def test_scan_results_format(self, tmp_path):
        """Verify the output JSON has expected top-level structure."""
        output_path = str(tmp_path / "scan_results.json")

        with patch("scheduler._get_watchlist_tickers", return_value=[
            {"ticker": "2330", "name": "TSMC", "source": "watchlist"},
            {"ticker": "2454", "name": "MediaTek", "source": "watchlist"},
        ]), patch("scheduler._get_portfolio_tickers", return_value=[]), \
             patch("data.scanner.scan_single_stock_deep", return_value=_mock_scan_result()), \
             patch("scheduler.SCAN_RESULTS_PATH", output_path):
            run_daily_market_scan()

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ("scan_time", "scheduler", "total_scanned", "results_count", "results"):
            assert key in data, f"Missing key: {key}"

        assert data["scheduler"] == "apscheduler"
        assert data["total_scanned"] == 2
        assert isinstance(data["results"], list)

        for entry in data["results"]:
            assert "ticker" in entry
            assert "name" in entry
            assert "scan" in entry
