"""Tests for ui/backtest_dashboard.py helper functions."""
import numpy as np
import pandas as pd
import pytest

from ui.backtest_dashboard import _build_yearly_table, _build_position_changes


# ── _build_yearly_table ──────────────────────────────────────────


class TestBuildYearlyTable:

    def test_none_equity(self):
        result = _build_yearly_table(None, pd.DataFrame())
        assert result.empty

    def test_empty_equity(self):
        result = _build_yearly_table(pd.Series(dtype=float), pd.DataFrame())
        assert result.empty

    def test_basic(self):
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        equity = pd.Series(np.linspace(1.0, 1.2, 50), index=dates)
        trades = pd.DataFrame({
            "entry_date": ["2024-01-05", "2024-02-01"],
            "return": [0.05, -0.02],
            "period": [10, 5],
            "mae": [-0.03, -0.05],
            "gmfe": [0.08, 0.01],
        })

        result = _build_yearly_table(equity, trades)
        assert len(result) == 1
        assert result.iloc[0]["年度"] == 2024
        assert result.iloc[0]["交易數"] == 2
        assert result.iloc[0]["年報酬%"] == pytest.approx(20.0, abs=0.1)
        assert result.iloc[0]["勝率%"] == 50.0
        assert pd.notna(result.iloc[0]["MAE%"])
        assert pd.notna(result.iloc[0]["MFE%"])

    def test_no_trades_for_year(self):
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        equity = pd.Series(np.linspace(1.0, 1.1, 50), index=dates)
        trades = pd.DataFrame(columns=["entry_date", "return", "period"])

        result = _build_yearly_table(equity, trades)
        assert len(result) == 1
        assert result.iloc[0]["交易數"] == 0
        assert result.iloc[0]["平均報酬%"] is None
        assert result.iloc[0]["勝率%"] is None

    def test_missing_mae_mfe_columns(self):
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        equity = pd.Series(np.linspace(1.0, 1.1, 50), index=dates)
        trades = pd.DataFrame({
            "entry_date": ["2024-01-10"],
            "return": [0.05],
            "period": [10],
        })

        result = _build_yearly_table(equity, trades)
        assert result.iloc[0]["MAE%"] is None
        assert result.iloc[0]["MFE%"] is None

    def test_single_day_sharpe_is_nan(self):
        dates = pd.DatetimeIndex(["2024-01-02"])
        equity = pd.Series([1.0], index=dates)
        trades = pd.DataFrame(columns=["entry_date", "return", "period"])

        result = _build_yearly_table(equity, trades)
        assert len(result) == 1
        assert np.isnan(result.iloc[0]["Sharpe"])


# ── _build_position_changes ──────────────────────────────────────


class TestBuildPositionChanges:

    def test_none_position(self):
        result = _build_position_changes(None)
        assert result.empty

    def test_empty_position(self):
        result = _build_position_changes(pd.DataFrame())
        assert result.empty

    def test_enter(self):
        dates = pd.bdate_range("2024-01-02", periods=2, freq="B")
        position = pd.DataFrame(
            {"2330": [0, 1], "2454": [0, 1]},
            index=dates,
        )

        result = _build_position_changes(position)
        assert len(result) == 1
        assert result.iloc[0]["異動"] == "新進場"
        assert "2330" in result.iloc[0]["明細"]
        assert result.iloc[0]["持倉數"] == 2

    def test_exit(self):
        dates = pd.bdate_range("2024-01-02", periods=2, freq="B")
        position = pd.DataFrame(
            {"2330": [1, 0], "2454": [1, 0]},
            index=dates,
        )

        result = _build_position_changes(position)
        # First row: enter, second row: exit
        assert len(result) == 2
        exit_row = result[result["異動"] == "出場"]
        assert len(exit_row) == 1
        assert exit_row.iloc[0]["持倉數"] == 0

    def test_replace(self):
        dates = pd.bdate_range("2024-01-02", periods=2, freq="B")
        position = pd.DataFrame(
            {"2330": [1, 0], "2454": [0, 1]},
            index=dates,
        )

        result = _build_position_changes(position)
        # Day 1: 2330 enters; Day 2: 2330 exits + 2454 enters = replace
        assert len(result) == 2
        replace_row = result[result["異動"] == "替換"]
        assert len(replace_row) == 1
        assert "OUT" in replace_row.iloc[0]["明細"]
        assert "IN" in replace_row.iloc[0]["明細"]
