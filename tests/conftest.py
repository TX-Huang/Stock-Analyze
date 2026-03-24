"""Shared fixtures for AI Invest HQ test suite."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_ohlcv_df():
    """100-row OHLCV DataFrame with realistic price data starting 2024-01-01."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=100, freq="B")
    # Random walk for close prices starting at 100
    returns = np.random.normal(0.0005, 0.02, size=100)
    close = 100.0 * np.cumprod(1 + returns)
    # Derive OHLV from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    open_ = close * (1 + np.random.normal(0, 0.005, 100))
    volume = np.random.randint(100_000, 5_000_000, size=100).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    # Ensure High >= Close >= Low invariant
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


@pytest.fixture
def sample_short_df():
    """5-row OHLCV DataFrame for edge-case testing."""
    dates = pd.bdate_range("2024-06-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0, 103.0, 104.0],
            "High": [103.0, 104.0, 103.5, 105.0, 106.0],
            "Low": [99.0, 100.5, 99.5, 101.0, 102.0],
            "Close": [102.0, 101.0, 103.0, 104.0, 105.0],
            "Volume": [1_000_000, 1_200_000, 900_000, 1_500_000, 1_100_000],
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_empty_df():
    """Empty DataFrame with correct OHLCV columns."""
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


@pytest.fixture
def sample_trade_log():
    """List of trade dicts for portfolio-level tests."""
    return [
        {"ticker": "2330", "shares": 1000, "entry_price": 580.0, "side": "long"},
        {"ticker": "2454", "shares": 500, "entry_price": 1050.0, "side": "long"},
        {"ticker": "2317", "shares": 2000, "entry_price": 105.0, "side": "long"},
    ]


@pytest.fixture
def mock_provider(sample_ohlcv_df):
    """Mock data provider that returns sample_ohlcv_df for any ticker."""
    provider = MagicMock()
    provider.get_historical_data.return_value = sample_ohlcv_df
    return provider
