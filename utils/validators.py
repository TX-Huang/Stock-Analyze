"""Input validation utilities for AI Invest HQ."""
import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Taiwan stock ticker: 4-6 digits, optionally followed by a single uppercase letter (e.g. 00631L)
_TW_TICKER_RE = re.compile(r'^\d{4,6}[A-Z]?$')
# US stock ticker: 1-5 uppercase letters, optionally with dots (e.g. BRK.B)
_US_TICKER_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

_VALID_PERIODS = {'1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'}


def validate_ticker(ticker: str, market: str = "TW") -> str:
    """Validate and normalize a stock ticker.

    Args:
        ticker: Raw ticker input
        market: "TW" for Taiwan stocks, "US" for US stocks

    Returns:
        Normalized ticker string

    Raises:
        ValueError: If ticker format is invalid
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"Ticker must be a non-empty string, got {ticker!r}")

    ticker = ticker.strip().upper()

    if market == "TW":
        # Strip common suffixes like .TW or .TWO
        ticker = re.sub(r'\.(TW|TWO)$', '', ticker)
        if not _TW_TICKER_RE.match(ticker):
            raise ValueError(
                f"Invalid TW ticker '{ticker}': expected 4-6 digits optionally "
                f"followed by a single letter (e.g. 2330, 00631L)"
            )
    elif market == "US":
        if not _US_TICKER_RE.match(ticker):
            raise ValueError(
                f"Invalid US ticker '{ticker}': expected 1-5 uppercase letters "
                f"(e.g. AAPL, BRK.B)"
            )
    else:
        raise ValueError(f"Unknown market '{market}': expected 'TW' or 'US'")

    logger.debug("Validated ticker: %s (market=%s)", ticker, market)
    return ticker


def validate_period(period: str) -> str:
    """Validate period string (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y).

    Args:
        period: Period string to validate

    Returns:
        Validated period string

    Raises:
        ValueError: If period is not in the accepted set
    """
    if not isinstance(period, str) or not period.strip():
        raise ValueError(f"Period must be a non-empty string, got {period!r}")

    period = period.strip().lower()

    if period not in _VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}': must be one of {sorted(_VALID_PERIODS)}"
        )

    logger.debug("Validated period: %s", period)
    return period


def validate_positive_float(val, name: str, min_val: float = 0.0, max_val: float = float('inf')) -> float:
    """Validate a positive float within range.

    Args:
        val: Value to validate (will be cast to float)
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive, default 0.0)
        max_val: Maximum allowed value (inclusive, default inf)

    Returns:
        Validated float value

    Raises:
        ValueError: If value is not a valid positive float in range
    """
    try:
        val = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number, got {val!r}")

    if pd.isna(val):
        raise ValueError(f"'{name}' must not be NaN")

    if val < min_val or val > max_val:
        raise ValueError(
            f"'{name}' must be between {min_val} and {max_val}, got {val}"
        )

    logger.debug("Validated %s: %s", name, val)
    return val


def validate_dataframe(df: pd.DataFrame, required_cols: list, min_rows: int = 1) -> pd.DataFrame:
    """Validate a DataFrame has required columns and minimum rows.

    Args:
        df: DataFrame to validate
        required_cols: List of column names that must be present
        min_rows: Minimum number of rows required (default 1)

    Returns:
        The validated DataFrame (unchanged)

    Raises:
        TypeError: If df is not a DataFrame
        ValueError: If required columns are missing or row count is insufficient
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame has {len(df)} rows, but at least {min_rows} required"
        )

    logger.debug(
        "Validated DataFrame: %d rows, columns %s present",
        len(df), required_cols,
    )
    return df


def validate_json_schema(data: dict, required_keys: list) -> dict:
    """Validate a dict has all required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of keys that must be present

    Returns:
        The validated dict (unchanged)

    Raises:
        TypeError: If data is not a dict
        ValueError: If required keys are missing
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(
            f"Dict is missing required keys: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    logger.debug("Validated dict schema: required keys %s present", required_keys)
    return data
