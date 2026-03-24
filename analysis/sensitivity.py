"""
Parameter sensitivity analysis (P2-5)
Sweep strategy parameters to analyze impact on performance metrics.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def single_param_sweep(
    run_strategy_fn,
    param_name: str,
    param_range: list,
    base_params: dict = None,
    metrics_fn=None,
) -> pd.DataFrame:
    """Sweep a single parameter and collect performance metrics.

    Args:
        run_strategy_fn: Callable that runs backtest given params dict, returns report
        param_name: Name of parameter to sweep
        param_range: List of values to test
        base_params: Base parameter dict (swept param overrides this)
        metrics_fn: Callable(report) -> dict of metrics. Default extracts standard stats.

    Returns:
        DataFrame with columns: [param_value, cagr, max_drawdown, sharpe, win_ratio, trade_count]
    """
    if base_params is None:
        base_params = {}

    if metrics_fn is None:
        metrics_fn = _default_metrics_extractor

    results = []
    total = len(param_range)

    for i, val in enumerate(param_range):
        logger.info(f"[Sensitivity] {param_name}={val} ({i+1}/{total})")
        params = {**base_params, param_name: val}

        try:
            report = run_strategy_fn(params)
            metrics = metrics_fn(report)
            metrics['param_value'] = val
            results.append(metrics)
        except Exception as e:
            logger.warning(f"[Sensitivity] {param_name}={val} failed: {e}")
            results.append({
                'param_value': val,
                'cagr': None,
                'max_drawdown': None,
                'sharpe': None,
                'win_ratio': None,
                'trade_count': None,
                'error': str(e),
            })

    return pd.DataFrame(results)


def dual_param_sweep(
    run_strategy_fn,
    param1_name: str,
    param1_range: list,
    param2_name: str,
    param2_range: list,
    base_params: dict = None,
    metric: str = 'sharpe',
    metrics_fn=None,
) -> pd.DataFrame:
    """Sweep two parameters and collect a target metric as a 2D grid.

    Args:
        run_strategy_fn: Callable that runs backtest
        param1_name, param1_range: First parameter
        param2_name, param2_range: Second parameter
        base_params: Base parameter dict
        metric: Which metric to track (e.g., 'sharpe', 'cagr')
        metrics_fn: Custom metrics extractor

    Returns:
        DataFrame with param1 as index, param2 as columns, metric as values
    """
    if base_params is None:
        base_params = {}

    if metrics_fn is None:
        metrics_fn = _default_metrics_extractor

    grid = np.full((len(param1_range), len(param2_range)), np.nan)
    total = len(param1_range) * len(param2_range)
    count = 0

    for i, v1 in enumerate(param1_range):
        for j, v2 in enumerate(param2_range):
            count += 1
            logger.info(f"[Sensitivity] {param1_name}={v1}, {param2_name}={v2} ({count}/{total})")
            params = {**base_params, param1_name: v1, param2_name: v2}

            try:
                report = run_strategy_fn(params)
                metrics = metrics_fn(report)
                grid[i, j] = metrics.get(metric, np.nan)
            except Exception as e:
                logger.warning(f"[Sensitivity] ({v1}, {v2}) failed: {e}")

    return pd.DataFrame(
        grid,
        index=pd.Index(param1_range, name=param1_name),
        columns=pd.Index(param2_range, name=param2_name),
    )


def _default_metrics_extractor(report) -> dict:
    """Extract standard metrics from a FinLab backtest report."""
    try:
        stats = report.get_stats()
        trades = report.get_trades()
        return {
            'cagr': float(stats.get('cagr', 0)) * 100,
            'max_drawdown': float(stats.get('max_drawdown', 0)) * 100,
            'sharpe': float(stats.get('daily_sharpe', 0)),
            'win_ratio': float(stats.get('win_ratio', 0)) * 100,
            'trade_count': len(trades) if trades is not None else 0,
            'avg_period': float(trades['period'].mean()) if trades is not None and len(trades) > 0 else 0,
        }
    except Exception as e:
        logger.warning(f"[Sensitivity] Metrics extraction failed: {e}")
        return {
            'cagr': None, 'max_drawdown': None, 'sharpe': None,
            'win_ratio': None, 'trade_count': None, 'avg_period': None,
        }


def generate_sweep_range(current_value: float, n_steps: int = 10, pct_range: float = 0.5) -> list:
    """Generate a parameter sweep range centered on the current value.

    Args:
        current_value: Current parameter value
        n_steps: Number of steps
        pct_range: Percentage range (0.5 = ±50%)

    Returns:
        List of evenly spaced values
    """
    low = current_value * (1 - pct_range)
    high = current_value * (1 + pct_range)
    return list(np.linspace(max(low, 0.001), high, n_steps).round(4))
