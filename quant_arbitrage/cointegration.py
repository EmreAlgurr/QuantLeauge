"""
Cointegration & Mean-Reversion Statistical Tools
------------------------------------------------
Implements:
- Augmented Dickey-Fuller (ADF) Unit Root Test
- Engle-Granger Two-Step Cointegration Test
- Hurst Exponent Calculation (H < 0.5 => Mean Reverting)
- Ornstein-Uhlenbeck Mean-Reversion Half-Life Estimation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def calculate_hurst_exponent(time_series: np.ndarray, max_lags: int = 100) -> float:
    """
    Computes the Hurst Exponent to determine long-term memory of a time series.
    - H < 0.5: Mean-reverting series (ideal for statistical arbitrage)
    - H = 0.5: Geometric Brownian motion (random walk)
    - H > 0.5: Trending / persistent series
    """
    ts = np.asarray(time_series, dtype=float)
    lags = range(2, min(max_lags, len(ts) // 4))
    
    # Calculate variance of lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Log-log linear regression slope
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0] * 2.0)


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimates the half-life of mean reversion using an Ornstein-Uhlenbeck process:
    d(Spread_t) = theta * (mu - Spread_{t-1}) * dt + sigma * dW_t
    
    Regression: delta_S = lambda * S_{t-1} + alpha
    Half-life = -ln(2) / lambda
    """
    spread_clean = spread.dropna()
    lagged_spread = spread_clean.shift(1).dropna()
    delta_spread = (spread_clean - lagged_spread).dropna()
    
    # Align indices
    idx = lagged_spread.index.intersection(delta_spread.index)
    x = lagged_spread.loc[idx].values
    y = delta_spread.loc[idx].values
    
    # OLS regression of delta_spread on lagged_spread
    x_with_const = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    lambda_val = beta[1]
    
    if lambda_val >= 0:
        return float("inf") # Non-mean-reverting
        
    half_life = -np.log(2.0) / lambda_val
    return float(half_life)


def test_cointegration(series_y: pd.Series, series_x: pd.Series) -> dict:
    """
    Runs Engle-Granger two-step cointegration test between Asset Y and Asset X.
    Returns:
    - score: t-statistic of residual ADF test
    - pvalue: Cointegration p-value (p < 0.05 indicates stationarity)
    - hedge_ratio: Static OLS beta
    - is_cointegrated: Boolean flag
    - half_life: Half-life in trading days
    - hurst: Hurst exponent of residual spread
    """
    # Align and drop missing
    df = pd.concat([series_y, series_x], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    
    score, pvalue, _ = coint(y, x)
    
    # Static OLS for initial hedge ratio
    x_const = np.column_stack([np.ones_like(x), x])
    ols_res = np.linalg.lstsq(x_const, y, rcond=None)[0]
    intercept = ols_res[0]
    hedge_ratio = ols_res[1]
    
    static_spread = pd.Series(y - (intercept + hedge_ratio * x), index=df.index)
    hl = calculate_half_life(static_spread)
    hurst = calculate_hurst_exponent(static_spread.values)
    
    return {
        "t_stat": float(score),
        "p_value": float(pvalue),
        "is_cointegrated": bool(pvalue < 0.05),
        "static_hedge_ratio": float(hedge_ratio),
        "static_intercept": float(intercept),
        "half_life_days": hl,
        "hurst_exponent": hurst
    }
