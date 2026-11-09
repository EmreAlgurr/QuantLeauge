"""
Historical Multi-Asset Market Data Pipeline
-------------------------------------------
Downloads 10-year historical daily price series with offline synthetic generator fallback.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_historical_pair(
    ticker_y: str = "KO",
    ticker_x: str = "PEP",
    start_date: str = "2014-01-01",
    end_date: str = "2024-01-01",
    cache_dir: str = "data_cache"
) -> tuple[pd.Series, pd.Series]:
    """
    Fetches adjusted close prices for a pair across 10 years.
    Caches locally to prevent repeated network hits.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker_y}_{ticker_x}_{start_date}_{end_date}.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df[ticker_y], df[ticker_x]
        
    try:
        data = yf.download([ticker_y, ticker_x], start=start_date, end=end_date, progress=False)["Close"]
        if data.empty or ticker_y not in data.columns or ticker_x not in data.columns:
            raise ValueError("Empty response from yfinance")
            
        data = data.dropna()
        data.to_csv(cache_file)
        return data[ticker_y], data[ticker_x]
    except Exception as e:
        print(f"[WARN] yfinance fetch failed ({e}). Generating 10-year synthetic cointegrated dataset...")
        return generate_synthetic_cointegrated_pair(n_days=2520, start_date=start_date)


def generate_synthetic_cointegrated_pair(
    n_days: int = 2520, # ~10 years
    start_date: str = "2014-01-01",
    true_beta: float = 1.25,
    true_alpha: float = 5.0,
    half_life: int = 20,
    vol_x: float = 0.015,
    vol_spread: float = 0.5
) -> tuple[pd.Series, pd.Series]:
    """
    Generates realistic 10-year cointegrated pair using Ornstein-Uhlenbeck spread:
    X_t = Geometric Brownian Motion
    Spread_t = OU process (Mean Reverting)
    Y_t = true_alpha + true_beta * X_t + Spread_t
    """
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")
    
    # 1. Asset X: GBM
    rets_x = np.random.normal(loc=0.0003, scale=vol_x, size=n_days)
    price_x = 100.0 * np.exp(np.cumsum(rets_x))
    
    # 2. Spread: Discrete Ornstein-Uhlenbeck process
    # S_t = S_{t-1} + theta * (mu - S_{t-1}) + sigma * e_t
    theta = np.log(2.0) / half_life
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = spread[t - 1] - theta * spread[t - 1] + np.random.normal(0, vol_spread)
        
    # 3. Asset Y: Cointegrated with X
    price_y = true_alpha + true_beta * price_x + spread
    
    series_y = pd.Series(price_y, index=dates, name="Asset_Y")
    series_x = pd.Series(price_x, index=dates, name="Asset_X")
    return series_y, series_x
