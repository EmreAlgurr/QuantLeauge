"""
simulation_engine.py — Offline Backtesting Engine
===================================================
Part of the Decentralized Algorithmic Trading Arena MVP.

This module generates mock hourly crypto price data, runs a Moving Average
Crossover strategy against it, and returns a clean result dictionary that
can later be posted on-chain via the TradingLeaderboard contract.

Design notes
------------
* Every data-related function accepts / returns a pandas DataFrame with a
  canonical schema (timestamp, open, high, low, close, volume) so you can
  swap `generate_mock_data()` for `load_csv_data("historical.csv")` with
  zero changes to the strategy layer.
* PnL is expressed as a percentage scaled by 100 (e.g. 1234 = 12.34 %)
  to match the int256 representation used in the Solidity contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# 1. DATA LAYER -- mock generation & CSV loader
# ---------------------------------------------------------------------------

def generate_mock_data(
    days: int = 30,
    freq: str = "h",
    seed: int = 42,
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """
    Create a realistic-looking hourly OHLCV DataFrame.

    Parameters
    ----------
    days : int
        Number of calendar days to simulate.
    freq : str
        Pandas frequency alias (default ``"h"`` = hourly).
    seed : int
        Random seed for reproducibility.
    start_price : float
        Opening price of the first candle.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)
    periods = days * 24  # hours in the requested window

    # --- timestamps ----------------------------------------------------------
    timestamps = pd.date_range(
        start="2025-01-01", periods=periods, freq=freq, tz="UTC"
    )

    # --- price path via geometric Brownian motion ----------------------------
    # mu (drift) and sigma (volatility) are annualised, then scaled to hourly.
    mu_annual = 0.0                       # neutral drift for fair backtests
    sigma_annual = 0.60                   # ~60 % annual vol - typical crypto
    dt = 1 / (365 * 24)                   # fraction of a year per bar

    log_returns = rng.normal(
        loc=mu_annual * dt,
        scale=sigma_annual * np.sqrt(dt),
        size=periods,
    )
    cumulative = np.exp(np.cumsum(log_returns))
    close_prices = start_price * cumulative

    # --- derive OHLV from the close path ------------------------------------
    noise = lambda shape: rng.uniform(0.001, 0.006, size=shape)  # noqa: E731

    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price

    high_prices = np.maximum(open_prices, close_prices) * (1 + noise(periods))
    low_prices = np.minimum(open_prices, close_prices) * (1 - noise(periods))

    volume = rng.integers(100, 5_000, size=periods).astype(float)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": volume,
        }
    )
    return df


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load historical OHLCV data from a CSV file.

    The CSV must contain at least the columns:
        timestamp, open, high, low, close, volume

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the canonical schema.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. STRATEGY LAYER -- Moving Average Crossover
# ---------------------------------------------------------------------------

def run_moving_average_strategy(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 30,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Simulate a simple SMA crossover strategy.

    Rules
    -----
    * **Buy signal** : fast SMA crosses *above* slow SMA  ->  go long (hold).
    * **Sell signal**: fast SMA crosses *below* slow SMA  ->  exit to cash.
    * Positions are fully invested (no partial sizing).
    * No short selling, no leverage, no transaction costs (MVP simplicity).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with at least a ``close`` column.
    fast_window : int
        Look-back period for the fast moving average.
    slow_window : int
        Look-back period for the slow moving average.
    initial_capital : float
        Starting portfolio value in USD.

    Returns
    -------
    dict
        ``algorithm_name``  - human-readable strategy identifier.
        ``pnl_percentage``  - total return scaled by 100 (int, for on-chain use).
        ``final_capital``   - ending portfolio value.
        ``total_trades``    - number of round-trip trades executed.
        ``win_rate``        - fraction of profitable trades (0.0-1.0).
        ``max_drawdown_pct``- worst peak-to-trough decline as a percentage.
    """
    # ── validate ────────────────────────────────────────────────────────────
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if len(df) < slow_window:
        raise ValueError(
            f"Need at least {slow_window} rows; got {len(df)}."
        )

    # ── indicator computation ───────────────────────────────────────────────
    # Work on a copy to avoid mutating the caller's data.
    data = df.copy()
    data["sma_fast"] = data["close"].rolling(window=fast_window).mean()
    data["sma_slow"] = data["close"].rolling(window=slow_window).mean()

    # ── signal generation ───────────────────────────────────────────────────
    # signal = 1 when fast > slow (bullish), 0 otherwise.
    data["signal"] = 0
    data.loc[data["sma_fast"] > data["sma_slow"], "signal"] = 1

    # position changes: +1 = entry, -1 = exit
    data["position_change"] = data["signal"].diff().fillna(0).astype(int)

    # ── trade simulation (bar-by-bar) ───────────────────────────────────────
    capital = initial_capital
    shares: float = 0.0
    in_position = False

    trades: list[dict] = []           # record each closed trade
    entry_price: float = 0.0
    equity_curve: list[float] = []    # for drawdown calculation

    for idx in range(len(data)):
        row = data.iloc[idx]
        price = row["close"]
        change = row["position_change"]

        # — entry ─────────────────────────────────────────────────────────
        if change == 1 and not in_position:
            shares = capital / price
            entry_price = price
            capital = 0.0
            in_position = True

        # — exit ──────────────────────────────────────────────────────────
        elif change == -1 and in_position:
            capital = shares * price
            pnl = (price - entry_price) / entry_price
            trades.append({"entry": entry_price, "exit": price, "pnl": pnl})
            shares = 0.0
            in_position = False

        # — mark-to-market equity ─────────────────────────────────────────
        current_equity = capital + shares * price
        equity_curve.append(current_equity)

    # If still holding at the end of the data, liquidate at last close.
    if in_position:
        last_price = data["close"].iloc[-1]
        capital = shares * last_price
        pnl = (last_price - entry_price) / entry_price
        trades.append({"entry": entry_price, "exit": last_price, "pnl": pnl})
        shares = 0.0
        equity_curve[-1] = capital

    final_capital = capital

    # ── performance metrics ─────────────────────────────────────────────────
    pnl_pct = ((final_capital - initial_capital) / initial_capital) * 100
    pnl_scaled = int(round(pnl_pct * 100))  # scale by 100 for Solidity int256

    total_trades = len(trades)
    winning = [t for t in trades if t["pnl"] > 0]
    win_rate = len(winning) / total_trades if total_trades > 0 else 0.0

    # Max drawdown from the equity curve
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown_pct = round(float(drawdown.min()) * 100, 2)

    algorithm_name = f"SMA-Crossover-{fast_window}x{slow_window}"

    return {
        "algorithm_name": algorithm_name,
        "pnl_percentage": pnl_scaled,       # int, scaled x100 for on-chain
        "final_capital": round(final_capital, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "max_drawdown_pct": max_drawdown_pct,
    }


# ---------------------------------------------------------------------------
# 3. RUNNER -- convenience wrapper
# ---------------------------------------------------------------------------

def run_simulation(
    csv_path: Optional[str] = None,
    days: int = 30,
    fast_window: int = 10,
    slow_window: int = 30,
    seed: int = 42,
) -> dict:
    """
    End-to-end pipeline: load data -> run strategy -> return result dict.

    Parameters
    ----------
    csv_path : str or None
        If provided, historical data is loaded from this CSV instead of
        generating mock data.
    days : int
        Number of days for mock data generation (ignored when csv_path is set).
    fast_window : int
        Fast SMA period.
    slow_window : int
        Slow SMA period.
    seed : int
        Random seed for mock data reproducibility.

    Returns
    -------
    dict
        Strategy result dictionary (see ``run_moving_average_strategy``).
    """
    if csv_path:
        print(f"[Engine] Loading historical data from: {csv_path}")
        df = load_csv_data(csv_path)
    else:
        print(f"[Engine] Generating {days}-day mock hourly data (seed={seed})...")
        df = generate_mock_data(days=days, seed=seed)

    print(f"[Engine] Data shape: {df.shape[0]} bars x {df.shape[1]} columns")
    print(f"[Engine] Date range: {df['timestamp'].iloc[0]}  -->  {df['timestamp'].iloc[-1]}")
    print(f"[Engine] Running SMA Crossover ({fast_window}/{slow_window})...")

    result = run_moving_average_strategy(
        df,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    print("\n==================================================")
    print("           SIMULATION RESULTS")
    print("==================================================")
    print(f"  Strategy      : {result['algorithm_name']}")
    print(f"  PnL (%)       : {result['pnl_percentage'] / 100:.2f}%")
    print(f"  PnL (scaled)  : {result['pnl_percentage']}  (ready for on-chain)")
    print(f"  Final Capital : ${result['final_capital']:,.2f}")
    print(f"  Total Trades  : {result['total_trades']}")
    print(f"  Win Rate      : {result['win_rate'] * 100:.1f}%")
    print(f"  Max Drawdown  : {result['max_drawdown_pct']:.2f}%")
    print("==================================================\n")

    return result


# ---------------------------------------------------------------------------
# 4. MAIN -- run when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_simulation()
