"""
Statistical Arbitrage & Pairs Trading Strategy Engine
------------------------------------------------------
Generates quantitative execution signals based on Kalman Filter spreads:
- Rolling Z-Score normalization: Z_t = (Spread_t - mu) / sigma
- Entry: |Z_t| >= entry_threshold (e.g. 2.0 std dev)
- Exit: |Z_t| <= exit_threshold (mean-reversion target, e.g. 0.5 std dev)
- Stop-Loss: |Z_t| >= stop_loss_threshold (structural break protection, e.g. 3.5 std dev)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .kalman_filter import KalmanFilterHedgeRatio


@dataclass
class StrategyParameters:
    lookback_window: int = 30
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    stop_loss_z_score: float = 3.5
    delta_kalman: float = 1e-4


class PairsTradingStrategy:
    def __init__(self, params: StrategyParameters = StrategyParameters()):
        self.params = params
        self.kalman = KalmanFilterHedgeRatio(delta=params.delta_kalman)

    def generate_signals(self, price_y: pd.Series, price_x: pd.Series) -> pd.DataFrame:
        """
        Runs Kalman filtering and computes trading positions:
         1: Long Spread  (Long Y, Short beta * X)
        -1: Short Spread (Short Y, Long beta * X)
         0: Flat / Neutral
        """
        df = pd.DataFrame({"y": price_y, "x": price_x}).dropna()
        
        # 1. Compute dynamic Kalman state
        kalman_df = self.kalman.filter_series(df["y"], df["x"])
        df["alpha"] = kalman_df["alpha"]
        df["beta"] = kalman_df["beta"]
        df["spread"] = kalman_df["spread"]
        
        # 2. Rolling Z-Score calculation
        rolling_mean = df["spread"].rolling(window=self.params.lookback_window).mean()
        rolling_std = df["spread"].rolling(window=self.params.lookback_window).std()
        
        df["z_score"] = (df["spread"] - rolling_mean) / (rolling_std + 1e-8)
        
        # 3. Position state machine
        positions = np.zeros(len(df))
        current_pos = 0 # 0: Flat, 1: Long spread, -1: Short spread
        
        z_vals = df["z_score"].values
        
        for i in range(len(df)):
            z = z_vals[i]
            if np.isnan(z):
                positions[i] = 0
                continue
                
            # If flat: check entry
            if current_pos == 0:
                if z <= -self.params.entry_z_score:
                    current_pos = 1 # Long Spread (undervalued Y, overvalued X)
                elif z >= self.params.entry_z_score:
                    current_pos = -1 # Short Spread (overvalued Y, undervalued X)
            # If in position: check exit or stop-loss
            elif current_pos == 1:
                # Stop loss if spread widens beyond threshold
                if z <= -self.params.stop_loss_z_score:
                    current_pos = 0 # Stop Loss
                # Mean-reversion take profit
                elif z >= -self.params.exit_z_score:
                    current_pos = 0
            elif current_pos == -1:
                # Stop loss
                if z >= self.params.stop_loss_z_score:
                    current_pos = 0 # Stop Loss
                # Mean-reversion take profit
                elif z <= self.params.exit_z_score:
                    current_pos = 0
                    
            positions[i] = current_pos
            
        df["position"] = positions
        df["position_shifted"] = df["position"].shift(1).fillna(0) # Next-bar execution to avoid lookahead
        return df
