"""
Dynamic Online Kalman Filter for Pairs Trading
----------------------------------------------
Implements a 2-State Online Kalman Filter to dynamically estimate:
- Time-varying hedge ratio: beta_t
- Dynamic intercept / alpha_t

Observation Equation:
    y_t = [1, x_t] * [alpha_t, beta_t]^T + v_t,   v_t ~ N(0, R)

State Transition Equation:
    [alpha_t, beta_t]^T = [alpha_{t-1}, beta_{t-1}]^T + w_t,   w_t ~ N(0, Q)

Eliminates lookahead bias inherent in static rolling OLS regressions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class KalmanFilterHedgeRatio:
    def __init__(self, delta: float = 1e-4, vt_variance: float = 1e-3):
        """
        Parameters:
        -----------
        delta : Process noise covariance scaling (transistion covariance Q).
                Higher delta allows faster adaptation to regime shifts; lower delta smooths.
        vt_variance : Initial measurement noise variance R.
        """
        self.delta = delta
        self.vt_variance = vt_variance
        
        # State: [alpha, beta]^T
        self.theta = np.zeros(2)
        
        # State covariance P
        self.P = np.zeros((2, 2))
        
        # Process noise covariance Q
        self.Q = (delta / (1.0 - delta)) * np.eye(2)
        
        # Measurement noise R
        self.R = vt_variance
        
        self.is_initialized = False

    def initialize(self, initial_alpha: float = 0.0, initial_beta: float = 1.0):
        self.theta = np.array([initial_alpha, initial_beta], dtype=float)
        self.P = np.eye(2) * 1.0
        self.is_initialized = True

    def update(self, y: float, x: float) -> tuple[float, float, float]:
        """
        Performs single-step Kalman predict-update.
        Parameters:
        -----------
        y : Current price of Asset Y
        x : Current price of Asset X
        
        Returns:
        --------
        (alpha_t, beta_t, error_t)
        """
        if not self.is_initialized:
            self.initialize(0.0, y / x if x != 0 else 1.0)
            
        # 1. State Prediction (A = Identity)
        # theta_{t|t-1} = theta_{t-1}
        # P_{t|t-1} = P_{t-1} + Q
        P_pred = self.P + self.Q
        
        # 2. Measurement Vector H = [1.0, x]
        H = np.array([1.0, x])
        
        # 3. Innovation (Measurement Residual)
        y_pred = np.dot(H, self.theta)
        error = y - y_pred
        
        # 4. Innovation Covariance S = H * P_pred * H^T + R
        S = float(np.dot(H, np.dot(P_pred, H))) + self.R
        
        # 5. Kalman Gain K = P_pred * H^T / S
        K = np.dot(P_pred, H) / S
        
        # 6. State Update: theta_t = theta_{t|t-1} + K * error
        self.theta = self.theta + K * error
        
        # 7. Covariance Update: P_t = (I - K * H) * P_{t|t-1}
        self.P = P_pred - np.outer(K, H).dot(P_pred)
        
        alpha_t = float(self.theta[0])
        beta_t = float(self.theta[1])
        return alpha_t, beta_t, float(error)

    def filter_series(self, series_y: pd.Series, series_x: pd.Series) -> pd.DataFrame:
        """
        Vectorized processing across time series history.
        Returns DataFrame with ['alpha', 'beta', 'spread', 'spread_std'].
        """
        df = pd.concat([series_y, series_x], axis=1).dropna()
        y_vals = df.iloc[:, 0].values
        x_vals = df.iloc[:, 1].values
        
        n = len(y_vals)
        alphas = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)
        
        for t in range(n):
            a_t, b_t, err_t = self.update(y_vals[t], x_vals[t])
            alphas[t] = a_t
            betas[t] = b_t
            # Dynamic spread = y - (alpha + beta * x)
            spreads[t] = y_vals[t] - (a_t + b_t * x_vals[t])
            
        result = pd.DataFrame(index=df.index)
        result["alpha"] = alphas
        result["beta"] = betas
        result["spread"] = spreads
        return result
