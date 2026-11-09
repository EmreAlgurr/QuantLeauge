"""
Quant Arbitrage & Pairs Trading Engine
"""
from .cointegration import test_cointegration, calculate_half_life, calculate_hurst_exponent
from .kalman_filter import KalmanFilterHedgeRatio
from .pairs_trading_bot import PairsTradingStrategy, StrategyParameters
from .backtester import Backtester, BacktestConfig
from .data_pipeline import fetch_historical_pair, generate_synthetic_cointegrated_pair

