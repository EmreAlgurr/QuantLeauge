#!/usr/bin/env python3
"""
Statistical Arbitrage & Pairs Trading Engine CLI
------------------------------------------------
Executes 10-Year Historical Backtest with Kalman Filter Hedge Ratio & Cost Models.
"""

from __future__ import annotations
import sys
import os
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant_arbitrage.data_pipeline import fetch_historical_pair
from quant_arbitrage.cointegration import test_cointegration
from quant_arbitrage.pairs_trading_bot import PairsTradingStrategy, StrategyParameters
from quant_arbitrage.backtester import Backtester, BacktestConfig


def print_banner():
    print("=" * 72)
    print("   STATISTICAL ARBITRAGE & KALMAN FILTER PAIRS TRADING BOT")
    print("   10-Year Quantitative Backtesting & Execution Engine")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Statistical Arbitrage & Pairs Trading Backtester")
    parser.add_argument("--ticker_y", type=str, default="EWA", help="Dependent ticker (default: EWA)")
    parser.add_argument("--ticker_x", type=str, default="EWC", help="Independent ticker (default: EWC)")
    parser.add_argument("--start", type=str, default="2014-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--entry_z", type=float, default=1.8, help="Entry Z-score threshold")
    parser.add_argument("--exit_z", type=float, default=0.4, help="Exit Z-score threshold")
    parser.add_argument("--leverage", type=float, default=2.0, help="Margin leverage multiplier")
    args = parser.parse_args()

    print_banner()

    ticker_y = args.ticker_y
    ticker_x = args.ticker_x
    print(f"\n[1/4] Ingesting 10-Year Historical Market Data ({ticker_y} & {ticker_x})...")
    series_y, series_x = fetch_historical_pair(
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        start_date=args.start,
        end_date=args.end
    )
    print(f"      -> Ingested {len(series_y)} daily bars from {series_y.index[0].strftime('%Y-%m-%d')} to {series_y.index[-1].strftime('%Y-%m-%d')}.")

    print("\n[2/4] Statistical Cointegration Analysis...")
    coint_res = test_cointegration(series_y, series_x)
    print(f"      -> Engle-Granger ADF t-stat: {coint_res['t_stat']:.4f}")
    print(f"      -> Cointegration p-value:    {coint_res['p_value']:.6f} ({'COINTEGRATED (p < 0.05)' if coint_res['is_cointegrated'] else 'REJECTED (p >= 0.05)'})")
    print(f"      -> Static OLS Hedge Ratio:   {coint_res['static_hedge_ratio']:.4f}")
    print(f"      -> OU Half-Life of Spread:   {coint_res['half_life_days']:.1f} trading days")
    print(f"      -> Hurst Exponent (H):       {coint_res['hurst_exponent']:.4f} ({'Mean-Reverting H < 0.5' if coint_res['hurst_exponent'] < 0.5 else 'Trending'})")

    print("\n[3/4] Running Dynamic Online Kalman Filter & Signal Generation...")
    strategy = PairsTradingStrategy(StrategyParameters(
        lookback_window=40,
        entry_z_score=args.entry_z,
        exit_z_score=args.exit_z,
        stop_loss_z_score=3.5,
        delta_kalman=1e-4
    ))
    signals = strategy.generate_signals(series_y, series_x)
    print(f"      -> Generated {len(signals)} trading days of signals.")
    print(f"      -> Final Kalman Dynamic Beta: {signals['beta'].iloc[-1]:.4f}")

    print("\n[4/4] Executing Institutional Backtest with Frictions & Borrow Fees...")
    backtester = Backtester(BacktestConfig(
        initial_capital=100_000.0,
        leverage=args.leverage,
        commission_bps=3.0,        # 3 bps institutional commission
        slippage_bps=2.0,          # 2 bps bid-ask slippage
        annual_borrow_rate=0.015,  # 1.5% annual short borrow financing
        risk_free_rate=0.02        # 2% benchmark
    ))
    results = backtester.run(signals)

    print("\n" + "=" * 72)
    print(f"          PERFORMANCE TEARSHEET: {ticker_y} / {ticker_x} (10 YEARS)         ")
    print("=" * 72)
    print(f" Initial Portfolio Value  : ${results['initial_capital']:,.2f}")
    print(f" Final Portfolio Value    : ${results['final_equity']:,.2f}")
    print(f" Cumulative Net Return    : {results['total_return_pct']:+.2f}%")
    print(f" Annualized Return (CAGR) : {results['cagr_pct']:+.2f}%")
    print(f" Annualized Volatility    : {results['annualized_volatility_pct']:.2f}%")
    print(f" Sharpe Ratio (Rf = 2%)   : {results['sharpe_ratio']:.2f}")
    print(f" Sortino Ratio            : {results['sortino_ratio']:.2f}")
    print(f" Maximum Drawdown (MDD)   : {results['max_drawdown_pct']:.2f}%")
    print(f" Calmar Ratio             : {results['calmar_ratio']:.2f}")
    print(f" Total Completed Trades   : {results['total_trades']}")
    print(f" Trade Win Rate           : {results['win_rate_pct']:.1f}%")
    print(f" Profit Factor            : {results['profit_factor']:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
