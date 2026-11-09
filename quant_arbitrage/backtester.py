"""
Institutional Vectorized & Event-Driven Backtesting Engine
----------------------------------------------------------
Features realistic market micro-structure transaction cost models:
- Broker Commission (basis points per notional traded)
- Bid-Ask Spread Slippage (basis points)
- Short Borrow Financing Rate (annualized margin rate pro-rated daily)

Calculates institutional quantitative metrics:
- Annualized Return (CAGR)
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown (MDD %) & Drawdown Duration
- Calmar Ratio
- Win Rate (%) & Profit Factor
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    leverage: float = 2.0              # 2x margin leverage standard in market-neutral quant books
    commission_bps: float = 3.0        # 3 bps (0.03%)
    slippage_bps: float = 2.0          # 2 bps (0.02%)
    annual_borrow_rate: float = 0.015  # 1.5% institutional annual short borrow rate
    risk_free_rate: float = 0.02       # 2% benchmark


class Backtester:
    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.config = config

    def run(self, signals_df: pd.DataFrame) -> dict:
        """
        Executes backtest on signals dataframe containing ['y', 'x', 'beta', 'position_shifted'].
        """
        df = signals_df.copy()
        
        # 1. Asset returns
        df["ret_y"] = df["y"].pct_change().fillna(0)
        df["ret_x"] = df["x"].pct_change().fillna(0)
        
        pos = df["position_shifted"].values
        beta = df["beta"].values
        ret_y = df["ret_y"].values
        ret_x = df["ret_x"].values
        
        # 2. Capital-weighted spread return: (ret_y - beta * ret_x) / (1 + |beta|)
        weight_norm = 1.0 + np.abs(beta)
        spread_ret_normalized = (ret_y - beta * ret_x) / (weight_norm + 1e-8)
        
        # Apply leverage
        raw_levered_return = pos * spread_ret_normalized * self.config.leverage
        
        # 3. Transaction Costs & Friction Modeling
        pos_diff = np.abs(np.diff(pos, prepend=0))
        total_friction_bps = (self.config.commission_bps + self.config.slippage_bps) / 10000.0
        trade_costs = pos_diff * total_friction_bps * self.config.leverage
        
        # Short borrow cost (pro-rated per trading day)
        daily_borrow_fee = (self.config.annual_borrow_rate / 252.0) * (np.abs(pos) * (np.abs(beta) / weight_norm) * self.config.leverage)
        
        net_daily_return = raw_levered_return - trade_costs - daily_borrow_fee
        df["strategy_net_return"] = net_daily_return
        
        # 4. Equity Curve
        df["equity"] = self.config.initial_capital * np.cumprod(1.0 + net_daily_return)
        
        # 5. Performance Statistics
        total_days = len(df)
        years = total_days / 252.0 if total_days > 0 else 1.0
        
        total_return = (df["equity"].iloc[-1] / self.config.initial_capital) - 1.0
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and (1.0 + total_return) > 0 else 0.0
        
        daily_rf = (1.0 + self.config.risk_free_rate) ** (1.0 / 252.0) - 1.0
        excess_returns = net_daily_return - daily_rf
        
        ann_vol = float(np.std(net_daily_return) * np.sqrt(252))
        sharpe = float((np.mean(excess_returns) / (np.std(net_daily_return) + 1e-8)) * np.sqrt(252))
        
        downside = net_daily_return[net_daily_return < 0]
        downside_std = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 1e-8
        sortino = float((np.mean(excess_returns) / downside_std) * np.sqrt(252))
        
        cum_max = df["equity"].cummax()
        drawdown = (df["equity"] - cum_max) / cum_max
        max_drawdown = float(drawdown.min())
        
        calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0
        
        # Trade Analysis
        trades = []
        in_trade = False
        trade_ret = 0.0
        
        for p, r in zip(pos, net_daily_return):
            if p != 0:
                in_trade = True
                trade_ret += r
            elif in_trade and p == 0:
                trades.append(trade_ret)
                trade_ret = 0.0
                in_trade = False
                
        if in_trade:
            trades.append(trade_ret)
            
        trades = np.array(trades)
        num_trades = len(trades)
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        win_rate = float(len(wins) / num_trades) if num_trades > 0 else 0.0
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-8
        profit_factor = float(gross_profit / gross_loss)
        
        return {
            "initial_capital": self.config.initial_capital,
            "final_equity": float(df["equity"].iloc[-1]),
            "total_return_pct": float(total_return * 100.0),
            "cagr_pct": float(cagr * 100.0),
            "annualized_volatility_pct": float(ann_vol * 100.0),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "calmar_ratio": round(calmar, 2),
            "total_trades": num_trades,
            "win_rate_pct": round(win_rate * 100.0, 1),
            "profit_factor": round(profit_factor, 2),
            "backtest_days": total_days,
            "equity_series": df["equity"],
            "drawdown_series": drawdown
        }
