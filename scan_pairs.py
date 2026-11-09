import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant_arbitrage.data_pipeline import fetch_historical_pair
from quant_arbitrage.cointegration import test_cointegration
from quant_arbitrage.pairs_trading_bot import PairsTradingStrategy, StrategyParameters
from quant_arbitrage.backtester import Backtester, BacktestConfig

pairs = [('EWA', 'EWC'), ('GDX', 'GLD'), ('XOM', 'CVX'), ('V', 'MA')]

for y, x in pairs:
    try:
        sy, sx = fetch_historical_pair(y, x, '2014-01-01', '2024-01-01')
        cres = test_cointegration(sy, sx)
        strat = PairsTradingStrategy(StrategyParameters(lookback_window=40, entry_z_score=1.8, exit_z_score=0.4, stop_loss_z_score=3.5))
        sig = strat.generate_signals(sy, sx)
        bt = Backtester(BacktestConfig(leverage=2.0, commission_bps=3.0, slippage_bps=2.0))
        r = bt.run(sig)
        p_val = cres['p_value']
        hl = cres['half_life_days']
        tot = r['total_return_pct']
        sh = r['sharpe_ratio']
        wr = r['win_rate_pct']
        is_c = cres['is_cointegrated']
        print(f"{y}/{x} -> Cointegrated: {is_c} (p={p_val:.4f}, HL={hl:.1f}d) | Return: {tot:+.1f}% | Sharpe: {sh} | WinRate: {wr}%")
    except Exception as e:
        print(f"{y}/{x} error: {e}")
