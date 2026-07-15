import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .portfolio_optimization import PortfolioOptimizer


class Backtester:
    """
    Out-of-sample backtesting framework for portfolio strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate
        self.optimizer = PortfolioOptimizer(risk_free_rate)
    
    def rolling_backtest(
        self,
        returns: pd.DataFrame,
        lookback_days: int = 252,
        rebalance_freq: int = 21,
        method: str = 'max_sharpe'
    ) -> pd.DataFrame:
        """
        Perform rolling window backtest with periodic rebalancing.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Full period returns
        lookback_days : int
            Training window size
        rebalance_freq : int
            Rebalancing frequency in trading days
        method : str
            Optimization method
        
        Returns:
        --------
        pd.DataFrame with backtest results
        """
        n_assets = returns.shape[1]
        dates = returns.index[lookback_days:]
        
        portfolio_returns = []
        weights_history = []
        
        for i in range(0, len(dates), rebalance_freq):
            if i + rebalance_freq > len(dates):
                break
            
            train_start = i
            train_end = i + lookback_days
            test_start = train_end
            test_end = min(train_end + rebalance_freq, len(dates))
            
            train_returns = returns.iloc[train_start:train_end]
            test_returns = returns.iloc[test_start:test_end]
            
            weights = self._get_optimal_weights(train_returns, method)
            
            for j in range(len(test_returns)):
                port_ret = np.dot(weights, test_returns.iloc[j])
                portfolio_returns.append({
                    'date': test_returns.index[j],
                    'return': port_ret
                })
                weights_history.append({
                    'date': test_returns.index[j],
                    **{f'weight_{returns.columns[k]}': weights[k] 
                       for k in range(n_assets)}
                })
        
        port_df = pd.DataFrame(portfolio_returns).set_index('date')
        weights_df = pd.DataFrame(weights_history).set_index('date')
        
        return self._calculate_metrics(port_df, weights_df)
    
    def _get_optimal_weights(
        self,
        returns: pd.DataFrame,
        method: str
    ) -> np.ndarray:
        """Get optimal weights based on method."""
        if method == 'max_sharpe':
            weights, _ = self.optimizer.optimize_max_sharpe(returns)
        elif method == 'min_variance':
            weights, _ = self.optimizer.optimize_min_variance(returns)
        elif method == 'risk_parity':
            weights, _ = self.optimizer.optimize_risk_parity(returns)
        elif method == 'equal_weight':
            weights, _ = self.optimizer.optimize_equal_weight(returns)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return weights
    
    def _calculate_metrics(
        self,
        port_returns: pd.DataFrame,
        weights_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate backtest metrics."""
        cumulative = (1 + port_returns['return']).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        metrics = pd.DataFrame({
            'return': port_returns['return'],
            'cumulative_return': cumulative,
            'drawdown': drawdown,
            'rolling_volatility': port_returns['return'].rolling(21).std() * np.sqrt(252)
        })
        
        return metrics
    
    def compare_strategies(
        self,
        returns: pd.DataFrame,
        methods: list,
        lookback_days: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """Compare multiple strategies."""
        results = {}
        for method in methods:
            results[method] = self.rolling_backtest(
                returns,
                lookback_days=lookback_days,
                method=method
            )
        return results
    
    def calculate_statistics(
        self,
        backtest_results: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate summary statistics for backtest."""
        returns = backtest_results['return']
        cumulative = backtest_results['cumulative_return']
        
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.rf) / annual_vol
        max_drawdown = backtest_results['drawdown'].min()
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        stats = {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar:.2f}"
        }
        
        return stats
