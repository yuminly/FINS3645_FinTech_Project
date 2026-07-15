import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization using various methods:
    - Maximum Sharpe Ratio (Mean-Variance Tangency)
    - Minimum Variance
    - Risk Parity
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate
    
    def optimize_max_sharpe(
        self,
        returns: pd.DataFrame,
        constraints: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Find the maximum Sharpe ratio portfolio (tangency portfolio).
        """
        n = returns.shape[1]
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.rf) / port_vol
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if constraints:
            for key, val in constraints.items():
                if key == 'max_weight':
                    constraints_list.append({'type': 'ineq', 'fun': lambda x: val - x})
                elif key == 'min_weight':
                    constraints_list.append({'type': 'ineq', 'fun': lambda x: x - val})
        
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.array([1/n] * n)
        
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        metrics = self._calculate_portfolio_metrics(result.x, returns)
        return result.x, metrics
    
    def optimize_min_variance(
        self,
        returns: pd.DataFrame,
        constraints: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Find the minimum variance portfolio.
        """
        n = returns.shape[1]
        cov_matrix = returns.cov() * 252
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.array([1/n] * n)
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        metrics = self._calculate_portfolio_metrics(result.x, returns)
        return result.x, metrics
    
    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        constraints: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Find the risk parity portfolio where each asset contributes equally to risk.
        """
        n = returns.shape[1]
        cov_matrix = returns.cov() * 252
        
        def risk_contribution_error(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_contrib = port_vol / n
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 1) for _ in range(n))
        initial_weights = np.array([1/n] * n)
        
        result = minimize(
            risk_contribution_error,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        metrics = self._calculate_portfolio_metrics(result.x, returns)
        return result.x, metrics
    
    def optimize_equal_weight(
        self,
        returns: pd.DataFrame
    ) -> Tuple[np.ndarray, dict]:
        """
        Create an equal-weight portfolio (baseline).
        """
        n = returns.shape[1]
        weights = np.array([1/n] * n)
        metrics = self._calculate_portfolio_metrics(weights, returns)
        return weights, metrics
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> dict:
        """Calculate portfolio performance metrics."""
        port_returns = returns.dot(weights)
        
        total_return = (1 + port_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(port_returns)) - 1
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.rf) / annual_vol
        
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'weights': weights,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'cumulative_returns': cumulative,
            'drawdowns': drawdown
        }
        
        return metrics
    
    def get_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the efficient frontier."""
        n = returns.shape[1]
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252
        
        target_returns = np.linspace(
            mean_returns.min(),
            mean_returns.max(),
            n_points
        )
        
        efficient_vols = []
        
        for target in target_returns:
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
            ]
            bounds = tuple((0, 1) for _ in range(n))
            initial_weights = np.array([1/n] * n)
            
            result = minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            efficient_vols.append(result.fun if result.success else np.nan)
        
        return target_returns, np.array(efficient_vols)
