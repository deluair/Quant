"""
Portfolio optimization using various methods.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory and advanced techniques.
    """
    
    def __init__(self):
        """Initialize Portfolio Optimizer."""
        pass
    
    def mean_variance_optimization(self, 
                                 returns: pd.DataFrame,
                                 target_return: Optional[float] = None,
                                 risk_aversion: float = 1.0,
                                 constraints: Optional[Dict] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform mean-variance optimization.
        
        Args:
            returns: Historical returns DataFrame
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints dictionary
            
        Returns:
            Dictionary with optimal weights and portfolio statistics
        """
        n_assets = len(returns.columns)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = mean_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1]  # Weights sum to 1
        
        # Add custom constraints
        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(weights >= 0)
            
            if 'max_weight' in constraints:
                constraints_list.append(weights <= constraints['max_weight'])
            
            if 'min_weight' in constraints:
                constraints_list.append(weights >= constraints['min_weight'])
            
            if 'sector_constraints' in constraints:
                for sector_constraint in constraints['sector_constraints']:
                    sector_indices = sector_constraint['indices']
                    max_weight = sector_constraint['max_weight']
                    constraints_list.append(cp.sum(weights[sector_indices]) <= max_weight)
        
        # Objective function
        if target_return is not None:
            # Minimize risk for target return
            constraints_list.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_risk)
        else:
            # Maximize utility (return - risk_aversion * risk)
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = weights.value
            portfolio_ret = mean_returns.T @ optimal_weights
            portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            sharpe_ratio = portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_ret,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'status': problem.status
            }
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
    
    def minimum_variance_portfolio(self, 
                                 returns: pd.DataFrame,
                                 constraints: Optional[Dict] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find minimum variance portfolio.
        
        Args:
            returns: Historical returns DataFrame
            constraints: Additional constraints
            
        Returns:
            Dictionary with minimum variance portfolio
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        
        weights = cp.Variable(n_assets)
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        constraints_list = [cp.sum(weights) == 1]
        
        if constraints and constraints.get('long_only', False):
            constraints_list.append(weights >= 0)
        
        objective = cp.Minimize(portfolio_risk)
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = weights.value
            mean_returns = returns.mean().values
            portfolio_ret = mean_returns.T @ optimal_weights
            portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_ret,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
            }
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
    
    def maximum_sharpe_portfolio(self, 
                               returns: pd.DataFrame,
                               risk_free_rate: float = 0.02,
                               constraints: Optional[Dict] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find maximum Sharpe ratio portfolio.
        
        Args:
            returns: Historical returns DataFrame
            risk_free_rate: Risk-free rate
            constraints: Additional constraints
            
        Returns:
            Dictionary with maximum Sharpe portfolio
        """
        n_assets = len(returns.columns)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Convert to excess returns
        excess_returns = mean_returns - risk_free_rate / 252
        
        # Use auxiliary variable approach for Sharpe maximization
        y = cp.Variable(n_assets)
        kappa = cp.Variable()
        
        constraints_list = [
            excess_returns.T @ y == 1,
            cp.sum(y) == kappa,
            kappa >= 0
        ]
        
        if constraints and constraints.get('long_only', False):
            constraints_list.append(y >= 0)
        
        objective = cp.Minimize(cp.quad_form(y, cov_matrix))
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_y = y.value
            optimal_kappa = kappa.value
            optimal_weights = optimal_y / optimal_kappa
            
            portfolio_ret = mean_returns.T @ optimal_weights
            portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            sharpe_ratio = (portfolio_ret - risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_ret,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio
            }
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
    
    def risk_parity_portfolio(self, 
                            returns: pd.DataFrame,
                            method: str = 'equal_risk_contribution') -> Dict[str, Union[np.ndarray, float]]:
        """
        Construct risk parity portfolio.
        
        Args:
            returns: Historical returns DataFrame
            method: 'equal_risk_contribution' or 'inverse_volatility'
            
        Returns:
            Dictionary with risk parity portfolio
        """
        if method == 'inverse_volatility':
            # Simple inverse volatility weighting
            volatilities = returns.std().values
            inv_vol_weights = 1 / volatilities
            weights = inv_vol_weights / inv_vol_weights.sum()
            
        elif method == 'equal_risk_contribution':
            # Equal risk contribution optimization
            cov_matrix = returns.cov().values
            n_assets = len(returns.columns)
            
            def risk_budget_objective(weights):
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                marginal_contrib = cov_matrix @ weights / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x}
            ]
            
            x0 = np.ones(n_assets) / n_assets
            result = minimize(risk_budget_objective, x0, method='SLSQP', constraints=constraints)
            
            if result.success:
                weights = result.x
            else:
                raise ValueError("Risk parity optimization failed")
        
        else:
            raise ValueError("Method must be 'equal_risk_contribution' or 'inverse_volatility'")
        
        # Calculate portfolio statistics
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        portfolio_ret = mean_returns.T @ weights
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        return {
            'weights': weights,
            'expected_return': portfolio_ret,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
        }
    
    def black_litterman_optimization(self, 
                                   returns: pd.DataFrame,
                                   market_caps: np.ndarray,
                                   views: Optional[Dict] = None,
                                   tau: float = 0.025,
                                   risk_aversion: float = 3.0) -> Dict[str, Union[np.ndarray, float]]:
        """
        Black-Litterman portfolio optimization.
        
        Args:
            returns: Historical returns DataFrame
            market_caps: Market capitalizations for equilibrium weights
            views: Dictionary with investor views {'assets': [indices], 'returns': [expected], 'confidence': [values]}
            tau: Scaling factor for uncertainty of prior
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dictionary with Black-Litterman portfolio
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        
        # Market capitalization weights (equilibrium portfolio)
        w_market = market_caps / market_caps.sum()
        
        # Implied equilibrium returns
        pi = risk_aversion * cov_matrix @ w_market
        
        # Prior distribution
        mu_prior = pi
        sigma_prior = tau * cov_matrix
        
        if views is not None:
            # Incorporate views
            P = np.zeros((len(views['assets']), n_assets))
            Q = np.array(views['returns'])
            
            for i, assets in enumerate(views['assets']):
                if isinstance(assets, int):
                    P[i, assets] = 1
                else:
                    for asset in assets:
                        P[i, asset] = 1 / len(assets)
            
            # View uncertainty matrix
            if 'confidence' in views:
                omega = np.diag(views['confidence'])
            else:
                omega = np.eye(len(Q)) * 0.01  # Default uncertainty
            
            # Black-Litterman formula
            tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
            p_omega_inv_p = P.T @ np.linalg.inv(omega) @ P
            
            sigma_bl = np.linalg.inv(tau_sigma_inv + p_omega_inv_p)
            mu_bl = sigma_bl @ (tau_sigma_inv @ mu_prior + P.T @ np.linalg.inv(omega) @ Q)
        else:
            # No views, use prior
            mu_bl = mu_prior
            sigma_bl = sigma_prior
        
        # Optimize portfolio with Black-Litterman inputs
        weights = cp.Variable(n_assets)
        portfolio_return = mu_bl.T @ weights
        portfolio_risk = cp.quad_form(weights, sigma_bl)
        
        constraints = [cp.sum(weights) == 1, weights >= 0]
        objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = weights.value
            portfolio_ret = mu_bl.T @ optimal_weights
            portfolio_vol = np.sqrt(optimal_weights.T @ sigma_bl @ optimal_weights)
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_ret,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0,
                'implied_returns': pi,
                'bl_returns': mu_bl,
                'bl_covariance': sigma_bl
            }
        else:
            raise ValueError(f"Black-Litterman optimization failed with status: {problem.status}")
    
    def robust_optimization(self, 
                          returns: pd.DataFrame,
                          uncertainty_set: str = 'box',
                          epsilon: float = 0.1) -> Dict[str, Union[np.ndarray, float]]:
        """
        Robust portfolio optimization under parameter uncertainty.
        
        Args:
            returns: Historical returns DataFrame
            uncertainty_set: Type of uncertainty set ('box', 'elliptical')
            epsilon: Size of uncertainty set
            
        Returns:
            Dictionary with robust portfolio
        """
        n_assets = len(returns.columns)
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        weights = cp.Variable(n_assets)
        
        if uncertainty_set == 'box':
            # Box uncertainty set
            worst_case_return = mean_returns.T @ weights - epsilon * cp.norm(weights, 1)
        elif uncertainty_set == 'elliptical':
            # Elliptical uncertainty set
            worst_case_return = mean_returns.T @ weights - epsilon * cp.norm(cov_matrix @ weights, 2)
        else:
            raise ValueError("Uncertainty set must be 'box' or 'elliptical'")
        
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        constraints = [cp.sum(weights) == 1, weights >= 0]
        objective = cp.Maximize(worst_case_return - 0.5 * portfolio_risk)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = weights.value
            portfolio_ret = mean_returns.T @ optimal_weights
            portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_ret,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0,
                'uncertainty_set': uncertainty_set,
                'epsilon': epsilon
            }
        else:
            raise ValueError(f"Robust optimization failed with status: {problem.status}")
    
    def portfolio_comparison(self, 
                           returns: pd.DataFrame,
                           methods: List[str] = None) -> pd.DataFrame:
        """
        Compare different portfolio optimization methods.
        
        Args:
            returns: Historical returns DataFrame
            methods: List of methods to compare
            
        Returns:
            DataFrame comparing different portfolios
        """
        if methods is None:
            methods = ['equal_weight', 'min_variance', 'max_sharpe', 'risk_parity']
        
        results = []
        
        for method in methods:
            try:
                if method == 'equal_weight':
                    n_assets = len(returns.columns)
                    weights = np.ones(n_assets) / n_assets
                    mean_returns = returns.mean().values
                    cov_matrix = returns.cov().values
                    portfolio_ret = mean_returns.T @ weights
                    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                    sharpe = portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
                    
                    result = {
                        'method': method,
                        'expected_return': portfolio_ret,
                        'volatility': portfolio_vol,
                        'sharpe_ratio': sharpe
                    }
                
                elif method == 'min_variance':
                    result = self.minimum_variance_portfolio(returns, {'long_only': True})
                    result['method'] = method
                
                elif method == 'max_sharpe':
                    result = self.maximum_sharpe_portfolio(returns, constraints={'long_only': True})
                    result['method'] = method
                
                elif method == 'risk_parity':
                    result = self.risk_parity_portfolio(returns)
                    result['method'] = method
                
                else:
                    continue
                
                results.append(result)
                
            except Exception as e:
                print(f"Error with method {method}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
