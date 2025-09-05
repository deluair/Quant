"""
Monte Carlo simulation for option pricing and risk analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class MonteCarlo:
    """
    Monte Carlo simulation for option pricing and risk modeling.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
    
    def geometric_brownian_motion(self, 
                                 S0: float, 
                                 mu: float, 
                                 sigma: float, 
                                 T: float, 
                                 dt: float, 
                                 num_paths: int = 10000) -> np.ndarray:
        """
        Generate stock price paths using Geometric Brownian Motion.
        
        Args:
            S0: Initial stock price
            mu: Drift rate
            sigma: Volatility
            T: Time horizon
            dt: Time step
            num_paths: Number of simulation paths
            
        Returns:
            Array of stock price paths
        """
        num_steps = int(T / dt)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        
        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(dt), (num_paths, num_steps))
        
        # Simulate paths
        for t in range(num_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW[:, t]
            )
        
        return paths
    
    def european_option_price(self, 
                            S0: float, 
                            K: float, 
                            T: float, 
                            r: float, 
                            sigma: float, 
                            option_type: str = 'call',
                            num_simulations: int = 100000) -> Dict[str, float]:
        """
        Price European option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with price and confidence interval
        """
        # Generate final stock prices
        Z = np.random.standard_normal(num_simulations)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate price and statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        
        # 95% confidence interval
        confidence_interval = 1.96 * std_error
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_lower': price - confidence_interval,
            'confidence_upper': price + confidence_interval,
            'num_simulations': num_simulations
        }
    
    def american_option_price(self, 
                            S0: float, 
                            K: float, 
                            T: float, 
                            r: float, 
                            sigma: float, 
                            option_type: str = 'put',
                            num_paths: int = 10000,
                            num_steps: int = 100) -> Dict[str, float]:
        """
        Price American option using Longstaff-Schwartz method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            num_paths: Number of simulation paths
            num_steps: Number of time steps
            
        Returns:
            Dictionary with option price
        """
        dt = T / num_steps
        discount = np.exp(-r * dt)
        
        # Generate stock price paths
        paths = self.geometric_brownian_motion(S0, r, sigma, T, dt, num_paths)
        
        # Initialize payoff matrix
        if option_type.lower() == 'call':
            payoffs = np.maximum(paths - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - paths, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Backward induction
        option_values = payoffs[:, -1].copy()
        
        for t in range(num_steps - 1, 0, -1):
            # In-the-money paths
            itm = payoffs[:, t] > 0
            
            if np.sum(itm) > 0:
                # Regression variables (polynomial basis)
                X = paths[itm, t]
                X_poly = np.column_stack([np.ones(len(X)), X, X**2])
                
                # Continuation values
                Y = discount * option_values[itm]
                
                # Regression
                try:
                    coeffs = np.linalg.lstsq(X_poly, Y, rcond=None)[0]
                    continuation_values = X_poly @ coeffs
                    
                    # Exercise decision
                    exercise = payoffs[itm, t] > continuation_values
                    option_values[itm] = np.where(exercise, 
                                                payoffs[itm, t], 
                                                discount * option_values[itm])
                except np.linalg.LinAlgError:
                    # If regression fails, use intrinsic value
                    option_values[itm] = np.maximum(payoffs[itm, t], 
                                                  discount * option_values[itm])
            
            # Discount all values
            option_values *= discount
        
        price = np.mean(option_values)
        std_error = np.std(option_values) / np.sqrt(num_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'num_paths': num_paths,
            'num_steps': num_steps
        }
    
    def asian_option_price(self, 
                          S0: float, 
                          K: float, 
                          T: float, 
                          r: float, 
                          sigma: float, 
                          option_type: str = 'call',
                          averaging_type: str = 'arithmetic',
                          num_simulations: int = 100000,
                          num_steps: int = 252) -> Dict[str, float]:
        """
        Price Asian option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            averaging_type: 'arithmetic' or 'geometric'
            num_simulations: Number of simulations
            num_steps: Number of averaging points
            
        Returns:
            Dictionary with option price
        """
        dt = T / num_steps
        
        # Generate stock price paths
        paths = self.geometric_brownian_motion(S0, r, sigma, T, dt, num_simulations)
        
        # Calculate averages
        if averaging_type == 'arithmetic':
            averages = np.mean(paths, axis=1)
        elif averaging_type == 'geometric':
            averages = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(averages - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - averages, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'averaging_type': averaging_type,
            'num_simulations': num_simulations
        }
    
    def barrier_option_price(self, 
                           S0: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           sigma: float, 
                           barrier: float,
                           barrier_type: str = 'up_and_out',
                           option_type: str = 'call',
                           num_simulations: int = 100000,
                           num_steps: int = 252) -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            barrier: Barrier level
            barrier_type: 'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in'
            option_type: 'call' or 'put'
            num_simulations: Number of simulations
            num_steps: Number of time steps
            
        Returns:
            Dictionary with option price
        """
        dt = T / num_steps
        
        # Generate stock price paths
        paths = self.geometric_brownian_motion(S0, r, sigma, T, dt, num_simulations)
        
        # Check barrier conditions
        if 'up' in barrier_type:
            barrier_hit = np.any(paths >= barrier, axis=1)
        else:  # down
            barrier_hit = np.any(paths <= barrier, axis=1)
        
        # Calculate payoffs based on barrier type
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            intrinsic_payoffs = np.maximum(final_prices - K, 0)
        elif option_type.lower() == 'put':
            intrinsic_payoffs = np.maximum(K - final_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        if 'out' in barrier_type:
            # Knock-out: payoff only if barrier not hit
            payoffs = np.where(barrier_hit, 0, intrinsic_payoffs)
        else:  # in
            # Knock-in: payoff only if barrier hit
            payoffs = np.where(barrier_hit, intrinsic_payoffs, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'barrier': barrier,
            'barrier_type': barrier_type,
            'num_simulations': num_simulations
        }
    
    def portfolio_var(self, 
                     returns: np.ndarray, 
                     weights: np.ndarray, 
                     confidence_level: float = 0.95,
                     num_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate portfolio VaR using Monte Carlo simulation.
        
        Args:
            returns: Historical returns matrix (assets x time)
            weights: Portfolio weights
            confidence_level: Confidence level for VaR
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with VaR metrics
        """
        # Calculate portfolio returns
        portfolio_returns = returns @ weights
        
        # Fit parameters
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Monte Carlo simulation
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # Calculate VaR
        var_level = 1 - confidence_level
        var = np.percentile(simulated_returns, var_level * 100)
        
        # Expected Shortfall (Conditional VaR)
        es = np.mean(simulated_returns[simulated_returns <= var])
        
        return {
            'var': -var,  # VaR is typically reported as positive loss
            'expected_shortfall': -es,
            'confidence_level': confidence_level,
            'num_simulations': num_simulations,
            'mean_return': mean_return,
            'volatility': std_return
        }
    
    def stress_testing(self, 
                      portfolio_value: float,
                      scenarios: Dict[str, Dict[str, float]],
                      num_simulations: int = 1000) -> pd.DataFrame:
        """
        Perform stress testing on portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            scenarios: Dictionary of stress scenarios
            num_simulations: Number of simulations per scenario
            
        Returns:
            DataFrame with stress test results
        """
        results = []
        
        for scenario_name, params in scenarios.items():
            # Extract scenario parameters
            shock_mean = params.get('shock_mean', 0)
            shock_std = params.get('shock_std', 0.1)
            correlation = params.get('correlation', 0)
            
            # Generate correlated shocks
            shocks = np.random.multivariate_normal(
                [shock_mean, shock_mean], 
                [[shock_std**2, correlation * shock_std**2],
                 [correlation * shock_std**2, shock_std**2]], 
                num_simulations
            )
            
            # Calculate portfolio values under stress
            stressed_values = portfolio_value * (1 + shocks[:, 0])
            losses = portfolio_value - stressed_values
            
            # Calculate statistics
            mean_loss = np.mean(losses)
            max_loss = np.max(losses)
            var_95 = np.percentile(losses, 95)
            
            results.append({
                'scenario': scenario_name,
                'mean_loss': mean_loss,
                'max_loss': max_loss,
                'var_95': var_95,
                'prob_loss_gt_10pct': np.mean(losses > 0.1 * portfolio_value)
            })
        
        return pd.DataFrame(results)
