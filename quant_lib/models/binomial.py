"""
Binomial tree option pricing model implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class BinomialTree:
    """
    Binomial tree model for option pricing (European and American options).
    """
    
    def __init__(self):
        """Initialize Binomial Tree calculator."""
        pass
    
    def _calculate_parameters(self, r: float, sigma: float, dt: float) -> Tuple[float, float, float]:
        """
        Calculate binomial tree parameters.
        
        Args:
            r: Risk-free rate
            sigma: Volatility
            dt: Time step
            
        Returns:
            Tuple of (u, d, p) where u=up factor, d=down factor, p=risk-neutral probability
        """
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        
        return u, d, p
    
    def european_option_price(self, 
                            S0: float, 
                            K: float, 
                            T: float, 
                            r: float, 
                            sigma: float, 
                            option_type: str = 'call',
                            n_steps: int = 100) -> Dict[str, float]:
        """
        Price European option using binomial tree.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            
        Returns:
            Dictionary with option price and tree details
        """
        dt = T / n_steps
        u, d, p = self._calculate_parameters(r, sigma, dt)
        discount = np.exp(-r * dt)
        
        # Initialize stock price tree
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Fill stock price tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Calculate terminal option values
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_tree[j, n_steps] = max(0, stock_tree[j, n_steps] - K)
            elif option_type.lower() == 'put':
                option_tree[j, n_steps] = max(0, K - stock_tree[j, n_steps])
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = discount * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )
        
        return {
            'price': option_tree[0, 0],
            'u': u,
            'd': d,
            'p': p,
            'n_steps': n_steps,
            'dt': dt
        }
    
    def american_option_price(self, 
                            S0: float, 
                            K: float, 
                            T: float, 
                            r: float, 
                            sigma: float, 
                            option_type: str = 'put',
                            n_steps: int = 100) -> Dict[str, float]:
        """
        Price American option using binomial tree.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            
        Returns:
            Dictionary with option price and early exercise boundary
        """
        dt = T / n_steps
        u, d, p = self._calculate_parameters(r, sigma, dt)
        discount = np.exp(-r * dt)
        
        # Initialize trees
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        exercise_boundary = []
        
        # Fill stock price tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        # Calculate terminal option values
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_tree[j, n_steps] = max(0, stock_tree[j, n_steps] - K)
            elif option_type.lower() == 'put':
                option_tree[j, n_steps] = max(0, K - stock_tree[j, n_steps])
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        # Backward induction with early exercise
        for i in range(n_steps - 1, -1, -1):
            exercise_nodes = []
            
            for j in range(i + 1):
                # Calculate continuation value
                continuation_value = discount * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )
                
                # Calculate intrinsic value
                if option_type.lower() == 'call':
                    intrinsic_value = max(0, stock_tree[j, i] - K)
                else:  # put
                    intrinsic_value = max(0, K - stock_tree[j, i])
                
                # American option: max of continuation and intrinsic
                option_tree[j, i] = max(continuation_value, intrinsic_value)
                
                # Check for early exercise
                if intrinsic_value > continuation_value:
                    exercise_nodes.append((j, stock_tree[j, i]))
            
            if exercise_nodes:
                exercise_boundary.append({
                    'time_step': i,
                    'time': i * dt,
                    'exercise_nodes': exercise_nodes
                })
        
        return {
            'price': option_tree[0, 0],
            'european_price': self.european_option_price(S0, K, T, r, sigma, option_type, n_steps)['price'],
            'early_exercise_premium': option_tree[0, 0] - self.european_option_price(S0, K, T, r, sigma, option_type, n_steps)['price'],
            'exercise_boundary': exercise_boundary,
            'u': u,
            'd': d,
            'p': p,
            'n_steps': n_steps
        }
    
    def calculate_greeks(self, 
                        S0: float, 
                        K: float, 
                        T: float, 
                        r: float, 
                        sigma: float, 
                        option_type: str = 'call',
                        n_steps: int = 100,
                        bump_size: float = 0.01) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            bump_size: Size of bump for finite differences
            
        Returns:
            Dictionary with Greeks
        """
        # Base price
        base_price = self.european_option_price(S0, K, T, r, sigma, option_type, n_steps)['price']
        
        # Delta: sensitivity to stock price
        price_up = self.european_option_price(S0 * (1 + bump_size), K, T, r, sigma, option_type, n_steps)['price']
        price_down = self.european_option_price(S0 * (1 - bump_size), K, T, r, sigma, option_type, n_steps)['price']
        delta = (price_up - price_down) / (2 * S0 * bump_size)
        
        # Gamma: second derivative with respect to stock price
        gamma = (price_up - 2 * base_price + price_down) / ((S0 * bump_size) ** 2)
        
        # Theta: sensitivity to time (negative of time decay)
        if T > bump_size:
            price_theta = self.european_option_price(S0, K, T - bump_size, r, sigma, option_type, n_steps)['price']
            theta = (price_theta - base_price) / bump_size
        else:
            theta = 0
        
        # Vega: sensitivity to volatility
        price_vega_up = self.european_option_price(S0, K, T, r, sigma + bump_size, option_type, n_steps)['price']
        price_vega_down = self.european_option_price(S0, K, T, r, sigma - bump_size, option_type, n_steps)['price']
        vega = (price_vega_up - price_vega_down) / (2 * bump_size)
        
        # Rho: sensitivity to interest rate
        price_rho_up = self.european_option_price(S0, K, T, r + bump_size, sigma, option_type, n_steps)['price']
        price_rho_down = self.european_option_price(S0, K, T, r - bump_size, sigma, option_type, n_steps)['price']
        rho = (price_rho_up - price_rho_down) / (2 * bump_size)
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def convergence_analysis(self, 
                           S0: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           sigma: float, 
                           option_type: str = 'call',
                           max_steps: int = 500,
                           step_increment: int = 10) -> pd.DataFrame:
        """
        Analyze convergence of binomial tree to Black-Scholes price.
        
        Args:
            S0, K, T, r, sigma: Option parameters
            option_type: 'call' or 'put'
            max_steps: Maximum number of steps to test
            step_increment: Increment between step counts
            
        Returns:
            DataFrame with convergence analysis
        """
        from .black_scholes import BlackScholes
        
        bs = BlackScholes()
        bs_price = bs.calculate_price(S0, K, T, r, sigma, option_type)
        
        results = []
        step_counts = range(step_increment, max_steps + 1, step_increment)
        
        for n_steps in step_counts:
            binomial_price = self.european_option_price(
                S0, K, T, r, sigma, option_type, n_steps
            )['price']
            
            error = abs(binomial_price - bs_price)
            relative_error = error / bs_price * 100
            
            results.append({
                'n_steps': n_steps,
                'binomial_price': binomial_price,
                'bs_price': bs_price,
                'absolute_error': error,
                'relative_error_pct': relative_error
            })
        
        return pd.DataFrame(results)
    
    def dividend_adjusted_price(self, 
                              S0: float, 
                              K: float, 
                              T: float, 
                              r: float, 
                              sigma: float, 
                              dividend_yield: float,
                              option_type: str = 'call',
                              n_steps: int = 100) -> Dict[str, float]:
        """
        Price option with continuous dividend yield.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            dividend_yield: Continuous dividend yield
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            
        Returns:
            Dictionary with option price
        """
        dt = T / n_steps
        
        # Adjust parameters for dividend yield
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Initialize trees
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Fill stock price tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        # Calculate terminal option values
        for j in range(n_steps + 1):
            if option_type.lower() == 'call':
                option_tree[j, n_steps] = max(0, stock_tree[j, n_steps] - K)
            elif option_type.lower() == 'put':
                option_tree[j, n_steps] = max(0, K - stock_tree[j, n_steps])
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = discount * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )
        
        return {
            'price': option_tree[0, 0],
            'dividend_yield': dividend_yield,
            'adjusted_u': u,
            'adjusted_d': d,
            'adjusted_p': p,
            'n_steps': n_steps
        }
