"""
Black-Scholes option pricing model implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback normal distribution functions
    class MockNorm:
        @staticmethod
        def cdf(x):
            # Approximation of normal CDF using error function
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        @staticmethod
        def pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    norm = MockNorm()

import warnings
warnings.filterwarnings('ignore')


class BlackScholes:
    """
    Black-Scholes option pricing model with Greeks calculation.
    """
    
    def __init__(self):
        """Initialize Black-Scholes calculator."""
        pass
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def calculate_price(self, 
                       S: float, 
                       K: float, 
                       T: float, 
                       r: float, 
                       sigma: float, 
                       option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    def calculate_greeks(self, 
                        S: float, 
                        K: float, 
                        T: float, 
                        r: float, 
                        sigma: float, 
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with Greeks values
        """
        if T <= 0:
            return {
                'delta': 0, 'gamma': 0, 'theta': 0, 
                'vega': 0, 'rho': 0, 'price': self.calculate_price(S, K, T, r, sigma, option_type)
            }
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        # Common terms
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        npd1 = norm.pdf(d1)
        
        if option_type.lower() == 'call':
            # Call option Greeks
            delta = nd1
            gamma = npd1 / (S * sigma * np.sqrt(T))
            theta = (-S * npd1 * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * nd2) / 365
            vega = S * npd1 * np.sqrt(T) / 100
            rho = K * T * np.exp(-r * T) * nd2 / 100
            
        elif option_type.lower() == 'put':
            # Put option Greeks
            delta = nd1 - 1
            gamma = npd1 / (S * sigma * np.sqrt(T))
            theta = (-S * npd1 * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = S * npd1 * np.sqrt(T) / 100
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        price = self.calculate_price(S, K, T, r, sigma, option_type)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, 
                          market_price: float, 
                          S: float, 
                          K: float, 
                          T: float, 
                          r: float, 
                          option_type: str = 'call',
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.2
        
        
        if not SCIPY_AVAILABLE:
            # Simple Newton-Raphson implementation
            for i in range(max_iterations):
                price = self.calculate_price(S, K, T, r, sigma, option_type)
                vega = self.calculate_greeks(S, K, T, r, sigma, option_type)['vega']
                
                if abs(vega) < 1e-10:
                    break
                    
                sigma_new = sigma - (price - market_price) / (vega * 100)
                
                if abs(sigma_new - sigma) < tolerance:
                    return max(sigma_new, 0.001)
                    
                sigma = max(sigma_new, 0.001)
            
            return sigma
        
        def objective(sigma):
            return self.calculate_price(S, K, T, r, sigma, option_type) - market_price
        
        try:
            # Initial bounds for volatility search
            vol_low, vol_high = 0.001, 5.0
            
            # Check if solution exists in bounds
            if objective(vol_low) * objective(vol_high) > 0:
                return np.nan
            
            implied_vol = brentq(objective, vol_low, vol_high, xtol=tolerance)
            return implied_vol
        except:
            return np.nan
    
    def option_chain(self, 
                    S: float, 
                    strikes: np.ndarray, 
                    T: float, 
                    r: float, 
                    sigma: float) -> pd.DataFrame:
        """
        Generate option chain for multiple strikes.
        
        Args:
            S: Current stock price
            strikes: Array of strike prices
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            DataFrame with option chain
        """
        results = []
        
        for K in strikes:
            call_greeks = self.calculate_greeks(S, K, T, r, sigma, 'call')
            put_greeks = self.calculate_greeks(S, K, T, r, sigma, 'put')
            
            results.append({
                'strike': K,
                'call_price': call_greeks['price'],
                'call_delta': call_greeks['delta'],
                'call_gamma': call_greeks['gamma'],
                'call_theta': call_greeks['theta'],
                'call_vega': call_greeks['vega'],
                'call_rho': call_greeks['rho'],
                'put_price': put_greeks['price'],
                'put_delta': put_greeks['delta'],
                'put_gamma': put_greeks['gamma'],
                'put_theta': put_greeks['theta'],
                'put_vega': put_greeks['vega'],
                'put_rho': put_greeks['rho']
            })
        
        return pd.DataFrame(results)
    
    def put_call_parity_check(self, 
                             call_price: float, 
                             put_price: float, 
                             S: float, 
                             K: float, 
                             T: float, 
                             r: float) -> Dict[str, float]:
        """
        Check put-call parity relationship.
        
        Args:
            call_price: Call option price
            put_price: Put option price
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            
        Returns:
            Dictionary with parity analysis
        """
        # Put-call parity: C - P = S - K*e^(-rT)
        theoretical_diff = S - K * np.exp(-r * T)
        actual_diff = call_price - put_price
        parity_violation = actual_diff - theoretical_diff
        
        return {
            'theoretical_difference': theoretical_diff,
            'actual_difference': actual_diff,
            'parity_violation': parity_violation,
            'arbitrage_opportunity': abs(parity_violation) > 0.01  # Threshold for arbitrage
        }
    
    def sensitivity_analysis(self, 
                           S: float, 
                           K: float, 
                           T: float, 
                           r: float, 
                           sigma: float, 
                           option_type: str = 'call',
                           parameter: str = 'S',
                           range_pct: float = 0.2,
                           num_points: int = 21) -> pd.DataFrame:
        """
        Perform sensitivity analysis on option price.
        
        Args:
            S, K, T, r, sigma: Black-Scholes parameters
            option_type: 'call' or 'put'
            parameter: Parameter to vary ('S', 'K', 'T', 'r', 'sigma')
            range_pct: Percentage range to vary parameter
            num_points: Number of points in analysis
            
        Returns:
            DataFrame with sensitivity analysis
        """
        base_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        base_value = base_params[parameter]
        
        # Create range of values
        if parameter == 'T':
            # For time, use absolute range
            param_range = np.linspace(max(0.01, base_value * (1 - range_pct)), 
                                    base_value * (1 + range_pct), num_points)
        else:
            param_range = np.linspace(base_value * (1 - range_pct), 
                                    base_value * (1 + range_pct), num_points)
        
        results = []
        
        for param_value in param_range:
            params = base_params.copy()
            params[parameter] = param_value
            
            price = self.calculate_price(
                params['S'], params['K'], params['T'], 
                params['r'], params['sigma'], option_type
            )
            
            greeks = self.calculate_greeks(
                params['S'], params['K'], params['T'], 
                params['r'], params['sigma'], option_type
            )
            
            results.append({
                parameter: param_value,
                'price': price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho']
            })
        
        return pd.DataFrame(results)
