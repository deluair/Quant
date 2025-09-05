"""
Capital Asset Pricing Model (CAPM) implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CAPM:
    """
    Capital Asset Pricing Model implementation for calculating expected returns,
    beta coefficients, and risk-adjusted performance metrics.
    """
    
    def __init__(self):
        """Initialize CAPM calculator."""
        pass
    
    def calculate_beta(self, 
                      asset_returns: pd.Series, 
                      market_returns: pd.Series,
                      risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate beta coefficient for an asset relative to market.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Dictionary with beta, alpha, R-squared, and other statistics
        """
        # Align the series and remove NaN values
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if aligned_data.empty:
            raise ValueError("No overlapping data between asset and market returns")
        
        asset_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / 252  # Assuming daily returns
        
        # Calculate excess returns
        asset_excess = asset_ret - rf_period
        market_excess = market_ret - rf_period
        
        # Linear regression: asset_excess = alpha + beta * market_excess
        slope, intercept, r_value, p_value, std_err = stats.linregress(market_excess, asset_excess)
        
        # Calculate additional statistics
        beta = slope
        alpha = intercept
        r_squared = r_value ** 2
        
        # Annualized alpha
        alpha_annual = alpha * 252
        
        # Standard error of beta
        beta_std_error = std_err
        
        # T-statistic for beta
        t_stat = beta / beta_std_error if beta_std_error != 0 else 0
        
        return {
            'beta': beta,
            'alpha': alpha,
            'alpha_annual': alpha_annual,
            'r_squared': r_squared,
            'p_value': p_value,
            'beta_std_error': beta_std_error,
            't_statistic': t_stat,
            'correlation': r_value
        }
    
    def expected_return(self, 
                       beta: float, 
                       market_return: float, 
                       risk_free_rate: float) -> float:
        """
        Calculate expected return using CAPM formula.
        
        Args:
            beta: Asset beta coefficient
            market_return: Expected market return
            risk_free_rate: Risk-free rate
            
        Returns:
            Expected return
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)
    
    def security_market_line(self, 
                           betas: np.ndarray, 
                           market_return: float, 
                           risk_free_rate: float) -> np.ndarray:
        """
        Calculate Security Market Line (SML) for given beta range.
        
        Args:
            betas: Array of beta values
            market_return: Expected market return
            risk_free_rate: Risk-free rate
            
        Returns:
            Array of expected returns
        """
        return risk_free_rate + betas * (market_return - risk_free_rate)
    
    def treynor_ratio(self, 
                     portfolio_return: float, 
                     beta: float, 
                     risk_free_rate: float) -> float:
        """
        Calculate Treynor ratio (risk-adjusted return per unit of systematic risk).
        
        Args:
            portfolio_return: Portfolio return
            beta: Portfolio beta
            risk_free_rate: Risk-free rate
            
        Returns:
            Treynor ratio
        """
        if beta == 0:
            return np.inf if portfolio_return > risk_free_rate else -np.inf
        return (portfolio_return - risk_free_rate) / beta
    
    def jensens_alpha(self, 
                     portfolio_return: float, 
                     beta: float, 
                     market_return: float, 
                     risk_free_rate: float) -> float:
        """
        Calculate Jensen's Alpha (portfolio's excess return over CAPM prediction).
        
        Args:
            portfolio_return: Actual portfolio return
            beta: Portfolio beta
            market_return: Market return
            risk_free_rate: Risk-free rate
            
        Returns:
            Jensen's Alpha
        """
        expected_return = self.expected_return(beta, market_return, risk_free_rate)
        return portfolio_return - expected_return
    
    def portfolio_beta(self, 
                      weights: np.ndarray, 
                      individual_betas: np.ndarray) -> float:
        """
        Calculate portfolio beta as weighted average of individual asset betas.
        
        Args:
            weights: Portfolio weights
            individual_betas: Individual asset betas
            
        Returns:
            Portfolio beta
        """
        return np.sum(weights * individual_betas)
    
    def multi_factor_model(self, 
                          asset_returns: pd.Series, 
                          factors: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate multi-factor model (extension of CAPM).
        
        Args:
            asset_returns: Asset return series
            factors: DataFrame with factor returns (columns are factors)
            
        Returns:
            Dictionary with factor loadings and statistics
        """
        # Align data and remove NaN values
        aligned_data = pd.concat([asset_returns, factors], axis=1).dropna()
        if aligned_data.empty:
            raise ValueError("No overlapping data between asset returns and factors")
        
        y = aligned_data.iloc[:, 0].values  # Asset returns
        X = aligned_data.iloc[:, 1:].values  # Factor returns
        
        # Add constant term for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Ordinary Least Squares regression
        try:
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            
            # Calculate R-squared
            y_pred = X_with_const @ coefficients
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Prepare results
            results = {
                'alpha': coefficients[0],
                'factor_loadings': coefficients[1:],
                'r_squared': r_squared,
                'factor_names': factors.columns.tolist()
            }
            
            return results
            
        except np.linalg.LinAlgError:
            raise ValueError("Unable to solve regression - check for multicollinearity")
    
    def rolling_beta(self, 
                    asset_returns: pd.Series, 
                    market_returns: pd.Series,
                    window: int = 252,
                    risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate rolling beta over time.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window size
            risk_free_rate: Risk-free rate
            
        Returns:
            Series of rolling beta values
        """
        # Align the series
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < window:
            raise ValueError(f"Insufficient data: need at least {window} observations")
        
        asset_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        rf_period = risk_free_rate / 252
        
        def calc_beta(window_data):
            if len(window_data) < 2:
                return np.nan
            asset_excess = window_data.iloc[:, 0] - rf_period
            market_excess = window_data.iloc[:, 1] - rf_period
            
            if market_excess.var() == 0:
                return np.nan
            
            return asset_excess.cov(market_excess) / market_excess.var()
        
        rolling_betas = aligned_data.rolling(window=window).apply(
            lambda x: calc_beta(x), raw=False
        )
        
        return rolling_betas.iloc[:, 0]  # Return only the beta series
    
    def capm_performance_attribution(self, 
                                   portfolio_returns: pd.Series,
                                   market_returns: pd.Series,
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Comprehensive CAPM-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market return series
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        # Calculate basic CAPM statistics
        capm_stats = self.calculate_beta(portfolio_returns, market_returns, risk_free_rate)
        
        # Calculate annualized returns
        portfolio_annual = (1 + portfolio_returns.mean()) ** 252 - 1
        market_annual = (1 + market_returns.mean()) ** 252 - 1
        
        # Calculate volatilities
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        market_vol = market_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratios
        portfolio_sharpe = (portfolio_annual - risk_free_rate) / portfolio_vol
        market_sharpe = (market_annual - risk_free_rate) / market_vol
        
        # Calculate Treynor ratio
        treynor = self.treynor_ratio(portfolio_annual, capm_stats['beta'], risk_free_rate)
        
        # Calculate Jensen's Alpha
        jensens_alpha = self.jensens_alpha(
            portfolio_annual, capm_stats['beta'], market_annual, risk_free_rate
        )
        
        return {
            **capm_stats,
            'portfolio_return_annual': portfolio_annual,
            'market_return_annual': market_annual,
            'portfolio_volatility': portfolio_vol,
            'market_volatility': market_vol,
            'portfolio_sharpe': portfolio_sharpe,
            'market_sharpe': market_sharpe,
            'treynor_ratio': treynor,
            'jensens_alpha': jensens_alpha,
            'information_ratio': capm_stats['alpha_annual'] / (portfolio_returns.std() * np.sqrt(252))
        }
