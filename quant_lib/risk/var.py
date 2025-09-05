"""
Value at Risk (VaR) and Expected Shortfall calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Union, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class VaR:
    """
    Value at Risk (VaR) and Expected Shortfall (ES) calculations using multiple methods.
    """
    
    def __init__(self):
        """Initialize VaR calculator."""
        pass
    
    def historical_var(self, 
                      returns: Union[pd.Series, pd.DataFrame], 
                      confidence_level: float = 0.95,
                      window: Optional[int] = None) -> Union[float, pd.Series]:
        """
        Calculate Historical VaR using empirical distribution.
        
        Args:
            returns: Return series or DataFrame
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            window: Rolling window size (None for full sample)
            
        Returns:
            VaR value(s)
        """
        if window is None:
            # Full sample VaR
            if isinstance(returns, pd.DataFrame):
                return returns.quantile(1 - confidence_level)
            else:
                return returns.quantile(1 - confidence_level)
        else:
            # Rolling VaR
            if isinstance(returns, pd.DataFrame):
                return returns.rolling(window=window).quantile(1 - confidence_level)
            else:
                return returns.rolling(window=window).quantile(1 - confidence_level)
    
    def parametric_var(self, 
                      returns: Union[pd.Series, pd.DataFrame], 
                      confidence_level: float = 0.95,
                      distribution: str = 'normal') -> Union[float, pd.Series]:
        """
        Calculate Parametric VaR assuming a specific distribution.
        
        Args:
            returns: Return series or DataFrame
            confidence_level: Confidence level
            distribution: 'normal' or 't' (Student's t-distribution)
            
        Returns:
            VaR value(s)
        """
        if isinstance(returns, pd.DataFrame):
            mean_returns = returns.mean()
            std_returns = returns.std()
        else:
            mean_returns = returns.mean()
            std_returns = returns.std()
        
        if distribution == 'normal':
            z_score = stats.norm.ppf(1 - confidence_level)
        elif distribution == 't':
            # Fit t-distribution
            if isinstance(returns, pd.DataFrame):
                # For DataFrame, use average degrees of freedom
                df_params = []
                for col in returns.columns:
                    params = stats.t.fit(returns[col].dropna())
                    df_params.append(params[0])
                avg_df = np.mean(df_params)
                z_score = stats.t.ppf(1 - confidence_level, avg_df)
            else:
                params = stats.t.fit(returns.dropna())
                df = params[0]
                z_score = stats.t.ppf(1 - confidence_level, df)
        else:
            raise ValueError("Distribution must be 'normal' or 't'")
        
        var = mean_returns + z_score * std_returns
        return var
    
    def monte_carlo_var(self, 
                       returns: Union[pd.Series, pd.DataFrame], 
                       confidence_level: float = 0.95,
                       num_simulations: int = 10000,
                       method: str = 'bootstrap') -> Union[float, pd.Series]:
        """
        Calculate Monte Carlo VaR.
        
        Args:
            returns: Return series or DataFrame
            confidence_level: Confidence level
            num_simulations: Number of Monte Carlo simulations
            method: 'bootstrap' or 'parametric'
            
        Returns:
            VaR value(s)
        """
        if method == 'bootstrap':
            # Bootstrap resampling
            if isinstance(returns, pd.DataFrame):
                simulated_returns = []
                for _ in range(num_simulations):
                    sample = returns.sample(n=len(returns), replace=True)
                    simulated_returns.append(sample.mean())
                simulated_returns = pd.DataFrame(simulated_returns)
                return simulated_returns.quantile(1 - confidence_level)
            else:
                simulated_returns = []
                for _ in range(num_simulations):
                    sample = returns.sample(n=len(returns), replace=True)
                    simulated_returns.append(sample.mean())
                return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric Monte Carlo
            if isinstance(returns, pd.DataFrame):
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                simulated_returns = np.random.multivariate_normal(
                    mean_returns, cov_matrix, num_simulations
                )
                return pd.Series(
                    np.percentile(simulated_returns, (1 - confidence_level) * 100, axis=0),
                    index=returns.columns
                )
            else:
                mean_return = returns.mean()
                std_return = returns.std()
                simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
                return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        else:
            raise ValueError("Method must be 'bootstrap' or 'parametric'")
    
    def expected_shortfall(self, 
                          returns: Union[pd.Series, pd.DataFrame], 
                          confidence_level: float = 0.95,
                          method: str = 'historical') -> Union[float, pd.Series]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Return series or DataFrame
            confidence_level: Confidence level
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            Expected Shortfall value(s)
        """
        if method == 'historical':
            var = self.historical_var(returns, confidence_level)
            
            if isinstance(returns, pd.DataFrame):
                es = {}
                for col in returns.columns:
                    mask = returns[col] <= var[col]
                    es[col] = returns[col][mask].mean()
                return pd.Series(es)
            else:
                mask = returns <= var
                return returns[mask].mean()
        
        elif method == 'parametric':
            if isinstance(returns, pd.DataFrame):
                mean_returns = returns.mean()
                std_returns = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                phi_z = stats.norm.pdf(z_score)
                es = mean_returns - std_returns * phi_z / (1 - confidence_level)
                return es
            else:
                mean_return = returns.mean()
                std_return = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                phi_z = stats.norm.pdf(z_score)
                return mean_return - std_return * phi_z / (1 - confidence_level)
        
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")
    
    def portfolio_var(self, 
                     returns: pd.DataFrame, 
                     weights: np.ndarray, 
                     confidence_level: float = 0.95,
                     method: str = 'parametric') -> Dict[str, float]:
        """
        Calculate portfolio VaR and component VaR.
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            confidence_level: Confidence level
            method: VaR calculation method
            
        Returns:
            Dictionary with portfolio VaR and component VaRs
        """
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Portfolio VaR
        if method == 'historical':
            portfolio_var = self.historical_var(portfolio_returns, confidence_level)
        elif method == 'parametric':
            portfolio_var = self.parametric_var(portfolio_returns, confidence_level)
        elif method == 'monte_carlo':
            portfolio_var = self.monte_carlo_var(portfolio_returns, confidence_level)
        else:
            raise ValueError("Invalid method")
        
        # Component VaR calculation
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # Marginal VaR
        marginal_var = (cov_matrix @ weights) / portfolio_vol
        z_score = stats.norm.ppf(1 - confidence_level)
        marginal_var = marginal_var * z_score
        
        # Component VaR
        component_var = weights * marginal_var
        
        return {
            'portfolio_var': portfolio_var,
            'component_var': dict(zip(returns.columns, component_var)),
            'marginal_var': dict(zip(returns.columns, marginal_var)),
            'portfolio_volatility': portfolio_vol
        }
    
    def backtesting(self, 
                   returns: pd.Series, 
                   var_estimates: pd.Series, 
                   confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Backtest VaR model performance.
        
        Args:
            returns: Actual returns
            var_estimates: VaR estimates
            confidence_level: Confidence level used for VaR
            
        Returns:
            Dictionary with backtesting results
        """
        # Align data
        aligned_data = pd.concat([returns, var_estimates], axis=1).dropna()
        actual_returns = aligned_data.iloc[:, 0]
        var_forecasts = aligned_data.iloc[:, 1]
        
        # VaR violations
        violations = actual_returns < var_forecasts
        violation_rate = violations.mean()
        expected_violation_rate = 1 - confidence_level
        
        # Number of violations
        num_violations = violations.sum()
        num_observations = len(violations)
        expected_violations = expected_violation_rate * num_observations
        
        # Kupiec POF test
        lr_pof = -2 * np.log(
            (expected_violation_rate ** num_violations) * 
            ((1 - expected_violation_rate) ** (num_observations - num_violations))
        ) + 2 * np.log(
            (violation_rate ** num_violations) * 
            ((1 - violation_rate) ** (num_observations - num_violations))
        )
        
        # Christoffersen Independence test
        violations_shifted = violations.shift(1).fillna(False)
        n00 = ((~violations) & (~violations_shifted)).sum()
        n01 = ((~violations) & violations_shifted).sum()
        n10 = (violations & (~violations_shifted)).sum()
        n11 = (violations & violations_shifted).sum()
        
        if n01 + n11 > 0 and n10 + n11 > 0:
            pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            pi = (n01 + n11) / (n00 + n01 + n10 + n11)
            
            if pi01 > 0 and pi11 > 0 and pi > 0:
                lr_ind = -2 * np.log(
                    ((1 - pi) ** (n00 + n10)) * (pi ** (n01 + n11))
                ) + 2 * np.log(
                    ((1 - pi01) ** n00) * (pi01 ** n01) * 
                    ((1 - pi11) ** n10) * (pi11 ** n11)
                )
            else:
                lr_ind = 0
        else:
            lr_ind = 0
        
        return {
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'num_violations': num_violations,
            'expected_violations': expected_violations,
            'kupiec_pof_stat': lr_pof,
            'christoffersen_ind_stat': lr_ind,
            'kupiec_pof_pvalue': 1 - stats.chi2.cdf(lr_pof, 1),
            'christoffersen_ind_pvalue': 1 - stats.chi2.cdf(lr_ind, 1)
        }
    
    def stress_var(self, 
                  returns: pd.DataFrame, 
                  stress_scenarios: Dict[str, np.ndarray],
                  confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calculate VaR under stress scenarios.
        
        Args:
            returns: Asset returns DataFrame
            stress_scenarios: Dictionary of stress scenario shocks
            confidence_level: Confidence level
            
        Returns:
            DataFrame with stress VaR results
        """
        results = []
        
        for scenario_name, shocks in stress_scenarios.items():
            # Apply stress shocks
            stressed_returns = returns + shocks
            
            # Calculate VaR for stressed returns
            stressed_var = self.historical_var(stressed_returns, confidence_level)
            normal_var = self.historical_var(returns, confidence_level)
            
            if isinstance(stressed_var, pd.Series):
                for asset in returns.columns:
                    results.append({
                        'scenario': scenario_name,
                        'asset': asset,
                        'normal_var': normal_var[asset],
                        'stressed_var': stressed_var[asset],
                        'var_change': stressed_var[asset] - normal_var[asset]
                    })
            else:
                results.append({
                    'scenario': scenario_name,
                    'asset': 'portfolio',
                    'normal_var': normal_var,
                    'stressed_var': stressed_var,
                    'var_change': stressed_var - normal_var
                })
        
        return pd.DataFrame(results)
