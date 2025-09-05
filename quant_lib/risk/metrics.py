"""
Risk metrics and performance measurement calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RiskMetrics:
    """
    Comprehensive risk metrics and performance measurement calculations.
    """
    
    def __init__(self):
        """Initialize RiskMetrics calculator."""
        pass
    
    def sharpe_ratio(self, 
                    returns: Union[pd.Series, float], 
                    risk_free_rate: float = 0.02,
                    periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series or annualized return
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        if isinstance(returns, pd.Series):
            excess_returns = returns - risk_free_rate / periods_per_year
            return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        else:
            # Assuming returns is already annualized
            return (returns - risk_free_rate) / returns  # Need volatility for proper calculation
    
    def sortino_ratio(self, 
                     returns: pd.Series, 
                     risk_free_rate: float = 0.02,
                     periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (downside deviation version of Sharpe ratio).
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    
    def calmar_ratio(self, 
                    returns: pd.Series, 
                    prices: pd.Series = None,
                    periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / maximum drawdown).
        
        Args:
            returns: Return series
            prices: Price series (optional, will be calculated from returns if not provided)
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        
        if prices is not None:
            # Compute drawdown directly from price series
            running_max = prices.expanding().max()
            drawdown = (prices - running_max) / running_max
            max_dd = drawdown.min()
        else:
            # Calculate cumulative returns and then max drawdown using returns series
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = drawdown.min()
        
        if max_dd == 0:
            return np.inf
        
        return annualized_return / abs(max_dd)
    
    def maximum_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related statistics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with drawdown statistics
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        if len(drawdown) == 0:
            return {
                'max_drawdown': 0,
                'max_drawdown_start': None,
                'max_drawdown_end': None,
                'recovery_date': None,
                'drawdown_duration': None,
                'recovery_duration': None
            }
        
        max_drawdown_end = drawdown.idxmin()
        # Determine the start date of the drawdown period using label-based slicing
        try:
            max_drawdown_start = running_max.loc[:max_drawdown_end].idxmax()
        except Exception:
            max_drawdown_start = None
        
        # Recovery time
        recovery_date = None
        try:
            end_pos = cumulative_returns.index.get_loc(max_drawdown_end)
            if end_pos < len(cumulative_returns) - 1:
                recovery_mask = cumulative_returns[max_drawdown_end:] >= running_max[max_drawdown_end]
                if recovery_mask.any():
                    recovery_date = recovery_mask.idxmax()
        except Exception:
            recovery_date = None
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_start': max_drawdown_start,
            'max_drawdown_end': max_drawdown_end,
            'recovery_date': recovery_date,
            'drawdown_duration': (max_drawdown_end - max_drawdown_start).days if hasattr(max_drawdown_end, 'days') else None,
            'recovery_duration': (recovery_date - max_drawdown_end).days if recovery_date and hasattr(recovery_date, 'days') else None
        }
    
    def volatility(self, 
                  returns: pd.Series, 
                  periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Return series
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)
    
    def skewness(self, returns: pd.Series) -> float:
        """
        Calculate skewness of returns.
        
        Args:
            returns: Return series
            
        Returns:
            Skewness
        """
        return returns.skew()
    
    def kurtosis(self, returns: pd.Series) -> float:
        """
        Calculate excess kurtosis of returns.
        
        Args:
            returns: Return series
            
        Returns:
            Excess kurtosis
        """
        return returns.kurtosis()
    
    def beta(self, 
            asset_returns: pd.Series, 
            market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
        """
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if aligned_data.empty:
            return np.nan
        
        asset_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        return asset_ret.cov(market_ret) / market_ret.var()
    
    def tracking_error(self, 
                      portfolio_returns: pd.Series, 
                      benchmark_returns: pd.Series,
                      periods_per_year: int = 252) -> float:
        """
        Calculate tracking error (annualized standard deviation of excess returns).
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized tracking error
        """
        excess_returns = portfolio_returns - benchmark_returns
        return excess_returns.std() * np.sqrt(periods_per_year)
    
    def information_ratio(self, 
                         portfolio_returns: pd.Series, 
                         benchmark_returns: pd.Series,
                         periods_per_year: int = 252) -> float:
        """
        Calculate information ratio (excess return / tracking error).
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            periods_per_year: Number of periods per year
            
        Returns:
            Information ratio
        """
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = self.tracking_error(portfolio_returns, benchmark_returns, periods_per_year)
        
        if tracking_error == 0:
            return np.inf if excess_returns.mean() > 0 else -np.inf
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / tracking_error
    
    def upside_capture(self, 
                      portfolio_returns: pd.Series, 
                      benchmark_returns: pd.Series) -> float:
        """
        Calculate upside capture ratio.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Upside capture ratio
        """
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_ret = aligned_data.iloc[:, 0]
        benchmark_ret = aligned_data.iloc[:, 1]
        
        up_market = benchmark_ret > 0
        if up_market.sum() == 0:
            return np.nan
        
        portfolio_up = portfolio_ret[up_market].mean()
        benchmark_up = benchmark_ret[up_market].mean()
        
        return portfolio_up / benchmark_up if benchmark_up != 0 else np.nan
    
    def downside_capture(self, 
                        portfolio_returns: pd.Series, 
                        benchmark_returns: pd.Series) -> float:
        """
        Calculate downside capture ratio.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Downside capture ratio
        """
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_ret = aligned_data.iloc[:, 0]
        benchmark_ret = aligned_data.iloc[:, 1]
        
        down_market = benchmark_ret < 0
        if down_market.sum() == 0:
            return np.nan
        
        portfolio_down = portfolio_ret[down_market].mean()
        benchmark_down = benchmark_ret[down_market].mean()
        
        return portfolio_down / benchmark_down if benchmark_down != 0 else np.nan
    
    def tail_ratio(self, returns: pd.Series, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio (average of top percentile / average of bottom percentile).
        
        Args:
            returns: Return series
            percentile: Percentile for tail calculation
            
        Returns:
            Tail ratio
        """
        top_tail = returns.quantile(1 - percentile)
        bottom_tail = returns.quantile(percentile)
        
        top_returns = returns[returns >= top_tail].mean()
        bottom_returns = returns[returns <= bottom_tail].mean()
        
        return abs(top_returns / bottom_returns) if bottom_returns != 0 else np.nan
    
    def comprehensive_risk_report(self, 
                                 returns: pd.Series, 
                                 benchmark_returns: Optional[pd.Series] = None,
                                 risk_free_rate: float = 0.02,
                                 periods_per_year: int = 252) -> Dict[str, float]:
        """
        Generate comprehensive risk and performance report.
        
        Args:
            returns: Return series
            benchmark_returns: Optional benchmark return series
            risk_free_rate: Risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Dictionary with comprehensive metrics
        """
        # Basic statistics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        volatility = self.volatility(returns, periods_per_year)
        
        # Risk metrics
        sharpe = self.sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = self.sortino_ratio(returns, risk_free_rate, periods_per_year)
        calmar = self.calmar_ratio(returns, prices=None, periods_per_year=periods_per_year)
        max_dd_info = self.maximum_drawdown(returns)
        
        # Distribution metrics
        skew = self.skewness(returns)
        kurt = self.kurtosis(returns)
        tail_r = self.tail_ratio(returns)
        
        report = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd_info['max_drawdown'],
            'skewness': skew,
            'kurtosis': kurt,
            'tail_ratio': tail_r,
            'best_month': returns.max(),
            'worst_month': returns.min(),
            'positive_months': (returns > 0).mean(),
            'win_rate': (returns > 0).mean()
        }
        
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_returns is not None:
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if not aligned_data.empty:
                portfolio_ret = aligned_data.iloc[:, 0]
                benchmark_ret = aligned_data.iloc[:, 1]
                
                report.update({
                    'beta': self.beta(portfolio_ret, benchmark_ret),
                    'tracking_error': self.tracking_error(portfolio_ret, benchmark_ret, periods_per_year),
                    'information_ratio': self.information_ratio(portfolio_ret, benchmark_ret, periods_per_year),
                    'upside_capture': self.upside_capture(portfolio_ret, benchmark_ret),
                    'downside_capture': self.downside_capture(portfolio_ret, benchmark_ret)
                })
        
        return report
    
    def rolling_metrics(self, 
                       returns: pd.Series, 
                       window: int = 252,
                       metrics: list = None) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
        Args:
            returns: Return series
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['volatility', 'sharpe_ratio', 'max_drawdown']
        
        results = pd.DataFrame(index=returns.index)
        
        for metric in metrics:
            if metric == 'volatility':
                results[metric] = returns.rolling(window).std() * np.sqrt(252)
            elif metric == 'sharpe_ratio':
                rolling_mean = returns.rolling(window).mean()
                rolling_std = returns.rolling(window).std()
                results[metric] = np.sqrt(252) * rolling_mean / rolling_std
            elif metric == 'max_drawdown':
                rolling_dd = []
                for i in range(window, len(returns) + 1):
                    window_returns = returns.iloc[i-window:i]
                    dd_info = self.maximum_drawdown(window_returns)
                    rolling_dd.append(dd_info['max_drawdown'])
                
                results[metric] = pd.Series(rolling_dd, index=returns.index[window-1:])
        
        return results
