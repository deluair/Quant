"""
Efficient frontier construction and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class EfficientFrontier:
    """
    Efficient frontier construction and visualization for portfolio optimization.
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize Efficient Frontier.
        
        Args:
            returns: Historical returns DataFrame
        """
        self.returns = returns
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()
    
    def calculate_portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio return and volatility for given weights.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (expected_return, volatility)
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_vol
    
    def generate_efficient_frontier(self, 
                                  num_portfolios: int = 100,
                                  return_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Args:
            num_portfolios: Number of portfolios to generate
            return_range: Tuple of (min_return, max_return) for frontier
            
        Returns:
            DataFrame with efficient frontier portfolios
        """
        from .optimizer import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        
        # Get minimum variance and maximum return portfolios for range
        min_var_portfolio = optimizer.minimum_variance_portfolio(self.returns, {'long_only': True})
        
        # Calculate maximum possible return (100% in highest return asset)
        max_return_asset = np.argmax(self.mean_returns)
        max_return = self.mean_returns[max_return_asset]
        
        if return_range is None:
            min_return = min_var_portfolio['expected_return']
            return_range = (min_return, max_return * 0.95)  # Slightly below max to avoid infeasibility
        
        target_returns = np.linspace(return_range[0], return_range[1], num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                result = optimizer.mean_variance_optimization(
                    self.returns, 
                    target_return=target_return,
                    constraints={'long_only': True}
                )
                
                efficient_portfolios.append({
                    'target_return': target_return,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
                
            except Exception as e:
                # Skip infeasible portfolios
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def plot_efficient_frontier(self, 
                              num_portfolios: int = 100,
                              show_assets: bool = True,
                              show_special_portfolios: bool = True,
                              interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot efficient frontier with individual assets and special portfolios.
        
        Args:
            num_portfolios: Number of portfolios for frontier
            show_assets: Whether to show individual assets
            show_special_portfolios: Whether to show special portfolios
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        # Generate efficient frontier
        efficient_df = self.generate_efficient_frontier(num_portfolios)
        
        if interactive:
            fig = go.Figure()
            
            # Plot efficient frontier
            fig.add_trace(go.Scatter(
                x=efficient_df['volatility'],
                y=efficient_df['expected_return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ))
            
            # Plot individual assets
            if show_assets:
                asset_returns = []
                asset_vols = []
                
                for i in range(self.n_assets):
                    weights = np.zeros(self.n_assets)
                    weights[i] = 1
                    ret, vol = self.calculate_portfolio_stats(weights)
                    asset_returns.append(ret)
                    asset_vols.append(vol)
                
                fig.add_trace(go.Scatter(
                    x=asset_vols,
                    y=asset_returns,
                    mode='markers',
                    name='Individual Assets',
                    text=self.asset_names,
                    marker=dict(size=10, color='red')
                ))
            
            # Plot special portfolios
            if show_special_portfolios:
                from .optimizer import PortfolioOptimizer
                optimizer = PortfolioOptimizer()
                
                try:
                    # Minimum variance portfolio
                    min_var = optimizer.minimum_variance_portfolio(self.returns, {'long_only': True})
                    fig.add_trace(go.Scatter(
                        x=[min_var['volatility']],
                        y=[min_var['expected_return']],
                        mode='markers',
                        name='Min Variance',
                        marker=dict(size=12, color='green', symbol='diamond')
                    ))
                    
                    # Maximum Sharpe ratio portfolio
                    max_sharpe = optimizer.maximum_sharpe_portfolio(self.returns, constraints={'long_only': True})
                    fig.add_trace(go.Scatter(
                        x=[max_sharpe['volatility']],
                        y=[max_sharpe['expected_return']],
                        mode='markers',
                        name='Max Sharpe',
                        marker=dict(size=12, color='orange', symbol='star')
                    ))
                    
                except Exception as e:
                    print(f"Could not plot special portfolios: {e}")
            
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                hovermode='closest'
            )
            
            return fig
        
        else:
            # Matplotlib plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot efficient frontier
            ax.plot(efficient_df['volatility'], efficient_df['expected_return'], 
                   'b-', linewidth=3, label='Efficient Frontier')
            
            # Plot individual assets
            if show_assets:
                asset_returns = []
                asset_vols = []
                
                for i in range(self.n_assets):
                    weights = np.zeros(self.n_assets)
                    weights[i] = 1
                    ret, vol = self.calculate_portfolio_stats(weights)
                    asset_returns.append(ret)
                    asset_vols.append(vol)
                
                ax.scatter(asset_vols, asset_returns, c='red', s=100, 
                          label='Individual Assets', alpha=0.7)
                
                # Add asset labels
                for i, name in enumerate(self.asset_names):
                    ax.annotate(name, (asset_vols[i], asset_returns[i]), 
                              xytext=(5, 5), textcoords='offset points')
            
            # Plot special portfolios
            if show_special_portfolios:
                from .optimizer import PortfolioOptimizer
                optimizer = PortfolioOptimizer()
                
                try:
                    min_var = optimizer.minimum_variance_portfolio(self.returns, {'long_only': True})
                    ax.scatter(min_var['volatility'], min_var['expected_return'], 
                             c='green', s=150, marker='D', label='Min Variance')
                    
                    max_sharpe = optimizer.maximum_sharpe_portfolio(self.returns, constraints={'long_only': True})
                    ax.scatter(max_sharpe['volatility'], max_sharpe['expected_return'], 
                             c='orange', s=150, marker='*', label='Max Sharpe')
                    
                except Exception as e:
                    print(f"Could not plot special portfolios: {e}")
            
            ax.set_xlabel('Volatility (Risk)')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def capital_allocation_line(self, 
                              risk_free_rate: float = 0.02,
                              tangency_portfolio: Optional[Dict] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate Capital Allocation Line (CAL) from risk-free asset to tangency portfolio.
        
        Args:
            risk_free_rate: Risk-free rate
            tangency_portfolio: Tangency portfolio (if None, calculate max Sharpe portfolio)
            
        Returns:
            Dictionary with CAL parameters
        """
        if tangency_portfolio is None:
            from .optimizer import PortfolioOptimizer
            optimizer = PortfolioOptimizer()
            tangency_portfolio = optimizer.maximum_sharpe_portfolio(
                self.returns, risk_free_rate, {'long_only': True}
            )
        
        # CAL slope (Sharpe ratio of tangency portfolio)
        cal_slope = (tangency_portfolio['expected_return'] - risk_free_rate / 252) / tangency_portfolio['volatility']
        
        # Generate CAL points
        volatilities = np.linspace(0, tangency_portfolio['volatility'] * 2, 100)
        cal_returns = risk_free_rate / 252 + cal_slope * volatilities
        
        return {
            'volatilities': volatilities,
            'returns': cal_returns,
            'slope': cal_slope,
            'tangency_portfolio': tangency_portfolio,
            'risk_free_rate': risk_free_rate / 252
        }
    
    def plot_cal_with_frontier(self, 
                             risk_free_rate: float = 0.02,
                             interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot efficient frontier with Capital Allocation Line.
        
        Args:
            risk_free_rate: Risk-free rate
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        # Generate efficient frontier
        efficient_df = self.generate_efficient_frontier()
        
        # Calculate CAL
        cal_data = self.capital_allocation_line(risk_free_rate)
        
        if interactive:
            fig = go.Figure()
            
            # Plot efficient frontier
            fig.add_trace(go.Scatter(
                x=efficient_df['volatility'],
                y=efficient_df['expected_return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ))
            
            # Plot CAL
            fig.add_trace(go.Scatter(
                x=cal_data['volatilities'],
                y=cal_data['returns'],
                mode='lines',
                name='Capital Allocation Line',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Plot risk-free rate
            fig.add_trace(go.Scatter(
                x=[0],
                y=[cal_data['risk_free_rate']],
                mode='markers',
                name='Risk-Free Asset',
                marker=dict(size=10, color='green')
            ))
            
            # Plot tangency portfolio
            tangency = cal_data['tangency_portfolio']
            fig.add_trace(go.Scatter(
                x=[tangency['volatility']],
                y=[tangency['expected_return']],
                mode='markers',
                name='Tangency Portfolio',
                marker=dict(size=12, color='orange', symbol='star')
            ))
            
            fig.update_layout(
                title='Efficient Frontier with Capital Allocation Line',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                hovermode='closest'
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot efficient frontier
            ax.plot(efficient_df['volatility'], efficient_df['expected_return'], 
                   'b-', linewidth=3, label='Efficient Frontier')
            
            # Plot CAL
            ax.plot(cal_data['volatilities'], cal_data['returns'], 
                   'r--', linewidth=2, label='Capital Allocation Line')
            
            # Plot risk-free rate
            ax.scatter(0, cal_data['risk_free_rate'], c='green', s=100, 
                      label='Risk-Free Asset')
            
            # Plot tangency portfolio
            tangency = cal_data['tangency_portfolio']
            ax.scatter(tangency['volatility'], tangency['expected_return'], 
                      c='orange', s=150, marker='*', label='Tangency Portfolio')
            
            ax.set_xlabel('Volatility (Risk)')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier with Capital Allocation Line')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def portfolio_composition_analysis(self, 
                                     efficient_df: pd.DataFrame,
                                     top_n: int = 5) -> pd.DataFrame:
        """
        Analyze portfolio composition along the efficient frontier.
        
        Args:
            efficient_df: Efficient frontier DataFrame
            top_n: Number of top assets to show
            
        Returns:
            DataFrame with portfolio compositions
        """
        compositions = []
        
        for idx, row in efficient_df.iterrows():
            weights = row['weights']
            
            # Get top N assets by weight
            weight_series = pd.Series(weights, index=self.asset_names)
            top_assets = weight_series.nlargest(top_n)
            
            composition = {
                'expected_return': row['expected_return'],
                'volatility': row['volatility'],
                'sharpe_ratio': row['sharpe_ratio']
            }
            
            for i, (asset, weight) in enumerate(top_assets.items()):
                composition[f'asset_{i+1}'] = asset
                composition[f'weight_{i+1}'] = weight
            
            compositions.append(composition)
        
        return pd.DataFrame(compositions)
    
    def risk_return_heatmap(self, 
                          grid_size: int = 20,
                          interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Create risk-return heatmap for two-asset portfolios.
        
        Args:
            grid_size: Size of the grid for heatmap
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if self.n_assets != 2:
            raise ValueError("Risk-return heatmap only works for 2-asset portfolios")
        
        # Create weight grid
        weights_1 = np.linspace(0, 1, grid_size)
        weights_2 = 1 - weights_1
        
        returns_grid = np.zeros((grid_size, grid_size))
        volatility_grid = np.zeros((grid_size, grid_size))
        sharpe_grid = np.zeros((grid_size, grid_size))
        
        for i, w1 in enumerate(weights_1):
            for j, w2 in enumerate(weights_2):
                if abs(w1 + w2 - 1) < 1e-6:  # Ensure weights sum to 1
                    weights = np.array([w1, w2])
                    ret, vol = self.calculate_portfolio_stats(weights)
                    sharpe = ret / vol if vol > 0 else 0
                    
                    returns_grid[i, j] = ret
                    volatility_grid[i, j] = vol
                    sharpe_grid[i, j] = sharpe
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=sharpe_grid,
                x=weights_2,
                y=weights_1,
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio')
            ))
            
            fig.update_layout(
                title='Portfolio Risk-Return Heatmap',
                xaxis_title=f'{self.asset_names[1]} Weight',
                yaxis_title=f'{self.asset_names[0]} Weight'
            )
            
            return fig
        
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Returns heatmap
            im1 = axes[0].imshow(returns_grid, cmap='viridis', aspect='auto')
            axes[0].set_title('Expected Returns')
            axes[0].set_xlabel(f'{self.asset_names[1]} Weight')
            axes[0].set_ylabel(f'{self.asset_names[0]} Weight')
            plt.colorbar(im1, ax=axes[0])
            
            # Volatility heatmap
            im2 = axes[1].imshow(volatility_grid, cmap='viridis', aspect='auto')
            axes[1].set_title('Volatility')
            axes[1].set_xlabel(f'{self.asset_names[1]} Weight')
            axes[1].set_ylabel(f'{self.asset_names[0]} Weight')
            plt.colorbar(im2, ax=axes[1])
            
            # Sharpe ratio heatmap
            im3 = axes[2].imshow(sharpe_grid, cmap='viridis', aspect='auto')
            axes[2].set_title('Sharpe Ratio')
            axes[2].set_xlabel(f'{self.asset_names[1]} Weight')
            axes[2].set_ylabel(f'{self.asset_names[0]} Weight')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            return fig
