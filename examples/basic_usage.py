"""
Basic usage examples for the Quantitative Finance Library.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from quant_lib import (
    MarketData, BlackScholes, CAPM, MonteCarlo, 
    VaR, PortfolioOptimizer, EfficientFrontier, ReturnPredictor
)
import pandas as pd
import numpy as np


def market_data_example():
    """Example of fetching and analyzing market data."""
    print("=== Market Data Example ===")
    
    # Initialize market data fetcher
    market_data = MarketData()
    
    # Fetch stock data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = market_data.get_stock_data(symbols, period='1y')
    
    print(f"Fetched data for {len(symbols)} symbols")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate returns
    returns = market_data.get_returns(data)
    print(f"Average daily returns:\n{returns.mean()}")
    
    # Get correlation matrix
    correlation = market_data.get_correlation_matrix(data)
    print(f"Correlation matrix:\n{correlation}")


def option_pricing_example():
    """Example of option pricing using different models."""
    print("\n=== Option Pricing Example ===")
    
    # Parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Black-Scholes model
    bs = BlackScholes()
    bs_price = bs.calculate_price(S, K, T, r, sigma, 'call')
    bs_greeks = bs.calculate_greeks(S, K, T, r, sigma, 'call')
    
    print(f"Black-Scholes Call Price: ${bs_price:.4f}")
    print(f"Delta: {bs_greeks['delta']:.4f}")
    print(f"Gamma: {bs_greeks['gamma']:.4f}")
    
    # Monte Carlo model
    mc = MonteCarlo(seed=42)
    mc_result = mc.european_option_price(S, K, T, r, sigma, 'call', 100000)
    
    print(f"Monte Carlo Call Price: ${mc_result['price']:.4f}")
    print(f"Standard Error: ${mc_result['std_error']:.4f}")
    
    # Binomial tree model
    from quant_lib.models.binomial import BinomialTree
    bt = BinomialTree()
    bt_result = bt.european_option_price(S, K, T, r, sigma, 'call', 100)
    
    print(f"Binomial Tree Call Price: ${bt_result['price']:.4f}")


def risk_analysis_example():
    """Example of risk analysis and VaR calculation."""
    print("\n=== Risk Analysis Example ===")
    
    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    # Calculate VaR
    var_calc = VaR()
    
    # Historical VaR
    historical_var = var_calc.historical_var(returns, confidence_level=0.95)
    print(f"Historical VaR (95%): {historical_var:.4f}")
    
    # Parametric VaR
    parametric_var = var_calc.parametric_var(returns, confidence_level=0.95)
    print(f"Parametric VaR (95%): {parametric_var:.4f}")
    
    # Expected Shortfall
    expected_shortfall = var_calc.expected_shortfall(returns, confidence_level=0.95)
    print(f"Expected Shortfall (95%): {expected_shortfall:.4f}")
    
    # Risk metrics
    from quant_lib.risk.metrics import RiskMetrics
    risk_metrics = RiskMetrics()
    
    sharpe_ratio = risk_metrics.sharpe_ratio(returns)
    max_dd = risk_metrics.maximum_drawdown(returns)
    
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_dd['max_drawdown']:.4f}")


def portfolio_optimization_example():
    """Example of portfolio optimization."""
    print("\n=== Portfolio Optimization Example ===")
    
    # Generate sample returns data for multiple assets
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    returns_data = {
        'AAPL': np.random.normal(0.0008, 0.025, 252),
        'GOOGL': np.random.normal(0.0006, 0.022, 252),
        'MSFT': np.random.normal(0.0007, 0.020, 252),
        'AMZN': np.random.normal(0.0005, 0.028, 252)
    }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Portfolio optimization
    optimizer = PortfolioOptimizer()
    
    # Maximum Sharpe ratio portfolio
    max_sharpe = optimizer.maximum_sharpe_portfolio(returns_df, constraints={'long_only': True})
    print("Maximum Sharpe Portfolio:")
    print(f"Expected Return: {max_sharpe['expected_return']:.4f}")
    print(f"Volatility: {max_sharpe['volatility']:.4f}")
    print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.4f}")
    print(f"Weights: {dict(zip(returns_df.columns, max_sharpe['weights']))}")
    
    # Minimum variance portfolio
    min_var = optimizer.minimum_variance_portfolio(returns_df, {'long_only': True})
    print("\nMinimum Variance Portfolio:")
    print(f"Expected Return: {min_var['expected_return']:.4f}")
    print(f"Volatility: {min_var['volatility']:.4f}")
    print(f"Weights: {dict(zip(returns_df.columns, min_var['weights']))}")
    
    # Efficient frontier
    ef = EfficientFrontier(returns_df)
    efficient_portfolios = ef.generate_efficient_frontier(20)
    print(f"\nGenerated {len(efficient_portfolios)} efficient portfolios")


def capm_analysis_example():
    """Example of CAPM analysis."""
    print("\n=== CAPM Analysis Example ===")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    market_returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
    asset_returns = pd.Series(0.2 + 1.2 * market_returns + np.random.normal(0, 0.01, 252), index=dates)
    
    # CAPM analysis
    capm = CAPM()
    capm_results = capm.calculate_beta(asset_returns, market_returns, risk_free_rate=0.02)
    
    print(f"Beta: {capm_results['beta']:.4f}")
    print(f"Alpha (annualized): {capm_results['alpha_annual']:.4f}")
    print(f"R-squared: {capm_results['r_squared']:.4f}")
    print(f"Correlation: {capm_results['correlation']:.4f}")
    
    # Expected return using CAPM
    expected_return = capm.expected_return(
        capm_results['beta'], 
        market_returns.mean() * 252, 
        0.02
    )
    print(f"Expected Return (CAPM): {expected_return:.4f}")


def ml_prediction_example():
    """Example of ML-based return prediction."""
    print("\n=== ML Prediction Example ===")
    
    try:
        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        
        # Simulate price data with trend and noise
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 500)))
        price_data = pd.DataFrame({
            'Close': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 500))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 500))),
            'Volume': np.random.randint(1000000, 10000000, 500)
        }, index=dates)
        
        # Feature engineering
        from quant_lib.ml.features import FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        features = feature_engineer.create_technical_features(price_data)
        target = feature_engineer.create_target_variable(price_data, horizon=1, target_type='return')
        
        print(f"Created {features.shape[1]} features")
        print(f"Target variable has {len(target)} observations")
        
        # ML prediction
        predictor = ReturnPredictor()
        data_splits = predictor.prepare_data(features, target, test_size=0.2)
        
        # Train models
        linear_models = predictor.train_linear_models(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val']
        )
        
        tree_models = predictor.train_tree_models(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val']
        )
        
        print(f"Trained {len(linear_models)} linear models")
        print(f"Trained {len(tree_models)} tree models")
        
        # Model comparison
        if predictor.performance_metrics:
            comparison = predictor.model_comparison()
            print("\nModel Performance:")
            print(comparison[['val_r2', 'val_mse']].round(4))
        
    except Exception as e:
        print(f"ML example error (likely missing dependencies): {str(e)}")


def main():
    """Run all examples."""
    print("Quantitative Finance Library - Basic Usage Examples")
    print("=" * 60)
    
    try:
        market_data_example()
    except Exception as e:
        print(f"Market data example error: {str(e)}")
    
    try:
        option_pricing_example()
    except Exception as e:
        print(f"Option pricing example error: {str(e)}")
    
    try:
        risk_analysis_example()
    except Exception as e:
        print(f"Risk analysis example error: {str(e)}")
    
    try:
        portfolio_optimization_example()
    except Exception as e:
        print(f"Portfolio optimization example error: {str(e)}")
    
    try:
        capm_analysis_example()
    except Exception as e:
        print(f"CAPM analysis example error: {str(e)}")
    
    try:
        ml_prediction_example()
    except Exception as e:
        print(f"ML prediction example error: {str(e)}")


if __name__ == "__main__":
    main()
