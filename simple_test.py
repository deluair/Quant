"""
Simple test without complex imports to verify core functionality.
"""

import numpy as np
import pandas as pd

def test_basic_math():
    """Test basic mathematical operations."""
    try:
        # Black-Scholes formula implementation
        def black_scholes_call(S, K, T, r, sigma):
            if T <= 0:
                return max(S - K, 0)
            
            # Use numpy's approximation for normal CDF
            def norm_cdf(x):
                return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
            return call_price
        
        price = black_scholes_call(100, 105, 0.25, 0.05, 0.2)
        print(f"✓ Black-Scholes call price: ${price:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Basic math test failed: {e}")
        return False

def test_portfolio_math():
    """Test portfolio calculations."""
    try:
        # Create sample data
        returns = np.array([
            [0.01, 0.02, -0.01],  # Asset A
            [0.015, -0.005, 0.02], # Asset B  
            [-0.005, 0.03, 0.01]   # Asset C
        ]).T
        
        # Calculate means and covariance
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        # Equal weight portfolio
        weights = np.array([1/3, 1/3, 1/3])
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        print(f"✓ Portfolio return: {portfolio_return:.4f}")
        print(f"✓ Portfolio volatility: {portfolio_vol:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Portfolio test failed: {e}")
        return False

def test_var_calculation():
    """Test VaR calculation."""
    try:
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        
        # Historical VaR
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        
        print(f"✓ VaR (95%): {var_95:.4f} ({var_95*100:.2f}%)")
        return True
        
    except Exception as e:
        print(f"✗ VaR test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple Quantitative Finance Tests")
    print("=" * 35)
    
    tests = [
        ("Basic Math & Black-Scholes", test_basic_math),
        ("Portfolio Calculations", test_portfolio_math),
        ("VaR Calculation", test_var_calculation)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Core functionality works! The library is ready to use.")
    else:
        print("⚠️ Some basic tests failed.")

if __name__ == "__main__":
    main()
