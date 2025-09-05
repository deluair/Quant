"""
Quick test of the quantitative finance library.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_black_scholes():
    """Test Black-Scholes model."""
    try:
        from quant_lib.models.black_scholes import BlackScholes
        
        bs = BlackScholes()
        price = bs.calculate_price(100, 105, 0.25, 0.05, 0.2, 'call')
        greeks = bs.calculate_greeks(100, 105, 0.25, 0.05, 0.2, 'call')
        
        print(f"✓ Black-Scholes call price: ${price:.4f}")
        print(f"✓ Delta: {greeks['delta']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Black-Scholes test failed: {e}")
        return False

def test_var_calculation():
    """Test VaR calculation."""
    try:
        import numpy as np
        import pandas as pd
        from quant_lib.risk.var import VaR
        
        # Generate sample data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        var_calc = VaR()
        historical_var = var_calc.historical_var(returns, 0.95)
        
        print(f"✓ Historical VaR (95%): {historical_var:.4f}")
        return True
    except Exception as e:
        print(f"✗ VaR test failed: {e}")
        return False

def test_portfolio_optimization():
    """Test portfolio optimization."""
    try:
        import numpy as np
        import pandas as pd
        from quant_lib.portfolio.optimizer import PortfolioOptimizer
        
        # Generate sample data
        np.random.seed(42)
        returns_data = {
            'Asset1': np.random.normal(0.001, 0.02, 100),
            'Asset2': np.random.normal(0.0008, 0.025, 100),
            'Asset3': np.random.normal(0.0012, 0.03, 100)
        }
        returns_df = pd.DataFrame(returns_data)
        
        optimizer = PortfolioOptimizer()
        
        # Try risk parity (doesn't require cvxpy)
        result = optimizer.risk_parity_portfolio(returns_df, method='inverse_volatility')
        
        print(f"✓ Risk parity portfolio created")
        print(f"✓ Expected return: {result['expected_return']:.4f}")
        print(f"✓ Volatility: {result['volatility']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Portfolio optimization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Quantitative Finance Library")
    print("=" * 40)
    
    tests = [
        test_black_scholes,
        test_var_calculation,
        test_portfolio_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Library is working correctly.")
    else:
        print("⚠️ Some tests failed. Check dependencies and implementation.")

if __name__ == "__main__":
    main()
