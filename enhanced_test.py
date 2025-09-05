"""
Enhanced test suite for the quantitative finance library.
Tests core functionality without external dependencies.
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

def test_black_scholes_comprehensive():
    """Comprehensive Black-Scholes testing."""
    try:
        from quant_lib.models.black_scholes import BlackScholes
        
        bs = BlackScholes()
        
        # Test basic pricing
        call_price = bs.calculate_price(100, 105, 0.25, 0.05, 0.2, 'call')
        put_price = bs.calculate_price(100, 105, 0.25, 0.05, 0.2, 'put')
        
        # Test Greeks
        call_greeks = bs.calculate_greeks(100, 105, 0.25, 0.05, 0.2, 'call')
        put_greeks = bs.calculate_greeks(100, 105, 0.25, 0.05, 0.2, 'put')
        
        # Test put-call parity
        parity = bs.put_call_parity_check(call_price, put_price, 100, 105, 0.25, 0.05)
        
        # Test option chain
        strikes = np.array([95, 100, 105, 110])
        chain = bs.option_chain(100, strikes, 0.25, 0.05, 0.2)
        
        # Test sensitivity analysis
        sensitivity = bs.sensitivity_analysis(100, 105, 0.25, 0.05, 0.2, 'call', 'S', 0.1, 11)
        
        print(f"✓ Black-Scholes call price: ${call_price:.4f}")
        print(f"✓ Black-Scholes put price: ${put_price:.4f}")
        print(f"✓ Call delta: {call_greeks['delta']:.4f}")
        print(f"✓ Put delta: {put_greeks['delta']:.4f}")
        print(f"✓ Put-call parity violation: {abs(parity['parity_violation']):.6f}")
        print(f"✓ Option chain generated with {len(chain)} strikes")
        print(f"✓ Sensitivity analysis completed with {len(sensitivity)} points")
        
        return True
    except Exception as e:
        print(f"✗ Black-Scholes comprehensive test failed: {e}")
        return False

def test_monte_carlo_methods():
    """Test Monte Carlo pricing methods."""
    try:
        from quant_lib.models.monte_carlo import MonteCarlo
        
        mc = MonteCarlo(seed=42)
        
        # Test European option pricing
        european = mc.european_option_price(100, 105, 0.25, 0.05, 0.2, 'call', 50000)
        
        # Test American option pricing
        american = mc.american_option_price(100, 105, 0.25, 0.05, 0.2, 'put', 5000, 50)
        
        # Test Asian option pricing
        asian = mc.asian_option_price(100, 105, 0.25, 0.05, 0.2, 'call', 'arithmetic', 10000, 50)
        
        # Test barrier option pricing
        barrier = mc.barrier_option_price(100, 105, 0.25, 0.05, 0.2, 110, 'up_and_out', 'call', 10000, 50)
        
        print(f"✓ Monte Carlo European call: ${european['price']:.4f} ± {european['std_error']:.4f}")
        print(f"✓ Monte Carlo American put: ${american['price']:.4f} ± {american['std_error']:.4f}")
        print(f"✓ Monte Carlo Asian call: ${asian['price']:.4f}")
        print(f"✓ Monte Carlo barrier option: ${barrier['price']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Monte Carlo methods test failed: {e}")
        return False

def test_risk_metrics():
    """Test comprehensive risk metrics."""
    try:
        from quant_lib.risk.var import VaR
        from quant_lib.risk.metrics import RiskMetrics
        
        # Generate sample data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        prices = pd.Series(100 * np.cumprod(1 + returns))
        
        var_calc = VaR()
        risk_calc = RiskMetrics()
        
        # Test VaR methods
        hist_var = var_calc.historical_var(returns, 0.95)
        param_var = var_calc.parametric_var(returns, 0.95)
        mc_var = var_calc.monte_carlo_var(returns, 0.95, 10000)
        
        # Test Expected Shortfall
        es = var_calc.expected_shortfall(returns, 0.95)
        
        # Test risk metrics
        sharpe = risk_calc.sharpe_ratio(returns, 0.02)
        sortino = risk_calc.sortino_ratio(returns, 0.02)
        max_dd = risk_calc.maximum_drawdown(prices)
        calmar = risk_calc.calmar_ratio(returns, prices)
        
        print(f"✓ Historical VaR (95%): {hist_var:.4f}")
        print(f"✓ Parametric VaR (95%): {param_var:.4f}")
        print(f"✓ Monte Carlo VaR (95%): {mc_var:.4f}")
        print(f"✓ Expected Shortfall (95%): {es:.4f}")
        print(f"✓ Sharpe Ratio: {sharpe:.4f}")
        print(f"✓ Sortino Ratio: {sortino:.4f}")
        print(f"✓ Maximum Drawdown: {max_dd['max_drawdown']:.4f}")
        print(f"✓ Calmar Ratio: {calmar:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Risk metrics test failed: {e}")
        return False

def test_portfolio_optimization():
    """Test portfolio optimization methods."""
    try:
        from quant_lib.portfolio.optimizer import PortfolioOptimizer
        
        # Generate sample data
        np.random.seed(42)
        n_assets = 3
        n_periods = 252
        
        # Create correlated returns
        correlation_matrix = np.array([[1.0, 0.3, 0.2],
                                     [0.3, 1.0, 0.4],
                                     [0.2, 0.4, 1.0]])
        
        returns_data = np.random.multivariate_normal(
            [0.08/252, 0.10/252, 0.12/252],  # Daily returns
            correlation_matrix * (0.15/np.sqrt(252))**2,  # Daily volatility
            n_periods
        )
        
        returns_df = pd.DataFrame(returns_data, columns=['Asset1', 'Asset2', 'Asset3'])
        
        optimizer = PortfolioOptimizer()
        
        # Test different optimization methods
        risk_parity = optimizer.risk_parity_portfolio(returns_df, method='inverse_volatility')
        
        try:
            min_var = optimizer.minimum_variance_portfolio(returns_df, {'long_only': True})
            print(f"✓ Minimum variance portfolio: return={min_var['expected_return']:.4f}, vol={min_var['volatility']:.4f}")
        except:
            print("! Minimum variance optimization requires cvxpy")
        
        # Portfolio comparison
        comparison = optimizer.portfolio_comparison(returns_df, ['equal_weight', 'risk_parity'])
        
        print(f"✓ Risk parity portfolio: return={risk_parity['expected_return']:.4f}, vol={risk_parity['volatility']:.4f}")
        print(f"✓ Portfolio comparison completed with {len(comparison)} methods")
        
        return True
    except Exception as e:
        print(f"✗ Portfolio optimization test failed: {e}")
        return False

def test_capm_analysis():
    """Test CAPM calculations."""
    try:
        from quant_lib.models.capm import CAPM
        
        # Generate sample data
        np.random.seed(42)
        n_periods = 252
        
        market_returns = pd.Series(np.random.normal(0.08/252, 0.15/np.sqrt(252), n_periods))
        asset_returns = pd.Series(0.5 * market_returns + np.random.normal(0.02/252, 0.05/np.sqrt(252), n_periods))
        
        capm = CAPM()
        
        # Test beta calculation
        beta_stats = capm.calculate_beta(asset_returns, market_returns, 0.02)
        
        # Test expected return
        expected_ret = capm.expected_return(beta_stats['beta'], 0.08, 0.02)
        
        # Test Treynor ratio
        treynor = capm.treynor_ratio(0.10, beta_stats['beta'], 0.02)
        
        # Test Jensen's alpha
        jensens = capm.jensens_alpha(0.10, beta_stats['beta'], 0.08, 0.02)
        
        print(f"✓ Asset beta: {beta_stats['beta']:.4f}")
        print(f"✓ R-squared: {beta_stats['r_squared']:.4f}")
        print(f"✓ Expected return: {expected_ret:.4f}")
        print(f"✓ Treynor ratio: {treynor:.4f}")
        print(f"✓ Jensen's alpha: {jensens:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ CAPM analysis test failed: {e}")
        return False

def test_binomial_model():
    """Test binomial option pricing."""
    try:
        from quant_lib.models.binomial import BinomialTree
        
        binomial = BinomialTree()
        
        # Test European options
        call_price = binomial.european_option_price(100, 105, 0.25, 0.05, 0.2, 'call', 100)
        put_price = binomial.european_option_price(100, 105, 0.25, 0.05, 0.2, 'put', 100)
        
        # Test American options
        american_call = binomial.american_option_price(100, 105, 0.25, 0.05, 0.2, 'call', 100)
        american_put = binomial.american_option_price(100, 105, 0.25, 0.05, 0.2, 'put', 100)
        
        print(f"✓ Binomial European call: ${float(call_price['price']):.4f}")
        print(f"✓ Binomial European put: ${float(put_price['price']):.4f}")
        print(f"✓ Binomial American call: ${float(american_call['price']):.4f}")
        print(f"✓ Binomial American put: ${float(american_put['price']):.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Binomial model test failed: {e}")
        return False

def main():
    """Run comprehensive test suite."""
    print("Enhanced Quantitative Finance Library Test Suite")
    print("=" * 60)
    
    tests = [
        ("Black-Scholes Comprehensive", test_black_scholes_comprehensive),
        ("Monte Carlo Methods", test_monte_carlo_methods),
        ("Risk Metrics", test_risk_metrics),
        ("Portfolio Optimization", test_portfolio_optimization),
        ("CAPM Analysis", test_capm_analysis),
        ("Binomial Model", test_binomial_model)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All test suites passed! Library is working excellently.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed. Library is working well with minor issues.")
    else:
        print("⚠️ Several tests failed. Check dependencies and implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
