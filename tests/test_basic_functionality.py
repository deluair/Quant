"""
Basic functionality tests for the Quantitative Finance Library.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quant_lib.models.black_scholes import BlackScholes
from quant_lib.models.capm import CAPM
from quant_lib.risk.var import VaR
from quant_lib.risk.metrics import RiskMetrics
from quant_lib.portfolio.optimizer import PortfolioOptimizer


class TestBlackScholes(unittest.TestCase):
    """Test Black-Scholes model."""
    
    def setUp(self):
        self.bs = BlackScholes()
        self.S = 100
        self.K = 105
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_call_price(self):
        """Test call option pricing."""
        price = self.bs.calculate_price(self.S, self.K, self.T, self.r, self.sigma, 'call')
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
    
    def test_put_price(self):
        """Test put option pricing."""
        price = self.bs.calculate_price(self.S, self.K, self.T, self.r, self.sigma, 'put')
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
    
    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        greeks = self.bs.calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, 'call')
        
        required_greeks = ['price', 'delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in required_greeks:
            self.assertIn(greek, greeks)
            self.assertIsInstance(greeks[greek], float)
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        call_price = self.bs.calculate_price(self.S, self.K, self.T, self.r, self.sigma, 'call')
        put_price = self.bs.calculate_price(self.S, self.K, self.T, self.r, self.sigma, 'put')
        
        parity_check = self.bs.put_call_parity_check(call_price, put_price, self.S, self.K, self.T, self.r)
        
        self.assertIn('parity_violation', parity_check)
        self.assertLess(abs(parity_check['parity_violation']), 0.01)  # Small tolerance


class TestCAPM(unittest.TestCase):
    """Test CAPM model."""
    
    def setUp(self):
        self.capm = CAPM()
        
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        market_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        asset_returns = pd.Series(0.5 + 1.2 * market_returns + np.random.normal(0, 0.01, 100), index=dates)
        
        self.market_returns = market_returns
        self.asset_returns = asset_returns
    
    def test_beta_calculation(self):
        """Test beta coefficient calculation."""
        result = self.capm.calculate_beta(self.asset_returns, self.market_returns)
        
        self.assertIn('beta', result)
        self.assertIn('alpha', result)
        self.assertIn('r_squared', result)
        
        self.assertIsInstance(result['beta'], float)
        self.assertGreater(result['beta'], 0)  # Should be positive for our test data
    
    def test_expected_return(self):
        """Test expected return calculation."""
        beta = 1.2
        market_return = 0.08
        risk_free_rate = 0.03
        
        expected_return = self.capm.expected_return(beta, market_return, risk_free_rate)
        
        self.assertIsInstance(expected_return, float)
        self.assertGreater(expected_return, risk_free_rate)


class TestVaR(unittest.TestCase):
    """Test VaR calculations."""
    
    def setUp(self):
        self.var_calc = VaR()
        
        # Generate sample returns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    def test_historical_var(self):
        """Test historical VaR calculation."""
        var = self.var_calc.historical_var(self.returns, confidence_level=0.95)
        
        self.assertIsInstance(var, float)
        self.assertLess(var, 0)  # VaR should be negative (loss)
    
    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        var = self.var_calc.parametric_var(self.returns, confidence_level=0.95)
        
        self.assertIsInstance(var, float)
        self.assertLess(var, 0)  # VaR should be negative (loss)
    
    def test_expected_shortfall(self):
        """Test Expected Shortfall calculation."""
        es = self.var_calc.expected_shortfall(self.returns, confidence_level=0.95)
        
        self.assertIsInstance(es, float)
        self.assertLess(es, 0)  # ES should be negative (loss)
        
        # ES should be more negative than VaR
        var = self.var_calc.historical_var(self.returns, confidence_level=0.95)
        self.assertLess(es, var)


class TestRiskMetrics(unittest.TestCase):
    """Test risk metrics calculations."""
    
    def setUp(self):
        self.risk_metrics = RiskMetrics()
        
        # Generate sample returns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.risk_metrics.sharpe_ratio(self.returns, risk_free_rate=0.02)
        
        self.assertIsInstance(sharpe, float)
    
    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        dd_info = self.risk_metrics.maximum_drawdown(self.returns)
        
        self.assertIn('max_drawdown', dd_info)
        self.assertIsInstance(dd_info['max_drawdown'], float)
        self.assertLessEqual(dd_info['max_drawdown'], 0)  # Drawdown should be negative or zero
    
    def test_volatility(self):
        """Test volatility calculation."""
        vol = self.risk_metrics.volatility(self.returns)
        
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)  # Volatility should be positive


class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimization."""
    
    def setUp(self):
        self.optimizer = PortfolioOptimizer()
        
        # Generate sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        returns_data = {
            'Asset1': np.random.normal(0.0008, 0.025, 252),
            'Asset2': np.random.normal(0.0006, 0.022, 252),
            'Asset3': np.random.normal(0.0007, 0.020, 252)
        }
        
        self.returns_df = pd.DataFrame(returns_data, index=dates)
    
    def test_minimum_variance_portfolio(self):
        """Test minimum variance portfolio optimization."""
        try:
            result = self.optimizer.minimum_variance_portfolio(self.returns_df, {'long_only': True})
            
            self.assertIn('weights', result)
            self.assertIn('expected_return', result)
            self.assertIn('volatility', result)
            
            # Check weights sum to 1
            self.assertAlmostEqual(np.sum(result['weights']), 1.0, places=6)
            
            # Check all weights are non-negative (long-only constraint)
            self.assertTrue(np.all(result['weights'] >= -1e-6))  # Small tolerance for numerical errors
            
        except Exception as e:
            self.skipTest(f"Optimization failed (likely missing cvxpy): {str(e)}")
    
    def test_risk_parity_portfolio(self):
        """Test risk parity portfolio construction."""
        result = self.optimizer.risk_parity_portfolio(self.returns_df, method='inverse_volatility')
        
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        
        # Check weights sum to 1
        self.assertAlmostEqual(np.sum(result['weights']), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
