"""
Quantitative Finance Library

A comprehensive Python library for quantitative finance analysis.
"""

# Core models (always available)
from .models.black_scholes import BlackScholes
from .models.capm import CAPM
from .models.monte_carlo import MonteCarlo
from .models.binomial import BinomialTree
from .risk.var import VaR
from .risk.metrics import RiskMetrics
from .portfolio.optimizer import PortfolioOptimizer
from .portfolio.efficient_frontier import EfficientFrontier

# Optional imports (may fail due to dependency issues)
try:
    from .data.market_data import MarketData
    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MarketData not available due to dependency issue: {e}")
    MarketData = None
    MARKET_DATA_AVAILABLE = False

try:
    from .ml.predictors import ReturnPredictor
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML features not available due to dependency issue: {e}")
    ReturnPredictor = None
    ML_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Quantitative Finance Team"

__all__ = [
    "BlackScholes", 
    "CAPM",
    "MonteCarlo",
    "BinomialTree",
    "VaR",
    "RiskMetrics",
    "PortfolioOptimizer",
    "EfficientFrontier",
    "MARKET_DATA_AVAILABLE",
    "ML_AVAILABLE"
]

# Add optional components to __all__ if available
if MARKET_DATA_AVAILABLE:
    __all__.append("MarketData")
if ML_AVAILABLE:
    __all__.append("ReturnPredictor")
