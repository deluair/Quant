"""
Quantitative Finance Library

A comprehensive Python library for quantitative finance analysis.
"""

from .data.market_data import MarketData
from .models.black_scholes import BlackScholes
from .models.capm import CAPM
from .models.monte_carlo import MonteCarlo
from .models.binomial import BinomialTree
from .risk.var import VaR
from .risk.metrics import RiskMetrics
from .portfolio.optimizer import PortfolioOptimizer
from .portfolio.efficient_frontier import EfficientFrontier
from .ml.predictors import ReturnPredictor

__version__ = "1.0.0"
__author__ = "Quantitative Finance Team"

__all__ = [
    "MarketData",
    "BlackScholes", 
    "CAPM",
    "MonteCarlo",
    "BinomialTree",
    "VaR",
    "RiskMetrics",
    "PortfolioOptimizer",
    "EfficientFrontier",
    "ReturnPredictor"
]
