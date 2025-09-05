# Quantitative Finance Library

A comprehensive Python library for quantitative finance analysis, featuring market data visualization, option pricing models, risk analysis, and machine learning-based return predictions.

## Features

### 📊 Market Data Dashboard
- Real-time market data fetching
- Interactive visualizations with Plotly
- Technical indicators and price analysis
- Multi-asset portfolio tracking

### 📈 Financial Models
- **CAPM (Capital Asset Pricing Model)**: Calculate expected returns and beta coefficients
- **Black-Scholes Model**: European option pricing with Greeks calculation
- **Monte Carlo Simulation**: Advanced option pricing and risk modeling
- **Binomial Tree Model**: American and European option pricing

### ⚠️ Risk Analysis
- **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo VaR
- **Expected Shortfall (ES)**: Conditional VaR calculations
- **Risk metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Stress testing**: Scenario analysis and backtesting

### 🎯 Portfolio Optimization
- **Modern Portfolio Theory**: Efficient frontier construction
- **Mean-variance optimization**: Risk-return optimization
- **Risk parity**: Equal risk contribution portfolios
- **Black-Litterman model**: Bayesian portfolio optimization

### 🤖 Machine Learning
- **Return Prediction**: LSTM, Random Forest, and ensemble models
- **Feature Engineering**: Technical indicators and market sentiment
- **Backtesting Framework**: Strategy performance evaluation
- **Model Validation**: Walk-forward analysis and cross-validation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from quant_lib import MarketData, BlackScholes, VaR, PortfolioOptimizer

# Fetch market data
data = MarketData()
prices = data.get_stock_data(['AAPL', 'GOOGL', 'MSFT'], period='1y')

# Calculate Black-Scholes option price
bs = BlackScholes()
option_price = bs.calculate_price(
    S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type='call'
)

# Calculate portfolio VaR
var_calc = VaR()
portfolio_var = var_calc.historical_var(prices, confidence=0.95)

# Optimize portfolio
optimizer = PortfolioOptimizer()
weights = optimizer.mean_variance_optimization(prices)
```

## Project Structure

```
quant_lib/
├── data/                   # Market data modules
│   ├── market_data.py     # Data fetching and processing
│   └── indicators.py      # Technical indicators
├── models/                # Financial models
│   ├── capm.py           # CAPM implementation
│   ├── black_scholes.py  # Black-Scholes model
│   ├── monte_carlo.py    # Monte Carlo simulations
│   └── binomial.py       # Binomial tree model
├── risk/                  # Risk analysis
│   ├── var.py            # Value at Risk calculations
│   └── metrics.py        # Risk metrics
├── portfolio/             # Portfolio optimization
│   ├── optimizer.py      # Portfolio optimization
│   └── efficient_frontier.py  # Efficient frontier
├── ml/                    # Machine learning models
│   ├── predictors.py     # Return prediction models
│   └── features.py       # Feature engineering
├── dashboard/             # Web dashboard
│   ├── app.py            # Streamlit/Dash app
│   └── components/       # Dashboard components
└── tests/                # Unit tests
```

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Plotly, Matplotlib, Seaborn
- Scikit-learn, TensorFlow/PyTorch
- yfinance, alpha_vantage
- Streamlit (for dashboard)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This library is for educational and research purposes only. Not intended for actual trading decisions without proper risk management and due diligence.
