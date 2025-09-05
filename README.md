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

Notes:
- Eventlet is included to resolve SSL issues on Python 3.12+ (wrap_socket deprecation). If you installed earlier, upgrade with: `pip install -U eventlet`.
- TensorFlow/PyTorch are optional and only required for the ML page. The rest of the library and dashboard work without them.

## Quick Start

```python
from quant_lib import BlackScholes, VaR, PortfolioOptimizer, MARKET_DATA_AVAILABLE
# MarketData is optional depending on your environment
try:
    from quant_lib import MarketData
except Exception:
    MarketData = None

# Fetch market data (requires yfinance). If MarketData is unavailable, skip this section.
if MARKET_DATA_AVAILABLE and MarketData is not None:
    data = MarketData()
    prices = data.get_stock_data(['AAPL', 'GOOGL', 'MSFT'], period='1y')

# Calculate Black-Scholes option price
bs = BlackScholes()
option_price = bs.calculate_price(
    S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type='call'
)

# Calculate portfolio VaR (requires a returns series/DataFrame)
var_calc = VaR()
# Example with synthetic returns:
# portfolio_returns = prices['Close_AAPL'].pct_change().dropna()
# portfolio_var = var_calc.historical_var(portfolio_returns, confidence_level=0.95)

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

enhanced_test.py          # Offline test suite (no external data needed)
run_dashboard.py          # Entry script to launch Streamlit dashboard
```

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Plotly, Matplotlib, Seaborn
- Scikit-learn, TensorFlow/PyTorch
- yfinance, alpha_vantage
- Streamlit (for dashboard)

Optional/Environment notes:
- TensorFlow is optional (used only by ML page). If it fails to load on Windows, the rest of the app still works.
- alpha_vantage is optional; `MarketData` will default to yfinance if alpha_vantage is missing.
- Eventlet>=0.40.3 is included to fix SSL on Python 3.12+.

Troubleshooting:
- SSL error about `ssl.wrap_socket`: upgrade/install eventlet: `pip install -U eventlet`.
- TensorFlow DLL error: ignore if not using ML; the ML page will be hidden automatically.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This library is for educational and research purposes only. Not intended for actual trading decisions without proper risk management and due diligence.

## Run the Dashboard

Windows (PowerShell):

```powershell
cd "C:\Users\mhossen\OneDrive - University of Tennessee\AI\Quant"
python -m pip install -r requirements.txt
streamlit run run_dashboard.py
```

Open `http://localhost:8501` in your browser.

To choose a custom port:

```powershell
streamlit run run_dashboard.py --server.port 8502
```

## Run the Offline Test Suite

```powershell
python enhanced_test.py
```

All six suites should pass without Internet access.
