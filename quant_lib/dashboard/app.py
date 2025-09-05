"""
Streamlit dashboard for quantitative finance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our quant library modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Prefer package-level imports with optional components guarded
from quant_lib import (
    BlackScholes,
    CAPM,
    MonteCarlo,
    BinomialTree,
    VaR,
    RiskMetrics,
    PortfolioOptimizer,
    EfficientFrontier,
    MARKET_DATA_AVAILABLE,
    ML_AVAILABLE,
)

try:
    from quant_lib import MarketData  # optional
except Exception:
    MarketData = None

try:
    from quant_lib.ml.predictors import ReturnPredictor  # optional
    from quant_lib.ml.features import FeatureEngineer    # optional
except Exception:
    ReturnPredictor = None
    FeatureEngineer = None


class QuantDashboard:
    """
    Interactive Streamlit dashboard for quantitative finance analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.market_data = MarketData() if MarketData else None
        self.bs_model = BlackScholes()
        self.capm = CAPM()
        self.monte_carlo = MonteCarlo()
        self.binomial = BinomialTree()
        self.var_calculator = VaR()
        self.risk_metrics = RiskMetrics()
        self.optimizer = PortfolioOptimizer()
        self.predictor = ReturnPredictor() if ReturnPredictor else None
        self.feature_engineer = FeatureEngineer() if FeatureEngineer else None
    
    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Quantitative Finance Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📈 Quantitative Finance Dashboard")
        st.markdown("---")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        pages = [
            "Option Pricing",
            "Risk Analysis",
            "Portfolio Optimization",
            "CAPM Analysis",
        ]
        if self.market_data is not None:
            pages.insert(0, "Market Data Analysis")
        if self.predictor is not None and self.feature_engineer is not None:
            pages.insert(4, "ML Predictions")

        page = st.sidebar.selectbox("Choose Analysis Type", pages)
        
        # Route to appropriate page
        if page == "Market Data Analysis":
            self.market_data_page()
        elif page == "Option Pricing":
            self.option_pricing_page()
        elif page == "Risk Analysis":
            self.risk_analysis_page()
        elif page == "Portfolio Optimization":
            self.portfolio_optimization_page()
        elif page == "ML Predictions":
            self.ml_predictions_page()
        elif page == "CAPM Analysis":
            self.capm_analysis_page()
    
    def market_data_page(self):
        """Market data analysis page."""
        st.header("📊 Market Data Analysis")
        if self.market_data is None:
            st.warning("Market data features are unavailable. Please install dependencies or provide API keys.")
            return
        
        # Stock selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols = st.text_input(
                "Stock Symbols (comma-separated)", 
                value="AAPL,GOOGL,MSFT,TSLA"
            ).split(',')
            symbols = [s.strip().upper() for s in symbols if s.strip()]
        
        with col2:
            period = st.selectbox(
                "Time Period", 
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3
            )
        
        with col3:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Price Chart", "Returns Analysis", "Correlation Matrix", "Technical Indicators"]
            )
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching market data..."):
                try:
                    # Fetch data
                    data = self.market_data.get_stock_data(symbols, period=period)
                    
                    if not data.empty:
                        st.success(f"Successfully fetched data for {len(symbols)} symbols")
                        
                        if analysis_type == "Price Chart":
                            self.plot_price_chart(data, symbols)
                        
                        elif analysis_type == "Returns Analysis":
                            self.plot_returns_analysis(data, symbols)
                        
                        elif analysis_type == "Correlation Matrix":
                            self.plot_correlation_matrix(data, symbols)
                        
                        elif analysis_type == "Technical Indicators":
                            self.plot_technical_indicators(data, symbols[0])
                    
                    else:
                        st.error("No data found for the specified symbols")
                
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    def plot_price_chart(self, data, symbols):
        """Plot price chart."""
        fig = go.Figure()
        
        for symbol in symbols:
            close_col = f"Close_{symbol}"
            if close_col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[close_col],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Stock Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_returns_analysis(self, data, symbols):
        """Plot returns analysis."""
        returns_data = {}
        
        for symbol in symbols:
            close_col = f"Close_{symbol}"
            if close_col in data.columns:
                returns = data[close_col].pct_change().dropna()
                returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        
        # Returns distribution
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cumulative Returns", "Returns Distribution", 
                          "Rolling Volatility", "Drawdown")
        )
        
        # Cumulative returns
        cumulative_returns = (1 + returns_df).cumprod()
        for symbol in symbols:
            if symbol in cumulative_returns.columns:
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[symbol],
                        mode='lines',
                        name=f"{symbol} Cumulative",
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Returns histogram
        for symbol in symbols[:2]:  # Limit to 2 for clarity
            if symbol in returns_df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=returns_df[symbol],
                        name=f"{symbol} Returns",
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Rolling volatility
        rolling_vol = returns_df.rolling(30).std() * np.sqrt(252)
        for symbol in symbols:
            if symbol in rolling_vol.columns:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol[symbol],
                        mode='lines',
                        name=f"{symbol} Vol",
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Drawdown
        for symbol in symbols:
            if symbol in cumulative_returns.columns:
                running_max = cumulative_returns[symbol].expanding().max()
                drawdown = (cumulative_returns[symbol] - running_max) / running_max
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        mode='lines',
                        name=f"{symbol} DD",
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, title="Returns Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        stats = []
        for symbol in symbols:
            if symbol in returns_df.columns:
                ret_series = returns_df[symbol]
                stats.append({
                    'Symbol': symbol,
                    'Annual Return': f"{ret_series.mean() * 252:.2%}",
                    'Volatility': f"{ret_series.std() * np.sqrt(252):.2%}",
                    'Sharpe Ratio': f"{ret_series.mean() / ret_series.std() * np.sqrt(252):.2f}",
                    'Max Drawdown': f"{((1 + ret_series).cumprod() / (1 + ret_series).cumprod().expanding().max() - 1).min():.2%}"
                })
        
        st.subheader("Performance Statistics")
        st.dataframe(pd.DataFrame(stats))
    
    def plot_correlation_matrix(self, data, symbols):
        """Plot correlation matrix."""
        returns_data = {}
        
        for symbol in symbols:
            close_col = f"Close_{symbol}"
            if close_col in data.columns:
                returns = data[close_col].pct_change().dropna()
                returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Returns Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_technical_indicators(self, data, symbol):
        """Plot technical indicators."""
        close_col = f"Close_{symbol}"
        high_col = f"High_{symbol}"
        low_col = f"Low_{symbol}"
        volume_col = f"Volume_{symbol}"
        
        if close_col not in data.columns:
            st.error(f"No data found for {symbol}")
            return
        
        # Create comprehensive features
        symbol_data = pd.DataFrame({
            'Close': data[close_col],
            'High': data[high_col] if high_col in data.columns else data[close_col],
            'Low': data[low_col] if low_col in data.columns else data[close_col],
            'Volume': data[volume_col] if volume_col in data.columns else pd.Series(index=data.index)
        })
        
        if self.feature_engineer is None:
            st.warning("Feature engineering module unavailable. Install ML extras to enable indicators.")
            return

        features = self.feature_engineer.create_technical_features(symbol_data)
        
        # Plot
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=(f"{symbol} Price & Moving Averages", "RSI", "MACD", "Bollinger Bands"),
            vertical_spacing=0.05
        )
        
        # Price and moving averages
        fig.add_trace(go.Scatter(x=features.index, y=features['Close'], name='Close', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=features.index, y=features['sma_20'], name='SMA 20', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=features.index, y=features['sma_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=features.index, y=features['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=features.index, y=features['macd'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=features.index, y=features['macd_signal'], name='Signal', line=dict(color='red')), row=3, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=features.index, y=features['Close'], name='Close', line=dict(color='black')), row=4, col=1)
        fig.add_trace(go.Scatter(x=features.index, y=features['bb_upper'], name='BB Upper', line=dict(color='red', dash='dash')), row=4, col=1)
        fig.add_trace(go.Scatter(x=features.index, y=features['bb_lower'], name='BB Lower', line=dict(color='green', dash='dash')), row=4, col=1)
        
        fig.update_layout(height=800, title=f"Technical Analysis - {symbol}")
        st.plotly_chart(fig, use_container_width=True)
    
    def option_pricing_page(self):
        """Option pricing page."""
        st.header("🎯 Option Pricing Models")
        
        # Input parameters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            S = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01)
            K = st.number_input("Strike Price ($)", value=105.0, min_value=0.01)
        
        with col2:
            T = st.number_input("Time to Expiration (years)", value=0.25, min_value=0.001, max_value=10.0)
            r = st.number_input("Risk-free Rate", value=0.05, min_value=0.0, max_value=1.0)
        
        with col3:
            sigma = st.number_input("Volatility", value=0.20, min_value=0.001, max_value=5.0)
            option_type = st.selectbox("Option Type", ["call", "put"])
        
        with col4:
            model_type = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])
            n_steps = st.number_input("Steps/Simulations", value=100, min_value=10, max_value=10000)
        
        if st.button("Calculate Option Price"):
            try:
                if model_type == "Black-Scholes":
                    result = self.bs_model.calculate_greeks(S, K, T, r, sigma, option_type)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Option Price", f"${result['price']:.4f}")
                        st.metric("Delta", f"{result['delta']:.4f}")
                    
                    with col2:
                        st.metric("Gamma", f"{result['gamma']:.4f}")
                        st.metric("Theta", f"{result['theta']:.4f}")
                    
                    with col3:
                        st.metric("Vega", f"{result['vega']:.4f}")
                        st.metric("Rho", f"{result['rho']:.4f}")
                
                elif model_type == "Binomial Tree":
                    if option_type == "call":
                        result = self.binomial.european_option_price(S, K, T, r, sigma, option_type, int(n_steps))
                    else:
                        result = self.binomial.american_option_price(S, K, T, r, sigma, option_type, int(n_steps))
                    
                    st.metric("Option Price", f"${result['price']:.4f}")
                
                elif model_type == "Monte Carlo":
                    result = self.monte_carlo.european_option_price(S, K, T, r, sigma, option_type, int(n_steps))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Option Price", f"${result['price']:.4f}")
                    with col2:
                        st.metric("Standard Error", f"${result['std_error']:.4f}")
                    
                    st.info(f"95% Confidence Interval: ${result['confidence_lower']:.4f} - ${result['confidence_upper']:.4f}")
                
                # Sensitivity analysis
                st.subheader("Sensitivity Analysis")
                sensitivity_param = st.selectbox("Parameter to Vary", ["S", "K", "T", "r", "sigma"])
                
                if st.button("Run Sensitivity Analysis"):
                    sensitivity_result = self.bs_model.sensitivity_analysis(
                        S, K, T, r, sigma, option_type, sensitivity_param
                    )
                    
                    fig = px.line(
                        sensitivity_result, 
                        x=sensitivity_param, 
                        y='price',
                        title=f"Option Price Sensitivity to {sensitivity_param}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error calculating option price: {str(e)}")
    
    def risk_analysis_page(self):
        """Risk analysis page."""
        st.header("⚠️ Risk Analysis")
        if self.market_data is None:
            st.warning("Market data features are unavailable. Please install dependencies to enable Risk Analysis.")
            return
        
        # Portfolio input
        st.subheader("Portfolio Setup")
        symbols_input = st.text_input("Stock Symbols (comma-separated)", value="AAPL,GOOGL,MSFT")
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        
        period = st.selectbox("Data Period", ["6mo", "1y", "2y", "3y"], index=1)
        confidence_level = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
        
        if st.button("Calculate Risk Metrics"):
            with st.spinner("Calculating risk metrics..."):
                try:
                    # Fetch data
                    data = self.market_data.get_stock_data(symbols, period=period)
                    
                    # Calculate returns
                    returns_data = {}
                    for symbol in symbols:
                        close_col = f"Close_{symbol}"
                        if close_col in data.columns:
                            returns = data[close_col].pct_change().dropna()
                            returns_data[symbol] = returns
                    
                    returns_df = pd.DataFrame(returns_data)
                    
                    # Equal weight portfolio
                    weights = np.array([1/len(symbols)] * len(symbols))
                    portfolio_returns = (returns_df * weights).sum(axis=1)
                    
                    # Calculate VaR
                    historical_var = self.var_calculator.historical_var(portfolio_returns, confidence_level)
                    parametric_var = self.var_calculator.parametric_var(portfolio_returns, confidence_level)
                    expected_shortfall = self.var_calculator.expected_shortfall(portfolio_returns, confidence_level)
                    
                    # Risk metrics
                    risk_report = self.risk_metrics.comprehensive_risk_report(portfolio_returns)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Historical VaR", f"{historical_var:.4f}")
                        st.metric("Parametric VaR", f"{parametric_var:.4f}")
                    
                    with col2:
                        st.metric("Expected Shortfall", f"{expected_shortfall:.4f}")
                        st.metric("Maximum Drawdown", f"{risk_report['max_drawdown']:.4f}")
                    
                    with col3:
                        st.metric("Sharpe Ratio", f"{risk_report['sharpe_ratio']:.4f}")
                        st.metric("Volatility", f"{risk_report['volatility']:.4f}")
                    
                    # VaR backtesting
                    st.subheader("VaR Backtesting")
                    var_estimates = pd.Series([historical_var] * len(portfolio_returns), index=portfolio_returns.index)
                    backtest_results = self.var_calculator.backtesting(portfolio_returns, var_estimates, confidence_level)
                    
                    st.write(f"Violation Rate: {backtest_results['violation_rate']:.2%}")
                    st.write(f"Expected Violation Rate: {backtest_results['expected_violation_rate']:.2%}")
                    
                    # Plot returns and VaR
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=portfolio_returns.index,
                        y=portfolio_returns,
                        mode='lines',
                        name='Portfolio Returns'
                    ))
                    fig.add_hline(y=historical_var, line_dash="dash", line_color="red", 
                                annotation_text=f"VaR ({confidence_level:.0%})")
                    
                    fig.update_layout(title="Portfolio Returns vs VaR", xaxis_title="Date", yaxis_title="Returns")
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in risk analysis: {str(e)}")
    
    def portfolio_optimization_page(self):
        """Portfolio optimization page."""
        st.header("🎯 Portfolio Optimization")
        if self.market_data is None:
            st.warning("Market data features are unavailable. Please install dependencies to enable Portfolio Optimization.")
            return
        
        # Input parameters
        symbols_input = st.text_input("Stock Symbols (comma-separated)", value="AAPL,GOOGL,MSFT,AMZN,TSLA")
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        
        period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1)
        optimization_method = st.selectbox(
            "Optimization Method", 
            ["Maximum Sharpe", "Minimum Variance", "Risk Parity", "Equal Weight"]
        )
        
        if st.button("Optimize Portfolio"):
            with st.spinner("Optimizing portfolio..."):
                try:
                    # Fetch data
                    data = self.market_data.get_stock_data(symbols, period=period)
                    
                    # Calculate returns
                    returns_data = {}
                    for symbol in symbols:
                        close_col = f"Close_{symbol}"
                        if close_col in data.columns:
                            returns = data[close_col].pct_change().dropna()
                            returns_data[symbol] = returns
                    
                    returns_df = pd.DataFrame(returns_data)
                    
                    # Optimize portfolio
                    if optimization_method == "Maximum Sharpe":
                        result = self.optimizer.maximum_sharpe_portfolio(returns_df, constraints={'long_only': True})
                    elif optimization_method == "Minimum Variance":
                        result = self.optimizer.minimum_variance_portfolio(returns_df, {'long_only': True})
                    elif optimization_method == "Risk Parity":
                        result = self.optimizer.risk_parity_portfolio(returns_df)
                    else:  # Equal Weight
                        weights = np.array([1/len(symbols)] * len(symbols))
                        mean_returns = returns_df.mean().values
                        cov_matrix = returns_df.cov().values
                        portfolio_ret = mean_returns.T @ weights
                        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                        result = {
                            'weights': weights,
                            'expected_return': portfolio_ret,
                            'volatility': portfolio_vol,
                            'sharpe_ratio': portfolio_ret / portfolio_vol
                        }
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Expected Return", f"{result['expected_return']:.4f}")
                    with col2:
                        st.metric("Volatility", f"{result['volatility']:.4f}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.4f}")
                    
                    # Portfolio weights
                    st.subheader("Optimal Weights")
                    weights_df = pd.DataFrame({
                        'Symbol': symbols,
                        'Weight': result['weights'],
                        'Weight %': [f"{w:.2%}" for w in result['weights']]
                    })
                    st.dataframe(weights_df)
                    
                    # Pie chart
                    fig = px.pie(
                        weights_df, 
                        values='Weight', 
                        names='Symbol',
                        title="Portfolio Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Efficient frontier
                    st.subheader("Efficient Frontier")
                    ef = EfficientFrontier(returns_df)
                    efficient_df = ef.generate_efficient_frontier(50)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=efficient_df['volatility'],
                        y=efficient_df['expected_return'],
                        mode='lines',
                        name='Efficient Frontier'
                    ))
                    
                    # Mark optimal portfolio
                    fig.add_trace(go.Scatter(
                        x=[result['volatility']],
                        y=[result['expected_return']],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name='Optimal Portfolio'
                    ))
                    
                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Volatility",
                        yaxis_title="Expected Return"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in portfolio optimization: {str(e)}")
    
    def ml_predictions_page(self):
        """ML predictions page."""
        st.header("🤖 ML Return Predictions")
        if self.predictor is None or self.feature_engineer is None or self.market_data is None:
            st.warning("ML features are unavailable. Ensure ML dependencies and market data are installed.")
            return
        
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        period = st.selectbox("Training Period", ["2y", "3y", "5y"], index=1)
        prediction_horizon = st.selectbox("Prediction Horizon", [1, 5, 10, 20], index=0)
        
        if st.button("Train Models & Predict"):
            with st.spinner("Training ML models..."):
                try:
                    # Fetch data
                    data = self.market_data.get_stock_data([symbol], period=period)
                    
                    # Prepare OHLCV data
                    ohlcv_data = pd.DataFrame({
                        'Close': data[f'Close_{symbol}'],
                        'High': data[f'High_{symbol}'],
                        'Low': data[f'Low_{symbol}'],
                        'Volume': data[f'Volume_{symbol}'] if f'Volume_{symbol}' in data.columns else pd.Series(index=data.index)
                    })
                    
                    # Create features
                    features = self.feature_engineer.create_comprehensive_features(ohlcv_data)
                    
                    # Create target
                    target = self.feature_engineer.create_target_variable(
                        ohlcv_data, horizon=prediction_horizon, target_type='return'
                    )
                    
                    # Prepare data
                    data_splits = self.predictor.prepare_data(features, target, test_size=0.2)
                    
                    # Train models
                    linear_models = self.predictor.train_linear_models(
                        data_splits['X_train'], data_splits['y_train'],
                        data_splits['X_val'], data_splits['y_val']
                    )
                    
                    tree_models = self.predictor.train_tree_models(
                        data_splits['X_train'], data_splits['y_train'],
                        data_splits['X_val'], data_splits['y_val']
                    )
                    
                    # Model comparison
                    st.subheader("Model Performance")
                    comparison_df = self.predictor.model_comparison()
                    st.dataframe(comparison_df)
                    
                    # Feature importance
                    if 'random_forest' in self.predictor.models:
                        st.subheader("Feature Importance (Random Forest)")
                        importance = self.predictor.get_feature_importance('random_forest', top_n=15)
                        
                        fig = px.bar(
                            x=importance.values,
                            y=importance.index,
                            orientation='h',
                            title="Top 15 Most Important Features"
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Predictions vs actual
                    if 'random_forest' in self.predictor.models:
                        test_predictions = self.predictor.predict_returns(data_splits['X_test'], 'random_forest')
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data_splits['y_test'].index,
                            y=data_splits['y_test'].values,
                            mode='lines',
                            name='Actual Returns'
                        ))
                        fig.add_trace(go.Scatter(
                            x=data_splits['y_test'].index,
                            y=test_predictions,
                            mode='lines',
                            name='Predicted Returns'
                        ))
                        
                        fig.update_layout(
                            title="Actual vs Predicted Returns",
                            xaxis_title="Date",
                            yaxis_title="Returns"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in ML predictions: {str(e)}")
    
    def capm_analysis_page(self):
        """CAPM analysis page."""
        st.header("📈 CAPM Analysis")
        if self.market_data is None:
            st.warning("Market data features are unavailable. Please install dependencies to enable CAPM Analysis.")
            return
        
        stock_symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        market_symbol = st.text_input("Market Index", value="SPY").upper()
        period = st.selectbox("Analysis Period", ["1y", "2y", "3y", "5y"], index=1)
        risk_free_rate = st.number_input("Risk-free Rate", value=0.02, min_value=0.0, max_value=0.1)
        
        if st.button("Run CAPM Analysis"):
            with st.spinner("Running CAPM analysis..."):
                try:
                    # Fetch data
                    stock_data = self.market_data.get_stock_data([stock_symbol], period=period)
                    market_data = self.market_data.get_stock_data([market_symbol], period=period)
                    
                    # Calculate returns
                    stock_returns = stock_data[f'Close_{stock_symbol}'].pct_change().dropna()
                    market_returns = market_data[f'Close_{market_symbol}'].pct_change().dropna()
                    
                    # CAPM analysis
                    capm_results = self.capm.calculate_beta(stock_returns, market_returns, risk_free_rate)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Beta", f"{capm_results['beta']:.4f}")
                        st.metric("Alpha (Annual)", f"{capm_results['alpha_annual']:.4f}")
                    
                    with col2:
                        st.metric("R-squared", f"{capm_results['r_squared']:.4f}")
                        st.metric("Correlation", f"{capm_results['correlation']:.4f}")
                    
                    with col3:
                        expected_return = self.capm.expected_return(
                            capm_results['beta'], 
                            market_returns.mean() * 252, 
                            risk_free_rate
                        )
                        st.metric("Expected Return (CAPM)", f"{expected_return:.4f}")
                        st.metric("P-value", f"{capm_results['p_value']:.4f}")
                    
                    # Scatter plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=market_returns,
                        y=stock_returns,
                        mode='markers',
                        name='Returns',
                        opacity=0.6
                    ))
                    
                    # Regression line
                    x_line = np.linspace(market_returns.min(), market_returns.max(), 100)
                    y_line = capm_results['alpha'] + capm_results['beta'] * x_line
                    
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        name=f'Regression Line (β={capm_results["beta"]:.3f})',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title=f"CAPM Regression: {stock_symbol} vs {market_symbol}",
                        xaxis_title=f"{market_symbol} Returns",
                        yaxis_title=f"{stock_symbol} Returns"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Rolling beta
                    st.subheader("Rolling Beta Analysis")
                    rolling_beta = self.capm.rolling_beta(stock_returns, market_returns, window=252)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rolling_beta.index,
                        y=rolling_beta,
                        mode='lines',
                        name='Rolling Beta (1Y)'
                    ))
                    fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Market Beta = 1")
                    
                    fig.update_layout(
                        title="Rolling Beta Over Time",
                        xaxis_title="Date",
                        yaxis_title="Beta"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in CAPM analysis: {str(e)}")


def main():
    """Main function to run the dashboard."""
    dashboard = QuantDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
