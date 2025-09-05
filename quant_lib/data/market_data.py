"""
Market data fetching and processing module.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import requests

# Make alpha_vantage optional
try:
    from alpha_vantage.timeseries import TimeSeries  # type: ignore
    ALPHA_VANTAGE_AVAILABLE = True
except Exception:
    TimeSeries = None  # type: ignore
    ALPHA_VANTAGE_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


class MarketData:
    """
    Market data fetching and processing class.
    Supports multiple data sources including Yahoo Finance and Alpha Vantage.
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize MarketData instance.
        
        Args:
            alpha_vantage_key: Optional Alpha Vantage API key for premium data
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.av_ts = None
        if alpha_vantage_key and ALPHA_VANTAGE_AVAILABLE:
            try:
                self.av_ts = TimeSeries(key=alpha_vantage_key, output_format='pandas')
            except Exception:
                self.av_ts = None
    
    def get_stock_data(self, 
                      symbols: Union[str, List[str]], 
                      period: str = '1y',
                      interval: str = '1d',
                      start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock price data for given symbols.
        
        Args:
            symbols: Stock symbol(s) to fetch
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock price data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        data_frames = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                if start and end:
                    data = ticker.history(start=start, end=end, interval=interval)
                else:
                    data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    data.columns = [f"{col}_{symbol}" for col in data.columns]
                    data_frames.append(data)
                else:
                    print(f"Warning: No data found for symbol {symbol}")
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        if data_frames:
            result = pd.concat(data_frames, axis=1)
            result.index.name = 'Date'
            return result
        else:
            return pd.DataFrame()
    
    def get_returns(self, 
                   prices: pd.DataFrame, 
                   method: str = 'simple',
                   period: int = 1) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame with price data
            method: Return calculation method ('simple', 'log')
            period: Period for return calculation
            
        Returns:
            DataFrame with calculated returns
        """
        if method == 'simple':
            returns = prices.pct_change(periods=period)
        elif method == 'log':
            returns = np.log(prices / prices.shift(period))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns.dropna()
    
    def get_market_cap(self, symbols: Union[str, List[str]]) -> Dict[str, float]:
        """
        Get market capitalization for given symbols.
        
        Args:
            symbols: Stock symbol(s)
            
        Returns:
            Dictionary with market cap data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        market_caps = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_caps[symbol] = info.get('marketCap', None)
            except Exception as e:
                print(f"Error fetching market cap for {symbol}: {str(e)}")
                market_caps[symbol] = None
        
        return market_caps
    
    def get_financial_ratios(self, symbol: str) -> Dict[str, float]:
        """
        Get key financial ratios for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with financial ratios
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            ratios = {
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None)
            }
            
            return ratios
            
        except Exception as e:
            print(f"Error fetching financial ratios for {symbol}: {str(e)}")
            return {}
    
    def get_risk_free_rate(self, period: str = '3m') -> float:
        """
        Get current risk-free rate (US Treasury rate).
        
        Args:
            period: Treasury period ('3m', '1y', '10y')
            
        Returns:
            Risk-free rate as decimal
        """
        # Map common periods to Yahoo Finance treasury tickers
        treasury_symbols = {
            '3m': '^IRX',   # 13 Week T-Bill
            '6m': '^IRX',
            '1y': '^FVX',   # 5 Year (closest available on Yahoo)
            '2y': '^FVX',
            '5y': '^FVX',   # 5 Year Treasury Yield Index
            '10y': '^TNX',  # 10 Year Treasury Note
        }
        
        try:
            symbol = treasury_symbols.get(period, '^IRX')
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d')
            
            if not data.empty:
                return data['Close'].iloc[-1] / 100  # Convert percentage to decimal
            else:
                return 0.02  # Default 2% if data unavailable
                
        except Exception as e:
            print(f"Error fetching risk-free rate: {str(e)}")
            return 0.02  # Default 2%
    
    def get_market_index(self, index: str = 'SPY', period: str = '1y') -> pd.DataFrame:
        """
        Get market index data for benchmarking.
        
        Args:
            index: Market index symbol (default: SPY for S&P 500)
            period: Data period
            
        Returns:
            DataFrame with index data
        """
        return self.get_stock_data(index, period=period)
    
    def get_correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for given price data.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            Correlation matrix
        """
        returns = self.get_returns(prices)
        return returns.corr()
    
    def get_volatility(self, 
                      prices: pd.DataFrame, 
                      window: int = 30,
                      annualized: bool = True) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            prices: DataFrame with price data
            window: Rolling window size
            annualized: Whether to annualize volatility
            
        Returns:
            DataFrame with volatility data
        """
        returns = self.get_returns(prices)
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            volatility = volatility * np.sqrt(252)  # Assuming 252 trading days per year
        
        return volatility
