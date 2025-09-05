"""
Feature engineering for financial machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for financial time series data.
    """
    
    def __init__(self):
        """Initialize Feature Engineer."""
        self.scalers = {}
    
    def create_technical_features(self, 
                                data: pd.DataFrame,
                                price_col: str = 'Close',
                                volume_col: str = 'Volume',
                                high_col: str = 'High',
                                low_col: str = 'Low') -> pd.DataFrame:
        """
        Create technical indicator features.
        
        Args:
            data: OHLCV DataFrame
            price_col: Column name for closing price
            volume_col: Column name for volume
            high_col: Column name for high price
            low_col: Column name for low price
            
        Returns:
            DataFrame with technical features
        """
        features = data.copy()
        
        # Price-based features
        features['sma_10'] = ta.trend.sma_indicator(data[price_col], window=10)
        features['sma_20'] = ta.trend.sma_indicator(data[price_col], window=20)
        features['sma_50'] = ta.trend.sma_indicator(data[price_col], window=50)
        features['ema_12'] = ta.trend.ema_indicator(data[price_col], window=12)
        features['ema_26'] = ta.trend.ema_indicator(data[price_col], window=26)
        
        # Momentum indicators
        features['rsi'] = ta.momentum.rsi(data[price_col], window=14)
        features['stoch'] = ta.momentum.stoch(data[high_col], data[low_col], data[price_col])
        features['williams_r'] = ta.momentum.williams_r(data[high_col], data[low_col], data[price_col])
        
        # MACD
        macd = ta.trend.MACD(data[price_col])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data[price_col])
        features['bb_upper'] = bb.bollinger_hband()
        features['bb_lower'] = bb.bollinger_lband()
        features['bb_middle'] = bb.bollinger_mavg()
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (data[price_col] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volatility indicators
        features['atr'] = ta.volatility.average_true_range(data[high_col], data[low_col], data[price_col])
        features['volatility_10'] = data[price_col].pct_change().rolling(10).std()
        features['volatility_20'] = data[price_col].pct_change().rolling(20).std()
        
        # Volume indicators
        if volume_col in data.columns:
            features['volume_sma'] = ta.volume.volume_sma(data[price_col], data[volume_col])
            features['vwap'] = ta.volume.volume_weighted_average_price(
                data[high_col], data[low_col], data[price_col], data[volume_col]
            )
            features['obv'] = ta.volume.on_balance_volume(data[price_col], data[volume_col])
            features['cmf'] = ta.volume.chaikin_money_flow(
                data[high_col], data[low_col], data[price_col], data[volume_col]
            )
        
        # Price patterns
        features['price_change'] = data[price_col].pct_change()
        features['price_change_2d'] = data[price_col].pct_change(2)
        features['price_change_5d'] = data[price_col].pct_change(5)
        
        # Support and resistance levels
        features['resistance_20'] = data[high_col].rolling(20).max()
        features['support_20'] = data[low_col].rolling(20).min()
        features['price_to_resistance'] = data[price_col] / features['resistance_20']
        features['price_to_support'] = data[price_col] / features['support_20']
        
        return features
    
    def create_lag_features(self, 
                          data: pd.DataFrame, 
                          columns: List[str],
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        features = data.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return features
    
    def create_rolling_features(self, 
                              data: pd.DataFrame, 
                              columns: List[str],
                              windows: List[int] = [5, 10, 20],
                              operations: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create rolling features for
            windows: Rolling window sizes
            operations: Rolling operations to apply
            
        Returns:
            DataFrame with rolling features
        """
        features = data.copy()
        
        for col in columns:
            if col in data.columns:
                for window in windows:
                    for op in operations:
                        if op == 'mean':
                            features[f'{col}_rolling_{window}_mean'] = data[col].rolling(window).mean()
                        elif op == 'std':
                            features[f'{col}_rolling_{window}_std'] = data[col].rolling(window).std()
                        elif op == 'min':
                            features[f'{col}_rolling_{window}_min'] = data[col].rolling(window).min()
                        elif op == 'max':
                            features[f'{col}_rolling_{window}_max'] = data[col].rolling(window).max()
                        elif op == 'skew':
                            features[f'{col}_rolling_{window}_skew'] = data[col].rolling(window).skew()
                        elif op == 'kurt':
                            features[f'{col}_rolling_{window}_kurt'] = data[col].rolling(window).kurt()
        
        return features
    
    def create_interaction_features(self, 
                                  data: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between pairs of variables.
        
        Args:
            data: Input DataFrame
            feature_pairs: List of tuples with feature pairs
            
        Returns:
            DataFrame with interaction features
        """
        features = data.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in data.columns and feat2 in data.columns:
                # Multiplication interaction
                features[f'{feat1}_x_{feat2}'] = data[feat1] * data[feat2]
                
                # Ratio interaction (avoid division by zero)
                features[f'{feat1}_div_{feat2}'] = data[feat1] / (data[feat2] + 1e-8)
                
                # Difference interaction
                features[f'{feat1}_minus_{feat2}'] = data[feat1] - data[feat2]
        
        return features
    
    def create_market_regime_features(self, 
                                    data: pd.DataFrame, 
                                    price_col: str = 'Close',
                                    volume_col: str = 'Volume') -> pd.DataFrame:
        """
        Create market regime features.
        
        Args:
            data: Input DataFrame
            price_col: Price column name
            volume_col: Volume column name
            
        Returns:
            DataFrame with market regime features
        """
        features = data.copy()
        
        # Trend features
        features['trend_5d'] = np.where(data[price_col] > data[price_col].shift(5), 1, 0)
        features['trend_20d'] = np.where(data[price_col] > data[price_col].shift(20), 1, 0)
        
        # Volatility regime
        vol_20d = data[price_col].pct_change().rolling(20).std()
        vol_60d = data[price_col].pct_change().rolling(60).std()
        features['high_vol_regime'] = np.where(vol_20d > vol_60d.quantile(0.8), 1, 0)
        
        # Volume regime
        if volume_col in data.columns:
            vol_ma_20 = data[volume_col].rolling(20).mean()
            features['high_volume_regime'] = np.where(data[volume_col] > vol_ma_20 * 1.5, 1, 0)
        
        # Market stress indicator
        returns = data[price_col].pct_change()
        features['market_stress'] = np.where(returns < returns.quantile(0.05), 1, 0)
        
        return features
    
    def create_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical time-based features.
        
        Args:
            data: Input DataFrame with datetime index
            
        Returns:
            DataFrame with cyclical features
        """
        features = data.copy()
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Day of week (Monday=0, Sunday=6)
            features['day_of_week'] = data.index.dayofweek
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
            
            # Month features
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Cyclical encoding
            features['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            features['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
            
            # Market calendar effects
            features['is_month_end'] = (data.index == data.index.to_period('M').end_time).astype(int)
            features['is_quarter_end'] = (data.index.month % 3 == 0).astype(int)
        
        return features
    
    def create_target_variable(self, 
                             data: pd.DataFrame, 
                             price_col: str = 'Close',
                             horizon: int = 1,
                             target_type: str = 'return') -> pd.Series:
        """
        Create target variable for prediction.
        
        Args:
            data: Input DataFrame
            price_col: Price column name
            horizon: Prediction horizon
            target_type: 'return', 'direction', or 'volatility'
            
        Returns:
            Target variable series
        """
        if target_type == 'return':
            # Future returns
            target = data[price_col].pct_change(horizon).shift(-horizon)
        
        elif target_type == 'direction':
            # Future price direction (1 for up, 0 for down)
            future_returns = data[price_col].pct_change(horizon).shift(-horizon)
            target = (future_returns > 0).astype(int)
        
        elif target_type == 'volatility':
            # Future realized volatility
            returns = data[price_col].pct_change()
            target = returns.rolling(horizon).std().shift(-horizon)
        
        else:
            raise ValueError("target_type must be 'return', 'direction', or 'volatility'")
        
        return target
    
    def scale_features(self, 
                      data: pd.DataFrame, 
                      method: str = 'standard',
                      fit_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Scale features using various methods.
        
        Args:
            data: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit_data: Data to fit scaler on (if None, use input data)
            
        Returns:
            Scaled DataFrame
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Fit scaler
        fit_data = fit_data if fit_data is not None else data
        numeric_cols = fit_data.select_dtypes(include=[np.number]).columns
        
        scaler.fit(fit_data[numeric_cols])
        self.scalers[method] = scaler
        
        # Transform data
        scaled_data = data.copy()
        scaled_data[numeric_cols] = scaler.transform(data[numeric_cols])
        
        return scaled_data
    
    def create_comprehensive_features(self, 
                                    data: pd.DataFrame,
                                    price_col: str = 'Close',
                                    volume_col: str = 'Volume',
                                    high_col: str = 'High',
                                    low_col: str = 'Low') -> pd.DataFrame:
        """
        Create comprehensive feature set combining all methods.
        
        Args:
            data: OHLCV DataFrame
            price_col: Price column name
            volume_col: Volume column name
            high_col: High price column name
            low_col: Low price column name
            
        Returns:
            DataFrame with comprehensive features
        """
        # Start with technical features
        features = self.create_technical_features(data, price_col, volume_col, high_col, low_col)
        
        # Add lag features for key indicators
        key_features = [price_col, 'rsi', 'macd', 'bb_position', 'atr']
        features = self.create_lag_features(features, key_features, [1, 2, 3, 5])
        
        # Add rolling features
        rolling_features = [price_col, 'price_change', 'rsi', 'volatility_10']
        features = self.create_rolling_features(features, rolling_features, [5, 10, 20])
        
        # Add interaction features
        interactions = [
            ('rsi', 'bb_position'),
            ('macd', 'price_change'),
            ('volatility_10', 'volume_sma')
        ]
        features = self.create_interaction_features(features, interactions)
        
        # Add market regime features
        features = self.create_market_regime_features(features, price_col, volume_col)
        
        # Add cyclical features
        features = self.create_cyclical_features(features)
        
        return features
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       method: str = 'correlation',
                       threshold: float = 0.05) -> List[str]:
        """
        Select features based on various criteria.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Selection method ('correlation', 'mutual_info', 'variance')
            threshold: Selection threshold
            
        Returns:
            List of selected feature names
        """
        if method == 'correlation':
            # Select features with correlation above threshold
            correlations = X.corrwith(y).abs()
            selected_features = correlations[correlations > threshold].index.tolist()
        
        elif method == 'variance':
            # Select features with variance above threshold
            variances = X.var()
            selected_features = variances[variances > threshold].index.tolist()
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            # Handle missing values
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            mi_scores = mutual_info_regression(X_clean, y_clean)
            mi_series = pd.Series(mi_scores, index=X.columns)
            selected_features = mi_series[mi_series > threshold].index.tolist()
        
        else:
            raise ValueError("Method must be 'correlation', 'mutual_info', or 'variance'")
        
        return selected_features
