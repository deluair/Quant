"""
Machine learning models for financial return prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


class ReturnPredictor:
    """
    Machine learning models for predicting financial returns.
    """
    
    def __init__(self):
        """Initialize Return Predictor."""
        self.models = {}
        self.feature_importance = {}
        self.predictions = {}
        self.performance_metrics = {}
    
    def prepare_data(self, 
                    features: pd.DataFrame, 
                    target: pd.Series,
                    test_size: float = 0.2,
                    validation_size: float = 0.1) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Prepare data for training with proper time series splits.
        
        Args:
            features: Feature DataFrame
            target: Target series
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Dictionary with train/validation/test splits
        """
        # Align features and target
        aligned_data = pd.concat([features, target], axis=1).dropna()
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Time series split (no shuffling)
        n_samples = len(X)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - validation_size))
        
        X_train = X.iloc[:val_start]
        y_train = y.iloc[:val_start]
        
        X_val = X.iloc[val_start:test_start]
        y_val = y.iloc[val_start:test_start]
        
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def train_linear_models(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train linear regression models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with trained models
        """
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        trained_models = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Store feature importance for regularized models
                if hasattr(model, 'coef_'):
                    self.feature_importance[name] = pd.Series(
                        model.coef_, index=X_train.columns
                    ).abs().sort_values(ascending=False)
                
                # Validation performance
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    self.performance_metrics[name] = {
                        'val_mse': mean_squared_error(y_val, val_pred),
                        'val_mae': mean_absolute_error(y_val, val_pred),
                        'val_r2': r2_score(y_val, val_pred)
                    }
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        self.models.update(trained_models)
        return trained_models
    
    def train_tree_models(self, 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         X_val: Optional[pd.DataFrame] = None,
                         y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train tree-based models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with trained models
        """
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        trained_models = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Store feature importance
                self.feature_importance[name] = pd.Series(
                    model.feature_importances_, index=X_train.columns
                ).sort_values(ascending=False)
                
                # Validation performance
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    self.performance_metrics[name] = {
                        'val_mse': mean_squared_error(y_val, val_pred),
                        'val_mae': mean_absolute_error(y_val, val_pred),
                        'val_r2': r2_score(y_val, val_pred)
                    }
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        self.models.update(trained_models)
        return trained_models
    
    def prepare_lstm_data(self, 
                         X: pd.DataFrame, 
                         y: pd.Series, 
                         lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            lookback: Number of time steps to look back
            
        Returns:
            Tuple of (X_lstm, y_lstm) arrays
        """
        X_values = X.values
        y_values = y.values
        
        X_lstm = []
        y_lstm = []
        
        for i in range(lookback, len(X_values)):
            X_lstm.append(X_values[i-lookback:i])
            y_lstm.append(y_values[i])
        
        return np.array(X_lstm), np.array(y_lstm)
    
    def train_lstm_model(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        lookback: int = 60,
                        epochs: int = 50,
                        batch_size: int = 32) -> tf.keras.Model:
        """
        Train LSTM model for time series prediction.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            lookback: LSTM lookback window
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Trained LSTM model
        """
        try:
            # Prepare LSTM data
            X_lstm_train, y_lstm_train = self.prepare_lstm_data(X_train, y_train, lookback)
            
            validation_data = None
            if X_val is not None and y_val is not None:
                X_lstm_val, y_lstm_val = self.prepare_lstm_data(X_val, y_val, lookback)
                validation_data = (X_lstm_val, y_lstm_val)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_lstm_train, y_lstm_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=0
            )
            
            self.models['lstm'] = model
            
            # Store training history
            if validation_data is not None:
                val_loss = min(history.history['val_loss'])
                self.performance_metrics['lstm'] = {
                    'val_mse': val_loss,
                    'val_mae': min(history.history['val_mae'])
                }
            
            return model
            
        except Exception as e:
            print(f"Error training LSTM: {str(e)}")
            return None
    
    def train_gru_model(self, 
                       X_train: pd.DataFrame, 
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       lookback: int = 60,
                       epochs: int = 50,
                       batch_size: int = 32) -> tf.keras.Model:
        """
        Train GRU model for time series prediction.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            lookback: GRU lookback window
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Trained GRU model
        """
        try:
            # Prepare data
            X_gru_train, y_gru_train = self.prepare_lstm_data(X_train, y_train, lookback)
            
            validation_data = None
            if X_val is not None and y_val is not None:
                X_gru_val, y_gru_val = self.prepare_lstm_data(X_val, y_val, lookback)
                validation_data = (X_gru_val, y_gru_val)
            
            # Build GRU model
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_gru_train, y_gru_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=0
            )
            
            self.models['gru'] = model
            
            if validation_data is not None:
                val_loss = min(history.history['val_loss'])
                self.performance_metrics['gru'] = {
                    'val_mse': val_loss,
                    'val_mae': min(history.history['val_mae'])
                }
            
            return model
            
        except Exception as e:
            print(f"Error training GRU: {str(e)}")
            return None
    
    def ensemble_predict(self, 
                        X: pd.DataFrame, 
                        models: List[str] = None,
                        weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Make ensemble predictions using multiple models.
        
        Args:
            X: Features for prediction
            models: List of model names to use
            weights: Weights for ensemble (if None, use equal weights)
            
        Returns:
            Ensemble predictions
        """
        if models is None:
            models = list(self.models.keys())
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        predictions = []
        
        for model_name in models:
            if model_name in self.models:
                model = self.models[model_name]
                
                if model_name in ['lstm', 'gru']:
                    # Handle RNN models (need sequence data)
                    # For simplicity, use last prediction or implement proper sequence handling
                    continue
                else:
                    pred = model.predict(X)
                    predictions.append(pred)
        
        if predictions:
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights[:len(predictions)])
            return ensemble_pred
        else:
            raise ValueError("No valid models found for ensemble prediction")
    
    def cross_validate_models(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            cv_folds: int = 5) -> pd.DataFrame:
        """
        Cross-validate models using time series splits.
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            
        Returns:
            DataFrame with CV results
        """
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        results = []
        
        # Test linear and tree models
        test_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        for name, model in test_models.items():
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, scoring='neg_mean_squared_error'
                )
                
                results.append({
                    'model': name,
                    'cv_mse_mean': -cv_scores.mean(),
                    'cv_mse_std': cv_scores.std(),
                    'cv_scores': cv_scores
                })
                
            except Exception as e:
                print(f"Error in CV for {name}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def predict_returns(self, 
                       X: pd.DataFrame, 
                       model_name: str = 'ensemble') -> np.ndarray:
        """
        Predict returns using specified model.
        
        Args:
            X: Features for prediction
            model_name: Name of model to use
            
        Returns:
            Predictions array
        """
        if model_name == 'ensemble':
            return self.ensemble_predict(X)
        elif model_name in self.models:
            model = self.models[model_name]
            return model.predict(X)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def evaluate_model(self, 
                      y_true: pd.Series, 
                      y_pred: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of model
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        # Directional accuracy
        y_true_direction = (y_true > 0).astype(int)
        y_pred_direction = (y_pred > 0).astype(int)
        metrics['directional_accuracy'] = (y_true_direction == y_pred_direction).mean()
        
        # Information coefficient
        metrics['ic'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        self.performance_metrics[model_name] = metrics
        return metrics
    
    def get_feature_importance(self, 
                             model_name: str, 
                             top_n: int = 20) -> pd.Series:
        """
        Get feature importance for specified model.
        
        Args:
            model_name: Name of model
            top_n: Number of top features to return
            
        Returns:
            Series with feature importance
        """
        if model_name in self.feature_importance:
            return self.feature_importance[model_name].head(top_n)
        else:
            raise ValueError(f"Feature importance not available for {model_name}")
    
    def model_comparison(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.performance_metrics:
            raise ValueError("No models have been evaluated yet")
        
        comparison_df = pd.DataFrame(self.performance_metrics).T
        comparison_df = comparison_df.sort_values('r2', ascending=False)
        
        return comparison_df
    
    def save_models(self, filepath: str):
        """
        Save trained models to file.
        
        Args:
            filepath: Path to save models
        """
        import joblib
        
        # Save sklearn models
        sklearn_models = {k: v for k, v in self.models.items() 
                         if k not in ['lstm', 'gru']}
        
        if sklearn_models:
            joblib.dump(sklearn_models, f"{filepath}_sklearn_models.pkl")
        
        # Save neural network models
        for name in ['lstm', 'gru']:
            if name in self.models:
                self.models[name].save(f"{filepath}_{name}_model.h5")
    
    def load_models(self, filepath: str):
        """
        Load trained models from file.
        
        Args:
            filepath: Path to load models from
        """
        import joblib
        import os
        
        # Load sklearn models
        sklearn_path = f"{filepath}_sklearn_models.pkl"
        if os.path.exists(sklearn_path):
            sklearn_models = joblib.load(sklearn_path)
            self.models.update(sklearn_models)
        
        # Load neural network models
        for name in ['lstm', 'gru']:
            model_path = f"{filepath}_{name}_model.h5"
            if os.path.exists(model_path):
                self.models[name] = tf.keras.models.load_model(model_path)
