"""
Machine Learning prediction framework for QuantumFinanceOpt.

This module implements ML models for return prediction and feature engineering
for financial time series data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from ..core.exceptions import ModelTrainingError, OptimizationError
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class MLPredictor:
    """Machine Learning predictor for portfolio optimization."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize MLPredictor with configuration."""
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.trained_models = {}
        
    def create_features(self, data: pd.DataFrame, lookback_periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create features for ML models from price/return data.
        
        Args:
            data: Price or return data
            lookback_periods: Periods for lagged features
        
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("Creating features for ML models")
            
            features = pd.DataFrame(index=data.index)
            
            for column in data.columns:
                series = data[column]
                
                # Lagged returns
                for lag in lookback_periods:
                    features[f'{column}_lag_{lag}'] = series.shift(lag)
                
                # Moving averages
                for window in [5, 10, 20]:
                    features[f'{column}_ma_{window}'] = series.rolling(window=window).mean()
                    features[f'{column}_ma_ratio_{window}'] = series / features[f'{column}_ma_{window}']
                
                # Volatility features
                features[f'{column}_vol_5'] = series.rolling(window=5).std()
                features[f'{column}_vol_20'] = series.rolling(window=20).std()
                
                # RSI
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features[f'{column}_rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = series.ewm(span=12).mean()
                ema_26 = series.ewm(span=26).mean()
                features[f'{column}_macd'] = ema_12 - ema_26
                features[f'{column}_macd_signal'] = features[f'{column}_macd'].ewm(span=9).mean()
                
                # Momentum
                features[f'{column}_momentum_5'] = series / series.shift(5) - 1
                features[f'{column}_momentum_10'] = series / series.shift(10) - 1
                
                # Bollinger Bands
                ma_20 = series.rolling(window=20).mean()
                std_20 = series.rolling(window=20).std()
                features[f'{column}_bb_upper'] = ma_20 + (std_20 * 2)
                features[f'{column}_bb_lower'] = ma_20 - (std_20 * 2)
                features[f'{column}_bb_position'] = (series - features[f'{column}_bb_lower']) / (
                    features[f'{column}_bb_upper'] - features[f'{column}_bb_lower']
                )
            
            # Cross-asset features
            if len(data.columns) > 1:
                # Correlation features
                for i, col1 in enumerate(data.columns):
                    for j, col2 in enumerate(data.columns[i+1:], i+1):
                        rolling_corr = data[col1].rolling(window=20).corr(data[col2])
                        features[f'{col1}_{col2}_corr_20'] = rolling_corr
                
                # Market features (using equal-weighted average)
                market_proxy = data.mean(axis=1)
                features['market_return'] = market_proxy
                features['market_vol'] = market_proxy.rolling(window=20).std()
                
                # Individual asset vs market
                for column in data.columns:
                    features[f'{column}_vs_market'] = data[column] - market_proxy
                    beta = data[column].rolling(window=60).cov(market_proxy) / market_proxy.rolling(window=60).var()
                    features[f'{column}_beta'] = beta
            
            # Remove NaN values
            features = features.dropna()
            
            self.feature_names = list(features.columns)
            logger.info(f"Created {len(self.feature_names)} features")
            
            return features
            
        except Exception as e:
            raise ModelTrainingError(f"Feature creation failed: {e}")
    
    def prepare_training_data(self, features: pd.DataFrame, targets: pd.DataFrame,
                            forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for ML models.
        
        Args:
            features: Feature data
            targets: Target returns
            forecast_horizon: Number of periods to forecast ahead
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            logger.info(f"Preparing training data with forecast horizon: {forecast_horizon}")
            
            # Align features and targets
            common_dates = features.index.intersection(targets.index)
            features_aligned = features.loc[common_dates]
            targets_aligned = targets.loc[common_dates]
            
            # Create target variables (future returns)
            y_data = []
            X_data = []
            
            for i in range(len(features_aligned) - forecast_horizon):
                X_data.append(features_aligned.iloc[i].values)
                y_data.append(targets_aligned.iloc[i + forecast_horizon].values)
            
            X = np.array(X_data)
            y = np.array(y_data)
            
            logger.info(f"Training data prepared. X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y, self.feature_names
            
        except Exception as e:
            raise ModelTrainingError(f"Training data preparation failed: {e}")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, asset_names: List[str]) -> Dict[str, Any]:
        """
        Train ensemble of ML models.
        
        Args:
            X: Feature matrix
            y: Target matrix
            asset_names: Names of assets
        
        Returns:
            Dictionary of trained models
        """
        try:
            logger.info("Training ML ensemble models")
            
            # Initialize models
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.01),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.random_seed,
                    n_jobs=self.config.n_jobs
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config.random_seed
                )
            }
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['features'] = scaler
            
            # Train models for each asset
            trained_models = {}
            model_scores = {}
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            for asset_idx, asset_name in enumerate(asset_names):
                logger.info(f"Training models for {asset_name}")
                
                y_asset = y[:, asset_idx] if y.ndim > 1 else y
                trained_models[asset_name] = {}
                model_scores[asset_name] = {}
                
                for model_name, model in models.items():
                    try:
                        # Cross-validation
                        cv_scores = cross_val_score(
                            model, X_scaled, y_asset, cv=tscv, 
                            scoring='neg_mean_squared_error', n_jobs=1
                        )
                        
                        # Train on full dataset
                        model.fit(X_scaled, y_asset)
                        
                        # Store model and scores
                        trained_models[asset_name][model_name] = model
                        model_scores[asset_name][model_name] = {
                            'cv_mse': -cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'train_score': model.score(X_scaled, y_asset)
                        }
                        
                        logger.info(f"{asset_name} - {model_name}: CV MSE = {-cv_scores.mean():.6f}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to train {model_name} for {asset_name}: {e}")
                        continue
            
            self.trained_models = trained_models
            
            results = {
                'models': trained_models,
                'scores': model_scores,
                'scaler': scaler,
                'feature_names': self.feature_names
            }
            
            logger.info("ML ensemble training completed")
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Ensemble training failed: {e}")
    
    def predict_returns(self, features: pd.DataFrame, asset_names: List[str],
                       ensemble_method: str = 'average') -> pd.DataFrame:
        """
        Predict returns using trained models.
        
        Args:
            features: Feature data for prediction
            asset_names: Names of assets
            ensemble_method: Method for combining predictions ('average', 'weighted')
        
        Returns:
            DataFrame with predicted returns
        """
        try:
            logger.info(f"Predicting returns using {ensemble_method} ensemble")
            
            if not self.trained_models:
                raise ModelTrainingError("No trained models available")
            
            # Scale features
            if 'features' not in self.scalers:
                raise ModelTrainingError("Feature scaler not available")
            
            X_scaled = self.scalers['features'].transform(features.values)
            
            predictions = pd.DataFrame(index=features.index, columns=asset_names)
            
            for asset_name in asset_names:
                if asset_name not in self.trained_models:
                    logger.warning(f"No trained models for {asset_name}")
                    continue
                
                asset_predictions = []
                model_weights = []
                
                for model_name, model in self.trained_models[asset_name].items():
                    try:
                        pred = model.predict(X_scaled)
                        asset_predictions.append(pred)
                        
                        # Weight based on training performance (inverse of MSE)
                        if ensemble_method == 'weighted':
                            mse = self.trained_models[asset_name][model_name].get('cv_mse', 1.0)
                            weight = 1.0 / (mse + 1e-8)
                            model_weights.append(weight)
                        else:
                            model_weights.append(1.0)
                            
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_name}: {e}")
                        continue
                
                if asset_predictions:
                    # Combine predictions
                    asset_predictions = np.array(asset_predictions)
                    model_weights = np.array(model_weights)
                    model_weights = model_weights / np.sum(model_weights)
                    
                    combined_pred = np.average(asset_predictions, axis=0, weights=model_weights)
                    predictions[asset_name] = combined_pred
                else:
                    logger.warning(f"No valid predictions for {asset_name}")
                    predictions[asset_name] = 0.0
            
            logger.info("Return predictions completed")
            return predictions
            
        except Exception as e:
            raise ModelTrainingError(f"Return prediction failed: {e}")
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for ML models.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to tune
        
        Returns:
            Best parameters and model
        """
        try:
            logger.info(f"Performing hyperparameter tuning for {model_type}")
            
            # Define parameter grids
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'lasso': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                'ridge': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            }
            
            if model_type not in param_grids:
                raise ModelTrainingError(f"Unknown model type: {model_type}")
            
            # Initialize model
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=self.config.random_seed)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(random_state=self.config.random_seed)
            elif model_type == 'lasso':
                model = Lasso()
            elif model_type == 'ridge':
                model = Ridge()
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Grid search
            grid_search = GridSearchCV(
                model,
                param_grids[model_type],
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=self.config.n_jobs,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_model': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.6f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Hyperparameter tuning failed: {e}")
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       asset_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            asset_names: Names of assets
        
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info("Evaluating ML models")
            
            if not self.trained_models:
                raise ModelTrainingError("No trained models available")
            
            # Scale test features
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            evaluation_results = {}
            
            for asset_idx, asset_name in enumerate(asset_names):
                if asset_name not in self.trained_models:
                    continue
                
                y_test_asset = y_test[:, asset_idx] if y_test.ndim > 1 else y_test
                evaluation_results[asset_name] = {}
                
                for model_name, model in self.trained_models[asset_name].items():
                    try:
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test_asset, y_pred)
                        mae = mean_absolute_error(y_test_asset, y_pred)
                        r2 = r2_score(y_test_asset, y_pred)
                        
                        # Directional accuracy
                        direction_actual = np.sign(y_test_asset)
                        direction_pred = np.sign(y_pred)
                        directional_accuracy = np.mean(direction_actual == direction_pred)
                        
                        evaluation_results[asset_name][model_name] = {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2,
                            'directional_accuracy': directional_accuracy
                        }
                        
                    except Exception as e:
                        logger.warning(f"Evaluation failed for {model_name}: {e}")
                        continue
            
            logger.info("Model evaluation completed")
            return evaluation_results
            
        except Exception as e:
            raise ModelTrainingError(f"Model evaluation failed: {e}")


# Unit tests for MLPredictor
def test_ml_predictor():
    """Unit tests for MLPredictor functionality."""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Generate correlated price data
    returns_data = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 200),
        'GOOGL': np.random.normal(0.0008, 0.025, 200),
        'MSFT': np.random.normal(0.0012, 0.018, 200)
    }, index=dates)
    
    # Convert to prices
    prices_data = (1 + returns_data).cumprod() * 100
    
    config = OptimizationConfig()
    ml_predictor = MLPredictor(config)
    
    try:
        # Test feature creation
        features = ml_predictor.create_features(returns_data)
        assert not features.empty, "Features should not be empty"
        assert len(ml_predictor.feature_names) > 0, "Feature names should be created"
        
        # Test training data preparation
        X, y, feature_names = ml_predictor.prepare_training_data(features, returns_data)
        assert X.shape[0] > 0, "Training data should not be empty"
        assert X.shape[1] == len(feature_names), "Feature dimensions should match"
        
        # Test ensemble training
        asset_names = list(returns_data.columns)
        training_results = ml_predictor.train_ensemble(X, y, asset_names)
        assert 'models' in training_results, "Should return trained models"
        assert len(training_results['models']) > 0, "Should train models for assets"
        
        # Test predictions
        predictions = ml_predictor.predict_returns(features.tail(10), asset_names)
        assert not predictions.empty, "Predictions should not be empty"
        assert predictions.shape[1] == len(asset_names), "Should predict for all assets"
        
        # Test model evaluation
        X_test, y_test = X[-20:], y[-20:]
        evaluation_results = ml_predictor.evaluate_models(X_test, y_test, asset_names)
        assert len(evaluation_results) > 0, "Should return evaluation results"
        
        print("All MLPredictor tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_ml_predictor()