"""
Advanced data preprocessing utilities for QuantumFinanceOpt.

This module provides additional preprocessing functions for financial data
including outlier detection, data transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class AdvancedPreprocessor:
    """Advanced preprocessing utilities for financial data."""
    
    def __init__(self):
        """Initialize AdvancedPreprocessor."""
        self.scalers = {}
        self.outlier_bounds = {}
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in financial data.
        
        Args:
            data: Input data
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        try:
            logger.info(f"Detecting outliers using {method} method")
            
            outliers = pd.DataFrame(False, index=data.index, columns=data.columns)
            
            for column in data.select_dtypes(include=[np.number]).columns:
                series = data[column].dropna()
                
                if method == 'iqr':
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers[column] = (data[column] < lower_bound) | (data[column] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(series))
                    outliers.loc[series.index, column] = z_scores > threshold
                    
                elif method == 'modified_zscore':
                    median = series.median()
                    mad = np.median(np.abs(series - median))
                    modified_z_scores = 0.6745 * (series - median) / mad
                    outliers.loc[series.index, column] = np.abs(modified_z_scores) > threshold
                
                # Store bounds for later use
                if method == 'iqr':
                    self.outlier_bounds[column] = (lower_bound, upper_bound)
            
            outlier_count = outliers.sum().sum()
            logger.info(f"Detected {outlier_count} outliers")
            
            return outliers
            
        except Exception as e:
            raise DataProcessingError(f"Outlier detection failed: {e}")
    
    def handle_outliers(self, data: pd.DataFrame, outliers: pd.DataFrame, 
                       method: str = 'winsorize') -> pd.DataFrame:
        """
        Handle detected outliers.
        
        Args:
            data: Input data
            outliers: Boolean DataFrame indicating outliers
            method: Handling method ('winsorize', 'remove', 'cap')
            
        Returns:
            Data with outliers handled
        """
        try:
            logger.info(f"Handling outliers using {method} method")
            
            processed_data = data.copy()
            
            if method == 'winsorize':
                # Winsorize at 5th and 95th percentiles
                for column in data.select_dtypes(include=[np.number]).columns:
                    lower_bound = data[column].quantile(0.05)
                    upper_bound = data[column].quantile(0.95)
                    processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)
                    
            elif method == 'remove':
                # Remove rows with any outliers
                outlier_rows = outliers.any(axis=1)
                processed_data = processed_data[~outlier_rows]
                
            elif method == 'cap':
                # Cap outliers at bounds
                for column in data.select_dtypes(include=[np.number]).columns:
                    if column in self.outlier_bounds:
                        lower_bound, upper_bound = self.outlier_bounds[column]
                        processed_data[column] = processed_data[column].clip(lower_bound, upper_bound)
            
            logger.info(f"Outlier handling completed. Shape: {processed_data.shape}")
            
            return processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Outlier handling failed: {e}")
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize financial data.
        
        Args:
            data: Input data
            method: Normalization method ('standard', 'robust', 'minmax')
            
        Returns:
            Normalized data
        """
        try:
            logger.info(f"Normalizing data using {method} method")
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                raise DataProcessingError(f"Unknown normalization method: {method}")
            
            # Fit and transform numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            normalized_data = data.copy()
            
            if len(numeric_columns) > 0:
                normalized_values = scaler.fit_transform(data[numeric_columns])
                normalized_data[numeric_columns] = normalized_values
                
                # Store scaler for inverse transformation
                self.scalers[method] = scaler
            
            logger.info("Data normalization completed")
            
            return normalized_data
            
        except Exception as e:
            raise DataProcessingError(f"Data normalization failed: {e}")
    
    def create_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators from price data.
        
        Args:
            prices: Price data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            logger.info("Creating technical indicators")
            
            indicators = pd.DataFrame(index=prices.index)
            
            for column in prices.columns:
                price_series = prices[column]
                
                # Moving averages
                indicators[f'{column}_MA_5'] = price_series.rolling(window=5).mean()
                indicators[f'{column}_MA_10'] = price_series.rolling(window=10).mean()
                indicators[f'{column}_MA_20'] = price_series.rolling(window=20).mean()
                
                # Exponential moving averages
                indicators[f'{column}_EMA_12'] = price_series.ewm(span=12).mean()
                indicators[f'{column}_EMA_26'] = price_series.ewm(span=26).mean()
                
                # MACD
                macd_line = indicators[f'{column}_EMA_12'] - indicators[f'{column}_EMA_26']
                indicators[f'{column}_MACD'] = macd_line
                indicators[f'{column}_MACD_Signal'] = macd_line.ewm(span=9).mean()
                
                # RSI
                delta = price_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators[f'{column}_RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                ma_20 = indicators[f'{column}_MA_20']
                std_20 = price_series.rolling(window=20).std()
                indicators[f'{column}_BB_Upper'] = ma_20 + (std_20 * 2)
                indicators[f'{column}_BB_Lower'] = ma_20 - (std_20 * 2)
                indicators[f'{column}_BB_Width'] = indicators[f'{column}_BB_Upper'] - indicators[f'{column}_BB_Lower']
                
                # Volatility
                indicators[f'{column}_Volatility'] = price_series.rolling(window=20).std()
                
                # Price momentum
                indicators[f'{column}_Momentum_5'] = price_series / price_series.shift(5) - 1
                indicators[f'{column}_Momentum_10'] = price_series / price_series.shift(10) - 1
            
            # Remove NaN values
            indicators = indicators.dropna()
            
            logger.info(f"Created {indicators.shape[1]} technical indicators")
            
            return indicators
            
        except Exception as e:
            raise DataProcessingError(f"Technical indicator creation failed: {e}")
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features from data.
        
        Args:
            data: Input data
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        try:
            logger.info(f"Creating lag features for periods: {lags}")
            
            lagged_features = pd.DataFrame(index=data.index)
            
            for column in data.columns:
                for lag in lags:
                    lagged_features[f'{column}_lag_{lag}'] = data[column].shift(lag)
            
            # Remove NaN values
            lagged_features = lagged_features.dropna()
            
            logger.info(f"Created {lagged_features.shape[1]} lagged features")
            
            return lagged_features
            
        except Exception as e:
            raise DataProcessingError(f"Lag feature creation failed: {e}")
    
    def validate_preprocessing_quality(self, original_data: pd.DataFrame, 
                                     processed_data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate preprocessing quality.
        
        Args:
            original_data: Original data before preprocessing
            processed_data: Data after preprocessing
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                'original_shape': original_data.shape,
                'processed_shape': processed_data.shape,
                'data_retention': processed_data.shape[0] / original_data.shape[0],
                'missing_data_reduction': {
                    'original': original_data.isnull().sum().sum(),
                    'processed': processed_data.isnull().sum().sum()
                }
            }
            
            # Statistical comparison for numeric columns
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                original_stats = original_data[numeric_cols].describe()
                processed_stats = processed_data[numeric_cols].describe()
                
                quality_metrics['statistical_changes'] = {
                    'mean_change': ((processed_stats.loc['mean'] - original_stats.loc['mean']) / 
                                   original_stats.loc['mean'] * 100).to_dict(),
                    'std_change': ((processed_stats.loc['std'] - original_stats.loc['std']) / 
                                  original_stats.loc['std'] * 100).to_dict()
                }
            
            logger.info("Preprocessing quality validation completed")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Preprocessing quality validation failed: {e}")
            return {'error': str(e)}


# Unit tests for AdvancedPreprocessor
def test_advanced_preprocessor():
    """Unit tests for AdvancedPreprocessor functionality."""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'AAPL': np.random.randn(100).cumsum() + 100,
        'GOOGL': np.random.randn(100).cumsum() + 1000
    }, index=dates)
    
    # Add some outliers
    test_data.iloc[10, 0] = 1000  # Extreme outlier
    test_data.iloc[20, 1] = 5000  # Extreme outlier
    
    preprocessor = AdvancedPreprocessor()
    
    try:
        # Test outlier detection
        outliers = preprocessor.detect_outliers(test_data)
        assert outliers.sum().sum() > 0, "Should detect outliers"
        
        # Test outlier handling
        cleaned_data = preprocessor.handle_outliers(test_data, outliers)
        assert not cleaned_data.empty, "Cleaned data should not be empty"
        
        # Test normalization
        normalized_data = preprocessor.normalize_data(test_data)
        assert not normalized_data.empty, "Normalized data should not be empty"
        
        # Test technical indicators
        indicators = preprocessor.create_technical_indicators(test_data)
        assert not indicators.empty, "Technical indicators should not be empty"
        
        # Test lag features
        lag_features = preprocessor.create_lag_features(test_data)
        assert not lag_features.empty, "Lag features should not be empty"
        
        # Test quality validation
        quality_metrics = preprocessor.validate_preprocessing_quality(test_data, cleaned_data)
        assert 'original_shape' in quality_metrics, "Quality metrics should contain shape info"
        
        print("All AdvancedPreprocessor tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_advanced_preprocessor()