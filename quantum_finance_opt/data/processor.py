"""
Data processing module for QuantumFinanceOpt.

This module handles data loading, preprocessing, validation, and simulation
for portfolio optimization.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

from ..core.exceptions import DataProcessingError, ValidationError
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processing class for QuantumFinanceOpt."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize DataProcessor with configuration."""
        self.config = config
        self.data = None
        self.returns = None
        self.tickers = None
        self.dates = None
        
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load historical price data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with price data
            
        Raises:
            DataProcessingError: If file loading fails
        """
        try:
            if not os.path.exists(csv_path):
                raise DataProcessingError(f"CSV file not found: {csv_path}")
            
            logger.info(f"Loading data from {csv_path}")
            
            # Try different common CSV formats
            try:
                # First try with date as index
                data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except:
                try:
                    # Try with date column
                    data = pd.read_csv(csv_path, parse_dates=['date'])
                    data.set_index('date', inplace=True)
                except:
                    # Try basic loading
                    data = pd.read_csv(csv_path)
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                        data.set_index('date', inplace=True)
                    else:
                        # Assume first column is date
                        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
                        data.set_index(data.columns[0], inplace=True)
            
            # Validate data structure
            self._validate_csv_data(data)
            
            logger.info(f"Successfully loaded data with shape {data.shape}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"Columns: {list(data.columns)}")
            
            return data
            
        except pd.errors.EmptyDataError:
            raise DataProcessingError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise DataProcessingError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise DataProcessingError(f"Failed to load CSV file: {e}")
    
    def _validate_csv_data(self, data: pd.DataFrame):
        """
        Validate CSV data structure and content.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check if data is empty
            if data.empty:
                raise ValidationError("Data is empty")
            
            # Check if index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValidationError("Index must be datetime")
            
            # Check for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValidationError("No numeric columns found")
            
            # Check for missing dates
            if data.index.duplicated().any():
                logger.warning("Duplicate dates found, removing duplicates")
                data = data[~data.index.duplicated(keep='first')]
            
            # Check data quality
            missing_pct = (data.isnull().sum() / len(data)) * 100
            if missing_pct.max() > 50:
                logger.warning(f"High missing data percentage: {missing_pct.max():.1f}%")
            
            # Check for negative prices
            if (data < 0).any().any():
                logger.warning("Negative values found in price data")
            
            logger.info("Data validation completed successfully")
            
        except Exception as e:
            raise ValidationError(f"Data validation failed: {e}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess loaded data for analysis.
        
        Args:
            data: Raw price data
            
        Returns:
            Preprocessed data
            
        Raises:
            DataProcessingError: If preprocessing fails
        """
        try:
            logger.info("Starting data preprocessing")
            
            # Make a copy to avoid modifying original
            processed_data = data.copy()
            
            # Sort by date
            processed_data = processed_data.sort_index()
            
            # Handle missing data
            processed_data = self.handle_missing_data(processed_data)
            
            # Remove columns with all NaN values
            processed_data = processed_data.dropna(axis=1, how='all')
            
            # Filter by date range if specified
            if self.config.start_date:
                start_date = pd.to_datetime(self.config.start_date)
                processed_data = processed_data[processed_data.index >= start_date]
            
            if self.config.end_date:
                end_date = pd.to_datetime(self.config.end_date)
                processed_data = processed_data[processed_data.index <= end_date]
            
            # Filter tickers if specified
            if self.config.tickers:
                available_tickers = [t for t in self.config.tickers if t in processed_data.columns]
                if not available_tickers:
                    raise DataProcessingError("None of the specified tickers found in data")
                processed_data = processed_data[available_tickers]
                
                missing_tickers = set(self.config.tickers) - set(available_tickers)
                if missing_tickers:
                    logger.warning(f"Missing tickers: {missing_tickers}")
            
            # Remove rows with all NaN values
            processed_data = processed_data.dropna(how='all')
            
            if processed_data.empty:
                raise DataProcessingError("No data remaining after preprocessing")
            
            # Store processed data
            self.data = processed_data
            self.tickers = list(processed_data.columns)
            self.dates = processed_data.index
            
            logger.info(f"Preprocessing completed. Final shape: {processed_data.shape}")
            
            return processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Data preprocessing failed: {e}")
    
    def handle_missing_data(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing data using specified method.
        
        Args:
            data: DataFrame with missing data
            method: Method to handle missing data ('forward_fill', 'interpolate', 'drop')
            
        Returns:
            DataFrame with missing data handled
        """
        try:
            logger.info(f"Handling missing data using method: {method}")
            
            missing_before = data.isnull().sum().sum()
            logger.info(f"Missing values before handling: {missing_before}")
            
            if method == 'forward_fill':
                # Forward fill then backward fill
                data = data.fillna(method='ffill').fillna(method='bfill')
            elif method == 'interpolate':
                # Linear interpolation
                data = data.interpolate(method='linear', limit_direction='both')
            elif method == 'drop':
                # Drop rows with any missing values
                data = data.dropna()
            else:
                logger.warning(f"Unknown method {method}, using forward_fill")
                data = data.fillna(method='ffill').fillna(method='bfill')
            
            missing_after = data.isnull().sum().sum()
            logger.info(f"Missing values after handling: {missing_after}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to handle missing data: {e}")
            return data
    
    def compute_returns(self, prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Compute returns from price data.
        
        Args:
            prices: Price data
            method: Return calculation method ('simple', 'log')
            
        Returns:
            DataFrame with returns
            
        Raises:
            DataProcessingError: If return calculation fails
        """
        try:
            logger.info(f"Computing {method} returns")
            
            if method == 'simple':
                returns = prices.pct_change()
            elif method == 'log':
                returns = np.log(prices / prices.shift(1))
            else:
                raise DataProcessingError(f"Unknown return method: {method}")
            
            # Remove first row (NaN values)
            returns = returns.dropna()
            
            # Check for infinite values
            if np.isinf(returns).any().any():
                logger.warning("Infinite values found in returns, replacing with NaN")
                returns = returns.replace([np.inf, -np.inf], np.nan)
                returns = returns.dropna()
            
            # Store returns
            self.returns = returns
            
            logger.info(f"Returns computed successfully. Shape: {returns.shape}")
            logger.info(f"Return statistics:\n{returns.describe()}")
            
            return returns
            
        except Exception as e:
            raise DataProcessingError(f"Failed to compute returns: {e}")
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data integrity and return quality metrics.
        
        Args:
            data: Data to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'shape': data.shape,
                'date_range': (data.index.min(), data.index.max()),
                'missing_data_pct': (data.isnull().sum() / len(data) * 100).to_dict(),
                'negative_values': (data < 0).sum().to_dict(),
                'zero_values': (data == 0).sum().to_dict(),
                'infinite_values': np.isinf(data).sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
            
            # Check for outliers (values beyond 3 standard deviations)
            outliers = {}
            for col in data.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers[col] = (z_scores > 3).sum()
            validation_results['outliers'] = outliers
            
            logger.info("Data integrity validation completed")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            return {'error': str(e)}


# Unit tests for DataProcessor
def test_data_processor():
    """Basic unit tests for DataProcessor functionality."""
    import tempfile
    
    # Create test configuration
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL'],
        start_date='2020-01-01',
        end_date='2020-12-31'
    )
    
    processor = DataProcessor(config)
    
    # Test CSV creation and loading
    test_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
        'AAPL': np.random.randn(366).cumsum() + 100,
        'GOOGL': np.random.randn(366).cumsum() + 1000
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        
        try:
            # Test loading
            loaded_data = processor.load_csv(f.name)
            assert not loaded_data.empty
            
            # Test preprocessing
            processed_data = processor.preprocess_data(loaded_data)
            assert not processed_data.empty
            
            # Test return calculation
            returns = processor.compute_returns(processed_data)
            assert not returns.empty
            
            # Test validation
            validation_results = processor.validate_data_integrity(processed_data)
            assert 'shape' in validation_results
            
            print("All DataProcessor tests passed!")
            
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    test_data_processor()