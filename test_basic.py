#!/usr/bin/env python3
"""
Basic functionality test for QuantumFinanceOpt.

This script tests core functionality without requiring all optional dependencies.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")
    
    try:
        from quantum_finance_opt.core.config import OptimizationConfig
        from quantum_finance_opt.core.exceptions import QuantumFinanceOptError
        from quantum_finance_opt.data.processor import DataProcessor
        from quantum_finance_opt.data.simulator import DataSimulator
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration management."""
    print("Testing configuration...")
    
    try:
        from quantum_finance_opt.core.config import OptimizationConfig
        
        # Test basic configuration
        config = OptimizationConfig(
            tickers=['AAPL', 'GOOGL'],
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        assert config.tickers == ['AAPL', 'GOOGL']
        assert config.start_date == '2020-01-01'
        assert config.budget > 0
        
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_simulation():
    """Test data simulation functionality."""
    print("Testing data simulation...")
    
    try:
        from quantum_finance_opt.core.config import OptimizationConfig
        from quantum_finance_opt.data.simulator import DataSimulator
        
        config = OptimizationConfig(
            tickers=['AAPL', 'GOOGL'],
            start_date='2020-01-01',
            end_date='2020-03-31'
        )
        
        simulator = DataSimulator(config)
        
        # Test price data generation
        price_data = simulator.generate_sample_data(['AAPL', 'GOOGL'], 90)
        assert not price_data.empty
        assert price_data.shape == (90, 2)
        
        # Test news data generation
        dates = ['2020-01-01', '2020-01-02']
        news_data = simulator.simulate_news_data(['AAPL'], dates)
        assert not news_data.empty
        
        # Test ESG scores
        esg_scores = simulator.simulate_esg_scores(['AAPL', 'GOOGL'])
        assert len(esg_scores) == 2
        assert all(0 <= score <= 1 for score in esg_scores.values())
        
        print("✓ Data simulation test passed")
        return True
    except Exception as e:
        print(f"✗ Data simulation test failed: {e}")
        return False

def test_data_processing():
    """Test data processing functionality."""
    print("Testing data processing...")
    
    try:
        from quantum_finance_opt.core.config import OptimizationConfig
        from quantum_finance_opt.data.processor import DataProcessor
        
        config = OptimizationConfig()
        processor = DataProcessor(config)
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 100,
            'GOOGL': np.random.randn(100).cumsum() + 1000
        }, index=dates)
        
        # Test preprocessing
        processed_data = processor.preprocess_data(test_data)
        assert not processed_data.empty
        
        # Test return calculation
        returns = processor.compute_returns(processed_data)
        assert not returns.empty
        assert returns.shape[0] == processed_data.shape[0] - 1  # One less due to diff
        
        # Test data validation
        validation_results = processor.validate_data_integrity(processed_data)
        assert 'shape' in validation_results
        
        print("✓ Data processing test passed")
        return True
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def test_basic_math():
    """Test basic mathematical operations."""
    print("Testing basic mathematical operations...")
    
    try:
        # Test portfolio weight calculations
        weights = np.array([0.4, 0.3, 0.3])
        assert abs(np.sum(weights) - 1.0) < 1e-10
        
        # Test return calculations
        prices = np.array([100, 101, 102, 101])
        returns = np.diff(prices) / prices[:-1]
        expected_returns = np.array([0.01, 0.0099, -0.0098])
        assert np.allclose(returns, expected_returns, atol=1e-4)
        
        # Test covariance calculation
        data = np.random.randn(100, 3)
        cov_matrix = np.cov(data.T)
        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric
        
        print("✓ Basic math test passed")
        return True
    except Exception as e:
        print(f"✗ Basic math test failed: {e}")
        return False

def test_file_operations():
    """Test file I/O operations."""
    print("Testing file operations...")
    
    try:
        # Test CSV creation and reading
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        test_data = pd.DataFrame({
            'AAPL': np.random.randn(10).cumsum() + 100,
            'GOOGL': np.random.randn(10).cumsum() + 1000
        }, index=dates)
        
        # Save to CSV
        test_file = 'test_data.csv'
        test_data.to_csv(test_file)
        
        # Read back
        loaded_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
        assert loaded_data.shape == test_data.shape
        
        # Clean up
        os.remove(test_file)
        
        print("✓ File operations test passed")
        return True
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("QuantumFinanceOpt - Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_simulation,
        test_data_processing,
        test_basic_math,
        test_file_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic tests passed! Core functionality is working.")
        return True
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)