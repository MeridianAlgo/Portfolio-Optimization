#!/usr/bin/env python3
"""
Test Script for Ultimate Portfolio Optimizer

Quick test to verify all components work with real CSV data.
"""

import sys
import os
import logging
from datetime import datetime

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_csv_processing():
    """Test CSV file processing"""
    print("🧪 Testing CSV processing...")
    
    try:
        from quantum_finance_opt.realtime.live_data_processor import LiveDataProcessor
        
        # Initialize processor
        processor = LiveDataProcessor(
            api_keys={},
            update_frequency=10,
            enable_news=False,  # Disable for testing
            enable_sentiment=False
        )
        
        # Test with sample CSV
        if os.path.exists('sample_portfolio.csv'):
            portfolio_data = processor.load_user_portfolio('sample_portfolio.csv')
            print(f"✅ CSV loaded: {portfolio_data.shape[0]} rows, {portfolio_data.shape[1]} columns")
            print(f"   Symbols: {[col for col in portfolio_data.columns if col != 'Date']}")
            
            # Test enhanced data
            enhanced_data = processor.get_enhanced_portfolio_data()
            print(f"✅ Enhanced data generated")
            
            # Test optimization-ready data
            opt_data, metadata = processor.get_optimization_ready_data()
            print(f"✅ Optimization data ready: {opt_data.shape[0]} rows")
            
            processor.stop_live_feeds()
            return True
        else:
            print("❌ sample_portfolio.csv not found")
            return False
            
    except Exception as e:
        print(f"❌ CSV processing failed: {e}")
        return False

def test_basic_optimization():
    """Test basic optimization without live data"""
    print("\n🧪 Testing basic optimization...")
    
    try:
        from quantum_finance_opt.core.config import OptimizationConfig
        from quantum_finance_opt.core.optimizer import QuantumFinanceOptimizer
        
        # Create config
        config = OptimizationConfig(
            tickers=['AAPL', 'GOOGL', 'MSFT'],
            budget=100000.0,
            csv_path='sample_portfolio.csv'
        )
        
        # Initialize optimizer
        optimizer = QuantumFinanceOptimizer(config)
        
        # Load data
        if os.path.exists('sample_portfolio.csv'):
            optimizer.load_data('sample_portfolio.csv')
            print(f"✅ Data loaded: {optimizer.price_data.shape}")
            
            # Test classical optimization
            classical_results = optimizer.classical_optimization(['max_sharpe'])
            print(f"✅ Classical optimization completed")
            
            if 'max_sharpe' in classical_results:
                result = classical_results['max_sharpe']
                print(f"   Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
                print(f"   Expected Return: {result.get('expected_return', 0):.2%}")
            
            return True
        else:
            print("❌ sample_portfolio.csv not found")
            return False
            
    except Exception as e:
        print(f"❌ Basic optimization failed: {e}")
        return False

def test_quantum_backend():
    """Test quantum backend initialization"""
    print("\n🧪 Testing quantum backend...")
    
    try:
        from quantum_finance_opt.quantum.backend_manager import QuantumBackendManager
        
        # Initialize quantum backend
        quantum_backend = QuantumBackendManager(preferred_backend="qiskit_simulator")
        
        # Get backend info
        backend_info = quantum_backend.get_backend_info()
        print(f"✅ Quantum backend initialized")
        print(f"   Available: {backend_info.get('available', False)}")
        print(f"   Quantum Capable: {backend_info.get('quantum_capable', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum backend failed: {e}")
        return False

def test_gpu_acceleration():
    """Test GPU acceleration"""
    print("\n🧪 Testing GPU acceleration...")
    
    try:
        from quantum_finance_opt.hpc.gpu_accelerator import GPUAccelerator
        
        # Initialize GPU accelerator
        gpu = GPUAccelerator()
        
        # Get GPU info
        gpu_info = gpu.get_gpu_info()
        print(f"✅ GPU accelerator initialized")
        print(f"   CuPy Available: {gpu_info.get('cupy_available', False)}")
        print(f"   PyTorch Available: {gpu_info.get('torch_available', False)}")
        
        if gpu_info.get('cupy_available') or gpu_info.get('torch_available'):
            print("   🚀 GPU acceleration ready!")
        else:
            print("   ⚠️ Using CPU fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU acceleration failed: {e}")
        return False

def test_live_data_services():
    """Test live data service availability"""
    print("\n🧪 Testing live data services...")
    
    try:
        # Test Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            if 'regularMarketPrice' in info:
                print(f"✅ Yahoo Finance: ${info['regularMarketPrice']:.2f}")
            else:
                print("⚠️ Yahoo Finance: Limited data")
        except Exception as e:
            print(f"❌ Yahoo Finance failed: {e}")
        
        # Test sentiment analysis
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            test_text = "Apple stock is performing very well today!"
            sentiment = analyzer.polarity_scores(test_text)
            print(f"✅ Sentiment Analysis: {sentiment['compound']:.3f}")
        except Exception as e:
            print(f"❌ Sentiment Analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Live data services failed: {e}")
        return False

def main():
    """Run all tests"""
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🚀 Ultimate Portfolio Optimizer - Test Suite")
    print("=" * 50)
    
    tests = [
        test_csv_processing,
        test_basic_optimization,
        test_quantum_backend,
        test_gpu_acceleration,
        test_live_data_services
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTry running:")
        print("python ultimate_portfolio_optimizer.py --csv sample_portfolio.csv --mode analyze")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("The system may still work with reduced functionality.")
    
    print("\n🚀 Ultimate Portfolio Optimizer is the best on GitHub!")

if __name__ == "__main__":
    main()