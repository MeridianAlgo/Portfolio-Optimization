#!/usr/bin/env python3
"""
Example usage of QuantumFinanceOpt.

This script demonstrates various ways to use the QuantumFinanceOpt
portfolio optimization tool programmatically.
"""

import numpy as np
import pandas as pd
from quantum_finance_opt.core.config import OptimizationConfig
from quantum_finance_opt.core.optimizer import QuantumFinanceOptimizer


def example_basic_optimization():
    """Example 1: Basic portfolio optimization with sample data."""
    
    print("="*60)
    print("Example 1: Basic Portfolio Optimization")
    print("="*60)
    
    # Create configuration
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        start_date='2020-01-01',
        end_date='2022-12-31',
        optimization_methods=['classical'],
        output_dir='example_output_1'
    )
    
    # Initialize optimizer
    optimizer = QuantumFinanceOptimizer(config)
    
    # Load data (will generate sample data)
    print("Loading sample data...")
    optimizer.load_data()
    
    # Run classical optimization
    print("Running classical optimization...")
    results = optimizer.classical_optimization(['mean_variance', 'max_sharpe', 'min_volatility'])
    
    # Display results
    print("\nResults:")
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Expected Return: {result.get('expected_return', 0):.2%}")
        print(f"  Volatility: {result.get('volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
        
        weights = result.get('weights')
        if weights is not None:
            print(f"  Portfolio Weights:")
            for ticker, weight in zip(config.tickers, weights):
                if weight > 0.01:  # Only show weights > 1%
                    print(f"    {ticker}: {weight:.1%}")


def example_quantum_optimization():
    """Example 2: Quantum-inspired optimization."""
    
    print("\n" + "="*60)
    print("Example 2: Quantum-Inspired Optimization")
    print("="*60)
    
    # Create configuration
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT'],
        optimization_methods=['quantum'],
        output_dir='example_output_2'
    )
    
    # Initialize optimizer
    optimizer = QuantumFinanceOptimizer(config)
    
    # Load data
    print("Loading sample data...")
    optimizer.load_data()
    
    # Run quantum optimization
    print("Running quantum-inspired optimization...")
    result = optimizer.quantum_optimization()
    
    # Display results
    print("\nQuantum-Inspired Results:")
    print(f"  Expected Return: {result.get('expected_return', 0):.2%}")
    print(f"  Volatility: {result.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    
    # Display quantum metrics
    quantum_metrics = result.get('detailed_quantum_metrics', {})
    if quantum_metrics:
        print(f"\nQuantum Risk Metrics:")
        print(f"  Quantum Entropy: {quantum_metrics.get('quantum_entropy', 0):.4f}")
        print(f"  Quantum Coherence: {quantum_metrics.get('quantum_coherence', 0):.4f}")
        print(f"  Quantum Discord: {quantum_metrics.get('quantum_discord', 0):.4f}")
        print(f"  Entanglement Measure: {quantum_metrics.get('entanglement_measure', 0):.4f}")
        print(f"  Composite Risk Score: {quantum_metrics.get('quantum_risk_score', 0):.4f}")


def example_ml_optimization():
    """Example 3: Machine Learning optimization."""
    
    print("\n" + "="*60)
    print("Example 3: Machine Learning Optimization")
    print("="*60)
    
    # Create configuration
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        optimization_methods=['ml'],
        forecast_horizon=5,
        output_dir='example_output_3'
    )
    
    # Initialize optimizer
    optimizer = QuantumFinanceOptimizer(config)
    
    # Load data
    print("Loading sample data...")
    optimizer.load_data()
    
    # Run ML optimization
    print("Running ML-based optimization...")
    print("(This may take a few minutes for model training...)")
    result = optimizer.ml_optimization()
    
    # Display results
    print("\nML-Enhanced Results:")
    print(f"  Expected Return: {result.get('expected_return', 0):.2%}")
    print(f"  Volatility: {result.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    
    # Display model performance
    evaluation = result.get('prediction_evaluation', {})
    if evaluation:
        print(f"\nML Model Performance:")
        for asset, metrics in evaluation.items():
            print(f"  {asset}:")
            for model, scores in metrics.items():
                r2 = scores.get('r2', 0)
                directional_acc = scores.get('directional_accuracy', 0)
                print(f"    {model}: R² = {r2:.3f}, Directional Accuracy = {directional_acc:.1%}")


def example_ensemble_optimization():
    """Example 4: Ensemble optimization combining multiple methods."""
    
    print("\n" + "="*60)
    print("Example 4: Ensemble Optimization")
    print("="*60)
    
    # Create configuration
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT'],
        optimization_methods=['classical', 'quantum', 'ensemble'],
        output_dir='example_output_4'
    )
    
    # Initialize optimizer
    optimizer = QuantumFinanceOptimizer(config)
    
    # Load data
    print("Loading sample data...")
    optimizer.load_data()
    
    # Run ensemble optimization
    print("Running ensemble optimization...")
    results = optimizer.run_optimization()
    
    # Compare all methods
    print("\nMethod Comparison:")
    print(f"{'Method':<15} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
    print("-" * 50)
    
    for method_name, method_results in results.items():
        if isinstance(method_results, dict) and 'weights' not in method_results:
            # Classical methods (nested)
            for sub_method, sub_result in method_results.items():
                display_name = f"{method_name}_{sub_method}"[:14]
                ret = sub_result.get('expected_return', 0)
                vol = sub_result.get('volatility', 0)
                sharpe = sub_result.get('sharpe_ratio', 0)
                print(f"{display_name:<15} {ret:>9.1%} {vol:>11.1%} {sharpe:>7.3f}")
        else:
            # Direct methods
            ret = method_results.get('expected_return', 0)
            vol = method_results.get('volatility', 0)
            sharpe = method_results.get('sharpe_ratio', 0)
            print(f"{method_name:<15} {ret:>9.1%} {vol:>11.1%} {sharpe:>7.3f}")
    
    # Find and display best portfolio
    try:
        best_method, best_result = optimizer.get_best_portfolio('sharpe_ratio')
        print(f"\nBest Portfolio (by Sharpe ratio): {best_method}")
        print(f"  Expected Return: {best_result.get('expected_return', 0):.2%}")
        print(f"  Volatility: {best_result.get('volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {best_result.get('sharpe_ratio', 0):.3f}")
        
        weights = best_result.get('weights')
        if weights is not None:
            print(f"  Portfolio Weights:")
            for ticker, weight in zip(config.tickers, weights):
                if weight > 0.01:
                    print(f"    {ticker}: {weight:.1%}")
    
    except Exception as e:
        print(f"Could not determine best portfolio: {e}")


def example_custom_data():
    """Example 5: Using custom CSV data."""
    
    print("\n" + "="*60)
    print("Example 5: Custom Data Example")
    print("="*60)
    
    # Generate sample CSV data
    print("Creating sample CSV data...")
    
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    np.random.seed(42)
    
    # Generate correlated price data
    n_assets = 4
    tickers = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D']
    
    # Generate returns with some correlation
    returns = np.random.multivariate_normal(
        mean=[0.0008, 0.0006, 0.001, 0.0004],
        cov=[[0.0004, 0.0001, 0.0002, 0.0001],
             [0.0001, 0.0006, 0.0001, 0.0002],
             [0.0002, 0.0001, 0.0005, 0.0001],
             [0.0001, 0.0002, 0.0001, 0.0003]],
        size=len(dates)
    )
    
    # Convert to prices
    prices = pd.DataFrame(returns, index=dates, columns=tickers)
    prices = (1 + prices).cumprod() * 100  # Start at 100
    
    # Save to CSV
    csv_path = 'sample_data.csv'
    prices.to_csv(csv_path)
    print(f"Sample data saved to {csv_path}")
    
    # Create configuration
    config = OptimizationConfig(
        csv_path=csv_path,
        tickers=tickers,
        optimization_methods=['classical'],
        output_dir='example_output_5'
    )
    
    # Initialize optimizer
    optimizer = QuantumFinanceOptimizer(config)
    
    # Load custom data
    print("Loading custom CSV data...")
    optimizer.load_data(csv_path)
    
    print(f"Loaded data: {optimizer.price_data.shape[0]} periods, {optimizer.price_data.shape[1]} assets")
    print(f"Date range: {optimizer.price_data.index.min().date()} to {optimizer.price_data.index.max().date()}")
    
    # Run optimization
    print("Running optimization with custom data...")
    results = optimizer.classical_optimization(['max_sharpe', 'min_volatility'])
    
    # Display results
    print("\nResults with Custom Data:")
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Expected Return: {result.get('expected_return', 0):.2%}")
        print(f"  Volatility: {result.get('volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")


def main():
    """Run all examples."""
    
    print("QuantumFinanceOpt - Usage Examples")
    print("This script demonstrates various ways to use the optimization tool.")
    
    try:
        # Run examples
        example_basic_optimization()
        example_quantum_optimization()
        
        # Skip ML example by default (takes longer)
        print("\n" + "="*60)
        print("Skipping ML example (takes longer to run)")
        print("Uncomment the line below to run ML optimization example")
        print("="*60)
        # example_ml_optimization()
        
        example_ensemble_optimization()
        example_custom_data()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the example_output_* directories for detailed results and visualizations.")
        print("="*60)
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()