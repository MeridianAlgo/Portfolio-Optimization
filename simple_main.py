#!/usr/bin/env python3
"""
Simplified QuantumFinanceOpt - Portfolio Optimization Tool

A simplified version that works with basic dependencies only.
Demonstrates core functionality without requiring advanced optimization libraries.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Any

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_finance_opt.core.config import OptimizationConfig
from quantum_finance_opt.data.processor import DataProcessor
from quantum_finance_opt.data.simulator import DataSimulator


class SimpleOptimizer:
    """Simplified portfolio optimizer using basic methods only."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.data_simulator = DataSimulator(config)
        self.price_data = None
        self.returns_data = None
        
    def load_data(self, csv_path: str = None):
        """Load or generate data."""
        if csv_path and os.path.exists(csv_path):
            self.price_data = self.data_processor.load_csv(csv_path)
            self.price_data = self.data_processor.preprocess_data(self.price_data)
        else:
            print("Generating sample data...")
            self.price_data, _, _ = self.data_simulator.generate_complete_dataset(save_to_disk=True)
        
        self.returns_data = self.data_processor.compute_returns(self.price_data)
        return self.price_data
    
    def equal_weight_portfolio(self) -> Dict[str, Any]:
        """Create equal-weight portfolio."""
        n_assets = len(self.returns_data.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Calculate metrics
        expected_returns = self.returns_data.mean().values * 252
        cov_matrix = self.returns_data.cov().values * 252
        
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'Equal Weight'
        }
    
    def market_cap_weight_portfolio(self) -> Dict[str, Any]:
        """Create market-cap weighted portfolio (simulated)."""
        # Simulate market caps based on average prices
        avg_prices = self.price_data.mean()
        market_caps = avg_prices * np.random.uniform(1e6, 1e9, len(avg_prices))  # Simulate shares outstanding
        weights = market_caps / market_caps.sum()
        
        # Calculate metrics
        expected_returns = self.returns_data.mean().values * 252
        cov_matrix = self.returns_data.cov().values * 252
        
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'Market Cap Weight'
        }
    
    def inverse_volatility_portfolio(self) -> Dict[str, Any]:
        """Create inverse volatility weighted portfolio."""
        volatilities = self.returns_data.std().values * np.sqrt(252)
        inv_vol_weights = 1 / volatilities
        weights = inv_vol_weights / inv_vol_weights.sum()
        
        # Calculate metrics
        expected_returns = self.returns_data.mean().values * 252
        cov_matrix = self.returns_data.cov().values * 252
        
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'Inverse Volatility'
        }
    
    def momentum_portfolio(self, lookback_months: int = 6) -> Dict[str, Any]:
        """Create momentum-based portfolio."""
        lookback_days = lookback_months * 21  # Approximate trading days per month
        
        if len(self.returns_data) < lookback_days:
            lookback_days = len(self.returns_data) // 2
        
        # Calculate momentum scores (cumulative returns over lookback period)
        momentum_returns = self.returns_data.tail(lookback_days).sum()
        
        # Create weights based on momentum (positive momentum gets higher weight)
        momentum_scores = np.maximum(momentum_returns.values, 0)  # Only positive momentum
        
        if momentum_scores.sum() > 0:
            weights = momentum_scores / momentum_scores.sum()
        else:
            # If no positive momentum, use equal weights
            weights = np.ones(len(momentum_scores)) / len(momentum_scores)
        
        # Calculate metrics
        expected_returns = self.returns_data.mean().values * 252
        cov_matrix = self.returns_data.cov().values * 252
        
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'Momentum'
        }
    
    def run_all_methods(self) -> Dict[str, Any]:
        """Run all available optimization methods."""
        results = {}
        
        print("Running portfolio optimization methods...")
        
        methods = [
            ('equal_weight', self.equal_weight_portfolio),
            ('market_cap', self.market_cap_weight_portfolio),
            ('inverse_vol', self.inverse_volatility_portfolio),
            ('momentum', self.momentum_portfolio)
        ]
        
        for method_name, method_func in methods:
            try:
                result = method_func()
                results[method_name] = result
                print(f"✓ {result['method']} completed")
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any]):
        """Create simple visualizations."""
        if not results:
            return
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Performance comparison
        methods = list(results.keys())
        returns = [results[m]['expected_return'] for m in methods]
        volatilities = [results[m]['volatility'] for m in methods]
        sharpe_ratios = [results[m]['sharpe_ratio'] for m in methods]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns
        ax1.bar(methods, returns, color='skyblue', alpha=0.7)
        ax1.set_title('Expected Returns')
        ax1.set_ylabel('Annual Return')
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(returns):
            ax1.text(i, v + 0.001, f'{v:.1%}', ha='center', va='bottom')
        
        # Volatility
        ax2.bar(methods, volatilities, color='lightcoral', alpha=0.7)
        ax2.set_title('Volatility')
        ax2.set_ylabel('Annual Volatility')
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(volatilities):
            ax2.text(i, v + 0.001, f'{v:.1%}', ha='center', va='bottom')
        
        # Sharpe Ratio
        ax3.bar(methods, sharpe_ratios, color='lightgreen', alpha=0.7)
        ax3.set_title('Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(sharpe_ratios):
            ax3.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        # Risk-Return Scatter
        ax4.scatter(volatilities, returns, s=100, alpha=0.7)
        for i, method in enumerate(methods):
            ax4.annotate(method, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Expected Return')
        ax4.set_title('Risk-Return Profile')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.output_dir, 'simple_optimization_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to: {plot_path}")
        
        # Portfolio weights plot
        fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (method_name, result) in enumerate(results.items()):
            if i >= 4:
                break
            
            ax = axes[i]
            weights = result['weights']
            tickers = list(self.price_data.columns)
            
            # Filter small weights
            min_weight = 0.01
            display_weights = []
            display_tickers = []
            others_weight = 0
            
            for j, (ticker, weight) in enumerate(zip(tickers, weights)):
                if weight >= min_weight:
                    display_weights.append(weight)
                    display_tickers.append(ticker)
                else:
                    others_weight += weight
            
            if others_weight > 0:
                display_weights.append(others_weight)
                display_tickers.append('Others')
            
            # Pie chart
            ax.pie(display_weights, labels=display_tickers, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{result["method"]} Portfolio')
        
        plt.tight_layout()
        
        weights_path = os.path.join(self.config.output_dir, 'portfolio_weights.png')
        plt.savefig(weights_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio weights plot saved to: {weights_path}")
        
        plt.close('all')


def main():
    """Main function for simplified optimizer."""
    parser = argparse.ArgumentParser(description='Simplified QuantumFinanceOpt')
    parser.add_argument('--csv-path', type=str, help='Path to CSV file')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    parser.add_argument('--output-dir', type=str, default='simple_output')
    parser.add_argument('--budget', type=float, default=100000)
    
    args = parser.parse_args()
    
    print("Simplified QuantumFinanceOpt - Portfolio Optimization")
    print("=" * 60)
    
    # Create configuration
    config = OptimizationConfig(
        tickers=args.tickers,
        output_dir=args.output_dir,
        budget=args.budget
    )
    
    # Initialize optimizer
    optimizer = SimpleOptimizer(config)
    
    # Load data
    print("Loading data...")
    optimizer.load_data(args.csv_path)
    print(f"Data loaded: {optimizer.price_data.shape[0]} periods, {optimizer.price_data.shape[1]} assets")
    
    # Run optimization
    results = optimizer.run_all_methods()
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"{'Method':<20} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
    print("-" * 55)
    
    for method_name, result in results.items():
        ret = result['expected_return']
        vol = result['volatility']
        sharpe = result['sharpe_ratio']
        print(f"{result['method']:<20} {ret:>9.1%} {vol:>11.1%} {sharpe:>7.3f}")
    
    # Find best portfolio
    if results:
        best_method = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
        best_result = results[best_method]
        
        print(f"\nBest Portfolio (by Sharpe ratio): {best_result['method']}")
        print(f"  Expected Return: {best_result['expected_return']:.2%}")
        print(f"  Volatility: {best_result['volatility']:.2%}")
        print(f"  Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
        
        print(f"  Portfolio Weights:")
        for ticker, weight in zip(config.tickers, best_result['weights']):
            if weight > 0.01:
                print(f"    {ticker}: {weight:.1%}")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    optimizer.plot_results(results)
    
    print(f"\nOptimization completed! Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()