#!/usr/bin/env python3
"""
QuantumFinanceOpt - Advanced Portfolio Optimization Tool

Main entry point for the QuantumFinanceOpt portfolio optimization system.
Provides command-line interface for running various optimization methods.
"""

import sys
import os
import argparse
import logging
from typing import List, Dict, Any
import traceback

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_finance_opt.core.config import OptimizationConfig
from quantum_finance_opt.core.optimizer import QuantumFinanceOptimizer
from quantum_finance_opt.core.exceptions import QuantumFinanceOptError
from quantum_finance_opt.visualization.plotter import VisualizationEngine


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    
    parser = argparse.ArgumentParser(
        description='QuantumFinanceOpt - Advanced Portfolio Optimization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization with sample data
  python main.py --methods classical quantum
  
  # Use custom CSV data
  python main.py --csv-path data/prices.csv --tickers AAPL GOOGL MSFT
  
  # Full optimization with custom parameters
  python main.py --methods classical quantum ml ensemble --budget 100000 --esg-threshold 0.6
  
  # Generate sample data and save
  python main.py --generate-data --output-dir results
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        '--csv-path', type=str,
        help='Path to CSV file with historical price data'
    )
    data_group.add_argument(
        '--tickers', nargs='+', 
        default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        help='List of ticker symbols (default: AAPL GOOGL MSFT AMZN)'
    )
    data_group.add_argument(
        '--start-date', type=str, default='2020-01-01',
        help='Start date for data (YYYY-MM-DD, default: 2020-01-01)'
    )
    data_group.add_argument(
        '--end-date', type=str, default='2023-12-31',
        help='End date for data (YYYY-MM-DD, default: 2023-12-31)'
    )
    data_group.add_argument(
        '--generate-data', action='store_true',
        help='Generate sample data instead of loading from file'
    )
    
    # Portfolio arguments
    portfolio_group = parser.add_argument_group('Portfolio Options')
    portfolio_group.add_argument(
        '--budget', type=float, default=100000.0,
        help='Portfolio budget (default: 100000)'
    )
    portfolio_group.add_argument(
        '--esg-threshold', type=float, default=0.5,
        help='Minimum ESG score threshold (0-1, default: 0.5)'
    )
    portfolio_group.add_argument(
        '--risk-free-rate', type=float, default=0.02,
        help='Risk-free rate for calculations (default: 0.02)'
    )
    portfolio_group.add_argument(
        '--rebalance-frequency', type=str, default='monthly',
        choices=['daily', 'weekly', 'monthly', 'quarterly'],
        help='Portfolio rebalancing frequency (default: monthly)'
    )
    
    # Optimization arguments
    opt_group = parser.add_argument_group('Optimization Options')
    opt_group.add_argument(
        '--methods', nargs='+',
        default=['classical', 'quantum', 'ensemble'],
        choices=['classical', 'quantum', 'ml', 'ensemble'],
        help='Optimization methods to use (default: classical quantum ensemble)'
    )
    opt_group.add_argument(
        '--classical-methods', nargs='+',
        default=['mean_variance', 'max_sharpe', 'min_volatility', 'hrp'],
        choices=['mean_variance', 'max_sharpe', 'min_volatility', 'hrp', 'black_litterman'],
        help='Classical optimization methods (default: mean_variance max_sharpe min_volatility hrp)'
    )
    opt_group.add_argument(
        '--forecast-horizon', type=int, default=5,
        help='Forecast horizon for ML models (default: 5)'
    )
    opt_group.add_argument(
        '--monte-carlo-runs', type=int, default=1000,
        help='Number of Monte Carlo simulation runs (default: 1000)'
    )
    
    # System arguments
    system_group = parser.add_argument_group('System Options')
    system_group.add_argument(
        '--output-dir', type=str, default='output',
        help='Output directory for results (default: output)'
    )
    system_group.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    system_group.add_argument(
        '--random-seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    system_group.add_argument(
        '--n-jobs', type=int, default=-1,
        help='Number of parallel jobs (-1 for all cores, default: -1)'
    )
    system_group.add_argument(
        '--config', type=str,
        help='Path to YAML configuration file'
    )
    system_group.add_argument(
        '--save-plots', action='store_true', default=True,
        help='Save visualization plots (default: True)'
    )
    system_group.add_argument(
        '--no-plots', action='store_true',
        help='Disable plot generation and saving'
    )
    
    return parser


def load_configuration(args: argparse.Namespace) -> OptimizationConfig:
    """Load configuration from arguments and config file."""
    
    try:
        if args.config:
            # Load from YAML file and override with CLI arguments
            config = OptimizationConfig.from_yaml(args.config)
            
            # Override with CLI arguments
            for key, value in vars(args).items():
                if value is not None and key != 'config':
                    # Convert hyphens to underscores for attribute names
                    attr_name = key.replace('-', '_')
                    if hasattr(config, attr_name):
                        setattr(config, attr_name, value)
        else:
            # Create from CLI arguments
            config = OptimizationConfig(
                csv_path=args.csv_path,
                tickers=args.tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                budget=args.budget,
                esg_threshold=args.esg_threshold,
                risk_free_rate=args.risk_free_rate,
                rebalance_frequency=args.rebalance_frequency,
                optimization_methods=args.methods,
                forecast_horizon=args.forecast_horizon,
                monte_carlo_runs=args.monte_carlo_runs,
                output_dir=args.output_dir,
                log_level=args.log_level,
                random_seed=args.random_seed,
                n_jobs=args.n_jobs,
                save_plots=args.save_plots and not args.no_plots
            )
        
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def print_results_summary(results: Dict[str, Any], config: OptimizationConfig):
    """Print a summary of optimization results."""
    
    print("\n" + "="*80)
    print("QUANTUMFINANCEOPT - OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nPortfolio Configuration:")
    print(f"  Assets: {', '.join(config.tickers)}")
    print(f"  Budget: ${config.budget:,.2f}")
    print(f"  ESG Threshold: {config.esg_threshold:.2f}")
    print(f"  Risk-free Rate: {config.risk_free_rate:.2%}")
    
    print(f"\nOptimization Methods: {', '.join(config.optimization_methods)}")
    
    # Results table
    print(f"\n{'Method':<25} {'Return':<12} {'Volatility':<12} {'Sharpe':<10} {'Top Holdings'}")
    print("-" * 80)
    
    for method_name, method_results in results.items():
        if isinstance(method_results, dict) and 'weights' not in method_results:
            # Nested results (e.g., classical methods)
            for sub_method, sub_result in method_results.items():
                display_name = f"{method_name}_{sub_method}"
                _print_result_row(display_name, sub_result, config.tickers)
        else:
            # Direct result
            _print_result_row(method_name, method_results, config.tickers)
    
    print("-" * 80)


def _print_result_row(method_name: str, result: Dict[str, Any], tickers: List[str]):
    """Print a single result row."""
    
    expected_return = result.get('expected_return')
    volatility = result.get('volatility')
    sharpe_ratio = result.get('sharpe_ratio')
    weights = result.get('weights')
    
    # Format values
    return_str = f"{expected_return:.2%}" if expected_return is not None else "N/A"
    vol_str = f"{volatility:.2%}" if volatility is not None else "N/A"
    sharpe_str = f"{sharpe_ratio:.3f}" if sharpe_ratio is not None else "N/A"
    
    # Top holdings
    if weights is not None and len(weights) == len(tickers):
        top_indices = np.argsort(weights)[-2:][::-1]  # Top 2 holdings
        top_holdings = [f"{tickers[i]}({weights[i]:.1%})" for i in top_indices]
        holdings_str = ", ".join(top_holdings)
    else:
        holdings_str = "N/A"
    
    print(f"{method_name:<25} {return_str:<12} {vol_str:<12} {sharpe_str:<10} {holdings_str}")


def generate_visualizations(optimizer: QuantumFinanceOptimizer, results: Dict[str, Any]):
    """Generate and save visualization plots."""
    
    try:
        print("\nGenerating visualizations...")
        
        # Initialize visualization engine
        viz_engine = VisualizationEngine(optimizer.config)
        
        # Generate comprehensive report
        viz_engine.generate_comprehensive_report(results, optimizer.price_data, optimizer.returns_data)
        
        print(f"Visualizations saved to {optimizer.config.output_dir}")
        
    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")


def main():
    """Main entry point for QuantumFinanceOpt."""
    
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Load configuration
        config = load_configuration(args)
        
        print("QuantumFinanceOpt - Advanced Portfolio Optimization Tool")
        print("=" * 60)
        
        # Initialize optimizer
        print("Initializing optimizer...")
        optimizer = QuantumFinanceOptimizer(config)
        
        # Load or generate data
        if args.generate_data or not args.csv_path:
            print("Generating sample data...")
        else:
            print(f"Loading data from {args.csv_path}...")
        
        optimizer.load_data(args.csv_path)
        
        print(f"Data loaded: {optimizer.price_data.shape[0]} periods, {optimizer.price_data.shape[1]} assets")
        print(f"Date range: {optimizer.price_data.index.min().date()} to {optimizer.price_data.index.max().date()}")
        
        # Run optimization
        print(f"\nRunning optimization with methods: {', '.join(config.optimization_methods)}")
        results = optimizer.run_optimization(config.optimization_methods)
        
        # Print results summary
        print_results_summary(results, config)
        
        # Find and highlight best portfolio
        try:
            best_method, best_result = optimizer.get_best_portfolio('sharpe_ratio')
            print(f"\nBest Portfolio (by Sharpe ratio): {best_method}")
            print(f"  Expected Return: {best_result.get('expected_return', 0):.2%}")
            print(f"  Volatility: {best_result.get('volatility', 0):.2%}")
            print(f"  Sharpe Ratio: {best_result.get('sharpe_ratio', 0):.3f}")
            
            # Print portfolio weights
            weights = best_result.get('weights')
            if weights is not None:
                print(f"  Portfolio Weights:")
                for i, (ticker, weight) in enumerate(zip(config.tickers, weights)):
                    if weight > 0.01:  # Only show weights > 1%
                        print(f"    {ticker}: {weight:.1%}")
        
        except Exception as e:
            print(f"Could not determine best portfolio: {e}")
        
        # Save results
        print(f"\nSaving results to {config.output_dir}...")
        optimizer.save_results()
        
        # Generate visualizations
        if config.save_plots:
            generate_visualizations(optimizer, results)
        
        print("\nOptimization completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(1)
        
    except QuantumFinanceOptError as e:
        print(f"\nQuantumFinanceOpt Error: {e}")
        if hasattr(e, 'details') and e.details:
            print(f"Details: {e.details}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()