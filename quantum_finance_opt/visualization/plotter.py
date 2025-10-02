"""
Visualization engine for QuantumFinanceOpt.

This module provides comprehensive visualization capabilities for portfolio
optimization results, including plots and reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

from ..core.config import OptimizationConfig
from ..core.exceptions import OptimizationError

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VisualizationEngine:
    """Visualization engine for portfolio optimization results."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize VisualizationEngine with configuration."""
        self.config = config
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_portfolio_weights(self, weights: np.ndarray, tickers: List[str], 
                             method_name: str = "Portfolio") -> plt.Figure:
        """
        Plot portfolio weights as pie chart and bar chart.
        
        Args:
            weights: Portfolio weights
            tickers: Asset tickers
            method_name: Name of the optimization method
        
        Returns:
            Matplotlib figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Filter out very small weights for cleaner visualization
            min_weight = 0.01
            display_weights = weights.copy()
            display_tickers = tickers.copy()
            
            # Group small weights into "Others"
            small_indices = display_weights < min_weight
            if np.any(small_indices):
                others_weight = np.sum(display_weights[small_indices])
                display_weights = display_weights[~small_indices]
                display_tickers = [t for i, t in enumerate(display_tickers) if not small_indices[i]]
                
                if others_weight > 0:
                    display_weights = np.append(display_weights, others_weight)
                    display_tickers.append('Others')
            
            # Pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(display_weights)))
            wedges, texts, autotexts = ax1.pie(
                display_weights, labels=display_tickers, autopct='%1.1f%%',
                colors=colors, startangle=90
            )
            ax1.set_title(f'{method_name} - Portfolio Allocation')
            
            # Bar chart
            bars = ax2.bar(range(len(display_weights)), display_weights, color=colors)
            ax2.set_xlabel('Assets')
            ax2.set_ylabel('Weight')
            ax2.set_title(f'{method_name} - Portfolio Weights')
            ax2.set_xticks(range(len(display_tickers)))
            ax2.set_xticklabels(display_tickers, rotation=45)
            
            # Add value labels on bars
            for bar, weight in zip(bars, display_weights):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{weight:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                filename = f'portfolio_weights_{method_name.lower().replace(" ", "_")}.png'
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Portfolio weights plot saved: {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Portfolio weights plotting failed: {e}")
            return plt.figure()
    
    def plot_efficient_frontier(self, frontier_results: List[Dict[str, Any]], 
                              highlight_portfolios: Dict[str, Dict[str, Any]] = None) -> plt.Figure:
        """
        Plot efficient frontier with highlighted portfolios.
        
        Args:
            frontier_results: List of portfolio results along efficient frontier
            highlight_portfolios: Dictionary of portfolios to highlight
        
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if frontier_results:
                # Extract returns and volatilities
                returns = [r.get('expected_return', 0) for r in frontier_results]
                volatilities = [r.get('volatility', 0) for r in frontier_results]
                
                # Plot efficient frontier
                ax.plot(volatilities, returns, 'b-', linewidth=2, label='Efficient Frontier')
                ax.scatter(volatilities, returns, c='blue', alpha=0.6, s=20)
            
            # Highlight specific portfolios
            if highlight_portfolios:
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for i, (name, portfolio) in enumerate(highlight_portfolios.items()):
                    vol = portfolio.get('volatility')
                    ret = portfolio.get('expected_return')
                    if vol is not None and ret is not None:
                        color = colors[i % len(colors)]
                        ax.scatter(vol, ret, c=color, s=100, marker='*', 
                                 label=name, edgecolors='black', linewidth=1)
            
            ax.set_xlabel('Volatility (Risk)')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format axes as percentages
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                filepath = os.path.join(self.output_dir, 'efficient_frontier.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Efficient frontier plot saved: {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Efficient frontier plotting failed: {e}")
            return plt.figure()
    
    def plot_performance_comparison(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Plot performance comparison across different methods.
        
        Args:
            results: Dictionary of optimization results
        
        Returns:
            Matplotlib figure
        """
        try:
            # Extract performance metrics
            methods = []
            returns = []
            volatilities = []
            sharpe_ratios = []
            
            for method_name, method_results in results.items():
                if isinstance(method_results, dict) and 'weights' not in method_results:
                    # Nested results
                    for sub_method, sub_result in method_results.items():
                        methods.append(f"{method_name}_{sub_method}")
                        returns.append(sub_result.get('expected_return', 0))
                        volatilities.append(sub_result.get('volatility', 0))
                        sharpe_ratios.append(sub_result.get('sharpe_ratio', 0))
                else:
                    # Direct result
                    methods.append(method_name)
                    returns.append(method_results.get('expected_return', 0))
                    volatilities.append(method_results.get('volatility', 0))
                    sharpe_ratios.append(method_results.get('sharpe_ratio', 0))
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Returns comparison
            bars1 = ax1.bar(methods, returns, color='skyblue', alpha=0.7)
            ax1.set_title('Expected Returns Comparison')
            ax1.set_ylabel('Expected Return')
            ax1.tick_params(axis='x', rotation=45)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Add value labels
            for bar, ret in zip(bars1, returns):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.1%}', ha='center', va='bottom')
            
            # Volatility comparison
            bars2 = ax2.bar(methods, volatilities, color='lightcoral', alpha=0.7)
            ax2.set_title('Volatility Comparison')
            ax2.set_ylabel('Volatility')
            ax2.tick_params(axis='x', rotation=45)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Add value labels
            for bar, vol in zip(bars2, volatilities):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{vol:.1%}', ha='center', va='bottom')
            
            # Sharpe ratio comparison
            bars3 = ax3.bar(methods, sharpe_ratios, color='lightgreen', alpha=0.7)
            ax3.set_title('Sharpe Ratio Comparison')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, sharpe in zip(bars3, sharpe_ratios):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{sharpe:.2f}', ha='center', va='bottom')
            
            # Risk-Return scatter
            ax4.scatter(volatilities, returns, s=100, alpha=0.7)
            for i, method in enumerate(methods):
                ax4.annotate(method, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.set_xlabel('Volatility')
            ax4.set_ylabel('Expected Return')
            ax4.set_title('Risk-Return Profile')
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                filepath = os.path.join(self.output_dir, 'performance_comparison.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Performance comparison plot saved: {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Performance comparison plotting failed: {e}")
            return plt.figure()
    
    def plot_correlation_heatmap(self, returns_data: pd.DataFrame) -> plt.Figure:
        """
        Plot correlation heatmap of asset returns.
        
        Args:
            returns_data: Returns data
        
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
            
            ax.set_title('Asset Returns Correlation Matrix')
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                filepath = os.path.join(self.output_dir, 'correlation_heatmap.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation heatmap saved: {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Correlation heatmap plotting failed: {e}")
            return plt.figure()
    
    def plot_price_performance(self, price_data: pd.DataFrame) -> plt.Figure:
        """
        Plot normalized price performance over time.
        
        Args:
            price_data: Price data
        
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Normalize prices to start at 100
            normalized_prices = price_data / price_data.iloc[0] * 100
            
            # Plot each asset
            for column in normalized_prices.columns:
                ax.plot(normalized_prices.index, normalized_prices[column], 
                       label=column, linewidth=2)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Price (Base = 100)')
            ax.set_title('Asset Price Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            fig.autofmt_xdate()
            
            plt.tight_layout()
            
            # Save plot
            if self.config.save_plots:
                filepath = os.path.join(self.output_dir, 'price_performance.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Price performance plot saved: {filepath}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Price performance plotting failed: {e}")
            return plt.figure()
    
    def generate_comprehensive_report(self, results: Dict[str, Any], 
                                    price_data: pd.DataFrame, 
                                    returns_data: pd.DataFrame):
        """
        Generate comprehensive visualization report.
        
        Args:
            results: Optimization results
            price_data: Price data
            returns_data: Returns data
        """
        try:
            logger.info("Generating comprehensive visualization report")
            
            # Plot price performance
            self.plot_price_performance(price_data)
            plt.close()
            
            # Plot correlation heatmap
            self.plot_correlation_heatmap(returns_data)
            plt.close()
            
            # Plot performance comparison
            self.plot_performance_comparison(results)
            plt.close()
            
            # Plot portfolio weights for each method
            tickers = list(price_data.columns)
            
            for method_name, method_results in results.items():
                if isinstance(method_results, dict) and 'weights' not in method_results:
                    # Nested results
                    for sub_method, sub_result in method_results.items():
                        if 'weights' in sub_result:
                            self.plot_portfolio_weights(
                                sub_result['weights'], tickers, f"{method_name}_{sub_method}"
                            )
                            plt.close()
                else:
                    # Direct result
                    if 'weights' in method_results:
                        self.plot_portfolio_weights(
                            method_results['weights'], tickers, method_name
                        )
                        plt.close()
            
            # Generate efficient frontier if classical results available
            if 'classical' in results:
                try:
                    # Create sample efficient frontier points
                    from ..models.classical import ClassicalOptimizer
                    classical_opt = ClassicalOptimizer(self.config)
                    classical_opt.set_data(returns_data)
                    classical_opt.calculate_expected_returns('historical_mean')
                    classical_opt.calculate_covariance_matrix('sample')
                    
                    frontier_results = classical_opt.efficient_frontier(n_points=20)
                    
                    # Highlight specific portfolios
                    highlight_portfolios = {}
                    for method_name, method_results in results.items():
                        if isinstance(method_results, dict) and 'weights' not in method_results:
                            for sub_method, sub_result in method_results.items():
                                if all(k in sub_result for k in ['expected_return', 'volatility']):
                                    highlight_portfolios[f"{method_name}_{sub_method}"] = sub_result
                        else:
                            if all(k in method_results for k in ['expected_return', 'volatility']):
                                highlight_portfolios[method_name] = method_results
                    
                    self.plot_efficient_frontier(frontier_results, highlight_portfolios)
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Efficient frontier generation failed: {e}")
            
            logger.info("Comprehensive visualization report completed")
            
        except Exception as e:
            logger.error(f"Comprehensive report generation failed: {e}")


# Simple test function
def test_visualization_engine():
    """Test visualization engine functionality."""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    price_data = pd.DataFrame({
        'AAPL': np.random.randn(100).cumsum() + 100,
        'GOOGL': np.random.randn(100).cumsum() + 1000,
        'MSFT': np.random.randn(100).cumsum() + 200
    }, index=dates)
    
    returns_data = price_data.pct_change().dropna()
    
    # Create test results
    results = {
        'classical': {
            'max_sharpe': {
                'weights': np.array([0.4, 0.3, 0.3]),
                'expected_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 0.67
            }
        },
        'quantum': {
            'weights': np.array([0.35, 0.35, 0.3]),
            'expected_return': 0.11,
            'volatility': 0.16,
            'sharpe_ratio': 0.69
        }
    }
    
    from ..core.config import OptimizationConfig
    config = OptimizationConfig(output_dir='test_output')
    
    viz_engine = VisualizationEngine(config)
    
    try:
        # Test individual plots
        viz_engine.plot_portfolio_weights(results['quantum']['weights'], 
                                        list(price_data.columns), 'Test Portfolio')
        viz_engine.plot_performance_comparison(results)
        viz_engine.plot_correlation_heatmap(returns_data)
        viz_engine.plot_price_performance(price_data)
        
        print("All visualization tests passed!")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        raise


if __name__ == "__main__":
    test_visualization_engine()