"""
Main optimizer class for QuantumFinanceOpt.

This module provides the central orchestrator that integrates all optimization
methods and provides a unified interface for portfolio optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
import os

from .config import OptimizationConfig, setup_logging
from .exceptions import OptimizationError, DataProcessingError, ModelTrainingError
from ..data.processor import DataProcessor
from ..data.simulator import DataSimulator
from ..models.classical import ClassicalOptimizer
from ..models.advanced_classical import AdvancedClassicalOptimizer
from ..models.quantum_risk import QuantumRiskModel
from ..models.ml_predictor import MLPredictor

logger = logging.getLogger(__name__)


class QuantumFinanceOptimizer:
    """Main optimizer class integrating all optimization methods."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize QuantumFinanceOptimizer with configuration."""
        self.config = config
        self.logger = setup_logging(config)
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.data_simulator = DataSimulator(config)
        self.classical_optimizer = ClassicalOptimizer(config)
        self.advanced_classical_optimizer = AdvancedClassicalOptimizer(config)
        self.quantum_risk_model = QuantumRiskModel(config)
        self.ml_predictor = MLPredictor(config)
        
        # Data storage
        self.price_data = None
        self.returns_data = None
        self.news_data = None
        self.esg_scores = None
        
        # Results storage
        self.optimization_results = {}
        
        logger.info("QuantumFinanceOptimizer initialized")
    
    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load or generate data for optimization.
        
        Args:
            csv_path: Path to CSV file (if None, generate sample data)
        
        Returns:
            Loaded price data
        """
        try:
            logger.info("Loading data for optimization")
            
            if csv_path and os.path.exists(csv_path):
                # Load real data
                self.price_data = self.data_processor.load_csv(csv_path)
                self.price_data = self.data_processor.preprocess_data(self.price_data)
                
                # Generate complementary data
                dates = self.price_data.index.strftime('%Y-%m-%d').tolist()
                self.news_data = self.data_simulator.simulate_news_data(
                    list(self.price_data.columns), dates
                )
                self.esg_scores = self.data_simulator.simulate_esg_scores(
                    list(self.price_data.columns)
                )
                
            else:
                # Generate sample data
                logger.info("Generating sample data")
                self.price_data, self.news_data, self.esg_scores = \
                    self.data_simulator.generate_complete_dataset(save_to_disk=True)
            
            # Calculate returns
            self.returns_data = self.data_processor.compute_returns(self.price_data)
            
            logger.info(f"Data loaded successfully. Shape: {self.price_data.shape}")
            logger.info(f"Date range: {self.price_data.index.min()} to {self.price_data.index.max()}")
            logger.info(f"Assets: {list(self.price_data.columns)}")
            
            return self.price_data
            
        except Exception as e:
            raise DataProcessingError(f"Data loading failed: {e}")
    
    def classical_optimization(self, methods: List[str] = None) -> Dict[str, Any]:
        """
        Perform classical portfolio optimization.
        
        Args:
            methods: List of methods to use
        
        Returns:
            Dictionary with optimization results
        """
        try:
            if self.returns_data is None:
                raise OptimizationError("Data not loaded")
            
            if methods is None:
                methods = ['mean_variance', 'max_sharpe', 'min_volatility', 'hrp']
            
            logger.info(f"Performing classical optimization with methods: {methods}")
            
            # Set up classical optimizer
            self.classical_optimizer.set_data(self.returns_data)
            expected_returns = self.classical_optimizer.calculate_expected_returns('historical_mean')
            covariance_matrix = self.classical_optimizer.calculate_covariance_matrix('sample')
            
            # Set up advanced classical optimizer
            self.advanced_classical_optimizer.set_data(
                self.returns_data, covariance_matrix, expected_returns
            )
            
            results = {}
            
            # Mean-variance optimization
            if 'mean_variance' in methods:
                try:
                    mv_result = self.classical_optimizer.mean_variance_optimization()
                    results['mean_variance'] = mv_result
                    logger.info(f"Mean-variance optimization completed. Sharpe: {mv_result.get('sharpe_ratio', 'N/A'):.4f}")
                except Exception as e:
                    logger.warning(f"Mean-variance optimization failed: {e}")
            
            # Maximum Sharpe ratio
            if 'max_sharpe' in methods:
                try:
                    sharpe_result = self.classical_optimizer.max_sharpe_optimization()
                    results['max_sharpe'] = sharpe_result
                    logger.info(f"Max Sharpe optimization completed. Sharpe: {sharpe_result.get('sharpe_ratio', 'N/A'):.4f}")
                except Exception as e:
                    logger.warning(f"Max Sharpe optimization failed: {e}")
            
            # Minimum volatility
            if 'min_volatility' in methods:
                try:
                    min_vol_result = self.classical_optimizer.min_volatility_optimization()
                    results['min_volatility'] = min_vol_result
                    logger.info(f"Min volatility optimization completed. Vol: {min_vol_result.get('volatility', 'N/A'):.4f}")
                except Exception as e:
                    logger.warning(f"Min volatility optimization failed: {e}")
            
            # Hierarchical Risk Parity
            if 'hrp' in methods:
                try:
                    hrp_result = self.advanced_classical_optimizer.hierarchical_risk_parity()
                    results['hrp'] = hrp_result
                    logger.info(f"HRP optimization completed. Vol: {hrp_result.get('volatility', 'N/A'):.4f}")
                except Exception as e:
                    logger.warning(f"HRP optimization failed: {e}")
            
            # Black-Litterman (with sample views)
            if 'black_litterman' in methods:
                try:
                    # Create sample views (top performing assets get positive views)
                    asset_returns = expected_returns
                    top_assets = np.argsort(asset_returns)[-2:]  # Top 2 assets
                    views = {}
                    for idx in top_assets:
                        asset_name = self.returns_data.columns[idx]
                        views[asset_name] = asset_returns[idx] * 1.2  # 20% higher expectation
                    
                    bl_result = self.advanced_classical_optimizer.black_litterman(views)
                    results['black_litterman'] = bl_result
                    logger.info(f"Black-Litterman optimization completed. Sharpe: {bl_result.get('sharpe_ratio', 'N/A'):.4f}")
                except Exception as e:
                    logger.warning(f"Black-Litterman optimization failed: {e}")
            
            self.optimization_results['classical'] = results
            logger.info("Classical optimization completed")
            
            return results
            
        except Exception as e:
            raise OptimizationError(f"Classical optimization failed: {e}")
    
    def quantum_optimization(self) -> Dict[str, Any]:
        """
        Perform quantum-inspired optimization.
        
        Returns:
            Dictionary with quantum optimization results
        """
        try:
            if self.returns_data is None:
                raise OptimizationError("Data not loaded")
            
            logger.info("Performing quantum-inspired optimization")
            
            # Set up quantum risk model
            self.quantum_risk_model.set_data(self.returns_data)
            
            # Calculate expected returns
            expected_returns = self.returns_data.mean().values * 252
            
            # Quantum risk-adjusted optimization
            quantum_result = self.quantum_risk_model.quantum_risk_adjusted_optimization(
                expected_returns, quantum_risk_weight=0.1
            )
            
            # Calculate comprehensive quantum metrics
            quantum_metrics = self.quantum_risk_model.calculate_quantum_risk_metrics(
                quantum_result['weights']
            )
            
            quantum_result['detailed_quantum_metrics'] = quantum_metrics
            
            self.optimization_results['quantum'] = quantum_result
            logger.info(f"Quantum optimization completed. Sharpe: {quantum_result.get('sharpe_ratio', 'N/A'):.4f}")
            
            return quantum_result
            
        except Exception as e:
            raise OptimizationError(f"Quantum optimization failed: {e}")
    
    def ml_optimization(self) -> Dict[str, Any]:
        """
        Perform ML-based optimization.
        
        Returns:
            Dictionary with ML optimization results
        """
        try:
            if self.returns_data is None:
                raise OptimizationError("Data not loaded")
            
            logger.info("Performing ML-based optimization")
            
            # Create features
            features = self.ml_predictor.create_features(self.returns_data)
            
            # Prepare training data
            X, y, feature_names = self.ml_predictor.prepare_training_data(
                features, self.returns_data, forecast_horizon=self.config.forecast_horizon
            )
            
            # Split data for training and testing
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train ensemble
            asset_names = list(self.returns_data.columns)
            training_results = self.ml_predictor.train_ensemble(X_train, y_train, asset_names)
            
            # Make predictions
            test_features = features.iloc[split_idx:split_idx+len(X_test)]
            predictions = self.ml_predictor.predict_returns(test_features, asset_names)
            
            # Use predictions as expected returns for optimization
            expected_returns = predictions.mean().values
            
            # Perform optimization with ML predictions
            self.classical_optimizer.set_data(self.returns_data)
            covariance_matrix = self.classical_optimizer.calculate_covariance_matrix('sample')
            self.classical_optimizer.expected_returns = expected_returns
            self.classical_optimizer.covariance_matrix = covariance_matrix
            
            ml_result = self.classical_optimizer.mean_variance_optimization()
            ml_result['method'] = 'ML-Enhanced'
            ml_result['predictions'] = predictions
            ml_result['training_scores'] = training_results['scores']
            
            # Evaluate predictions
            evaluation_results = self.ml_predictor.evaluate_models(X_test, y_test, asset_names)
            ml_result['prediction_evaluation'] = evaluation_results
            
            self.optimization_results['ml'] = ml_result
            logger.info(f"ML optimization completed. Sharpe: {ml_result.get('sharpe_ratio', 'N/A'):.4f}")
            
            return ml_result
            
        except Exception as e:
            raise OptimizationError(f"ML optimization failed: {e}")
    
    def ensemble_optimization(self, methods: List[str] = None, 
                            ensemble_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform ensemble optimization combining multiple methods.
        
        Args:
            methods: List of methods to ensemble
            ensemble_weights: Weights for combining methods
        
        Returns:
            Dictionary with ensemble optimization results
        """
        try:
            if methods is None:
                methods = ['classical', 'quantum', 'ml']
            
            logger.info(f"Performing ensemble optimization with methods: {methods}")
            
            # Run individual optimizations if not already done
            individual_results = {}
            
            if 'classical' in methods:
                if 'classical' not in self.optimization_results:
                    self.classical_optimization(['max_sharpe'])
                individual_results['classical'] = self.optimization_results['classical']['max_sharpe']
            
            if 'quantum' in methods:
                if 'quantum' not in self.optimization_results:
                    self.quantum_optimization()
                individual_results['quantum'] = self.optimization_results['quantum']
            
            if 'ml' in methods:
                if 'ml' not in self.optimization_results:
                    self.ml_optimization()
                individual_results['ml'] = self.optimization_results['ml']
            
            # Default ensemble weights
            if ensemble_weights is None:
                ensemble_weights = {method: 1.0/len(methods) for method in methods}
            
            # Normalize weights
            total_weight = sum(ensemble_weights.values())
            ensemble_weights = {k: v/total_weight for k, v in ensemble_weights.items()}
            
            # Combine portfolio weights
            n_assets = len(self.returns_data.columns)
            ensemble_weights_portfolio = np.zeros(n_assets)
            
            for method, weight in ensemble_weights.items():
                if method in individual_results:
                    method_weights = individual_results[method]['weights']
                    ensemble_weights_portfolio += weight * method_weights
            
            # Calculate ensemble portfolio metrics
            if self.classical_optimizer.expected_returns is not None and \
               self.classical_optimizer.covariance_matrix is not None:
                
                expected_returns = self.classical_optimizer.expected_returns
                covariance_matrix = self.classical_optimizer.covariance_matrix
                
                portfolio_return = float(expected_returns.T @ ensemble_weights_portfolio)
                portfolio_volatility = float(np.sqrt(
                    ensemble_weights_portfolio.T @ covariance_matrix @ ensemble_weights_portfolio
                ))
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            else:
                portfolio_return = None
                portfolio_volatility = None
                sharpe_ratio = None
            
            ensemble_result = {
                'weights': ensemble_weights_portfolio,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'method': 'Ensemble',
                'component_methods': methods,
                'ensemble_weights': ensemble_weights,
                'individual_results': individual_results
            }
            
            self.optimization_results['ensemble'] = ensemble_result
            logger.info(f"Ensemble optimization completed. Sharpe: {sharpe_ratio:.4f}" if sharpe_ratio else "Ensemble optimization completed")
            
            return ensemble_result
            
        except Exception as e:
            raise OptimizationError(f"Ensemble optimization failed: {e}")
    
    def run_optimization(self, methods: List[str] = None) -> Dict[str, Any]:
        """
        Run complete optimization workflow.
        
        Args:
            methods: List of optimization methods to run
        
        Returns:
            Dictionary with all optimization results
        """
        try:
            if methods is None:
                methods = self.config.optimization_methods
            
            logger.info(f"Running complete optimization workflow with methods: {methods}")
            
            # Load data if not already loaded
            if self.price_data is None:
                self.load_data(self.config.csv_path)
            
            results = {}
            
            # Run requested optimizations
            if 'classical' in methods:
                results['classical'] = self.classical_optimization()
            
            if 'quantum' in methods:
                results['quantum'] = self.quantum_optimization()
            
            if 'ml' in methods:
                results['ml'] = self.ml_optimization()
            
            if 'ensemble' in methods:
                ensemble_methods = [m for m in ['classical', 'quantum', 'ml'] if m in methods]
                if ensemble_methods:
                    results['ensemble'] = self.ensemble_optimization(ensemble_methods)
            
            # Store all results
            self.optimization_results.update(results)
            
            logger.info("Complete optimization workflow finished")
            return results
            
        except Exception as e:
            raise OptimizationError(f"Optimization workflow failed: {e}")
    
    def get_best_portfolio(self, metric: str = 'sharpe_ratio') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best portfolio based on specified metric.
        
        Args:
            metric: Metric to optimize for ('sharpe_ratio', 'volatility', 'expected_return')
        
        Returns:
            Tuple of (method_name, portfolio_result)
        """
        try:
            if not self.optimization_results:
                raise OptimizationError("No optimization results available")
            
            best_method = None
            best_result = None
            best_value = None
            
            for method_name, method_results in self.optimization_results.items():
                # Handle nested results (e.g., classical methods)
                if isinstance(method_results, dict) and 'weights' not in method_results:
                    for sub_method, sub_result in method_results.items():
                        if metric in sub_result and sub_result[metric] is not None:
                            value = sub_result[metric]
                            
                            if best_value is None or \
                               (metric == 'sharpe_ratio' and value > best_value) or \
                               (metric == 'volatility' and value < best_value) or \
                               (metric == 'expected_return' and value > best_value):
                                best_value = value
                                best_method = f"{method_name}_{sub_method}"
                                best_result = sub_result
                else:
                    # Direct result
                    if metric in method_results and method_results[metric] is not None:
                        value = method_results[metric]
                        
                        if best_value is None or \
                           (metric == 'sharpe_ratio' and value > best_value) or \
                           (metric == 'volatility' and value < best_value) or \
                           (metric == 'expected_return' and value > best_value):
                            best_value = value
                            best_method = method_name
                            best_result = method_results
            
            if best_method is None:
                raise OptimizationError(f"No results found for metric: {metric}")
            
            logger.info(f"Best portfolio by {metric}: {best_method} ({best_value:.4f})")
            return best_method, best_result
            
        except Exception as e:
            raise OptimizationError(f"Best portfolio selection failed: {e}")
    
    def save_results(self, output_path: str = None):
        """
        Save optimization results to files.
        
        Args:
            output_path: Path to save results (if None, use config output_dir)
        """
        try:
            if output_path is None:
                output_path = self.config.output_dir
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save results summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_path, f'optimization_results_{timestamp}.csv')
            
            # Create summary DataFrame
            summary_data = []
            
            for method_name, method_results in self.optimization_results.items():
                if isinstance(method_results, dict) and 'weights' not in method_results:
                    # Nested results
                    for sub_method, sub_result in method_results.items():
                        summary_data.append({
                            'method': f"{method_name}_{sub_method}",
                            'expected_return': sub_result.get('expected_return'),
                            'volatility': sub_result.get('volatility'),
                            'sharpe_ratio': sub_result.get('sharpe_ratio'),
                            'weights': str(sub_result.get('weights'))
                        })
                else:
                    # Direct result
                    summary_data.append({
                        'method': method_name,
                        'expected_return': method_results.get('expected_return'),
                        'volatility': method_results.get('volatility'),
                        'sharpe_ratio': method_results.get('sharpe_ratio'),
                        'weights': str(method_results.get('weights'))
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(results_file, index=False)
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Unit tests for QuantumFinanceOptimizer
def test_quantum_finance_optimizer():
    """Unit tests for QuantumFinanceOptimizer functionality."""
    
    config = OptimizationConfig(
        tickers=['AAPL', 'GOOGL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2020-06-30',
        optimization_methods=['classical'],
        random_seed=42
    )
    
    optimizer = QuantumFinanceOptimizer(config)
    
    try:
        # Test data loading (will generate sample data)
        price_data = optimizer.load_data()
        assert not price_data.empty, "Price data should not be empty"
        assert optimizer.returns_data is not None, "Returns data should be calculated"
        
        # Test classical optimization
        classical_results = optimizer.classical_optimization(['mean_variance', 'max_sharpe'])
        assert 'mean_variance' in classical_results, "Should include mean-variance results"
        assert 'max_sharpe' in classical_results, "Should include max Sharpe results"
        
        # Test quantum optimization
        quantum_results = optimizer.quantum_optimization()
        assert 'weights' in quantum_results, "Should return portfolio weights"
        assert 'quantum_metrics' in quantum_results, "Should include quantum metrics"
        
        # Test ML optimization
        ml_results = optimizer.ml_optimization()
        assert 'weights' in ml_results, "Should return portfolio weights"
        assert 'predictions' in ml_results, "Should include predictions"
        
        # Test ensemble optimization
        ensemble_results = optimizer.ensemble_optimization(['classical', 'quantum'])
        assert 'weights' in ensemble_results, "Should return ensemble weights"
        assert 'component_methods' in ensemble_results, "Should list component methods"
        
        # Test best portfolio selection
        best_method, best_result = optimizer.get_best_portfolio('sharpe_ratio')
        assert best_method is not None, "Should find best portfolio"
        assert 'weights' in best_result, "Best result should have weights"
        
        print("All QuantumFinanceOptimizer tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_quantum_finance_optimizer()