"""
Advanced classical portfolio optimization methods for QuantumFinanceOpt.

This module implements advanced techniques including Hierarchical Risk Parity,
Black-Litterman model, and other sophisticated portfolio construction methods.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import warnings

from ..core.exceptions import OptimizationError, ModelTrainingError
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class AdvancedClassicalOptimizer:
    """Advanced classical portfolio optimization methods."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize AdvancedClassicalOptimizer with configuration."""
        self.config = config
        self.returns_data = None
        self.covariance_matrix = None
        self.expected_returns = None
        
    def set_data(self, returns: pd.DataFrame, covariance_matrix: np.ndarray = None,
                expected_returns: np.ndarray = None):
        """Set data for optimization."""
        try:
            self.returns_data = returns.copy()
            if covariance_matrix is not None:
                self.covariance_matrix = covariance_matrix
            if expected_returns is not None:
                self.expected_returns = expected_returns
            logger.info(f"Set data with shape: {returns.shape}")
        except Exception as e:
            raise OptimizationError(f"Failed to set data: {e}")
    
    def hierarchical_risk_parity(self, method: str = 'single') -> Dict[str, Union[np.ndarray, float]]:
        """
        Implement Hierarchical Risk Parity (HRP) portfolio optimization.
        
        Args:
            method: Linkage method for hierarchical clustering
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info(f"Performing HRP optimization with {method} linkage")
            
            if self.returns_data is None:
                raise OptimizationError("Returns data not set")
            
            # Calculate correlation matrix
            corr_matrix = self.returns_data.corr()
            
            # Convert correlation to distance matrix
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method=method)
            
            # Get sorted indices from clustering
            sorted_indices = self._get_quasi_diag(linkage_matrix)
            
            # Calculate HRP weights
            hrp_weights = self._get_hrp_weights(sorted_indices, corr_matrix)
            
            # Calculate portfolio metrics
            if self.covariance_matrix is not None:
                portfolio_volatility = float(np.sqrt(hrp_weights.T @ self.covariance_matrix @ hrp_weights))
            else:
                cov_matrix = self.returns_data.cov().values * 252
                portfolio_volatility = float(np.sqrt(hrp_weights.T @ cov_matrix @ hrp_weights))
            
            portfolio_return = None
            sharpe_ratio = None
            if self.expected_returns is not None:
                portfolio_return = float(self.expected_returns.T @ hrp_weights)
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            
            results = {
                'weights': hrp_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'clustering_order': sorted_indices,
                'method': 'HRP'
            }
            
            logger.info(f"HRP optimization completed. Volatility: {portfolio_volatility:.4f}")
            return results
            
        except Exception as e:
            raise OptimizationError(f"HRP optimization failed: {e}")
    
    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from hierarchical clustering."""
        try:
            n_assets = linkage_matrix.shape[0] + 1
            
            # Get the order of assets from the dendrogram
            def get_cluster_order(node_id, n_assets):
                if node_id < n_assets:
                    return [node_id]
                else:
                    left_child = int(linkage_matrix[node_id - n_assets, 0])
                    right_child = int(linkage_matrix[node_id - n_assets, 1])
                    return (get_cluster_order(left_child, n_assets) + 
                           get_cluster_order(right_child, n_assets))
            
            # Start from the root node
            root_node = 2 * n_assets - 2
            sorted_indices = get_cluster_order(root_node, n_assets)
            
            return sorted_indices
            
        except Exception as e:
            logger.warning(f"Failed to get quasi-diagonal order: {e}")
            return list(range(linkage_matrix.shape[0] + 1))
    
    def _get_hrp_weights(self, sorted_indices: List[int], corr_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate HRP weights using recursive bisection."""
        try:
            n_assets = len(sorted_indices)
            weights = pd.Series(1.0, index=sorted_indices)
            
            # Recursive function to calculate weights
            def _get_cluster_var(indices):
                """Calculate cluster variance."""
                if len(indices) == 1:
                    return self.returns_data.iloc[:, indices].var().iloc[0]
                else:
                    sub_corr = corr_matrix.iloc[indices, indices]
                    sub_weights = np.ones(len(indices)) / len(indices)
                    cluster_var = sub_weights.T @ sub_corr.values @ sub_weights
                    return cluster_var
            
            def _get_rec_bipart(indices):
                """Recursive bisection to calculate weights."""
                if len(indices) == 1:
                    return
                
                # Split the cluster into two parts
                mid = len(indices) // 2
                left_indices = indices[:mid]
                right_indices = indices[mid:]
                
                # Calculate cluster variances
                left_var = _get_cluster_var(left_indices)
                right_var = _get_cluster_var(right_indices)
                
                # Calculate allocation weights
                total_var = left_var + right_var
                if total_var > 0:
                    left_weight = right_var / total_var
                    right_weight = left_var / total_var
                else:
                    left_weight = 0.5
                    right_weight = 0.5
                
                # Update weights
                weights[left_indices] *= left_weight
                weights[right_indices] *= right_weight
                
                # Recurse on sub-clusters
                _get_rec_bipart(left_indices)
                _get_rec_bipart(right_indices)
            
            # Start recursive bisection
            _get_rec_bipart(sorted_indices)
            
            # Convert to numpy array in original order
            final_weights = np.zeros(n_assets)
            for i, idx in enumerate(sorted_indices):
                final_weights[idx] = weights.iloc[i]
            
            return final_weights
            
        except Exception as e:
            logger.warning(f"HRP weight calculation failed: {e}")
            return np.ones(len(sorted_indices)) / len(sorted_indices)
    
    def black_litterman(self, views: Dict[str, float], view_confidences: Dict[str, float] = None,
                       tau: float = 0.025, market_cap_weights: np.ndarray = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Implement Black-Litterman model.
        
        Args:
            views: Dictionary of asset views {asset_name: expected_return}
            view_confidences: Dictionary of view confidences {asset_name: confidence}
            tau: Scaling factor for uncertainty of prior
            market_cap_weights: Market capitalization weights (if None, use equal weights)
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing Black-Litterman optimization")
            
            if self.returns_data is None:
                raise OptimizationError("Returns data not set")
            
            n_assets = len(self.returns_data.columns)
            asset_names = list(self.returns_data.columns)
            
            # Calculate covariance matrix if not provided
            if self.covariance_matrix is None:
                cov_matrix = self.returns_data.cov().values * 252
            else:
                cov_matrix = self.covariance_matrix
            
            # Market cap weights (use equal weights if not provided)
            if market_cap_weights is None:
                w_market = np.ones(n_assets) / n_assets
            else:
                w_market = market_cap_weights
            
            # Implied equilibrium returns (reverse optimization)
            risk_aversion = 3.0  # Typical risk aversion parameter
            pi = risk_aversion * cov_matrix @ w_market
            
            # Set up views
            view_assets = [asset for asset in views.keys() if asset in asset_names]
            if not view_assets:
                raise OptimizationError("No valid views provided")
            
            k = len(view_assets)  # Number of views
            
            # Picking matrix P (which assets the views relate to)
            P = np.zeros((k, n_assets))
            Q = np.zeros(k)  # View returns
            
            for i, asset in enumerate(view_assets):
                asset_idx = asset_names.index(asset)
                P[i, asset_idx] = 1.0
                Q[i] = views[asset]
            
            # Uncertainty matrix Omega
            if view_confidences is None:
                # Default: diagonal matrix with tau * P * Sigma * P'
                Omega = tau * P @ cov_matrix @ P.T
            else:
                Omega = np.zeros((k, k))
                for i, asset in enumerate(view_assets):
                    confidence = view_confidences.get(asset, 1.0)
                    Omega[i, i] = tau * cov_matrix[asset_names.index(asset), asset_names.index(asset)] / confidence
            
            # Black-Litterman formula
            tau_sigma = tau * cov_matrix
            
            # New expected returns
            M1 = np.linalg.inv(tau_sigma)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            M3 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
            
            mu_bl = np.linalg.inv(M1 + M2) @ M3
            
            # New covariance matrix
            sigma_bl = np.linalg.inv(M1 + M2)
            
            # Optimize portfolio using Black-Litterman inputs
            from .classical import ClassicalOptimizer
            
            classical_opt = ClassicalOptimizer(self.config)
            classical_opt.expected_returns = mu_bl
            classical_opt.covariance_matrix = sigma_bl
            
            # Use mean-variance optimization with Black-Litterman inputs
            bl_result = classical_opt.mean_variance_optimization()
            
            bl_result['method'] = 'Black-Litterman'
            bl_result['views'] = views
            bl_result['implied_returns'] = pi
            bl_result['bl_returns'] = mu_bl
            
            logger.info("Black-Litterman optimization completed")
            return bl_result
            
        except Exception as e:
            raise OptimizationError(f"Black-Litterman optimization failed: {e}")
    
    def hierarchical_equal_risk_contribution(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Implement Hierarchical Equal Risk Contribution (HERC).
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing HERC optimization")
            
            if self.returns_data is None:
                raise OptimizationError("Returns data not set")
            
            # First, get HRP clustering structure
            hrp_result = self.hierarchical_risk_parity()
            sorted_indices = hrp_result['clustering_order']
            
            # Calculate covariance matrix
            if self.covariance_matrix is None:
                cov_matrix = self.returns_data.cov().values * 252
            else:
                cov_matrix = self.covariance_matrix
            
            # Calculate HERC weights using risk contribution approach
            herc_weights = self._calculate_herc_weights(sorted_indices, cov_matrix)
            
            # Calculate portfolio metrics
            portfolio_volatility = float(np.sqrt(herc_weights.T @ cov_matrix @ herc_weights))
            
            portfolio_return = None
            sharpe_ratio = None
            if self.expected_returns is not None:
                portfolio_return = float(self.expected_returns.T @ herc_weights)
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            
            results = {
                'weights': herc_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'clustering_order': sorted_indices,
                'method': 'HERC'
            }
            
            logger.info(f"HERC optimization completed. Volatility: {portfolio_volatility:.4f}")
            return results
            
        except Exception as e:
            raise OptimizationError(f"HERC optimization failed: {e}")
    
    def _calculate_herc_weights(self, sorted_indices: List[int], cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate HERC weights based on equal risk contribution."""
        try:
            n_assets = len(sorted_indices)
            
            # Initialize weights
            weights = np.ones(n_assets) / n_assets
            
            # Iterative algorithm to achieve equal risk contribution
            max_iterations = 100
            tolerance = 1e-6
            
            for iteration in range(max_iterations):
                # Calculate risk contributions
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                marginal_contrib = cov_matrix @ weights / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # Target risk contribution (equal for all assets)
                target_risk = portfolio_vol / n_assets
                
                # Update weights based on risk contribution difference
                risk_diff = risk_contrib - target_risk
                
                # Check convergence
                if np.max(np.abs(risk_diff)) < tolerance:
                    break
                
                # Update weights (simple gradient-based approach)
                learning_rate = 0.1
                weights = weights - learning_rate * risk_diff
                
                # Ensure weights are positive and sum to 1
                weights = np.maximum(weights, 1e-8)
                weights = weights / np.sum(weights)
            
            logger.info(f"HERC converged after {iteration + 1} iterations")
            return weights
            
        except Exception as e:
            logger.warning(f"HERC weight calculation failed: {e}")
            return np.ones(len(sorted_indices)) / len(sorted_indices)
    
    def critical_line_algorithm(self, target_returns: List[float] = None) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Implement Critical Line Algorithm for efficient frontier.
        
        Args:
            target_returns: List of target returns for efficient frontier
        
        Returns:
            List of portfolio results along the efficient frontier
        """
        try:
            logger.info("Performing Critical Line Algorithm optimization")
            
            if self.expected_returns is None or self.covariance_matrix is None:
                raise OptimizationError("Expected returns and covariance matrix must be set")
            
            n_assets = len(self.expected_returns)
            
            if target_returns is None:
                # Generate target returns automatically
                min_return = np.min(self.expected_returns)
                max_return = np.max(self.expected_returns)
                target_returns = np.linspace(min_return, max_return, 20)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    # Solve quadratic programming problem
                    # min 0.5 * w' * Sigma * w
                    # s.t. mu' * w = target_return
                    #      1' * w = 1
                    #      w >= 0
                    
                    # Set up matrices for QP solver
                    P = 2 * self.covariance_matrix  # Quadratic term
                    q = np.zeros(n_assets)  # Linear term
                    
                    # Equality constraints: Aw = b
                    A_eq = np.vstack([
                        self.expected_returns.reshape(1, -1),  # Return constraint
                        np.ones((1, n_assets))  # Budget constraint
                    ])
                    b_eq = np.array([target_return, 1.0])
                    
                    # Inequality constraints: Gw <= h (w >= 0 -> -w <= 0)
                    G = -np.eye(n_assets)
                    h = np.zeros(n_assets)
                    
                    # Solve using scipy
                    def objective(w):
                        return 0.5 * w.T @ self.covariance_matrix @ w
                    
                    def eq_constraint1(w):
                        return self.expected_returns.T @ w - target_return
                    
                    def eq_constraint2(w):
                        return np.sum(w) - 1.0
                    
                    constraints = [
                        {'type': 'eq', 'fun': eq_constraint1},
                        {'type': 'eq', 'fun': eq_constraint2}
                    ]
                    
                    bounds = [(0, 1) for _ in range(n_assets)]
                    x0 = np.ones(n_assets) / n_assets
                    
                    result = minimize(
                        objective,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': self.config.max_iterations}
                    )
                    
                    if result.success:
                        weights = result.x
                        portfolio_return = float(self.expected_returns.T @ weights)
                        portfolio_volatility = float(np.sqrt(weights.T @ self.covariance_matrix @ weights))
                        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
                        
                        efficient_portfolios.append({
                            'weights': weights,
                            'expected_return': portfolio_return,
                            'volatility': portfolio_volatility,
                            'sharpe_ratio': sharpe_ratio,
                            'target_return': target_return,
                            'method': 'CLA'
                        })
                
                except:
                    # Skip infeasible points
                    continue
            
            logger.info(f"CLA generated {len(efficient_portfolios)} efficient portfolios")
            return efficient_portfolios
            
        except Exception as e:
            raise OptimizationError(f"Critical Line Algorithm failed: {e}")


# Unit tests for AdvancedClassicalOptimizer
def test_advanced_classical_optimizer():
    """Unit tests for AdvancedClassicalOptimizer functionality."""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns_data = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 252),
        'GOOGL': np.random.normal(0.0008, 0.025, 252),
        'MSFT': np.random.normal(0.0012, 0.018, 252),
        'AMZN': np.random.normal(0.0009, 0.022, 252)
    }, index=dates)
    
    config = OptimizationConfig()
    optimizer = AdvancedClassicalOptimizer(config)
    
    # Set up data
    cov_matrix = returns_data.cov().values * 252
    expected_returns = returns_data.mean().values * 252
    
    try:
        # Test data setting
        optimizer.set_data(returns_data, cov_matrix, expected_returns)
        assert optimizer.returns_data is not None, "Returns data should be set"
        
        # Test HRP optimization
        hrp_result = optimizer.hierarchical_risk_parity()
        assert 'weights' in hrp_result, "HRP should return weights"
        assert abs(np.sum(hrp_result['weights']) - 1.0) < 1e-6, "HRP weights should sum to 1"
        
        # Test HERC optimization
        herc_result = optimizer.hierarchical_equal_risk_contribution()
        assert 'weights' in herc_result, "HERC should return weights"
        assert abs(np.sum(herc_result['weights']) - 1.0) < 1e-6, "HERC weights should sum to 1"
        
        # Test Black-Litterman
        views = {'AAPL': 0.15, 'GOOGL': 0.12}
        bl_result = optimizer.black_litterman(views)
        assert 'weights' in bl_result, "Black-Litterman should return weights"
        assert abs(np.sum(bl_result['weights']) - 1.0) < 1e-6, "BL weights should sum to 1"
        
        # Test Critical Line Algorithm
        cla_results = optimizer.critical_line_algorithm()
        assert len(cla_results) > 0, "CLA should return efficient portfolios"
        
        print("All AdvancedClassicalOptimizer tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_advanced_classical_optimizer()