"""
Classical portfolio optimization methods for QuantumFinanceOpt.

This module implements traditional portfolio optimization techniques including
mean-variance optimization, CAPM, and various risk models.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.optimize import minimize
from scipy import linalg
import cvxpy as cp

from ..core.exceptions import OptimizationError, ModelTrainingError
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class ClassicalOptimizer:
    """Classical portfolio optimization methods."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize ClassicalOptimizer with configuration."""
        self.config = config
        self.returns_data = None
        self.expected_returns = None
        self.covariance_matrix = None
        
    def set_data(self, returns: pd.DataFrame):
        """Set returns data for optimization."""
        try:
            self.returns_data = returns.copy()
            logger.info(f"Set returns data with shape: {returns.shape}")
        except Exception as e:
            raise OptimizationError(f"Failed to set data: {e}")
    
    def calculate_expected_returns(self, method: str = 'historical_mean') -> np.ndarray:
        """
        Calculate expected returns using various methods.
        
        Args:
            method: Method for calculating expected returns
                   ('historical_mean', 'capm', 'exponential_weighted')
        
        Returns:
            Array of expected returns
        """
        try:
            if self.returns_data is None:
                raise OptimizationError("Returns data not set")
            
            logger.info(f"Calculating expected returns using {method}")
            
            if method == 'historical_mean':
                expected_returns = self.returns_data.mean().values
                
            elif method == 'capm':
                expected_returns = self._calculate_capm_returns()
                
            elif method == 'exponential_weighted':
                # Use exponentially weighted moving average
                span = min(60, len(self.returns_data) // 4)  # Adaptive span
                expected_returns = self.returns_data.ewm(span=span).mean().iloc[-1].values
                
            else:
                raise OptimizationError(f"Unknown expected returns method: {method}")
            
            # Annualize returns (assuming daily data)
            expected_returns = expected_returns * 252
            
            self.expected_returns = expected_returns
            logger.info(f"Expected returns calculated: {expected_returns}")
            
            return expected_returns
            
        except Exception as e:
            raise OptimizationError(f"Expected returns calculation failed: {e}")
    
    def _calculate_capm_returns(self) -> np.ndarray:
        """Calculate expected returns using CAPM."""
        try:
            # Use equal-weighted market proxy
            market_returns = self.returns_data.mean(axis=1)
            
            expected_returns = []
            
            for asset in self.returns_data.columns:
                asset_returns = self.returns_data[asset].dropna()
                
                # Calculate beta
                covariance = np.cov(asset_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 1.0
                
                # CAPM expected return
                market_premium = market_returns.mean() - self.config.risk_free_rate / 252
                expected_return = self.config.risk_free_rate / 252 + beta * market_premium
                
                expected_returns.append(expected_return)
            
            return np.array(expected_returns)
            
        except Exception as e:
            raise OptimizationError(f"CAPM calculation failed: {e}")
    
    def calculate_covariance_matrix(self, method: str = 'sample') -> np.ndarray:
        """
        Calculate covariance matrix using various methods.
        
        Args:
            method: Method for covariance estimation
                   ('sample', 'exponential', 'ledoit_wolf', 'semicovariance')
        
        Returns:
            Covariance matrix
        """
        try:
            if self.returns_data is None:
                raise OptimizationError("Returns data not set")
            
            logger.info(f"Calculating covariance matrix using {method}")
            
            if method == 'sample':
                cov_matrix = self.returns_data.cov().values
                
            elif method == 'exponential':
                # Exponentially weighted covariance
                span = min(60, len(self.returns_data) // 4)
                cov_matrix = self.returns_data.ewm(span=span).cov().iloc[-len(self.returns_data.columns):].values
                
            elif method == 'ledoit_wolf':
                cov_matrix = self._ledoit_wolf_shrinkage()
                
            elif method == 'semicovariance':
                cov_matrix = self._calculate_semicovariance()
                
            else:
                raise OptimizationError(f"Unknown covariance method: {method}")
            
            # Annualize covariance (assuming daily data)
            cov_matrix = cov_matrix * 252
            
            # Ensure positive semi-definite
            cov_matrix = self._ensure_positive_semidefinite(cov_matrix)
            
            self.covariance_matrix = cov_matrix
            logger.info(f"Covariance matrix calculated with shape: {cov_matrix.shape}")
            
            return cov_matrix
            
        except Exception as e:
            raise OptimizationError(f"Covariance matrix calculation failed: {e}")
    
    def _ledoit_wolf_shrinkage(self) -> np.ndarray:
        """Implement Ledoit-Wolf shrinkage estimator."""
        try:
            from sklearn.covariance import LedoitWolf
            
            lw = LedoitWolf()
            cov_matrix = lw.fit(self.returns_data.fillna(0)).covariance_
            
            return cov_matrix
            
        except ImportError:
            logger.warning("sklearn not available, using sample covariance")
            return self.returns_data.cov().values
        except Exception as e:
            logger.warning(f"Ledoit-Wolf shrinkage failed: {e}, using sample covariance")
            return self.returns_data.cov().values
    
    def _calculate_semicovariance(self) -> np.ndarray:
        """Calculate semicovariance matrix (downside risk)."""
        try:
            # Calculate mean returns
            mean_returns = self.returns_data.mean()
            
            # Get downside deviations
            downside_deviations = self.returns_data - mean_returns
            downside_deviations = downside_deviations.where(downside_deviations < 0, 0)
            
            # Calculate semicovariance
            semicov_matrix = downside_deviations.cov().values
            
            return semicov_matrix
            
        except Exception as e:
            logger.warning(f"Semicovariance calculation failed: {e}, using sample covariance")
            return self.returns_data.cov().values
    
    def _ensure_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive semi-definite."""
        try:
            # Check if already positive semi-definite
            eigenvals = np.linalg.eigvals(matrix)
            if np.all(eigenvals >= -1e-8):  # Allow small numerical errors
                return matrix
            
            # Fix using eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Set minimum eigenvalue
            
            fixed_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            logger.info("Fixed non-positive semi-definite covariance matrix")
            return fixed_matrix
            
        except Exception as e:
            logger.warning(f"Failed to fix covariance matrix: {e}")
            return matrix
    
    def mean_variance_optimization(self, target_return: Optional[float] = None,
                                 risk_aversion: float = 1.0) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform mean-variance optimization.
        
        Args:
            target_return: Target return for efficient frontier point
            risk_aversion: Risk aversion parameter for utility maximization
        
        Returns:
            Dictionary with optimization results
        """
        try:
            if self.expected_returns is None or self.covariance_matrix is None:
                raise OptimizationError("Expected returns and covariance matrix must be calculated first")
            
            n_assets = len(self.expected_returns)
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Define objective and constraints
            portfolio_return = self.expected_returns.T @ weights
            portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
            
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0  # Long-only constraint
            ]
            
            if target_return is not None:
                # Efficient frontier optimization
                objective = cp.Minimize(portfolio_variance)
                constraints.append(portfolio_return >= target_return)
                logger.info(f"Optimizing for target return: {target_return}")
            else:
                # Utility maximization
                objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
                logger.info(f"Optimizing utility with risk aversion: {risk_aversion}")
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = weights.value
                
                # Calculate portfolio metrics
                portfolio_return_val = float(self.expected_returns.T @ optimal_weights)
                portfolio_volatility = float(np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights))
                sharpe_ratio = (portfolio_return_val - self.config.risk_free_rate) / portfolio_volatility
                
                results = {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return_val,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'status': problem.status
                }
                
                logger.info(f"Optimization successful. Sharpe ratio: {sharpe_ratio:.4f}")
                return results
            else:
                raise OptimizationError(f"Optimization failed with status: {problem.status}")
                
        except Exception as e:
            raise OptimizationError(f"Mean-variance optimization failed: {e}")
    
    def max_sharpe_optimization(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find portfolio with maximum Sharpe ratio.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Optimizing for maximum Sharpe ratio")
            
            if self.expected_returns is None or self.covariance_matrix is None:
                raise OptimizationError("Expected returns and covariance matrix must be calculated first")
            
            n_assets = len(self.expected_returns)
            
            # Use scipy optimization for max Sharpe (more stable)
            def negative_sharpe(weights):
                portfolio_return = np.sum(self.expected_returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
                
                if portfolio_volatility == 0:
                    return -np.inf
                
                sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
                return -sharpe  # Negative because we minimize
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = float(np.sum(self.expected_returns * optimal_weights))
                portfolio_volatility = float(np.sqrt(np.dot(optimal_weights.T, 
                                                           np.dot(self.covariance_matrix, optimal_weights))))
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
                
                results = {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'status': 'optimal'
                }
                
                logger.info(f"Max Sharpe optimization successful. Sharpe ratio: {sharpe_ratio:.4f}")
                return results
            else:
                raise OptimizationError(f"Max Sharpe optimization failed: {result.message}")
                
        except Exception as e:
            raise OptimizationError(f"Max Sharpe optimization failed: {e}")
    
    def min_volatility_optimization(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find minimum volatility portfolio.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Optimizing for minimum volatility")
            
            if self.covariance_matrix is None:
                raise OptimizationError("Covariance matrix must be calculated first")
            
            n_assets = len(self.covariance_matrix)
            
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Objective: minimize portfolio variance
            portfolio_variance = cp.quad_form(weights, self.covariance_matrix)
            objective = cp.Minimize(portfolio_variance)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0  # Long-only constraint
            ]
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = weights.value
                
                # Calculate portfolio metrics
                portfolio_volatility = float(np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights))
                
                if self.expected_returns is not None:
                    portfolio_return = float(self.expected_returns.T @ optimal_weights)
                    sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
                else:
                    portfolio_return = None
                    sharpe_ratio = None
                
                results = {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'status': problem.status
                }
                
                logger.info(f"Min volatility optimization successful. Volatility: {portfolio_volatility:.4f}")
                return results
            else:
                raise OptimizationError(f"Min volatility optimization failed with status: {problem.status}")
                
        except Exception as e:
            raise OptimizationError(f"Min volatility optimization failed: {e}")
    
    def efficient_frontier(self, n_points: int = 50) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Generate efficient frontier.
        
        Args:
            n_points: Number of points on the efficient frontier
        
        Returns:
            List of portfolio results along the efficient frontier
        """
        try:
            logger.info(f"Generating efficient frontier with {n_points} points")
            
            if self.expected_returns is None or self.covariance_matrix is None:
                raise OptimizationError("Expected returns and covariance matrix must be calculated first")
            
            # Find min and max return portfolios
            min_vol_result = self.min_volatility_optimization()
            max_return = np.max(self.expected_returns)
            min_return = min_vol_result['expected_return']
            
            # Generate target returns
            target_returns = np.linspace(min_return, max_return * 0.95, n_points)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    result = self.mean_variance_optimization(target_return=target_return)
                    efficient_portfolios.append(result)
                except:
                    # Skip infeasible points
                    continue
            
            logger.info(f"Generated {len(efficient_portfolios)} efficient frontier points")
            return efficient_portfolios
            
        except Exception as e:
            raise OptimizationError(f"Efficient frontier generation failed: {e}")


# Unit tests for ClassicalOptimizer
def test_classical_optimizer():
    """Unit tests for ClassicalOptimizer functionality."""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns_data = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 252),
        'GOOGL': np.random.normal(0.0008, 0.025, 252),
        'MSFT': np.random.normal(0.0012, 0.018, 252)
    }, index=dates)
    
    config = OptimizationConfig()
    optimizer = ClassicalOptimizer(config)
    
    try:
        # Test data setting
        optimizer.set_data(returns_data)
        assert optimizer.returns_data is not None, "Returns data should be set"
        
        # Test expected returns calculation
        expected_returns = optimizer.calculate_expected_returns('historical_mean')
        assert len(expected_returns) == 3, "Should calculate returns for all assets"
        
        # Test covariance matrix calculation
        cov_matrix = optimizer.calculate_covariance_matrix('sample')
        assert cov_matrix.shape == (3, 3), "Covariance matrix should be 3x3"
        
        # Test mean-variance optimization
        mv_result = optimizer.mean_variance_optimization()
        assert 'weights' in mv_result, "Should return portfolio weights"
        assert abs(np.sum(mv_result['weights']) - 1.0) < 1e-6, "Weights should sum to 1"
        
        # Test max Sharpe optimization
        sharpe_result = optimizer.max_sharpe_optimization()
        assert 'sharpe_ratio' in sharpe_result, "Should return Sharpe ratio"
        
        # Test min volatility optimization
        min_vol_result = optimizer.min_volatility_optimization()
        assert 'volatility' in min_vol_result, "Should return volatility"
        
        # Test efficient frontier
        frontier = optimizer.efficient_frontier(n_points=10)
        assert len(frontier) > 0, "Should generate efficient frontier points"
        
        print("All ClassicalOptimizer tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_classical_optimizer()