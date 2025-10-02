"""
Quantum-inspired risk modeling for QuantumFinanceOpt.

This module implements quantum-inspired risk measures using quantum information
theory concepts to enhance traditional portfolio risk assessment.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

try:
    import qutip as qt
    from qutip import Qobj, entropy_vn, concurrence
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("QuTiP not available. Quantum risk features will use classical approximations.")

from ..core.exceptions import OptimizationError, ModelTrainingError
from ..core.config import OptimizationConfig

logger = logging.getLogger(__name__)


class QuantumRiskModel:
    """Quantum-inspired risk modeling for portfolio optimization."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize QuantumRiskModel with configuration."""
        self.config = config
        self.covariance_matrix = None
        self.returns_data = None
        self.quantum_state = None
        
    def set_data(self, returns: pd.DataFrame, covariance_matrix: np.ndarray = None):
        """Set data for quantum risk modeling."""
        try:
            self.returns_data = returns.copy()
            if covariance_matrix is not None:
                self.covariance_matrix = covariance_matrix
            else:
                self.covariance_matrix = returns.cov().values * 252
            
            logger.info(f"Set data for quantum risk modeling. Shape: {returns.shape}")
        except Exception as e:
            raise OptimizationError(f"Failed to set data: {e}")
    
    def calculate_quantum_entropy(self, weights: np.ndarray = None) -> float:
        """
        Calculate quantum entropy as a risk measure.
        
        Args:
            weights: Portfolio weights (if None, use equal weights)
        
        Returns:
            Quantum entropy value
        """
        try:
            if self.covariance_matrix is None:
                raise OptimizationError("Covariance matrix not set")
            
            logger.info("Calculating quantum entropy risk measure")
            
            if weights is None:
                weights = np.ones(len(self.covariance_matrix)) / len(self.covariance_matrix)
            
            if QUTIP_AVAILABLE:
                return self._calculate_qutip_entropy(weights)
            else:
                return self._calculate_classical_entropy_approximation(weights)
                
        except Exception as e:
            raise OptimizationError(f"Quantum entropy calculation failed: {e}")
    
    def _calculate_qutip_entropy(self, weights: np.ndarray) -> float:
        """Calculate quantum entropy using QuTiP."""
        try:
            # Create quantum state from covariance matrix eigenvalues
            eigenvals, eigenvecs = np.linalg.eigh(self.covariance_matrix)
            
            # Ensure positive eigenvalues
            eigenvals = np.maximum(eigenvals, 1e-10)
            
            # Normalize eigenvalues to create probability distribution
            eigenvals_normalized = eigenvals / np.sum(eigenvals)
            
            # Weight the eigenvalues by portfolio weights
            weighted_eigenvals = np.zeros_like(eigenvals_normalized)
            for i, weight in enumerate(weights):
                if i < len(weighted_eigenvals):
                    weighted_eigenvals[i] = weight * eigenvals_normalized[i]
            
            # Renormalize
            if np.sum(weighted_eigenvals) > 0:
                weighted_eigenvals = weighted_eigenvals / np.sum(weighted_eigenvals)
            else:
                weighted_eigenvals = np.ones_like(weighted_eigenvals) / len(weighted_eigenvals)
            
            # Create density matrix
            n_dim = min(len(weighted_eigenvals), 8)  # Limit dimension for computational efficiency
            density_matrix = np.diag(weighted_eigenvals[:n_dim])
            
            # Create QuTiP quantum object
            rho = Qobj(density_matrix)
            
            # Calculate von Neumann entropy
            entropy = entropy_vn(rho, base=2)  # Use base 2 for bits
            
            logger.info(f"Quantum entropy calculated: {entropy:.6f}")
            return float(entropy)
            
        except Exception as e:
            logger.warning(f"QuTiP entropy calculation failed: {e}, using classical approximation")
            return self._calculate_classical_entropy_approximation(weights)
    
    def _calculate_classical_entropy_approximation(self, weights: np.ndarray) -> float:
        """Calculate classical approximation of quantum entropy."""
        try:
            # Use Shannon entropy of eigenvalue distribution as approximation
            eigenvals, _ = np.linalg.eigh(self.covariance_matrix)
            eigenvals = np.maximum(eigenvals, 1e-10)
            
            # Weight eigenvalues by portfolio weights
            weighted_eigenvals = np.zeros_like(eigenvals)
            for i, weight in enumerate(weights):
                if i < len(weighted_eigenvals):
                    weighted_eigenvals[i] = weight * eigenvals[i]
            
            # Normalize to probability distribution
            if np.sum(weighted_eigenvals) > 0:
                prob_dist = weighted_eigenvals / np.sum(weighted_eigenvals)
            else:
                prob_dist = np.ones_like(weighted_eigenvals) / len(weighted_eigenvals)
            
            # Calculate Shannon entropy
            entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
            
            logger.info(f"Classical entropy approximation calculated: {entropy:.6f}")
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Classical entropy approximation failed: {e}")
            return 0.0
    
    def calculate_quantum_coherence(self, weights: np.ndarray = None) -> float:
        """
        Calculate quantum coherence as a diversification measure.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Quantum coherence value
        """
        try:
            if self.covariance_matrix is None:
                raise OptimizationError("Covariance matrix not set")
            
            logger.info("Calculating quantum coherence measure")
            
            if weights is None:
                weights = np.ones(len(self.covariance_matrix)) / len(self.covariance_matrix)
            
            # Calculate correlation matrix
            std_devs = np.sqrt(np.diag(self.covariance_matrix))
            corr_matrix = self.covariance_matrix / np.outer(std_devs, std_devs)
            
            # Weight the correlation matrix
            weighted_corr = np.zeros_like(corr_matrix)
            for i in range(len(weights)):
                for j in range(len(weights)):
                    weighted_corr[i, j] = weights[i] * weights[j] * corr_matrix[i, j]
            
            # Calculate coherence as sum of off-diagonal elements
            coherence = np.sum(np.abs(weighted_corr)) - np.sum(np.abs(np.diag(weighted_corr)))
            
            logger.info(f"Quantum coherence calculated: {coherence:.6f}")
            return float(coherence)
            
        except Exception as e:
            raise OptimizationError(f"Quantum coherence calculation failed: {e}")
    
    def calculate_quantum_discord(self, weights: np.ndarray = None) -> float:
        """
        Calculate quantum discord as a measure of non-classical correlations.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Quantum discord value
        """
        try:
            if self.returns_data is None:
                raise OptimizationError("Returns data not set")
            
            logger.info("Calculating quantum discord measure")
            
            if weights is None:
                weights = np.ones(len(self.returns_data.columns)) / len(self.returns_data.columns)
            
            # Calculate mutual information between asset pairs
            discord_sum = 0.0
            n_assets = len(weights)
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    # Calculate mutual information between assets i and j
                    asset_i_returns = self.returns_data.iloc[:, i].values
                    asset_j_returns = self.returns_data.iloc[:, j].values
                    
                    # Discretize returns for mutual information calculation
                    bins = min(10, len(asset_i_returns) // 10)
                    
                    hist_i, _ = np.histogram(asset_i_returns, bins=bins, density=True)
                    hist_j, _ = np.histogram(asset_j_returns, bins=bins, density=True)
                    hist_ij, _, _ = np.histogram2d(asset_i_returns, asset_j_returns, bins=bins, density=True)
                    
                    # Calculate mutual information
                    mi = 0.0
                    for x in range(bins):
                        for y in range(bins):
                            if hist_ij[x, y] > 0 and hist_i[x] > 0 and hist_j[y] > 0:
                                mi += hist_ij[x, y] * np.log2(hist_ij[x, y] / (hist_i[x] * hist_j[y]))
                    
                    # Weight by portfolio weights
                    weighted_mi = weights[i] * weights[j] * mi
                    discord_sum += weighted_mi
            
            logger.info(f"Quantum discord calculated: {discord_sum:.6f}")
            return float(discord_sum)
            
        except Exception as e:
            logger.warning(f"Quantum discord calculation failed: {e}")
            return 0.0
    
    def calculate_entanglement_measure(self, weights: np.ndarray = None) -> float:
        """
        Calculate entanglement measure for portfolio assets.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Entanglement measure value
        """
        try:
            if self.covariance_matrix is None:
                raise OptimizationError("Covariance matrix not set")
            
            logger.info("Calculating entanglement measure")
            
            if weights is None:
                weights = np.ones(len(self.covariance_matrix)) / len(self.covariance_matrix)
            
            if QUTIP_AVAILABLE and len(weights) >= 2:
                return self._calculate_qutip_entanglement(weights)
            else:
                return self._calculate_classical_entanglement_approximation(weights)
                
        except Exception as e:
            raise OptimizationError(f"Entanglement measure calculation failed: {e}")
    
    def _calculate_qutip_entanglement(self, weights: np.ndarray) -> float:
        """Calculate entanglement using QuTiP."""
        try:
            # For simplicity, calculate pairwise entanglement for top weighted assets
            n_top_assets = min(4, len(weights))  # Limit for computational efficiency
            top_indices = np.argsort(weights)[-n_top_assets:]
            
            total_entanglement = 0.0
            pair_count = 0
            
            for i in range(len(top_indices)):
                for j in range(i + 1, len(top_indices)):
                    idx_i, idx_j = top_indices[i], top_indices[j]
                    
                    # Create two-qubit state based on correlation
                    corr = self.covariance_matrix[idx_i, idx_j] / np.sqrt(
                        self.covariance_matrix[idx_i, idx_i] * self.covariance_matrix[idx_j, idx_j]
                    )
                    
                    # Map correlation to entanglement parameter
                    theta = np.arccos(np.abs(corr)) if np.abs(corr) <= 1 else 0
                    
                    # Create entangled state
                    state = np.cos(theta/2) * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + \
                           np.sin(theta/2) * qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
                    
                    # Calculate concurrence (entanglement measure)
                    rho = state * state.dag()
                    entanglement = concurrence(rho)
                    
                    # Weight by portfolio weights
                    weighted_entanglement = weights[idx_i] * weights[idx_j] * entanglement
                    total_entanglement += weighted_entanglement
                    pair_count += 1
            
            if pair_count > 0:
                average_entanglement = total_entanglement / pair_count
            else:
                average_entanglement = 0.0
            
            logger.info(f"QuTiP entanglement calculated: {average_entanglement:.6f}")
            return float(average_entanglement)
            
        except Exception as e:
            logger.warning(f"QuTiP entanglement calculation failed: {e}, using classical approximation")
            return self._calculate_classical_entanglement_approximation(weights)
    
    def _calculate_classical_entanglement_approximation(self, weights: np.ndarray) -> float:
        """Calculate classical approximation of entanglement."""
        try:
            # Use correlation-based measure as approximation
            corr_matrix = self.covariance_matrix / np.sqrt(
                np.outer(np.diag(self.covariance_matrix), np.diag(self.covariance_matrix))
            )
            
            # Calculate weighted average of squared correlations
            entanglement = 0.0
            total_weight = 0.0
            
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    correlation_strength = corr_matrix[i, j] ** 2
                    weight_product = weights[i] * weights[j]
                    entanglement += weight_product * correlation_strength
                    total_weight += weight_product
            
            if total_weight > 0:
                entanglement = entanglement / total_weight
            
            logger.info(f"Classical entanglement approximation calculated: {entanglement:.6f}")
            return float(entanglement)
            
        except Exception as e:
            logger.error(f"Classical entanglement approximation failed: {e}")
            return 0.0
    
    def calculate_quantum_risk_metrics(self, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive quantum risk metrics.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Dictionary of quantum risk metrics
        """
        try:
            logger.info("Calculating comprehensive quantum risk metrics")
            
            if weights is None:
                weights = np.ones(len(self.covariance_matrix)) / len(self.covariance_matrix)
            
            metrics = {}
            
            # Calculate quantum entropy
            try:
                metrics['quantum_entropy'] = self.calculate_quantum_entropy(weights)
            except Exception as e:
                logger.warning(f"Quantum entropy calculation failed: {e}")
                metrics['quantum_entropy'] = 0.0
            
            # Calculate quantum coherence
            try:
                metrics['quantum_coherence'] = self.calculate_quantum_coherence(weights)
            except Exception as e:
                logger.warning(f"Quantum coherence calculation failed: {e}")
                metrics['quantum_coherence'] = 0.0
            
            # Calculate quantum discord
            try:
                metrics['quantum_discord'] = self.calculate_quantum_discord(weights)
            except Exception as e:
                logger.warning(f"Quantum discord calculation failed: {e}")
                metrics['quantum_discord'] = 0.0
            
            # Calculate entanglement measure
            try:
                metrics['entanglement_measure'] = self.calculate_entanglement_measure(weights)
            except Exception as e:
                logger.warning(f"Entanglement measure calculation failed: {e}")
                metrics['entanglement_measure'] = 0.0
            
            # Calculate composite quantum risk score
            metrics['quantum_risk_score'] = (
                0.4 * metrics['quantum_entropy'] +
                0.3 * metrics['quantum_coherence'] +
                0.2 * metrics['quantum_discord'] +
                0.1 * metrics['entanglement_measure']
            )
            
            logger.info(f"Quantum risk metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            raise OptimizationError(f"Quantum risk metrics calculation failed: {e}")
    
    def quantum_risk_adjusted_optimization(self, expected_returns: np.ndarray,
                                         quantum_risk_weight: float = 0.1) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform portfolio optimization with quantum risk adjustment.
        
        Args:
            expected_returns: Expected returns for assets
            quantum_risk_weight: Weight for quantum risk in objective function
        
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Performing quantum risk-adjusted optimization")
            
            if self.covariance_matrix is None:
                raise OptimizationError("Covariance matrix not set")
            
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            def objective(weights):
                # Traditional mean-variance objective
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_variance = weights.T @ self.covariance_matrix @ weights
                
                # Quantum risk penalty
                quantum_metrics = self.calculate_quantum_risk_metrics(weights)
                quantum_risk_penalty = quantum_risk_weight * quantum_metrics['quantum_risk_score']
                
                # Combined objective (minimize risk, maximize return)
                return portfolio_variance + quantum_risk_penalty - portfolio_return
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = float(np.sum(expected_returns * optimal_weights))
                portfolio_volatility = float(np.sqrt(optimal_weights.T @ self.covariance_matrix @ optimal_weights))
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
                
                # Calculate quantum metrics
                quantum_metrics = self.calculate_quantum_risk_metrics(optimal_weights)
                
                results = {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'quantum_metrics': quantum_metrics,
                    'method': 'Quantum Risk Adjusted'
                }
                
                logger.info(f"Quantum risk-adjusted optimization completed. Sharpe: {sharpe_ratio:.4f}")
                return results
            else:
                raise OptimizationError(f"Quantum optimization failed: {result.message}")
                
        except Exception as e:
            raise OptimizationError(f"Quantum risk-adjusted optimization failed: {e}")


# Unit tests for QuantumRiskModel
def test_quantum_risk_model():
    """Unit tests for QuantumRiskModel functionality."""
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns_data = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 100),
        'GOOGL': np.random.normal(0.0008, 0.025, 100),
        'MSFT': np.random.normal(0.0012, 0.018, 100)
    }, index=dates)
    
    config = OptimizationConfig()
    quantum_model = QuantumRiskModel(config)
    
    try:
        # Test data setting
        quantum_model.set_data(returns_data)
        assert quantum_model.returns_data is not None, "Returns data should be set"
        assert quantum_model.covariance_matrix is not None, "Covariance matrix should be calculated"
        
        # Test quantum entropy calculation
        entropy = quantum_model.calculate_quantum_entropy()
        assert entropy >= 0, "Quantum entropy should be non-negative"
        
        # Test quantum coherence calculation
        coherence = quantum_model.calculate_quantum_coherence()
        assert coherence >= 0, "Quantum coherence should be non-negative"
        
        # Test quantum discord calculation
        discord = quantum_model.calculate_quantum_discord()
        assert discord >= 0, "Quantum discord should be non-negative"
        
        # Test entanglement measure calculation
        entanglement = quantum_model.calculate_entanglement_measure()
        assert entanglement >= 0, "Entanglement measure should be non-negative"
        
        # Test comprehensive quantum risk metrics
        metrics = quantum_model.calculate_quantum_risk_metrics()
        assert 'quantum_entropy' in metrics, "Should include quantum entropy"
        assert 'quantum_risk_score' in metrics, "Should include composite risk score"
        
        # Test quantum risk-adjusted optimization
        expected_returns = returns_data.mean().values * 252
        result = quantum_model.quantum_risk_adjusted_optimization(expected_returns)
        assert 'weights' in result, "Should return portfolio weights"
        assert abs(np.sum(result['weights']) - 1.0) < 1e-6, "Weights should sum to 1"
        
        print("All QuantumRiskModel tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_quantum_risk_model()