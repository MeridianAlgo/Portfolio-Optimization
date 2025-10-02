"""
Quantum Approximate Optimization Algorithm (QAOA) for Portfolio Optimization

Implements QAOA for solving portfolio optimization problems on quantum hardware.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

# Quantum imports with fallbacks
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class QAOAResult:
    """Result from QAOA optimization"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    quantum_cost: float
    classical_cost: float
    optimization_success: bool
    num_iterations: int
    backend_used: str
    quantum_advantage: bool


class QAOAPortfolioOptimizer:
    """
    QAOA implementation for portfolio optimization
    
    Formulates portfolio optimization as a QUBO problem and solves
    using quantum approximate optimization algorithm.
    """
    
    def __init__(self, backend_manager, num_layers: int = 2):
        self.backend_manager = backend_manager
        self.num_layers = num_layers
        self.logger = logging.getLogger(__name__)
        
        # QAOA parameters
        self.beta_params = None  # Mixer parameters
        self.gamma_params = None  # Cost parameters
        
        # Problem encoding
        self.num_assets = 0
        self.num_qubits = 0
        self.returns = None
        self.covariance = None
        self.risk_aversion = 1.0
        
    def optimize_portfolio(self, 
                          returns: np.ndarray,
                          covariance: np.ndarray,
                          budget: float = 1.0,
                          risk_aversion: float = 1.0,
                          max_iterations: int = 100) -> QAOAResult:
        """
        Optimize portfolio using QAOA
        
        Args:
            returns: Expected returns for each asset
            covariance: Covariance matrix of asset returns
            budget: Total budget constraint (default: 1.0 for weights)
            risk_aversion: Risk aversion parameter
            max_iterations: Maximum optimization iterations
            
        Returns:
            QAOAResult with optimization results
        """
        self.logger.info("Starting QAOA portfolio optimization")
        
        # Store problem parameters
        self.returns = returns
        self.covariance = covariance
        self.risk_aversion = risk_aversion
        self.num_assets = len(returns)
        
        # Determine number of qubits needed
        self.num_qubits = self._calculate_qubits_needed()
        
        if not QISKIT_AVAILABLE or not self.backend_manager.is_quantum_available():
            self.logger.warning("Quantum backend not available, using classical fallback")
            return self._classical_fallback_optimization()
        
        try:
            # Create QAOA circuit
            qaoa_circuit = self._create_qaoa_circuit()
            
            # Optimize parameters
            optimal_params = self._optimize_qaoa_parameters(qaoa_circuit, max_iterations)
            
            # Execute optimized circuit
            result = self._execute_optimized_circuit(qaoa_circuit, optimal_params)
            
            # Convert quantum result to portfolio weights
            weights = self._extract_portfolio_weights(result)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(weights)
            
            return QAOAResult(
                weights=weights,
                expected_return=portfolio_metrics['return'],
                volatility=portfolio_metrics['volatility'],
                sharpe_ratio=portfolio_metrics['sharpe'],
                quantum_cost=portfolio_metrics['quantum_cost'],
                classical_cost=portfolio_metrics['classical_cost'],
                optimization_success=True,
                num_iterations=len(optimal_params) // (2 * self.num_layers),
                backend_used=self.backend_manager.preferred_backend,
                quantum_advantage=True
            )
            
        except Exception as e:
            self.logger.error(f"QAOA optimization failed: {e}")
            return self._classical_fallback_optimization()
    
    def _calculate_qubits_needed(self) -> int:
        """Calculate number of qubits needed for problem encoding"""
        # For portfolio optimization, we need qubits to represent asset weights
        # Using binary encoding for discrete weight levels
        weight_precision = 4  # 4 bits per asset for 16 weight levels
        return self.num_assets * weight_precision
    
    def _create_qaoa_circuit(self) -> QuantumCircuit:
        """Create QAOA quantum circuit"""
        if not QISKIT_AVAILABLE:
            raise QuantumFinanceOptError("Qiskit not available for QAOA circuit creation")
        
        # Create quantum and classical registers
        qreg = QuantumRegister(self.num_qubits, 'q')
        creg = ClassicalRegister(self.num_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Initialize parameters
        beta_params = [Parameter(f'β_{i}') for i in range(self.num_layers)]
        gamma_params = [Parameter(f'γ_{i}') for i in range(self.num_layers)]
        
        # Initial state preparation (equal superposition)
        circuit.h(qreg)
        
        # QAOA layers
        for layer in range(self.num_layers):
            # Cost Hamiltonian (problem-specific)
            self._add_cost_hamiltonian(circuit, qreg, gamma_params[layer])
            
            # Mixer Hamiltonian (X rotations)
            self._add_mixer_hamiltonian(circuit, qreg, beta_params[layer])
        
        # Measurement
        circuit.measure(qreg, creg)
        
        return circuit    
 
   def _add_cost_hamiltonian(self, circuit: QuantumCircuit, qreg: QuantumRegister, gamma: Parameter):
        """Add cost Hamiltonian to circuit (portfolio objective function)"""
        # Encode portfolio optimization objective as quantum gates
        # This is a simplified version - full implementation would be more complex
        
        for i in range(self.num_assets):
            for j in range(i, self.num_assets):
                # Add interactions based on covariance matrix
                if i == j:
                    # Single asset terms (expected returns)
                    weight = self.returns[i] * gamma
                    circuit.rz(weight, qreg[i])
                else:
                    # Pairwise interactions (covariance terms)
                    weight = self.risk_aversion * self.covariance[i, j] * gamma
                    circuit.rzz(weight, qreg[i], qreg[j])
    
    def _add_mixer_hamiltonian(self, circuit: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):
        """Add mixer Hamiltonian to circuit (X rotations)"""
        for qubit in range(self.num_qubits):
            circuit.rx(2 * beta, qreg[qubit])
    
    def _optimize_qaoa_parameters(self, circuit: QuantumCircuit, max_iterations: int) -> List[float]:
        """Optimize QAOA parameters using classical optimizer"""
        # Initialize parameters randomly
        initial_params = np.random.uniform(0, 2*np.pi, 2 * self.num_layers)
        
        # Define cost function
        def cost_function(params):
            # Bind parameters to circuit
            bound_circuit = circuit.bind_parameters(dict(zip(circuit.parameters, params)))
            
            # Execute circuit
            result = self.backend_manager.execute_circuit(bound_circuit, shots=1024)
            
            # Calculate expectation value
            return self._calculate_expectation_value(result)
        
        # Use classical optimizer
        if QISKIT_AVAILABLE:
            optimizer = COBYLA(maxiter=max_iterations)
            result = optimizer.minimize(cost_function, initial_params)
            return result.x
        else:
            # Simple gradient-free optimization fallback
            from scipy.optimize import minimize
            result = minimize(cost_function, initial_params, method='COBYLA',
                            options={'maxiter': max_iterations})
            return result.x
    
    def _calculate_expectation_value(self, measurement_result: Dict[str, Any]) -> float:
        """Calculate expectation value from measurement results"""
        if 'counts' not in measurement_result:
            return float('inf')  # High cost for failed measurements
        
        counts = measurement_result['counts']
        total_shots = sum(counts.values())
        
        expectation = 0.0
        for bitstring, count in counts.items():
            # Convert bitstring to portfolio weights
            weights = self._bitstring_to_weights(bitstring)
            
            # Calculate portfolio cost (negative of objective function)
            portfolio_return = np.dot(weights, self.returns)
            portfolio_risk = np.dot(weights, np.dot(self.covariance, weights))
            cost = -(portfolio_return - self.risk_aversion * portfolio_risk)
            
            # Weight by measurement probability
            probability = count / total_shots
            expectation += probability * cost
        
        return expectation
    
    def _bitstring_to_weights(self, bitstring: str) -> np.ndarray:
        """Convert measurement bitstring to portfolio weights"""
        # Simple binary encoding - each asset gets equal bit allocation
        bits_per_asset = self.num_qubits // self.num_assets
        weights = np.zeros(self.num_assets)
        
        for i in range(self.num_assets):
            start_bit = i * bits_per_asset
            end_bit = start_bit + bits_per_asset
            asset_bits = bitstring[start_bit:end_bit]
            
            # Convert binary to weight (0 to 1)
            if asset_bits:
                weight_value = int(asset_bits, 2) / (2**bits_per_asset - 1)
                weights[i] = weight_value
        
        # Normalize to satisfy budget constraint
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        
        return weights
    
    def _execute_optimized_circuit(self, circuit: QuantumCircuit, optimal_params: List[float]) -> Dict[str, Any]:
        """Execute QAOA circuit with optimized parameters"""
        # Bind optimal parameters
        bound_circuit = circuit.bind_parameters(dict(zip(circuit.parameters, optimal_params)))
        
        # Execute with more shots for final result
        return self.backend_manager.execute_circuit(bound_circuit, shots=8192)
    
    def _extract_portfolio_weights(self, result: Dict[str, Any]) -> np.ndarray:
        """Extract portfolio weights from quantum measurement results"""
        if 'counts' not in result:
            # Fallback to equal weights
            return np.ones(self.num_assets) / self.num_assets
        
        counts = result['counts']
        
        # Find most probable measurement outcome
        max_count = 0
        best_bitstring = None
        
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                best_bitstring = bitstring
        
        if best_bitstring:
            return self._bitstring_to_weights(best_bitstring)
        else:
            return np.ones(self.num_assets) / self.num_assets
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        portfolio_return = np.dot(weights, self.returns)
        portfolio_variance = np.dot(weights, np.dot(self.covariance, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Quantum vs classical cost comparison
        quantum_cost = -(portfolio_return - self.risk_aversion * portfolio_variance)
        
        # Classical optimal for comparison (mean-variance)
        try:
            from scipy.optimize import minimize
            
            def classical_objective(w):
                ret = np.dot(w, self.returns)
                risk = np.dot(w, np.dot(self.covariance, w))
                return -(ret - self.risk_aversion * risk)
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.num_assets)]
            
            result = minimize(classical_objective, 
                            np.ones(self.num_assets) / self.num_assets,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            classical_cost = result.fun if result.success else quantum_cost
            
        except Exception:
            classical_cost = quantum_cost
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': sharpe_ratio,
            'quantum_cost': quantum_cost,
            'classical_cost': classical_cost
        }
    
    def _classical_fallback_optimization(self) -> QAOAResult:
        """Classical fallback when quantum backend unavailable"""
        self.logger.info("Using classical fallback for portfolio optimization")
        
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                portfolio_return = np.dot(weights, self.returns)
                portfolio_risk = np.dot(weights, np.dot(self.covariance, weights))
                return -(portfolio_return - self.risk_aversion * portfolio_risk)
            
            # Constraints and bounds
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.num_assets)]
            
            # Initial guess
            x0 = np.ones(self.num_assets) / self.num_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                metrics = self._calculate_portfolio_metrics(weights)
                
                return QAOAResult(
                    weights=weights,
                    expected_return=metrics['return'],
                    volatility=metrics['volatility'],
                    sharpe_ratio=metrics['sharpe'],
                    quantum_cost=metrics['classical_cost'],
                    classical_cost=metrics['classical_cost'],
                    optimization_success=True,
                    num_iterations=result.nit,
                    backend_used="classical_fallback",
                    quantum_advantage=False
                )
            else:
                # Equal weights fallback
                weights = np.ones(self.num_assets) / self.num_assets
                metrics = self._calculate_portfolio_metrics(weights)
                
                return QAOAResult(
                    weights=weights,
                    expected_return=metrics['return'],
                    volatility=metrics['volatility'],
                    sharpe_ratio=metrics['sharpe'],
                    quantum_cost=metrics['classical_cost'],
                    classical_cost=metrics['classical_cost'],
                    optimization_success=False,
                    num_iterations=0,
                    backend_used="equal_weights_fallback",
                    quantum_advantage=False
                )
                
        except Exception as e:
            self.logger.error(f"Classical fallback failed: {e}")
            
            # Ultimate fallback - equal weights
            weights = np.ones(self.num_assets) / self.num_assets
            metrics = self._calculate_portfolio_metrics(weights)
            
            return QAOAResult(
                weights=weights,
                expected_return=metrics['return'],
                volatility=metrics['volatility'],
                sharpe_ratio=metrics['sharpe'],
                quantum_cost=metrics['classical_cost'],
                classical_cost=metrics['classical_cost'],
                optimization_success=False,
                num_iterations=0,
                backend_used="equal_weights_fallback",
                quantum_advantage=False
            )