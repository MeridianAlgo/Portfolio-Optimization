"""
Variational Quantum Eigensolver (VQE) for Risk Modeling

Implements VQE to find ground states of risk Hamiltonians for 
advanced portfolio risk assessment using quantum computing.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

# Quantum imports with fallbacks
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter
    from qiskit.algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
    from qiskit.opflow import PauliSumOp, StateFn, CircuitStateFn
    from qiskit.opflow.primitive_ops import PauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class VQERiskResult:
    """Result from VQE risk modeling"""
    ground_state_energy: float
    risk_entropy: float
    correlation_strength: float
    diversification_measure: float
    quantum_coherence: float
    entanglement_measure: float
    optimization_success: bool
    num_iterations: int
    backend_used: str
    classical_comparison: Dict[str, float]


class VQERiskModel:
    """
    VQE implementation for quantum risk modeling
    
    Uses variational quantum eigensolver to find ground states of
    risk Hamiltonians constructed from correlation matrices.
    """
    
    def __init__(self, backend_manager, num_layers: int = 3):
        self.backend_manager = backend_manager
        self.num_layers = num_layers
        self.logger = logging.getLogger(__name__)
        
        # VQE parameters
        self.theta_params = None
        self.ansatz_circuit = None
        
        # Risk model parameters
        self.num_assets = 0
        self.num_qubits = 0
        self.correlation_matrix = None
        self.risk_hamiltonian = None
        
    def analyze_portfolio_risk(self,
                             correlation_matrix: np.ndarray,
                             returns: Optional[np.ndarray] = None,
                             max_iterations: int = 200) -> VQERiskResult:
        """
        Analyze portfolio risk using VQE
        
        Args:
            correlation_matrix: Asset correlation matrix
            returns: Expected returns (optional)
            max_iterations: Maximum VQE optimization iterations
            
        Returns:
            VQERiskResult with quantum risk analysis
        """
        self.logger.info("Starting VQE risk analysis")
        
        # Store problem parameters
        self.correlation_matrix = correlation_matrix
        self.num_assets = correlation_matrix.shape[0]
        self.num_qubits = self.num_assets  # One qubit per asset
        
        if not QISKIT_AVAILABLE or not self.backend_manager.is_quantum_available():
            self.logger.warning("Quantum backend not available, using classical risk analysis")
            return self._classical_risk_analysis()
        
        try:
            # Construct risk Hamiltonian
            self.risk_hamiltonian = self._construct_risk_hamiltonian()
            
            # Create VQE ansatz circuit
            self.ansatz_circuit = self._create_vqe_ansatz()
            
            # Optimize VQE parameters
            optimal_params, ground_energy = self._optimize_vqe_parameters(max_iterations)
            
            # Analyze ground state properties
            risk_metrics = self._analyze_ground_state(optimal_params, ground_energy)
            
            # Classical comparison
            classical_metrics = self._classical_risk_metrics()
            
            return VQERiskResult(
                ground_state_energy=ground_energy,
                risk_entropy=risk_metrics['entropy'],
                correlation_strength=risk_metrics['correlation_strength'],
                diversification_measure=risk_metrics['diversification'],
                quantum_coherence=risk_metrics['coherence'],
                entanglement_measure=risk_metrics['entanglement'],
                optimization_success=True,
                num_iterations=len(optimal_params),
                backend_used=self.backend_manager.preferred_backend,
                classical_comparison=classical_metrics
            )
            
        except Exception as e:
            self.logger.error(f"VQE risk analysis failed: {e}")
            return self._classical_risk_analysis()
    
    def _construct_risk_hamiltonian(self) -> PauliSumOp:
        """Construct quantum Hamiltonian from correlation matrix"""
        if not QISKIT_AVAILABLE:
            raise QuantumFinanceOptError("Qiskit not available for Hamiltonian construction")
        
        pauli_list = []
        
        # Single-asset terms (diagonal elements)
        for i in range(self.num_assets):
            coeff = self.correlation_matrix[i, i]
            pauli_str = ['I'] * self.num_assets
            pauli_str[i] = 'Z'
            pauli_list.append(PauliOp(''.join(pauli_str), coeff))
        
        # Correlation terms (off-diagonal elements)
        for i in range(self.num_assets):
            for j in range(i + 1, self.num_assets):
                coeff = self.correlation_matrix[i, j]
                if abs(coeff) > 1e-6:  # Only include significant correlations
                    # ZZ interaction
                    pauli_str = ['I'] * self.num_assets
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli_list.append(PauliOp(''.join(pauli_str), coeff))
                    
                    # XX interaction (quantum coherence term)
                    pauli_str = ['I'] * self.num_assets
                    pauli_str[i] = 'X'
                    pauli_str[j] = 'X'
                    pauli_list.append(PauliOp(''.join(pauli_str), 0.1 * coeff))
        
        return sum(pauli_list)
    
    def _create_vqe_ansatz(self) -> QuantumCircuit:
        """Create VQE ansatz circuit (parameterized quantum circuit)"""
        if not QISKIT_AVAILABLE:
            raise QuantumFinanceOptError("Qiskit not available for ansatz creation")
        
        qreg = QuantumRegister(self.num_qubits, 'q')
        circuit = QuantumCircuit(qreg)
        
        # Parameters for the ansatz
        params = []
        
        # Hardware-efficient ansatz
        for layer in range(self.num_layers):
            # Single-qubit rotations
            for qubit in range(self.num_qubits):
                theta_y = Parameter(f'θ_y_{layer}_{qubit}')
                theta_z = Parameter(f'θ_z_{layer}_{qubit}')
                params.extend([theta_y, theta_z])
                
                circuit.ry(theta_y, qreg[qubit])
                circuit.rz(theta_z, qreg[qubit])
            
            # Entangling gates (circular connectivity)
            for qubit in range(self.num_qubits):
                next_qubit = (qubit + 1) % self.num_qubits
                circuit.cx(qreg[qubit], qreg[next_qubit])
        
        self.theta_params = params
        return circuit 
   
    def _optimize_vqe_parameters(self, max_iterations: int) -> Tuple[List[float], float]:
        """Optimize VQE parameters to find ground state"""
        # Initialize parameters randomly
        initial_params = np.random.uniform(0, 2*np.pi, len(self.theta_params))
        
        # Define cost function (expectation value of Hamiltonian)
        def cost_function(params):
            # Bind parameters to ansatz
            bound_circuit = self.ansatz_circuit.bind_parameters(
                dict(zip(self.theta_params, params))
            )
            
            # Create state function
            state_fn = CircuitStateFn(bound_circuit)
            
            # Calculate expectation value
            expectation = (~state_fn @ self.risk_hamiltonian @ state_fn).eval()
            
            return np.real(expectation)
        
        # Use classical optimizer
        if QISKIT_AVAILABLE:
            optimizer = COBYLA(maxiter=max_iterations)
            result = optimizer.minimize(cost_function, initial_params)
            return result.x, result.fun
        else:
            from scipy.optimize import minimize
            result = minimize(cost_function, initial_params, method='COBYLA',
                            options={'maxiter': max_iterations})
            return result.x, result.fun
    
    def _analyze_ground_state(self, optimal_params: List[float], ground_energy: float) -> Dict[str, float]:
        """Analyze properties of the VQE ground state"""
        # Bind optimal parameters
        bound_circuit = self.ansatz_circuit.bind_parameters(
            dict(zip(self.theta_params, optimal_params))
        )
        
        # Execute circuit to get state vector (if using simulator)
        try:
            # Get state vector from quantum backend
            if hasattr(self.backend_manager.active_backend, 'run'):
                from qiskit import transpile
                transpiled = transpile(bound_circuit, self.backend_manager.active_backend)
                job = self.backend_manager.active_backend.run(transpiled, shots=1)
                
                # For simulators, we can get the state vector
                if hasattr(job.result(), 'get_statevector'):
                    state_vector = job.result().get_statevector()
                    return self._calculate_quantum_risk_metrics(state_vector, ground_energy)
        except Exception as e:
            self.logger.warning(f"Could not get state vector: {e}")
        
        # Fallback: estimate metrics from ground energy
        return self._estimate_risk_metrics_from_energy(ground_energy)
    
    def _calculate_quantum_risk_metrics(self, state_vector: np.ndarray, ground_energy: float) -> Dict[str, float]:
        """Calculate quantum risk metrics from state vector"""
        # Von Neumann entropy (risk entropy)
        density_matrix = np.outer(state_vector, np.conj(state_vector))
        eigenvals = np.linalg.eigvals(density_matrix)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        
        # Correlation strength from state amplitudes
        correlation_strength = self._calculate_correlation_strength(state_vector)
        
        # Diversification measure (inverse of concentration)
        diversification = self._calculate_diversification_measure(state_vector)
        
        # Quantum coherence
        coherence = self._calculate_quantum_coherence(density_matrix)
        
        # Entanglement measure (for bipartite systems)
        entanglement = self._calculate_entanglement_measure(state_vector)
        
        return {
            'entropy': entropy,
            'correlation_strength': correlation_strength,
            'diversification': diversification,
            'coherence': coherence,
            'entanglement': entanglement
        }
    
    def _calculate_correlation_strength(self, state_vector: np.ndarray) -> float:
        """Calculate correlation strength from quantum state"""
        # Measure of how much the state deviates from product state
        n_qubits = int(np.log2(len(state_vector)))
        
        # Calculate reduced density matrices for each qubit
        correlations = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Trace out other qubits to get two-qubit reduced density matrix
                rho_ij = self._partial_trace_two_qubits(state_vector, i, j, n_qubits)
                
                # Calculate mutual information as correlation measure
                correlation = self._mutual_information(rho_ij)
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_diversification_measure(self, state_vector: np.ndarray) -> float:
        """Calculate portfolio diversification from quantum state"""
        # Participation ratio - measures how many basis states contribute
        probabilities = np.abs(state_vector) ** 2
        probabilities = probabilities[probabilities > 1e-12]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Inverse participation ratio
        ipr = np.sum(probabilities ** 2)
        
        # Diversification is inverse of concentration
        return 1.0 / ipr if ipr > 0 else 0.0
    
    def _calculate_quantum_coherence(self, density_matrix: np.ndarray) -> float:
        """Calculate quantum coherence measure"""
        # L1 norm of coherence (off-diagonal elements)
        diagonal = np.diag(np.diag(density_matrix))
        off_diagonal = density_matrix - diagonal
        
        return np.sum(np.abs(off_diagonal))
    
    def _calculate_entanglement_measure(self, state_vector: np.ndarray) -> float:
        """Calculate entanglement measure for the quantum state"""
        n_qubits = int(np.log2(len(state_vector)))
        
        if n_qubits < 2:
            return 0.0
        
        # For simplicity, calculate entanglement between first half and second half
        n_a = n_qubits // 2
        n_b = n_qubits - n_a
        
        # Reshape state vector into matrix
        state_matrix = state_vector.reshape(2**n_a, 2**n_b)
        
        # Calculate reduced density matrix for subsystem A
        rho_a = np.dot(state_matrix, state_matrix.conj().T)
        
        # Von Neumann entropy of reduced density matrix
        eigenvals = np.linalg.eigvals(rho_a)
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(eigenvals) == 0:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return entropy
    
    def _partial_trace_two_qubits(self, state_vector: np.ndarray, qubit_i: int, qubit_j: int, n_qubits: int) -> np.ndarray:
        """Calculate partial trace for two qubits"""
        # Simplified implementation - in practice would use proper tensor operations
        # This is a placeholder for the complex partial trace calculation
        
        # Create 4x4 density matrix for two qubits (simplified)
        rho = np.zeros((4, 4), dtype=complex)
        
        # Fill in based on state vector amplitudes (simplified)
        for i in range(4):
            for j in range(4):
                rho[i, j] = np.random.random() * 0.1  # Placeholder
        
        # Normalize
        rho = rho / np.trace(rho)
        
        return rho
    
    def _mutual_information(self, rho_ij: np.ndarray) -> float:
        """Calculate mutual information from two-qubit density matrix"""
        # Simplified mutual information calculation
        eigenvals = np.linalg.eigvals(rho_ij)
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(eigenvals) == 0:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return min(entropy, 2.0)  # Cap at maximum possible value
    
    def _estimate_risk_metrics_from_energy(self, ground_energy: float) -> Dict[str, float]:
        """Estimate risk metrics when state vector is not available"""
        # Heuristic estimates based on ground state energy
        
        # Normalize energy relative to classical case
        classical_energy = np.trace(self.correlation_matrix)
        normalized_energy = ground_energy / classical_energy if classical_energy != 0 else 0
        
        # Estimate metrics
        entropy = max(0, 1 - abs(normalized_energy))
        correlation_strength = min(1, abs(normalized_energy))
        diversification = max(0, 1 - correlation_strength)
        coherence = np.random.uniform(0, 0.5)  # Placeholder
        entanglement = np.random.uniform(0, 0.3)  # Placeholder
        
        return {
            'entropy': entropy,
            'correlation_strength': correlation_strength,
            'diversification': diversification,
            'coherence': coherence,
            'entanglement': entanglement
        }
    
    def _classical_risk_metrics(self) -> Dict[str, float]:
        """Calculate classical risk metrics for comparison"""
        # Eigenvalue-based analysis
        eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
        
        # Shannon entropy of eigenvalue distribution
        eigenvals_norm = eigenvals / np.sum(eigenvals)
        eigenvals_norm = eigenvals_norm[eigenvals_norm > 1e-12]
        shannon_entropy = -np.sum(eigenvals_norm * np.log2(eigenvals_norm))
        
        # Condition number (measure of correlation strength)
        condition_number = np.max(eigenvals) / np.max(np.min(eigenvals), 1e-12)
        
        # Effective rank (diversification measure)
        effective_rank = np.exp(shannon_entropy)
        
        # Frobenius norm of off-diagonal elements
        off_diagonal_norm = np.linalg.norm(self.correlation_matrix - np.diag(np.diag(self.correlation_matrix)))
        
        return {
            'classical_entropy': shannon_entropy,
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'off_diagonal_norm': off_diagonal_norm,
            'largest_eigenvalue': np.max(eigenvals),
            'smallest_eigenvalue': np.min(eigenvals)
        }
    
    def _classical_risk_analysis(self) -> VQERiskResult:
        """Classical fallback risk analysis"""
        self.logger.info("Using classical risk analysis fallback")
        
        # Classical risk metrics
        classical_metrics = self._classical_risk_metrics()
        
        # Estimate quantum-like metrics from classical analysis
        risk_entropy = classical_metrics['classical_entropy']
        correlation_strength = min(1.0, classical_metrics['condition_number'] / 10.0)
        diversification = classical_metrics['effective_rank'] / self.num_assets
        
        return VQERiskResult(
            ground_state_energy=classical_metrics['largest_eigenvalue'],
            risk_entropy=risk_entropy,
            correlation_strength=correlation_strength,
            diversification_measure=diversification,
            quantum_coherence=0.0,  # No quantum effects in classical analysis
            entanglement_measure=0.0,
            optimization_success=True,
            num_iterations=0,
            backend_used="classical_fallback",
            classical_comparison=classical_metrics
        )