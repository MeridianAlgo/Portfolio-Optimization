"""
Quantum Backend Manager

Provides abstraction layer for multiple quantum computing backends
including IBM Qiskit, Google Cirq, PennyLane, and D-Wave.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np

# Quantum computing imports with fallbacks
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers import Backend
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available - quantum features will use classical fallbacks")

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False
    logging.warning("Cirq not available")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logging.warning("PennyLane not available")

try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.cloud import Client
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    logging.warning("D-Wave Ocean SDK not available")


class QuantumBackendType(Enum):
    """Supported quantum backend types"""
    QISKIT_SIMULATOR = "qiskit_simulator"
    QISKIT_HARDWARE = "qiskit_hardware"
    CIRQ_SIMULATOR = "cirq_simulator"
    PENNYLANE = "pennylane"
    DWAVE_ANNEALER = "dwave_annealer"
    CLASSICAL_FALLBACK = "classical_fallback"


class QuantumBackendManager:
    """
    Manages quantum computing backends and provides unified interface
    """
    
    def __init__(self, preferred_backend: str = "qiskit_simulator"):
        self.logger = logging.getLogger(__name__)
        self.preferred_backend = preferred_backend
        self.available_backends = self._detect_available_backends()
        self.active_backend = None
        self._initialize_backend()
    
    def _detect_available_backends(self) -> List[QuantumBackendType]:
        """Detect which quantum backends are available"""
        available = []
        
        if QISKIT_AVAILABLE:
            available.extend([
                QuantumBackendType.QISKIT_SIMULATOR,
                QuantumBackendType.QISKIT_HARDWARE
            ])
        
        if CIRQ_AVAILABLE:
            available.append(QuantumBackendType.CIRQ_SIMULATOR)
        
        if PENNYLANE_AVAILABLE:
            available.append(QuantumBackendType.PENNYLANE)
        
        if DWAVE_AVAILABLE:
            available.append(QuantumBackendType.DWAVE_ANNEALER)
        
        # Always have classical fallback
        available.append(QuantumBackendType.CLASSICAL_FALLBACK)
        
        return available    
  
  def _initialize_backend(self):
        """Initialize the preferred quantum backend"""
        try:
            backend_type = QuantumBackendType(self.preferred_backend)
            if backend_type in self.available_backends:
                self.active_backend = self._setup_backend(backend_type)
                self.logger.info(f"Initialized quantum backend: {backend_type.value}")
            else:
                self.logger.warning(f"Preferred backend {self.preferred_backend} not available")
                self._fallback_to_available_backend()
        except ValueError:
            self.logger.error(f"Invalid backend type: {self.preferred_backend}")
            self._fallback_to_available_backend()
    
    def _fallback_to_available_backend(self):
        """Fallback to first available backend"""
        if self.available_backends:
            fallback_backend = self.available_backends[0]
            self.active_backend = self._setup_backend(fallback_backend)
            self.logger.info(f"Using fallback backend: {fallback_backend.value}")
        else:
            self.logger.error("No quantum backends available")
            self.active_backend = None
    
    def _setup_backend(self, backend_type: QuantumBackendType) -> Optional[Any]:
        """Setup specific quantum backend"""
        if backend_type == QuantumBackendType.QISKIT_SIMULATOR:
            return AerSimulator()
        
        elif backend_type == QuantumBackendType.QISKIT_HARDWARE:
            # This would require IBM Quantum account setup
            try:
                from qiskit import IBMQ
                IBMQ.load_account()
                provider = IBMQ.get_provider()
                return provider.get_backend('ibmq_qasm_simulator')
            except Exception as e:
                self.logger.warning(f"IBM Quantum hardware not available: {e}")
                return AerSimulator()
        
        elif backend_type == QuantumBackendType.CIRQ_SIMULATOR:
            return cirq.Simulator()
        
        elif backend_type == QuantumBackendType.PENNYLANE:
            return qml.device('default.qubit', wires=10)
        
        elif backend_type == QuantumBackendType.DWAVE_ANNEALER:
            try:
                return EmbeddingComposite(DWaveSampler())
            except Exception as e:
                self.logger.warning(f"D-Wave annealer not available: {e}")
                return None
        
        elif backend_type == QuantumBackendType.CLASSICAL_FALLBACK:
            return "classical"
        
        return None
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend"""
        if not self.active_backend:
            return {"backend": "none", "available": False}
        
        backend_info = {
            "backend": self.preferred_backend,
            "available": True,
            "quantum_capable": self.active_backend != "classical"
        }
        
        if QISKIT_AVAILABLE and hasattr(self.active_backend, 'configuration'):
            config = self.active_backend.configuration()
            backend_info.update({
                "num_qubits": getattr(config, 'n_qubits', 'unknown'),
                "coupling_map": getattr(config, 'coupling_map', None),
                "basis_gates": getattr(config, 'basis_gates', [])
            })
        
        return backend_info
    
    def execute_circuit(self, circuit: Any, shots: int = 1024) -> Dict[str, Any]:
        """Execute quantum circuit on active backend"""
        if not self.active_backend:
            raise RuntimeError("No quantum backend available")
        
        if self.active_backend == "classical":
            return self._classical_fallback_execution(circuit, shots)
        
        # Execute on quantum backend
        if QISKIT_AVAILABLE and isinstance(self.active_backend, (AerSimulator, Backend)):
            job = self.active_backend.run(circuit, shots=shots)
            result = job.result()
            return {"counts": result.get_counts(), "success": True}
        
        # Add other backend execution logic here
        return {"error": "Backend execution not implemented", "success": False}
    
    def _classical_fallback_execution(self, circuit: Any, shots: int) -> Dict[str, Any]:
        """Classical simulation fallback"""
        # Implement classical approximation of quantum algorithms
        self.logger.info("Using classical fallback for quantum circuit execution")
        
        # For now, return mock results - this would be replaced with actual classical algorithms
        return {
            "counts": {"0" * 10: shots},  # Mock result
            "success": True,
            "classical_fallback": True
        }
    
    def is_quantum_available(self) -> bool:
        """Check if true quantum computing is available"""
        return (self.active_backend is not None and 
                self.active_backend != "classical")
    
    def get_quantum_advantage_estimate(self, problem_size: int) -> Dict[str, Any]:
        """Estimate potential quantum advantage for given problem size"""
        if not self.is_quantum_available():
            return {
                "quantum_advantage": False,
                "reason": "No quantum backend available",
                "recommendation": "Use classical algorithms"
            }
        
        # Rough heuristic for quantum advantage
        if problem_size < 50:
            return {
                "quantum_advantage": False,
                "reason": "Problem size too small for quantum advantage",
                "recommendation": "Use classical algorithms"
            }
        elif problem_size < 200:
            return {
                "quantum_advantage": True,
                "reason": "Moderate quantum advantage expected",
                "recommendation": "Try quantum algorithms with classical fallback"
            }
        else:
            return {
                "quantum_advantage": True,
                "reason": "Significant quantum advantage expected",
                "recommendation": "Prioritize quantum algorithms"
            }