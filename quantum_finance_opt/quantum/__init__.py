"""
Quantum Computing Module for QuantumFinanceOpt

This module provides quantum computing capabilities including:
- Quantum Approximate Optimization Algorithm (QAOA)
- Variational Quantum Eigensolver (VQE)
- Quantum Annealing
- Quantum Machine Learning
"""

from .backend_manager import QuantumBackendManager
from .qaoa_optimizer import QAOAPortfolioOptimizer
from .vqe_risk_model import VQERiskModel
from .quantum_annealing import QuantumAnnealingOptimizer
from .quantum_ml import QuantumMLService

__all__ = [
    'QuantumBackendManager',
    'QAOAPortfolioOptimizer', 
    'VQERiskModel',
    'QuantumAnnealingOptimizer',
    'QuantumMLService'
]