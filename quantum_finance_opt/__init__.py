"""
QuantumFinanceOpt - Advanced Portfolio Optimization Tool

A comprehensive portfolio optimization framework integrating classical methods,
machine learning, deep learning, and reinforcement learning approaches with
quantum-inspired risk modeling and transformer-based forecasting.
"""

__version__ = "1.0.0"
__author__ = "QuantumFinanceOpt Team"

# Import core components that don't require optional dependencies
from .core.config import OptimizationConfig
from .core.exceptions import *

# Try to import main optimizer, but handle missing dependencies gracefully
try:
    from .core.optimizer import QuantumFinanceOptimizer
    FULL_FUNCTIONALITY = True
except ImportError as e:
    import warnings
    warnings.warn(f"Some dependencies missing: {e}. Full functionality not available.")
    QuantumFinanceOptimizer = None
    FULL_FUNCTIONALITY = False

__all__ = [
    'OptimizationConfig',
    'QuantumFinanceOptError',
    'DataProcessingError',
    'ModelTrainingError',
    'OptimizationError',
    'FULL_FUNCTIONALITY'
]

if FULL_FUNCTIONALITY:
    __all__.append('QuantumFinanceOptimizer')