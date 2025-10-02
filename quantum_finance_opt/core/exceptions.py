"""
Custom exception classes for QuantumFinanceOpt.

This module defines the exception hierarchy used throughout the application
for proper error handling and debugging.
"""

class QuantumFinanceOptError(Exception):
    """Base exception for QuantumFinanceOpt."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class DataProcessingError(QuantumFinanceOptError):
    """Raised when data processing fails."""
    pass


class ModelTrainingError(QuantumFinanceOptError):
    """Raised when model training fails."""
    pass


class OptimizationError(QuantumFinanceOptError):
    """Raised when optimization fails to converge."""
    pass


class ConfigurationError(QuantumFinanceOptError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(QuantumFinanceOptError):
    """Raised when data validation fails."""
    pass


class ResourceError(QuantumFinanceOptError):
    """Raised when system resources are insufficient."""
    pass


class ModelLoadingError(QuantumFinanceOptError):
    """Raised when model loading fails."""
    pass