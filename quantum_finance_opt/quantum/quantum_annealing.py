"""
Quantum Annealing for Portfolio Optimization

Implements D-Wave quantum annealing for large-scale portfolio optimization
using QUBO (Quadratic Unconstrained Binary Optimization) formulation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd

# D-Wave imports with fallbacks
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.cloud import Client
    import dimod
    from dimod import BinaryQuadraticModel
    import dwave_networkx as dnx
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class QuantumAnnealingResult:
    """Result from quantum annealing optimization"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    energy: float
    chain_break_fraction: float
    timing: Dict[str, float]
    num_reads: int
    success_probability: float
    backend_used: str
    quantum_advantage: bool


class QuantumAnnealingOptimizer:
    """
    D-Wave quantum annealing implementation for portfolio optimization
    
    Formulates portfolio optimization as QUBO problem and solves using
    D-Wave quantum annealer or classical simulated annealing fallback.
    """
    
    def __init__(self, backend_manager):
        self.backend_manager = backend_manager
        self.logger = logging.getLogger(__name__)
        
        # Annealing parameters
        self.num_reads = 1000
        self.chain_strength = None
        self.annealing_time = 20  # microseconds
        
        # Problem encoding
        self.num_assets = 0
        self.precision_bits = 4  # Bits per asset weight
        self.returns = None
        self.covariance = None
        self.risk_aversion = 1.0
        
        # QUBO model
        self.qubo_model = None
        self.variable_mapping = {}
    
    def optimize_portfolio(self,
                          returns: np.ndarray,
                          covariance: np.ndarray,
                          budget: float = 1.0,
                          risk_aversion: float = 1.0,
                          num_reads: int = 1000) -> QuantumAnnealingResult:
        """
        Optimize portfolio using quantum annealing
        
        Args:
            returns: Expected returns for each asset
            covariance: Covariance matrix of asset returns
            budget: Total budget constraint
            risk_aversion: Risk aversion parameter
            num_reads: Number of annealing reads
            
        Returns:
            QuantumAnnealingResult with optimization results
        """
        self.logger.info("Starting quantum annealing portfolio optimization")
        
        # Store problem parameters
        self.returns = returns
        self.covariance = covariance
        self.risk_aversion = risk_aversion
        self.num_assets = len(returns)
        self.num_reads = num_reads
        
        if not DWAVE_AVAILABLE:
            self.logger.warning("D-Wave not available, using simulated annealing fallback")
            return self._simulated_annealing_fallback()
        
        try:
            # Create QUBO formulation
            self._create_qubo_model()
            
            # Solve using quantum annealer
            solution = self._solve_with_quantum_annealer()
            
            # Extract portfolio weights
            weights = self._extract_weights_from_solution(solution)
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Analyze solution quality
            quality_metrics = self._analyze_solution_quality(solution)
            
            return QuantumAnnealingResult(
                weights=weights,
                expected_return=metrics['return'],
                volatility=metrics['volatility'],
                sharpe_ratio=metrics['sharpe'],
                energy=solution.first.energy,
                chain_break_fraction=quality_metrics['chain_break_fraction'],
                timing=quality_metrics['timing'],
                num_reads=len(solution),
                success_probability=quality_metrics['success_probability'],
                backend_used="dwave_annealer",
                quantum_advantage=True
            )
            
        except Exception as e:
            self.logger.error(f"Quantum annealing failed: {e}")
            return self._simulated_annealing_fallback()
    
    def _create_qubo_model(self):
        """Create QUBO formulation of portfolio optimization problem"""
        # Binary variables: x_ij represents j-th bit of i-th asset weight
        total_vars = self.num_assets * self.precision_bits
        
        # Initialize QUBO matrix
        Q = {}
        
        # Create variable mapping
        self.variable_mapping = {}
        var_index = 0
        for asset in range(self.num_assets):
            for bit in range(self.precision_bits):
                var_name = f"x_{asset}_{bit}"
                self.variable_mapping[var_name] = var_index
                var_index += 1
        
        # Objective function terms
        self._add_return_terms(Q)
        self._add_risk_terms(Q)
        self._add_budget_constraint(Q)
        
        # Create BQM
        self.qubo_model = dimod.BinaryQuadraticModel.from_qubo(Q)
        
        self.logger.info(f"Created QUBO model with {len(self.qubo_model.variables)} variables")
    
    def _add_return_terms(self, Q: Dict):
        """Add expected return terms to QUBO"""
        for asset in range(self.num_assets):
            for bit in range(self.precision_bits):
                var_name = f"x_{asset}_{bit}"
                var_idx = self.variable_mapping[var_name]
                
                # Weight of this bit in the asset allocation
                bit_weight = 2**bit / (2**self.precision_bits - 1)
                
                # Return contribution (negative because we maximize returns)
                return_coeff = -self.returns[asset] * bit_weight
                Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0) + return_coeff   
 
    def _add_risk_terms(self, Q: Dict):
        """Add risk (covariance) terms to QUBO"""
        for asset_i in range(self.num_assets):
            for asset_j in range(self.num_assets):
                cov_ij = self.covariance[asset_i, asset_j]
                
                for bit_i in range(self.precision_bits):
                    for bit_j in range(self.precision_bits):
                        var_i = f"x_{asset_i}_{bit_i}"
                        var_j = f"x_{asset_j}_{bit_j}"
                        
                        idx_i = self.variable_mapping[var_i]
                        idx_j = self.variable_mapping[var_j]
                        
                        # Bit weights
                        weight_i = 2**bit_i / (2**self.precision_bits - 1)
                        weight_j = 2**bit_j / (2**self.precision_bits - 1)
                        
                        # Risk contribution
                        risk_coeff = self.risk_aversion * cov_ij * weight_i * weight_j
                        
                        if idx_i == idx_j:
                            Q[(idx_i, idx_i)] = Q.get((idx_i, idx_i), 0) + risk_coeff
                        else:
                            key = (min(idx_i, idx_j), max(idx_i, idx_j))
                            Q[key] = Q.get(key, 0) + risk_coeff
    
    def _add_budget_constraint(self, Q: Dict, penalty_strength: float = 10.0):
        """Add budget constraint as penalty term"""
        # Budget constraint: sum of weights = 1
        # Penalty: penalty_strength * (sum_weights - 1)^2
        
        # Linear terms: -2 * penalty_strength * sum_weights
        for asset in range(self.num_assets):
            for bit in range(self.precision_bits):
                var_name = f"x_{asset}_{bit}"
                var_idx = self.variable_mapping[var_name]
                
                bit_weight = 2**bit / (2**self.precision_bits - 1)
                linear_coeff = -2 * penalty_strength * bit_weight
                
                Q[(var_idx, var_idx)] = Q.get((var_idx, var_idx), 0) + linear_coeff
        
        # Quadratic terms: penalty_strength * sum_i sum_j w_i * w_j
        for asset_i in range(self.num_assets):
            for asset_j in range(self.num_assets):
                for bit_i in range(self.precision_bits):
                    for bit_j in range(self.precision_bits):
                        var_i = f"x_{asset_i}_{bit_i}"
                        var_j = f"x_{asset_j}_{bit_j}"
                        
                        idx_i = self.variable_mapping[var_i]
                        idx_j = self.variable_mapping[var_j]
                        
                        weight_i = 2**bit_i / (2**self.precision_bits - 1)
                        weight_j = 2**bit_j / (2**self.precision_bits - 1)
                        
                        quad_coeff = penalty_strength * weight_i * weight_j
                        
                        if idx_i == idx_j:
                            Q[(idx_i, idx_i)] = Q.get((idx_i, idx_i), 0) + quad_coeff
                        else:
                            key = (min(idx_i, idx_j), max(idx_i, idx_j))
                            Q[key] = Q.get(key, 0) + quad_coeff
        
        # Constant term: penalty_strength
        # (This doesn't affect optimization but is part of the penalty)
    
    def _solve_with_quantum_annealer(self):
        """Solve QUBO using D-Wave quantum annealer"""
        try:
            # Get D-Wave sampler
            sampler = EmbeddingComposite(DWaveSampler())
            
            # Set chain strength automatically if not specified
            if self.chain_strength is None:
                self.chain_strength = max(abs(bias) for bias in self.qubo_model.linear.values())
                if self.qubo_model.quadratic:
                    max_quad = max(abs(bias) for bias in self.qubo_model.quadratic.values())
                    self.chain_strength = max(self.chain_strength, max_quad)
            
            # Sample from the annealer
            response = sampler.sample(
                self.qubo_model,
                num_reads=self.num_reads,
                chain_strength=self.chain_strength,
                annealing_time=self.annealing_time,
                label="Portfolio Optimization"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"D-Wave sampling failed: {e}")
            raise
    
    def _extract_weights_from_solution(self, solution) -> np.ndarray:
        """Extract portfolio weights from annealing solution"""
        # Get the best solution
        best_sample = solution.first.sample
        
        # Convert binary variables back to weights
        weights = np.zeros(self.num_assets)
        
        for asset in range(self.num_assets):
            asset_weight = 0.0
            for bit in range(self.precision_bits):
                var_name = f"x_{asset}_{bit}"
                var_idx = self.variable_mapping[var_name]
                
                if var_idx in best_sample and best_sample[var_idx] == 1:
                    bit_weight = 2**bit / (2**self.precision_bits - 1)
                    asset_weight += bit_weight
            
            weights[asset] = asset_weight
        
        # Normalize weights to satisfy budget constraint
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # Fallback to equal weights
            weights = np.ones(self.num_assets) / self.num_assets
        
        return weights
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        portfolio_return = np.dot(weights, self.returns)
        portfolio_variance = np.dot(weights, np.dot(self.covariance, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': sharpe_ratio
        }
    
    def _analyze_solution_quality(self, solution) -> Dict[str, Any]:
        """Analyze quality of annealing solution"""
        # Chain break fraction
        try:
            chain_break_fraction = solution.data_vectors['chain_break_fraction'][0]
        except (KeyError, IndexError):
            chain_break_fraction = 0.0
        
        # Timing information
        try:
            timing = solution.info.get('timing', {})
        except AttributeError:
            timing = {}
        
        # Success probability (fraction of reads that found the best energy)
        best_energy = solution.first.energy
        num_best = sum(1 for sample in solution.data() if abs(sample.energy - best_energy) < 1e-6)
        success_probability = num_best / len(solution)
        
        return {
            'chain_break_fraction': chain_break_fraction,
            'timing': timing,
            'success_probability': success_probability
        }
    
    def _simulated_annealing_fallback(self) -> QuantumAnnealingResult:
        """Classical simulated annealing fallback"""
        self.logger.info("Using simulated annealing fallback")
        
        try:
            # Use dimod's simulated annealing sampler
            sampler = dimod.SimulatedAnnealingSampler()
            
            # Create simple QUBO if not already created
            if self.qubo_model is None:
                self._create_qubo_model()
            
            # Sample using simulated annealing
            response = sampler.sample(
                self.qubo_model,
                num_reads=self.num_reads,
                seed=42
            )
            
            # Extract weights
            weights = self._extract_weights_from_solution(response)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights)
            
            return QuantumAnnealingResult(
                weights=weights,
                expected_return=metrics['return'],
                volatility=metrics['volatility'],
                sharpe_ratio=metrics['sharpe'],
                energy=response.first.energy,
                chain_break_fraction=0.0,  # No chains in simulated annealing
                timing={'total_time': 0.0},
                num_reads=len(response),
                success_probability=1.0 / len(response),
                backend_used="simulated_annealing",
                quantum_advantage=False
            )
            
        except Exception as e:
            self.logger.error(f"Simulated annealing fallback failed: {e}")
            return self._equal_weights_fallback()
    
    def _equal_weights_fallback(self) -> QuantumAnnealingResult:
        """Ultimate fallback to equal weights"""
        self.logger.warning("Using equal weights fallback")
        
        weights = np.ones(self.num_assets) / self.num_assets
        metrics = self._calculate_portfolio_metrics(weights)
        
        return QuantumAnnealingResult(
            weights=weights,
            expected_return=metrics['return'],
            volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe'],
            energy=0.0,
            chain_break_fraction=0.0,
            timing={'total_time': 0.0},
            num_reads=1,
            success_probability=1.0,
            backend_used="equal_weights_fallback",
            quantum_advantage=False
        )
    
    def get_qubo_info(self) -> Dict[str, Any]:
        """Get information about the QUBO formulation"""
        if self.qubo_model is None:
            return {"status": "not_created"}
        
        return {
            "status": "created",
            "num_variables": len(self.qubo_model.variables),
            "num_interactions": len(self.qubo_model.quadratic),
            "energy_range": (
                min(self.qubo_model.linear.values()) if self.qubo_model.linear else 0,
                max(self.qubo_model.linear.values()) if self.qubo_model.linear else 0
            ),
            "variable_mapping": self.variable_mapping
        }
    
    def estimate_annealing_time(self, problem_size: int) -> Dict[str, float]:
        """Estimate optimal annealing time for problem size"""
        # Heuristic estimates based on problem complexity
        base_time = 20  # microseconds
        
        if problem_size < 100:
            recommended_time = base_time
        elif problem_size < 500:
            recommended_time = base_time * 2
        else:
            recommended_time = base_time * 5
        
        return {
            "recommended_annealing_time": recommended_time,
            "min_annealing_time": base_time,
            "max_annealing_time": 2000,  # D-Wave hardware limit
            "problem_complexity": "high" if problem_size > 500 else "medium" if problem_size > 100 else "low"
        }