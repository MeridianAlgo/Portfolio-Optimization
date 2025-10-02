"""
Streaming Portfolio Optimizer

Provides real-time portfolio optimization with incremental updates
based on streaming market data.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import threading
import time

from .data_stream_manager import MarketDataPoint
from ..core.optimizer import QuantumFinanceOptimizer
from ..core.config import OptimizationConfig
from ..core.exceptions import QuantumFinanceOptError


@dataclass
class RebalanceSignal:
    """Signal to trigger portfolio rebalancing"""
    trigger_type: str  # 'time', 'threshold', 'volatility', 'news'
    timestamp: datetime
    affected_symbols: List[str]
    severity: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class PortfolioUpdate:
    """Portfolio update result"""
    timestamp: datetime
    old_weights: np.ndarray
    new_weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    turnover: float
    transaction_costs: float
    optimization_method: str
    execution_time: float
    success: bool


class StreamingOptimizer:
    """
    Real-time portfolio optimizer with streaming data integration
    """
    
    def __init__(self, 
                 config: OptimizationConfig,
                 data_stream_manager,
                 update_frequency: str = "5m",
                 volatility_threshold: float = 0.02,
                 return_threshold: float = 0.01):
        
        self.config = config
        self.data_stream_manager = data_stream_manager
        self.update_frequency = update_frequency
        self.volatility_threshold = volatility_threshold
        self.return_threshold = return_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Core optimizer
        self.optimizer = QuantumFinanceOptimizer(config)
        
        # Portfolio state
        self.current_weights = None
        self.current_prices = {}
        self.price_history = {symbol: deque(maxlen=100) for symbol in config.tickers}
        self.returns_history = {symbol: deque(maxlen=50) for symbol in config.tickers}
        
        # Streaming data buffers
        self.data_buffer = deque(maxlen=1000)
        self.last_optimization_time = None
        self.last_prices = {}
        
        # Rebalancing triggers
        self.rebalance_callbacks = []
        self.update_callbacks = []
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'avg_execution_time': 0.0,
            'avg_turnover': 0.0,
            'total_transaction_costs': 0.0
        }
        
        # Threading for background processing
        self.processing_thread = None
        self.stop_signal = threading.Event()
        
        # Initialize
        self._initialize_portfolio()
        self._setup_data_callbacks()
    
    def _initialize_portfolio(self):
        """Initialize portfolio with equal weights"""
        n_assets = len(self.config.tickers)
        self.current_weights = np.ones(n_assets) / n_assets
        self.last_optimization_time = datetime.now()
        
        self.logger.info(f"Initialized portfolio with equal weights for {n_assets} assets")
    
    def _setup_data_callbacks(self):
        """Setup callbacks for streaming data"""
        self.data_stream_manager.add_data_callback(self._on_new_data)
        self.data_stream_manager.add_error_callback(self._on_stream_error)
    
    def start_streaming_optimization(self):
        """Start real-time optimization process"""
        self.logger.info("Starting streaming portfolio optimization")
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._optimization_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Streaming optimization started")
    
    def stop_streaming_optimization(self):
        """Stop real-time optimization process"""
        self.logger.info("Stopping streaming optimization")
        
        self.stop_signal.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("Streaming optimization stopped")
    
    def _on_new_data(self, data_point: MarketDataPoint):
        """Handle new market data point"""
        if data_point.symbol in self.config.tickers:
            # Update current prices
            self.current_prices[data_point.symbol] = data_point.price
            
            # Add to price history
            self.price_history[data_point.symbol].append({
                'timestamp': data_point.timestamp,
                'price': data_point.price,
                'volume': data_point.volume
            })
            
            # Calculate returns if we have previous price
            if data_point.symbol in self.last_prices:
                last_price = self.last_prices[data_point.symbol]
                if last_price > 0:
                    return_pct = (data_point.price - last_price) / last_price
                    self.returns_history[data_point.symbol].append({
                        'timestamp': data_point.timestamp,
                        'return': return_pct
                    })
            
            self.last_prices[data_point.symbol] = data_point.price
            
            # Add to data buffer
            self.data_buffer.append(data_point)
            
            # Check for rebalancing triggers
            self._check_rebalancing_triggers(data_point)
    
    def _on_stream_error(self, stream_id: str, error: Exception):
        """Handle streaming data errors"""
        self.logger.error(f"Stream error in {stream_id}: {error}")
        
        # Could implement fallback data sources or error recovery here
    
    def _check_rebalancing_triggers(self, data_point: MarketDataPoint):
        """Check if rebalancing should be triggered"""
        triggers = []
        
        # Time-based trigger
        if self._should_rebalance_by_time():
            triggers.append(RebalanceSignal(
                trigger_type='time',
                timestamp=datetime.now(),
                affected_symbols=self.config.tickers,
                severity=0.3,
                metadata={'frequency': self.update_frequency}
            ))
        
        # Volatility-based trigger
        volatility_trigger = self._check_volatility_trigger(data_point)
        if volatility_trigger:
            triggers.append(volatility_trigger)
        
        # Return threshold trigger
        return_trigger = self._check_return_trigger(data_point)
        if return_trigger:
            triggers.append(return_trigger)
        
        # Notify callbacks about triggers
        for trigger in triggers:
            for callback in self.rebalance_callbacks:
                try:
                    callback(trigger)
                except Exception as e:
                    self.logger.error(f"Rebalance callback error: {e}")
    
    def _should_rebalance_by_time(self) -> bool:
        """Check if enough time has passed for rebalancing"""
        if not self.last_optimization_time:
            return True
        
        time_diff = datetime.now() - self.last_optimization_time
        
        if self.update_frequency.endswith('s'):
            threshold = timedelta(seconds=int(self.update_frequency[:-1]))
        elif self.update_frequency.endswith('m'):
            threshold = timedelta(minutes=int(self.update_frequency[:-1]))
        elif self.update_frequency.endswith('h'):
            threshold = timedelta(hours=int(self.update_frequency[:-1]))
        else:
            threshold = timedelta(minutes=5)  # Default
        
        return time_diff >= threshold
    
    def _check_volatility_trigger(self, data_point: MarketDataPoint) -> Optional[RebalanceSignal]:
        """Check for volatility-based rebalancing trigger"""
        symbol = data_point.symbol
        
        if len(self.returns_history[symbol]) < 10:
            return None
        
        # Calculate recent volatility
        recent_returns = [r['return'] for r in list(self.returns_history[symbol])[-10:]]
        volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
        
        if volatility > self.volatility_threshold:
            return RebalanceSignal(
                trigger_type='volatility',
                timestamp=datetime.now(),
                affected_symbols=[symbol],
                severity=min(1.0, volatility / self.volatility_threshold),
                metadata={'volatility': volatility, 'threshold': self.volatility_threshold}
            )
        
        return None
    
    def _check_return_trigger(self, data_point: MarketDataPoint) -> Optional[RebalanceSignal]:
        """Check for return-based rebalancing trigger"""
        symbol = data_point.symbol
        
        if len(self.returns_history[symbol]) < 5:
            return None
        
        # Calculate cumulative return over recent period
        recent_returns = [r['return'] for r in list(self.returns_history[symbol])[-5:]]
        cumulative_return = np.prod([1 + r for r in recent_returns]) - 1
        
        if abs(cumulative_return) > self.return_threshold:
            return RebalanceSignal(
                trigger_type='return',
                timestamp=datetime.now(),
                affected_symbols=[symbol],
                severity=min(1.0, abs(cumulative_return) / self.return_threshold),
                metadata={'return': cumulative_return, 'threshold': self.return_threshold}
            )
        
        return None 
   
    def _optimization_loop(self):
        """Main optimization loop running in background thread"""
        self.logger.info("Starting optimization loop")
        
        while not self.stop_signal.is_set():
            try:
                # Check if we have enough data for optimization
                if self._has_sufficient_data():
                    # Perform portfolio optimization
                    update_result = self._perform_optimization()
                    
                    if update_result and update_result.success:
                        # Update portfolio weights
                        self.current_weights = update_result.new_weights
                        self.last_optimization_time = update_result.timestamp
                        
                        # Track performance
                        self._update_performance_metrics(update_result)
                        
                        # Store in history
                        self.optimization_history.append(update_result)
                        
                        # Notify callbacks
                        for callback in self.update_callbacks:
                            try:
                                callback(update_result)
                            except Exception as e:
                                self.logger.error(f"Update callback error: {e}")
                        
                        self.logger.info(f"Portfolio updated - Sharpe: {update_result.sharpe_ratio:.3f}, "
                                       f"Turnover: {update_result.turnover:.3f}")
                
                # Sleep before next iteration
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for optimization"""
        # Need current prices for all assets
        if len(self.current_prices) < len(self.config.tickers):
            return False
        
        # Need some return history
        min_history_length = 5
        for symbol in self.config.tickers:
            if len(self.returns_history[symbol]) < min_history_length:
                return False
        
        return True
    
    def _perform_optimization(self) -> Optional[PortfolioUpdate]:
        """Perform portfolio optimization with current data"""
        start_time = time.time()
        
        try:
            # Prepare data for optimization
            returns_data, covariance_data = self._prepare_optimization_data()
            
            if returns_data is None or covariance_data is None:
                return None
            
            # Store old weights
            old_weights = self.current_weights.copy()
            
            # Update optimizer data
            self.optimizer.returns_data = returns_data
            
            # Choose optimization method based on market conditions
            optimization_method = self._select_optimization_method()
            
            # Run optimization
            if optimization_method == 'quantum':
                results = self.optimizer.quantum_optimization()
            elif optimization_method == 'classical':
                results = self.optimizer.classical_optimization(['max_sharpe'])
            else:
                results = self.optimizer.ensemble_optimization(['classical', 'quantum'])
            
            # Extract best result
            if isinstance(results, dict):
                if 'max_sharpe' in results:
                    best_result = results['max_sharpe']
                elif 'quantum' in results:
                    best_result = results['quantum']
                else:
                    best_result = list(results.values())[0]
            else:
                best_result = results
            
            new_weights = best_result.get('weights', old_weights)
            
            # Calculate turnover and transaction costs
            turnover = np.sum(np.abs(new_weights - old_weights))
            transaction_costs = self._calculate_transaction_costs(old_weights, new_weights)
            
            # Create update result
            execution_time = time.time() - start_time
            
            update_result = PortfolioUpdate(
                timestamp=datetime.now(),
                old_weights=old_weights,
                new_weights=new_weights,
                expected_return=best_result.get('expected_return', 0.0),
                volatility=best_result.get('volatility', 0.0),
                sharpe_ratio=best_result.get('sharpe_ratio', 0.0),
                turnover=turnover,
                transaction_costs=transaction_costs,
                optimization_method=optimization_method,
                execution_time=execution_time,
                success=True
            )
            
            return update_result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            
            execution_time = time.time() - start_time
            
            return PortfolioUpdate(
                timestamp=datetime.now(),
                old_weights=self.current_weights,
                new_weights=self.current_weights,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                turnover=0.0,
                transaction_costs=0.0,
                optimization_method='failed',
                execution_time=execution_time,
                success=False
            )
    
    def _prepare_optimization_data(self) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
        """Prepare returns and covariance data for optimization"""
        try:
            # Collect returns data
            returns_dict = {}
            min_length = float('inf')
            
            for symbol in self.config.tickers:
                if symbol in self.returns_history and len(self.returns_history[symbol]) > 0:
                    returns = [r['return'] for r in self.returns_history[symbol]]
                    returns_dict[symbol] = returns
                    min_length = min(min_length, len(returns))
                else:
                    return None, None
            
            if min_length < 5:  # Need at least 5 observations
                return None, None
            
            # Truncate all series to same length
            for symbol in returns_dict:
                returns_dict[symbol] = returns_dict[symbol][-min_length:]
            
            # Create DataFrame
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate covariance matrix
            covariance_matrix = returns_df.cov().values
            
            # Ensure positive definite
            eigenvals = np.linalg.eigvals(covariance_matrix)
            if np.min(eigenvals) <= 0:
                # Add small diagonal term to make positive definite
                covariance_matrix += np.eye(len(self.config.tickers)) * 1e-6
            
            return returns_df, covariance_matrix
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None, None
    
    def _select_optimization_method(self) -> str:
        """Select optimization method based on market conditions"""
        # Simple heuristic - could be made more sophisticated
        
        # Check market volatility
        avg_volatility = self._calculate_average_volatility()
        
        if avg_volatility > 0.3:  # High volatility
            return 'quantum'  # Quantum methods may handle uncertainty better
        elif avg_volatility > 0.15:  # Medium volatility
            return 'ensemble'  # Use ensemble for robustness
        else:  # Low volatility
            return 'classical'  # Classical methods sufficient
    
    def _calculate_average_volatility(self) -> float:
        """Calculate average volatility across all assets"""
        volatilities = []
        
        for symbol in self.config.tickers:
            if len(self.returns_history[symbol]) >= 10:
                returns = [r['return'] for r in list(self.returns_history[symbol])[-10:]]
                vol = np.std(returns) * np.sqrt(252)  # Annualized
                volatilities.append(vol)
        
        return np.mean(volatilities) if volatilities else 0.2  # Default
    
    def _calculate_transaction_costs(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Calculate estimated transaction costs"""
        # Simple transaction cost model
        turnover = np.sum(np.abs(new_weights - old_weights))
        
        # Assume 0.1% transaction cost (10 basis points)
        transaction_cost_rate = 0.001
        
        return turnover * transaction_cost_rate * self.config.budget
    
    def _update_performance_metrics(self, update_result: PortfolioUpdate):
        """Update performance tracking metrics"""
        metrics = self.performance_metrics
        
        metrics['total_updates'] += 1
        
        if update_result.success:
            metrics['successful_updates'] += 1
        
        # Update averages
        n = metrics['total_updates']
        metrics['avg_execution_time'] = ((n - 1) * metrics['avg_execution_time'] + update_result.execution_time) / n
        metrics['avg_turnover'] = ((n - 1) * metrics['avg_turnover'] + update_result.turnover) / n
        metrics['total_transaction_costs'] += update_result.transaction_costs
    
    def add_rebalance_callback(self, callback: Callable[[RebalanceSignal], None]):
        """Add callback for rebalancing signals"""
        self.rebalance_callbacks.append(callback)
    
    def add_update_callback(self, callback: Callable[[PortfolioUpdate], None]):
        """Add callback for portfolio updates"""
        self.update_callbacks.append(callback)
    
    def get_current_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        portfolio_value = 0.0
        positions = {}
        
        for i, symbol in enumerate(self.config.tickers):
            weight = self.current_weights[i]
            price = self.current_prices.get(symbol, 0.0)
            position_value = weight * self.config.budget
            
            positions[symbol] = {
                'weight': weight,
                'value': position_value,
                'shares': position_value / price if price > 0 else 0,
                'current_price': price
            }
            
            portfolio_value += position_value
        
        return {
            'timestamp': datetime.now(),
            'total_value': portfolio_value,
            'positions': positions,
            'last_optimization': self.last_optimization_time,
            'performance_metrics': self.performance_metrics
        }
    
    def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        recent_history = self.optimization_history[-limit:] if limit > 0 else self.optimization_history
        
        return [asdict(update) for update in recent_history]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.optimization_history:
            return {'status': 'no_data'}
        
        # Calculate performance metrics
        sharpe_ratios = [u.sharpe_ratio for u in self.optimization_history if u.success]
        turnovers = [u.turnover for u in self.optimization_history if u.success]
        execution_times = [u.execution_time for u in self.optimization_history if u.success]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': len(sharpe_ratios),
            'success_rate': len(sharpe_ratios) / len(self.optimization_history) if self.optimization_history else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_turnover': np.mean(turnovers) if turnovers else 0,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'total_transaction_costs': self.performance_metrics['total_transaction_costs'],
            'current_weights': self.current_weights.tolist(),
            'last_update': self.last_optimization_time.isoformat() if self.last_optimization_time else None
        }
    
    def force_rebalance(self, reason: str = "manual"):
        """Force immediate portfolio rebalancing"""
        self.logger.info(f"Forcing rebalance: {reason}")
        
        # Create manual rebalance signal
        signal = RebalanceSignal(
            trigger_type='manual',
            timestamp=datetime.now(),
            affected_symbols=self.config.tickers,
            severity=1.0,
            metadata={'reason': reason}
        )
        
        # Notify callbacks
        for callback in self.rebalance_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.logger.error(f"Rebalance callback error: {e}")
        
        # Reset last optimization time to trigger immediate rebalance
        self.last_optimization_time = datetime.now() - timedelta(hours=1)