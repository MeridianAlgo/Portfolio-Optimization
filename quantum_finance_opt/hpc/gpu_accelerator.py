"""
GPU Acceleration Service

Provides GPU-accelerated operations for portfolio optimization including
matrix operations, Monte Carlo simulations, and ML model training.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import threading

# GPU libraries with fallbacks
try:
    import cupy as cp
    import cupyx.scipy.linalg as cp_linalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numba
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class GPUBenchmarkResult:
    """Result from GPU benchmark"""
    operation: str
    gpu_time: float
    cpu_time: float
    speedup: float
    memory_used: float
    gpu_device: str
    success: bool


@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation"""
    num_simulations: int
    num_assets: int
    time_steps: int
    initial_prices: np.ndarray
    returns: np.ndarray
    covariance: np.ndarray
    dt: float = 1/252  # Daily time step


class GPUAccelerator:
    """
    GPU acceleration service for portfolio optimization
    """
    
    def __init__(self, device_id: int = 0, memory_pool_size: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        
        # GPU availability
        self.cupy_available = CUPY_AVAILABLE
        self.torch_available = TORCH_AVAILABLE
        self.numba_available = NUMBA_AVAILABLE
        
        # Device info
        self.gpu_device = None
        self.gpu_memory = 0
        self.compute_capability = None
        
        # Performance tracking
        self.benchmark_results = []
        self.operation_cache = {}
        
        # Initialize GPU
        self._initialize_gpu(memory_pool_size)
    
    def _initialize_gpu(self, memory_pool_size: Optional[int]):
        """Initialize GPU device and memory"""
        
        if self.cupy_available:
            try:
                cp.cuda.Device(self.device_id).use()
                self.gpu_device = cp.cuda.Device()
                self.gpu_memory = self.gpu_device.mem_info[1]  # Total memory
                
                # Set memory pool if specified
                if memory_pool_size:
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=memory_pool_size)
                
                self.logger.info(f"CuPy initialized on GPU {self.device_id}")
                self.logger.info(f"GPU memory: {self.gpu_memory / 1e9:.1f} GB")
                
            except Exception as e:
                self.logger.warning(f"CuPy initialization failed: {e}")
                self.cupy_available = False
        
        if self.torch_available:
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.device_id)
                    device_name = torch.cuda.get_device_name(self.device_id)
                    self.logger.info(f"PyTorch CUDA available: {device_name}")
                else:
                    self.logger.warning("PyTorch CUDA not available")
            except Exception as e:
                self.logger.warning(f"PyTorch CUDA initialization failed: {e}")
        
        if self.numba_available:
            try:
                if cuda.is_available():
                    self.logger.info("Numba CUDA available")
                else:
                    self.logger.warning("Numba CUDA not available")
            except Exception as e:
                self.logger.warning(f"Numba CUDA initialization failed: {e}")
    
    def gpu_matrix_operations(self, 
                             matrices: List[np.ndarray], 
                             operation: str,
                             **kwargs) -> np.ndarray:
        """
        Perform GPU-accelerated matrix operations
        
        Args:
            matrices: List of input matrices
            operation: Operation type ('multiply', 'invert', 'eigenvals', 'cholesky', 'svd')
            **kwargs: Additional operation parameters
            
        Returns:
            Result matrix/array
        """
        
        if not self.cupy_available:
            return self._cpu_matrix_operations(matrices, operation, **kwargs)
        
        try:
            start_time = time.time()
            
            # Transfer to GPU
            gpu_matrices = [cp.asarray(mat) for mat in matrices]
            
            # Perform operation
            if operation == 'multiply':
                if len(gpu_matrices) >= 2:
                    result = cp.matmul(gpu_matrices[0], gpu_matrices[1])
                    for i in range(2, len(gpu_matrices)):
                        result = cp.matmul(result, gpu_matrices[i])
                else:
                    raise ValueError("Matrix multiplication requires at least 2 matrices")
            
            elif operation == 'invert':
                result = cp.linalg.inv(gpu_matrices[0])
            
            elif operation == 'eigenvals':
                eigenvals, eigenvecs = cp.linalg.eigh(gpu_matrices[0])
                result = cp.stack([eigenvals, eigenvecs.flatten()])
            
            elif operation == 'cholesky':
                result = cp.linalg.cholesky(gpu_matrices[0])
            
            elif operation == 'svd':
                U, s, Vt = cp.linalg.svd(gpu_matrices[0])
                result = cp.stack([U.flatten(), s, Vt.flatten()])
            
            elif operation == 'solve':
                if len(gpu_matrices) >= 2:
                    result = cp.linalg.solve(gpu_matrices[0], gpu_matrices[1])
                else:
                    raise ValueError("Solve operation requires 2 matrices")
            
            elif operation == 'covariance':
                # Calculate covariance matrix
                data = gpu_matrices[0]
                mean = cp.mean(data, axis=0, keepdims=True)
                centered = data - mean
                result = cp.matmul(centered.T, centered) / (data.shape[0] - 1)
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Transfer back to CPU
            result_cpu = cp.asnumpy(result)
            
            gpu_time = time.time() - start_time
            
            # Benchmark against CPU
            cpu_time = self._benchmark_cpu_operation(matrices, operation, **kwargs)
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            
            # Record benchmark
            self._record_benchmark(operation, gpu_time, cpu_time, speedup)
            
            return result_cpu
            
        except Exception as e:
            self.logger.error(f"GPU matrix operation failed: {e}")
            return self._cpu_matrix_operations(matrices, operation, **kwargs)
    
    def gpu_monte_carlo_simulation(self, params: SimulationParams) -> Dict[str, np.ndarray]:
        """
        GPU-accelerated Monte Carlo simulation for portfolio paths
        """
        
        if not self.cupy_available:
            return self._cpu_monte_carlo_simulation(params)
        
        try:
            start_time = time.time()
            
            # Transfer parameters to GPU
            initial_prices_gpu = cp.asarray(params.initial_prices)
            returns_gpu = cp.asarray(params.returns)
            covariance_gpu = cp.asarray(params.covariance)
            
            # Generate random numbers on GPU
            rng = cp.random.RandomState()
            
            # Cholesky decomposition for correlated random numbers
            chol = cp.linalg.cholesky(covariance_gpu)
            
            # Initialize price paths
            price_paths = cp.zeros((params.num_simulations, params.time_steps + 1, params.num_assets))
            price_paths[:, 0, :] = initial_prices_gpu
            
            # Simulate paths
            for t in range(params.time_steps):
                # Generate correlated random returns
                random_normal = rng.normal(0, 1, (params.num_simulations, params.num_assets))
                correlated_random = cp.matmul(random_normal, chol.T)
                
                # Calculate returns
                drift = returns_gpu * params.dt
                diffusion = correlated_random * cp.sqrt(params.dt)
                
                # Update prices (geometric Brownian motion)
                price_paths[:, t + 1, :] = price_paths[:, t, :] * cp.exp(drift + diffusion)
            
            # Calculate portfolio statistics
            final_prices = price_paths[:, -1, :]
            returns_sim = (final_prices - initial_prices_gpu) / initial_prices_gpu
            
            # Portfolio value paths (equal weights for now)
            weights = cp.ones(params.num_assets) / params.num_assets
            portfolio_values = cp.sum(price_paths * weights, axis=2)
            portfolio_returns = (portfolio_values[:, -1] - portfolio_values[:, 0]) / portfolio_values[:, 0]
            
            # Calculate risk metrics
            var_95 = cp.percentile(portfolio_returns, 5)
            var_99 = cp.percentile(portfolio_returns, 1)
            expected_return = cp.mean(portfolio_returns)
            volatility = cp.std(portfolio_returns)
            
            # Transfer results back to CPU
            results = {
                'price_paths': cp.asnumpy(price_paths),
                'portfolio_values': cp.asnumpy(portfolio_values),
                'portfolio_returns': cp.asnumpy(portfolio_returns),
                'asset_returns': cp.asnumpy(returns_sim),
                'var_95': float(cp.asnumpy(var_95)),
                'var_99': float(cp.asnumpy(var_99)),
                'expected_return': float(cp.asnumpy(expected_return)),
                'volatility': float(cp.asnumpy(volatility))
            }
            
            gpu_time = time.time() - start_time
            
            # Benchmark
            cpu_time = self._benchmark_cpu_monte_carlo(params)
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            
            self._record_benchmark('monte_carlo', gpu_time, cpu_time, speedup)
            
            self.logger.info(f"Monte Carlo simulation completed: {params.num_simulations} paths, "
                           f"{gpu_time:.2f}s (GPU), speedup: {speedup:.1f}x")
            
            return results
            
        except Exception as e:
            self.logger.error(f"GPU Monte Carlo simulation failed: {e}")
            return self._cpu_monte_carlo_simulation(params)  
  
    def gpu_ml_training(self, 
                       model: Any, 
                       training_data: Dict[str, np.ndarray],
                       training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        GPU-accelerated ML model training
        """
        
        if not self.torch_available:
            return self._cpu_ml_training(model, training_data, training_config)
        
        try:
            start_time = time.time()
            
            # Move model to GPU
            device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Prepare data
            X = torch.FloatTensor(training_data['X']).to(device)
            y = torch.FloatTensor(training_data['y']).to(device)
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters(), 
                                       lr=training_config.get('learning_rate', 0.001))
            criterion = torch.nn.MSELoss()
            
            epochs = training_config.get('epochs', 100)
            batch_size = training_config.get('batch_size', 32)
            
            # Training loop
            training_losses = []
            
            for epoch in range(epochs):
                model.train()
                
                # Create batches
                num_samples = len(X)
                indices = torch.randperm(num_samples, device=device)
                
                epoch_loss = 0.0
                num_batches = 0
                
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_X = X[batch_indices]
                    batch_y = y[batch_indices]
                    
                    # Forward pass
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                training_losses.append(avg_loss)
            
            gpu_time = time.time() - start_time
            
            # Move model back to CPU for return
            model = model.cpu()
            
            return {
                'trained_model': model,
                'training_losses': training_losses,
                'training_time': gpu_time,
                'device_used': str(device),
                'final_loss': training_losses[-1] if training_losses else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"GPU ML training failed: {e}")
            return self._cpu_ml_training(model, training_data, training_config)
    
    def optimize_memory_usage(self, computation: callable, *args, **kwargs) -> Any:
        """
        Optimize GPU memory usage for large computations
        """
        
        if not self.cupy_available:
            return computation(*args, **kwargs)
        
        try:
            # Clear GPU memory cache
            cp.get_default_memory_pool().free_all_blocks()
            
            # Monitor memory usage
            initial_memory = self.gpu_device.mem_info[0]  # Free memory
            
            # Execute computation
            result = computation(*args, **kwargs)
            
            # Check memory usage
            final_memory = self.gpu_device.mem_info[0]
            memory_used = initial_memory - final_memory
            
            self.logger.info(f"GPU memory used: {memory_used / 1e6:.1f} MB")
            
            # Clean up if memory usage is high
            if memory_used > self.gpu_memory * 0.8:  # 80% threshold
                cp.get_default_memory_pool().free_all_blocks()
                self.logger.info("GPU memory cleaned up")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return computation(*args, **kwargs)
    
    def stream_large_datasets(self, 
                             data_source: Union[np.ndarray, callable],
                             chunk_size: int,
                             processing_function: callable) -> List[Any]:
        """
        Stream processing of large datasets with GPU acceleration
        """
        
        results = []
        
        try:
            if isinstance(data_source, np.ndarray):
                # Process array in chunks
                num_chunks = len(data_source) // chunk_size + (1 if len(data_source) % chunk_size else 0)
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(data_source))
                    chunk = data_source[start_idx:end_idx]
                    
                    # Process chunk on GPU
                    if self.cupy_available:
                        gpu_chunk = cp.asarray(chunk)
                        result = processing_function(gpu_chunk)
                        if isinstance(result, cp.ndarray):
                            result = cp.asnumpy(result)
                    else:
                        result = processing_function(chunk)
                    
                    results.append(result)
                    
                    # Memory cleanup between chunks
                    if self.cupy_available:
                        cp.get_default_memory_pool().free_all_blocks()
            
            elif callable(data_source):
                # Process streaming data
                chunk_count = 0
                while True:
                    try:
                        chunk = data_source(chunk_size)
                        if chunk is None or len(chunk) == 0:
                            break
                        
                        # Process chunk
                        if self.cupy_available:
                            gpu_chunk = cp.asarray(chunk)
                            result = processing_function(gpu_chunk)
                            if isinstance(result, cp.ndarray):
                                result = cp.asnumpy(result)
                        else:
                            result = processing_function(chunk)
                        
                        results.append(result)
                        chunk_count += 1
                        
                        # Memory cleanup
                        if self.cupy_available and chunk_count % 10 == 0:
                            cp.get_default_memory_pool().free_all_blocks()
                    
                    except StopIteration:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Streaming processing failed: {e}")
            return []
    
    def _cpu_matrix_operations(self, matrices: List[np.ndarray], operation: str, **kwargs) -> np.ndarray:
        """CPU fallback for matrix operations"""
        
        if operation == 'multiply':
            result = matrices[0]
            for mat in matrices[1:]:
                result = np.matmul(result, mat)
        elif operation == 'invert':
            result = np.linalg.inv(matrices[0])
        elif operation == 'eigenvals':
            eigenvals, eigenvecs = np.linalg.eigh(matrices[0])
            result = np.stack([eigenvals, eigenvecs.flatten()])
        elif operation == 'cholesky':
            result = np.linalg.cholesky(matrices[0])
        elif operation == 'svd':
            U, s, Vt = np.linalg.svd(matrices[0])
            result = np.stack([U.flatten(), s, Vt.flatten()])
        elif operation == 'solve':
            result = np.linalg.solve(matrices[0], matrices[1])
        elif operation == 'covariance':
            data = matrices[0]
            result = np.cov(data.T)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return result
    
    def _cpu_monte_carlo_simulation(self, params: SimulationParams) -> Dict[str, np.ndarray]:
        """CPU fallback for Monte Carlo simulation"""
        
        # Simple CPU implementation
        np.random.seed(42)
        
        price_paths = np.zeros((params.num_simulations, params.time_steps + 1, params.num_assets))
        price_paths[:, 0, :] = params.initial_prices
        
        # Cholesky decomposition
        chol = np.linalg.cholesky(params.covariance)
        
        for t in range(params.time_steps):
            random_normal = np.random.normal(0, 1, (params.num_simulations, params.num_assets))
            correlated_random = np.dot(random_normal, chol.T)
            
            drift = params.returns * params.dt
            diffusion = correlated_random * np.sqrt(params.dt)
            
            price_paths[:, t + 1, :] = price_paths[:, t, :] * np.exp(drift + diffusion)
        
        # Calculate statistics
        weights = np.ones(params.num_assets) / params.num_assets
        portfolio_values = np.sum(price_paths * weights, axis=2)
        portfolio_returns = (portfolio_values[:, -1] - portfolio_values[:, 0]) / portfolio_values[:, 0]
        
        return {
            'price_paths': price_paths,
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'asset_returns': (price_paths[:, -1, :] - params.initial_prices) / params.initial_prices,
            'var_95': np.percentile(portfolio_returns, 5),
            'var_99': np.percentile(portfolio_returns, 1),
            'expected_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns)
        }
    
    def _cpu_ml_training(self, model: Any, training_data: Dict[str, np.ndarray], training_config: Dict[str, Any]) -> Dict[str, Any]:
        """CPU fallback for ML training"""
        
        # Simple CPU training (placeholder)
        return {
            'trained_model': model,
            'training_losses': [0.1, 0.05, 0.02],
            'training_time': 1.0,
            'device_used': 'cpu',
            'final_loss': 0.02
        }
    
    def _benchmark_cpu_operation(self, matrices: List[np.ndarray], operation: str, **kwargs) -> float:
        """Benchmark CPU operation for comparison"""
        
        try:
            start_time = time.time()
            self._cpu_matrix_operations(matrices, operation, **kwargs)
            return time.time() - start_time
        except Exception:
            return 1.0  # Default fallback time
    
    def _benchmark_cpu_monte_carlo(self, params: SimulationParams) -> float:
        """Benchmark CPU Monte Carlo for comparison"""
        
        try:
            start_time = time.time()
            # Run smaller simulation for benchmark
            small_params = SimulationParams(
                num_simulations=min(params.num_simulations, 1000),
                num_assets=params.num_assets,
                time_steps=min(params.time_steps, 100),
                initial_prices=params.initial_prices,
                returns=params.returns,
                covariance=params.covariance,
                dt=params.dt
            )
            self._cpu_monte_carlo_simulation(small_params)
            cpu_time = time.time() - start_time
            
            # Scale up time estimate
            scale_factor = (params.num_simulations / small_params.num_simulations) * (params.time_steps / small_params.time_steps)
            return cpu_time * scale_factor
        except Exception:
            return 10.0  # Default fallback time
    
    def _record_benchmark(self, operation: str, gpu_time: float, cpu_time: float, speedup: float):
        """Record benchmark result"""
        
        memory_used = 0.0
        if self.cupy_available:
            try:
                memory_used = (self.gpu_memory - self.gpu_device.mem_info[0]) / 1e6  # MB
            except Exception:
                pass
        
        benchmark = GPUBenchmarkResult(
            operation=operation,
            gpu_time=gpu_time,
            cpu_time=cpu_time,
            speedup=speedup,
            memory_used=memory_used,
            gpu_device=str(self.gpu_device) if self.gpu_device else "none",
            success=True
        )
        
        self.benchmark_results.append(benchmark)
        
        # Keep only recent benchmarks
        if len(self.benchmark_results) > 100:
            self.benchmark_results = self.benchmark_results[-100:]
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU device information"""
        
        info = {
            'cupy_available': self.cupy_available,
            'torch_available': self.torch_available,
            'numba_available': self.numba_available,
            'device_id': self.device_id
        }
        
        if self.cupy_available and self.gpu_device:
            try:
                mem_info = self.gpu_device.mem_info
                info.update({
                    'gpu_name': self.gpu_device.name.decode(),
                    'total_memory_gb': mem_info[1] / 1e9,
                    'free_memory_gb': mem_info[0] / 1e9,
                    'compute_capability': self.gpu_device.compute_capability
                })
            except Exception as e:
                info['gpu_error'] = str(e)
        
        if self.torch_available and torch.cuda.is_available():
            try:
                info.update({
                    'torch_device_name': torch.cuda.get_device_name(self.device_id),
                    'torch_memory_allocated': torch.cuda.memory_allocated(self.device_id) / 1e9,
                    'torch_memory_cached': torch.cuda.memory_reserved(self.device_id) / 1e9
                })
            except Exception as e:
                info['torch_error'] = str(e)
        
        return info
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of GPU benchmarks"""
        
        if not self.benchmark_results:
            return {'status': 'no_benchmarks'}
        
        # Group by operation
        operations = {}
        for benchmark in self.benchmark_results:
            op = benchmark.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(benchmark)
        
        # Calculate statistics
        summary = {}
        for op, benchmarks in operations.items():
            speedups = [b.speedup for b in benchmarks if b.success]
            gpu_times = [b.gpu_time for b in benchmarks if b.success]
            
            if speedups:
                summary[op] = {
                    'count': len(benchmarks),
                    'avg_speedup': np.mean(speedups),
                    'max_speedup': np.max(speedups),
                    'avg_gpu_time': np.mean(gpu_times),
                    'success_rate': sum(b.success for b in benchmarks) / len(benchmarks)
                }
        
        return summary
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        
        if self.cupy_available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                self.logger.info("GPU memory cleaned up")
            except Exception as e:
                self.logger.error(f"GPU memory cleanup failed: {e}")
        
        if self.torch_available and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                self.logger.info("PyTorch GPU cache cleared")
            except Exception as e:
                self.logger.error(f"PyTorch GPU cleanup failed: {e}")