#!/usr/bin/env python3
"""
QuantumFinanceOpt Advanced - Next-Generation Portfolio Optimizer

Main entry point for the advanced quantum finance optimization system with
real-time data, quantum computing, AI models, and institutional features.
"""

import sys
import os
import argparse
import logging
import asyncio
from typing import List, Dict, Any
import traceback
from datetime import datetime

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_finance_opt.core.config import OptimizationConfig
from quantum_finance_opt.core.optimizer import QuantumFinanceOptimizer
from quantum_finance_opt.core.exceptions import QuantumFinanceOptError

# Advanced modules
from quantum_finance_opt.quantum.backend_manager import QuantumBackendManager
from quantum_finance_opt.quantum.qaoa_optimizer import QAOAPortfolioOptimizer
from quantum_finance_opt.quantum.vqe_risk_model import VQERiskModel
from quantum_finance_opt.quantum.quantum_annealing import QuantumAnnealingOptimizer

from quantum_finance_opt.realtime.data_stream_manager import DataStreamManager, StreamConfig
from quantum_finance_opt.realtime.streaming_optimizer import StreamingOptimizer

from quantum_finance_opt.models.transformer_forecasting import TransformerForecastingService
from quantum_finance_opt.hpc.gpu_accelerator import GPUAccelerator, SimulationParams

from quantum_finance_opt.dashboard.dashboard_app import DashboardApp


class AdvancedQuantumFinanceOptimizer:
    """
    Advanced quantum finance optimizer with all cutting-edge features
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.base_optimizer = None
        self.quantum_backend = None
        self.gpu_accelerator = None
        
        # Quantum optimizers
        self.qaoa_optimizer = None
        self.vqe_risk_model = None
        self.quantum_annealer = None
        
        # Real-time components
        self.data_stream_manager = None
        self.streaming_optimizer = None
        
        # AI/ML components
        self.transformer_service = None
        
        # Dashboard
        self.dashboard = None
        
        # Results storage
        self.optimization_results = {}
        self.performance_metrics = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        self.logger.info("Initializing QuantumFinanceOpt Advanced System")
        
        try:
            # Core optimizer
            self.base_optimizer = QuantumFinanceOptimizer(self.config)
            self.logger.info("✓ Core optimizer initialized")
            
            # Quantum backend
            self.quantum_backend = QuantumBackendManager(preferred_backend="qiskit_simulator")
            self.logger.info("✓ Quantum backend initialized")
            
            # GPU accelerator
            try:
                self.gpu_accelerator = GPUAccelerator()
                gpu_info = self.gpu_accelerator.get_gpu_info()
                if gpu_info.get('cupy_available') or gpu_info.get('torch_available'):
                    self.logger.info("✓ GPU acceleration available")
                else:
                    self.logger.info("⚠ GPU acceleration not available, using CPU")
            except Exception as e:
                self.logger.warning(f"GPU initialization failed: {e}")
            
            # Quantum optimizers
            if self.quantum_backend.is_quantum_available():
                self.qaoa_optimizer = QAOAPortfolioOptimizer(self.quantum_backend)
                self.vqe_risk_model = VQERiskModel(self.quantum_backend)
                self.quantum_annealer = QuantumAnnealingOptimizer(self.quantum_backend)
                self.logger.info("✓ Quantum optimizers initialized")
            else:
                self.logger.info("⚠ Quantum computing not available, using classical fallbacks")
            
            # Real-time components
            self.data_stream_manager = DataStreamManager()
            self.streaming_optimizer = StreamingOptimizer(
                config=self.config,
                data_stream_manager=self.data_stream_manager
            )
            self.logger.info("✓ Real-time components initialized")
            
            # AI/ML components
            transformer_config = {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6
            }
            self.transformer_service = TransformerForecastingService(transformer_config)
            self.logger.info("✓ AI/ML components initialized")
            
            # Dashboard
            self.dashboard = DashboardApp(self.config)
            self.logger.info("✓ Dashboard initialized")
            
            self.logger.info("🚀 All components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def run_comprehensive_optimization(self, methods: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive optimization using all available methods
        """
        
        if methods is None:
            methods = ['classical', 'quantum', 'ai', 'ensemble']
        
        self.logger.info(f"Running comprehensive optimization with methods: {methods}")
        
        results = {}
        
        # Load data
        self.base_optimizer.load_data()
        
        # Classical optimization
        if 'classical' in methods:
            self.logger.info("Running classical optimization...")
            classical_results = self.base_optimizer.classical_optimization([
                'mean_variance', 'max_sharpe', 'min_volatility', 'hrp'
            ])
            results['classical'] = classical_results
        
        # Quantum optimization
        if 'quantum' in methods and self.quantum_backend.is_quantum_available():
            self.logger.info("Running quantum optimization...")
            
            # QAOA optimization
            if self.qaoa_optimizer:
                try:
                    qaoa_result = self.qaoa_optimizer.optimize_portfolio(
                        returns=self.base_optimizer.returns_data.mean().values,
                        covariance=self.base_optimizer.returns_data.cov().values
                    )
                    results['qaoa'] = {
                        'weights': qaoa_result.weights,
                        'expected_return': qaoa_result.expected_return,
                        'volatility': qaoa_result.volatility,
                        'sharpe_ratio': qaoa_result.sharpe_ratio,
                        'quantum_advantage': qaoa_result.quantum_advantage
                    }
                except Exception as e:
                    self.logger.error(f"QAOA optimization failed: {e}")
            
            # VQE risk analysis
            if self.vqe_risk_model:
                try:
                    vqe_result = self.vqe_risk_model.analyze_portfolio_risk(
                        correlation_matrix=self.base_optimizer.returns_data.corr().values
                    )
                    results['vqe_risk'] = {
                        'risk_entropy': vqe_result.risk_entropy,
                        'correlation_strength': vqe_result.correlation_strength,
                        'diversification_measure': vqe_result.diversification_measure,
                        'quantum_coherence': vqe_result.quantum_coherence
                    }
                except Exception as e:
                    self.logger.error(f"VQE risk analysis failed: {e}")
            
            # Quantum annealing
            if self.quantum_annealer:
                try:
                    annealing_result = self.quantum_annealer.optimize_portfolio(
                        returns=self.base_optimizer.returns_data.mean().values,
                        covariance=self.base_optimizer.returns_data.cov().values
                    )
                    results['quantum_annealing'] = {
                        'weights': annealing_result.weights,
                        'expected_return': annealing_result.expected_return,
                        'volatility': annealing_result.volatility,
                        'sharpe_ratio': annealing_result.sharpe_ratio,
                        'energy': annealing_result.energy
                    }
                except Exception as e:
                    self.logger.error(f"Quantum annealing failed: {e}")
        
        # AI/ML optimization
        if 'ai' in methods:
            self.logger.info("Running AI/ML optimization...")
            
            try:
                # Prepare features for transformer
                features_df = self.transformer_service.prepare_features(
                    self.base_optimizer.price_data
                )
                
                # Train transformer model (simplified for demo)
                if len(features_df) > 100:  # Need sufficient data
                    training_result = self.transformer_service.train_multi_horizon_model(
                        features_df=features_df,
                        target_symbols=self.config.tickers[:2],  # Limit for demo
                        epochs=10  # Quick training
                    )
                    
                    # Generate forecasts
                    forecast_result = self.transformer_service.forecast_returns(
                        recent_data=self.base_optimizer.price_data.tail(60),
                        symbols=self.config.tickers
                    )
                    
                    results['transformer'] = {
                        'training_loss': training_result.get('final_loss', 0),
                        'predictions': forecast_result.predictions.tolist(),
                        'model_confidence': forecast_result.model_confidence
                    }
                
            except Exception as e:
                self.logger.error(f"AI/ML optimization failed: {e}")
        
        # GPU-accelerated Monte Carlo
        if self.gpu_accelerator and 'monte_carlo' in methods:
            self.logger.info("Running GPU-accelerated Monte Carlo simulation...")
            
            try:
                sim_params = SimulationParams(
                    num_simulations=10000,
                    num_assets=len(self.config.tickers),
                    time_steps=252,  # 1 year
                    initial_prices=self.base_optimizer.price_data.iloc[-1].values,
                    returns=self.base_optimizer.returns_data.mean().values,
                    covariance=self.base_optimizer.returns_data.cov().values
                )
                
                mc_results = self.gpu_accelerator.gpu_monte_carlo_simulation(sim_params)
                
                results['monte_carlo'] = {
                    'var_95': mc_results['var_95'],
                    'var_99': mc_results['var_99'],
                    'expected_return': mc_results['expected_return'],
                    'volatility': mc_results['volatility']
                }
                
            except Exception as e:
                self.logger.error(f"Monte Carlo simulation failed: {e}")
        
        # Ensemble optimization
        if 'ensemble' in methods and len(results) > 1:
            self.logger.info("Running ensemble optimization...")
            
            try:
                ensemble_result = self.base_optimizer.ensemble_optimization(['classical', 'quantum'])
                results['ensemble'] = ensemble_result
            except Exception as e:
                self.logger.error(f"Ensemble optimization failed: {e}")
        
        self.optimization_results = results
        return results
    
    def start_real_time_optimization(self):
        """Start real-time optimization with streaming data"""
        
        self.logger.info("Starting real-time optimization...")
        
        try:
            # Setup data streams
            stream_config = StreamConfig(
                provider="mock",
                symbols=self.config.tickers,
                data_types=["price", "volume"],
                update_frequency="5s"
            )
            
            self.data_stream_manager.add_stream("main_stream", stream_config)
            self.data_stream_manager.start_stream("main_stream")
            
            # Start streaming optimization
            self.streaming_optimizer.start_streaming_optimization()
            
            self.logger.info("✓ Real-time optimization started")
            
        except Exception as e:
            self.logger.error(f"Real-time optimization startup failed: {e}")
    
    def launch_dashboard(self, dashboard_type: str = "streamlit", port: int = 8501):
        """Launch interactive dashboard"""
        
        self.logger.info(f"Launching {dashboard_type} dashboard on port {port}")
        
        try:
            if dashboard_type == "streamlit":
                self.dashboard.run_streamlit_dashboard(port)
            elif dashboard_type == "dash":
                self.dashboard.run_dash_dashboard(port)
            else:
                raise ValueError(f"Unsupported dashboard type: {dashboard_type}")
                
        except Exception as e:
            self.logger.error(f"Dashboard launch failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Quantum backend status
        if self.quantum_backend:
            status['components']['quantum'] = self.quantum_backend.get_backend_info()
        
        # GPU status
        if self.gpu_accelerator:
            status['components']['gpu'] = self.gpu_accelerator.get_gpu_info()
        
        # Streaming status
        if self.data_stream_manager:
            status['components']['streaming'] = self.data_stream_manager.get_stream_status()
        
        # Performance metrics
        if self.streaming_optimizer:
            status['components']['performance'] = self.streaming_optimizer.get_performance_summary()
        
        return status
    
    def cleanup(self):
        """Cleanup all system resources"""
        
        self.logger.info("Cleaning up system resources...")
        
        try:
            if self.streaming_optimizer:
                self.streaming_optimizer.stop_streaming_optimization()
            
            if self.data_stream_manager:
                self.data_stream_manager.stop_all_streams()
            
            if self.gpu_accelerator:
                self.gpu_accelerator.cleanup_gpu_memory()
            
            if self.dashboard:
                self.dashboard.cleanup()
            
            self.logger.info("✓ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description='QuantumFinanceOpt Advanced - Next-Generation Portfolio Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive optimization
  python advanced_main.py --mode optimize --methods classical quantum ai
  
  # Start real-time optimization
  python advanced_main.py --mode realtime --tickers AAPL GOOGL MSFT
  
  # Launch dashboard
  python advanced_main.py --mode dashboard --dashboard-type streamlit --port 8501
  
  # Full system with all features
  python advanced_main.py --mode full --methods classical quantum ai ensemble monte_carlo
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str, 
        choices=['optimize', 'realtime', 'dashboard', 'full'],
        default='optimize',
        help='Operation mode (default: optimize)'
    )
    
    # Data options
    parser.add_argument('--csv-path', type=str, help='Path to CSV data file')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    parser.add_argument('--budget', type=float, default=100000.0)
    
    # Optimization options
    parser.add_argument(
        '--methods', nargs='+',
        choices=['classical', 'quantum', 'ai', 'ensemble', 'monte_carlo'],
        default=['classical', 'quantum'],
        help='Optimization methods to use'
    )
    
    # Dashboard options
    parser.add_argument('--dashboard-type', choices=['streamlit', 'dash'], default='streamlit')
    parser.add_argument('--port', type=int, default=8501)
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--output-dir', type=str, default='advanced_output')
    
    return parser


def main():
    """Main entry point"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = OptimizationConfig(
            tickers=args.tickers,
            budget=args.budget,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            optimization_methods=args.methods
        )
        
        # Initialize advanced optimizer
        logger.info("🚀 Starting QuantumFinanceOpt Advanced System")
        optimizer = AdvancedQuantumFinanceOptimizer(config)
        
        # Execute based on mode
        if args.mode == 'optimize':
            logger.info("Running comprehensive optimization...")
            results = optimizer.run_comprehensive_optimization(args.methods)
            
            # Print results summary
            print("\n" + "="*80)
            print("QUANTUMFINANCEOPT ADVANCED - OPTIMIZATION RESULTS")
            print("="*80)
            
            for method, result in results.items():
                print(f"\n{method.upper()}:")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
        
        elif args.mode == 'realtime':
            logger.info("Starting real-time optimization...")
            optimizer.start_real_time_optimization()
            
            # Keep running
            try:
                while True:
                    status = optimizer.get_system_status()
                    logger.info(f"System running... Components: {len(status['components'])}")
                    import time
                    time.sleep(30)
            except KeyboardInterrupt:
                logger.info("Stopping real-time optimization...")
        
        elif args.mode == 'dashboard':
            logger.info("Launching dashboard...")
            optimizer.launch_dashboard(args.dashboard_type, args.port)
        
        elif args.mode == 'full':
            logger.info("Running full system with all features...")
            
            # Start real-time optimization
            optimizer.start_real_time_optimization()
            
            # Run comprehensive optimization
            results = optimizer.run_comprehensive_optimization(args.methods)
            
            # Launch dashboard in background
            import threading
            dashboard_thread = threading.Thread(
                target=optimizer.launch_dashboard,
                args=(args.dashboard_type, args.port)
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            
            # Print system status
            status = optimizer.get_system_status()
            print("\n" + "="*80)
            print("SYSTEM STATUS")
            print("="*80)
            for component, info in status['components'].items():
                print(f"{component.upper()}: {info}")
            
            print(f"\nDashboard available at: http://localhost:{args.port}")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    import time
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
        
        # Cleanup
        optimizer.cleanup()
        logger.info("✓ QuantumFinanceOpt Advanced completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()