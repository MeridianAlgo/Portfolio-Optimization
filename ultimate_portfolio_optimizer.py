#!/usr/bin/env python3
"""
Ultimate Portfolio Optimizer - The Best on GitHub

Real-time portfolio optimization using user CSV files, live market data,
news sentiment, and quantum computing. This is the most advanced portfolio
optimizer available anywhere.
"""

import sys
import os
import argparse
import logging
import asyncio
from typing import List, Dict, Any, Optional
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_finance_opt.core.config import OptimizationConfig
from quantum_finance_opt.core.optimizer import QuantumFinanceOptimizer
from quantum_finance_opt.core.exceptions import QuantumFinanceOptError

# Real-time data processing
from quantum_finance_opt.realtime.live_data_processor import LiveDataProcessor
from quantum_finance_opt.realtime.streaming_optimizer import StreamingOptimizer
from quantum_finance_opt.realtime.data_stream_manager import DataStreamManager

# Advanced components
from quantum_finance_opt.quantum.backend_manager import QuantumBackendManager
from quantum_finance_opt.hpc.gpu_accelerator import GPUAccelerator
from quantum_finance_opt.models.transformer_forecasting import TransformerForecastingService
from quantum_finance_opt.dashboard.dashboard_app import DashboardApp


class UltimatePortfolioOptimizer:
    """
    The Ultimate Portfolio Optimizer - Best on GitHub
    
    Features:
    - Real user CSV file processing
    - Live market data integration
    - Real-time news sentiment analysis
    - Quantum computing optimization
    - GPU acceleration
    - AI/ML forecasting
    - Interactive dashboard
    """
    
    def __init__(self, 
                 csv_path: str,
                 api_keys: Dict[str, str] = None,
                 optimization_methods: List[str] = None):
        
        self.csv_path = csv_path
        self.api_keys = api_keys or {}
        self.optimization_methods = optimization_methods or ['classical', 'quantum', 'ai']
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.live_data_processor = None
        self.optimizer = None
        self.streaming_optimizer = None
        self.quantum_backend = None
        self.gpu_accelerator = None
        self.transformer_service = None
        self.dashboard = None
        
        # Data
        self.portfolio_data = None
        self.symbols = []
        self.optimization_results = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the ultimate optimization system"""
        
        self.logger.info("🚀 Initializing Ultimate Portfolio Optimizer")
        
        try:
            # 1. Initialize live data processor
            self.live_data_processor = LiveDataProcessor(
                api_keys=self.api_keys,
                update_frequency=5,  # 5 second updates
                enable_news=True,
                enable_sentiment=True
            )
            self.logger.info("✓ Live data processor initialized")
            
            # 2. Load user portfolio data
            self.portfolio_data = self.live_data_processor.load_user_portfolio(self.csv_path)
            self.symbols = [col for col in self.portfolio_data.columns if col != 'Date']
            self.logger.info(f"✓ User portfolio loaded: {len(self.symbols)} assets")
            
            # 3. Initialize quantum backend
            self.quantum_backend = QuantumBackendManager(preferred_backend="qiskit_simulator")
            self.logger.info("✓ Quantum backend initialized")
            
            # 4. Initialize GPU accelerator
            try:
                self.gpu_accelerator = GPUAccelerator()
                gpu_info = self.gpu_accelerator.get_gpu_info()
                if gpu_info.get('cupy_available') or gpu_info.get('torch_available'):
                    self.logger.info("✓ GPU acceleration available")
                else:
                    self.logger.info("⚠ GPU acceleration not available, using CPU")
            except Exception as e:
                self.logger.warning(f"GPU initialization failed: {e}")
            
            # 5. Initialize AI/ML components
            transformer_config = {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6
            }
            self.transformer_service = TransformerForecastingService(transformer_config)
            self.logger.info("✓ AI/ML components initialized")
            
            # 6. Create optimization config
            config = OptimizationConfig(
                tickers=self.symbols,
                budget=100000.0,
                optimization_methods=self.optimization_methods,
                csv_path=self.csv_path
            )
            
            # 7. Initialize core optimizer
            self.optimizer = QuantumFinanceOptimizer(config)
            self.logger.info("✓ Core optimizer initialized")
            
            # 8. Initialize streaming optimizer
            data_stream_manager = DataStreamManager()
            self.streaming_optimizer = StreamingOptimizer(
                config=config,
                data_stream_manager=data_stream_manager,
                update_frequency="30s"  # 30 second rebalancing
            )
            self.logger.info("✓ Streaming optimizer initialized")
            
            # 9. Initialize dashboard
            self.dashboard = DashboardApp(config)
            self.logger.info("✓ Dashboard initialized")
            
            self.logger.info("🎉 Ultimate Portfolio Optimizer ready!")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive portfolio analysis with all available methods
        """
        
        self.logger.info("🔍 Running comprehensive portfolio analysis...")
        
        results = {
            'timestamp': datetime.now(),
            'portfolio_info': {
                'symbols': self.symbols,
                'data_points': len(self.portfolio_data),
                'date_range': {
                    'start': self.portfolio_data['Date'].min(),
                    'end': self.portfolio_data['Date'].max()
                }
            }
        }
        
        # 1. Get enhanced data with live feeds
        try:
            enhanced_data = self.live_data_processor.get_enhanced_portfolio_data()
            results['live_data'] = {
                'live_prices_available': len(enhanced_data['live_prices']) > 0,
                'news_sentiment_available': len(enhanced_data['news_sentiment']) > 0,
                'market_indicators_available': len(enhanced_data['market_indicators']) > 0,
                'data_quality_score': enhanced_data.get('data_quality_score', 0.0)
            }
            
            # Add sentiment summary
            if enhanced_data['news_sentiment']:
                sentiment_summary = {}
                for symbol, sentiment_data in enhanced_data['news_sentiment'].items():
                    sentiment_summary[symbol] = {
                        'avg_sentiment': sentiment_data['avg_vader_sentiment'],
                        'news_count': sentiment_data['news_count']
                    }
                results['sentiment_analysis'] = sentiment_summary
            
        except Exception as e:
            self.logger.error(f"Enhanced data retrieval failed: {e}")
            results['live_data'] = {'error': str(e)}
        
        # 2. Prepare data for optimization
        try:
            opt_data, metadata = self.live_data_processor.get_optimization_ready_data()
            
            # Load data into optimizer
            self.optimizer.price_data = opt_data.set_index('Date')
            self.optimizer.returns_data = self.optimizer.price_data.pct_change().dropna()
            
            results['data_preparation'] = {
                'success': True,
                'enhanced_data_points': len(opt_data),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            results['data_preparation'] = {'success': False, 'error': str(e)}
            return results
        
        # 3. Classical optimization
        if 'classical' in self.optimization_methods:
            try:
                self.logger.info("Running classical optimization...")
                classical_results = self.optimizer.classical_optimization([
                    'mean_variance', 'max_sharpe', 'min_volatility', 'hrp'
                ])
                results['classical'] = classical_results
                self.logger.info("✓ Classical optimization completed")
            except Exception as e:
                self.logger.error(f"Classical optimization failed: {e}")
                results['classical'] = {'error': str(e)}
        
        # 4. Quantum optimization
        if 'quantum' in self.optimization_methods and self.quantum_backend.is_quantum_available():
            try:
                self.logger.info("Running quantum optimization...")
                quantum_results = self.optimizer.quantum_optimization()
                results['quantum'] = quantum_results
                self.logger.info("✓ Quantum optimization completed")
            except Exception as e:
                self.logger.error(f"Quantum optimization failed: {e}")
                results['quantum'] = {'error': str(e)}
        
        # 5. AI/ML optimization
        if 'ai' in self.optimization_methods:
            try:
                self.logger.info("Running AI/ML optimization...")
                
                # Prepare features for transformer
                features_df = self.transformer_service.prepare_features(
                    self.optimizer.price_data
                )
                
                if len(features_df) > 100:  # Need sufficient data
                    # Quick training for real-time use
                    training_result = self.transformer_service.train_multi_horizon_model(
                        features_df=features_df,
                        target_symbols=self.symbols[:3],  # Limit for speed
                        epochs=20,  # Quick training
                        batch_size=16
                    )
                    
                    # Generate forecasts
                    forecast_result = self.transformer_service.forecast_returns(
                        recent_data=self.optimizer.price_data.tail(60),
                        symbols=self.symbols
                    )
                    
                    results['ai_ml'] = {
                        'training_success': True,
                        'final_loss': training_result.get('final_loss', 0),
                        'predictions': forecast_result.predictions.tolist(),
                        'model_confidence': forecast_result.model_confidence,
                        'forecast_horizon': forecast_result.forecast_horizon
                    }
                    
                    self.logger.info("✓ AI/ML optimization completed")
                else:
                    results['ai_ml'] = {'error': 'Insufficient data for AI training'}
                    
            except Exception as e:
                self.logger.error(f"AI/ML optimization failed: {e}")
                results['ai_ml'] = {'error': str(e)}
        
        # 6. GPU-accelerated Monte Carlo
        if self.gpu_accelerator:
            try:
                self.logger.info("Running GPU-accelerated Monte Carlo...")
                
                from quantum_finance_opt.hpc.gpu_accelerator import SimulationParams
                
                sim_params = SimulationParams(
                    num_simulations=25000,  # Large simulation
                    num_assets=len(self.symbols),
                    time_steps=252,  # 1 year
                    initial_prices=self.optimizer.price_data.iloc[-1].values,
                    returns=self.optimizer.returns_data.mean().values,
                    covariance=self.optimizer.returns_data.cov().values
                )
                
                mc_results = self.gpu_accelerator.gpu_monte_carlo_simulation(sim_params)
                
                results['monte_carlo'] = {
                    'num_simulations': sim_params.num_simulations,
                    'var_95': mc_results['var_95'],
                    'var_99': mc_results['var_99'],
                    'expected_return': mc_results['expected_return'],
                    'volatility': mc_results['volatility']
                }
                
                self.logger.info("✓ Monte Carlo simulation completed")
                
            except Exception as e:
                self.logger.error(f"Monte Carlo simulation failed: {e}")
                results['monte_carlo'] = {'error': str(e)}
        
        # 7. Ensemble optimization
        if len([m for m in self.optimization_methods if m in results and 'error' not in results[m]]) > 1:
            try:
                self.logger.info("Running ensemble optimization...")
                ensemble_results = self.optimizer.ensemble_optimization(['classical', 'quantum'])
                results['ensemble'] = ensemble_results
                self.logger.info("✓ Ensemble optimization completed")
            except Exception as e:
                self.logger.error(f"Ensemble optimization failed: {e}")
                results['ensemble'] = {'error': str(e)}
        
        # 8. Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        self.optimization_results = results
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio recommendations based on all analysis"""
        
        recommendations = {
            'timestamp': datetime.now(),
            'best_method': None,
            'recommended_weights': None,
            'risk_assessment': 'medium',
            'sentiment_impact': 'neutral',
            'market_conditions': 'normal'
        }
        
        # Find best performing method
        best_sharpe = -999
        best_method = None
        best_weights = None
        
        for method, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                if method == 'classical':
                    for sub_method, sub_result in result.items():
                        if isinstance(sub_result, dict) and 'sharpe_ratio' in sub_result:
                            if sub_result['sharpe_ratio'] > best_sharpe:
                                best_sharpe = sub_result['sharpe_ratio']
                                best_method = f"classical_{sub_method}"
                                best_weights = sub_result.get('weights')
                
                elif 'sharpe_ratio' in result:
                    if result['sharpe_ratio'] > best_sharpe:
                        best_sharpe = result['sharpe_ratio']
                        best_method = method
                        best_weights = result.get('weights')
        
        recommendations['best_method'] = best_method
        recommendations['best_sharpe_ratio'] = best_sharpe
        
        if best_weights is not None:
            recommendations['recommended_weights'] = {
                symbol: float(weight) for symbol, weight in zip(self.symbols, best_weights)
            }
        
        # Risk assessment based on Monte Carlo
        if 'monte_carlo' in results and 'error' not in results['monte_carlo']:
            var_95 = results['monte_carlo']['var_95']
            if var_95 < -0.1:
                recommendations['risk_assessment'] = 'high'
            elif var_95 < -0.05:
                recommendations['risk_assessment'] = 'medium'
            else:
                recommendations['risk_assessment'] = 'low'
        
        # Sentiment impact
        if 'sentiment_analysis' in results:
            avg_sentiment = np.mean([data['avg_sentiment'] for data in results['sentiment_analysis'].values()])
            if avg_sentiment > 0.1:
                recommendations['sentiment_impact'] = 'positive'
            elif avg_sentiment < -0.1:
                recommendations['sentiment_impact'] = 'negative'
            else:
                recommendations['sentiment_impact'] = 'neutral'
        
        return recommendations
    
    def start_real_time_optimization(self):
        """Start real-time portfolio optimization"""
        
        self.logger.info("🔄 Starting real-time optimization...")
        
        try:
            # Start streaming optimization
            self.streaming_optimizer.start_streaming_optimization()
            
            self.logger.info("✓ Real-time optimization started")
            
            # Keep running and show status updates
            import time
            while True:
                try:
                    # Get current status
                    portfolio_status = self.streaming_optimizer.get_current_portfolio()
                    performance_summary = self.streaming_optimizer.get_performance_summary()
                    
                    self.logger.info(f"Portfolio Value: ${portfolio_status.get('total_value', 0):,.2f}")
                    self.logger.info(f"Success Rate: {performance_summary.get('success_rate', 0):.1%}")
                    self.logger.info(f"Avg Sharpe: {performance_summary.get('avg_sharpe_ratio', 0):.3f}")
                    
                    time.sleep(30)  # Status update every 30 seconds
                    
                except KeyboardInterrupt:
                    self.logger.info("Stopping real-time optimization...")
                    break
                    
        except Exception as e:
            self.logger.error(f"Real-time optimization failed: {e}")
    
    def launch_dashboard(self, port: int = 8501):
        """Launch interactive dashboard"""
        
        self.logger.info(f"🖥️ Launching dashboard on port {port}")
        
        try:
            self.dashboard.run_streamlit_dashboard(port)
        except Exception as e:
            self.logger.error(f"Dashboard launch failed: {e}")
    
    def save_results(self, output_path: str = None):
        """Save optimization results to file"""
        
        if not self.optimization_results:
            self.logger.warning("No results to save")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ultimate_portfolio_results_{timestamp}.json"
        
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            serializable_results = convert_numpy(self.optimization_results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"✓ Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def print_results_summary(self):
        """Print a beautiful results summary"""
        
        if not self.optimization_results:
            print("No results available")
            return
        
        print("\n" + "="*100)
        print("🚀 ULTIMATE PORTFOLIO OPTIMIZER - RESULTS SUMMARY")
        print("="*100)
        
        # Portfolio info
        portfolio_info = self.optimization_results.get('portfolio_info', {})
        print(f"\n📊 Portfolio Information:")
        print(f"   Assets: {', '.join(portfolio_info.get('symbols', []))}")
        print(f"   Data Points: {portfolio_info.get('data_points', 0)}")
        print(f"   Date Range: {portfolio_info.get('date_range', {}).get('start')} to {portfolio_info.get('date_range', {}).get('end')}")
        
        # Live data status
        live_data = self.optimization_results.get('live_data', {})
        print(f"\n📡 Live Data Status:")
        print(f"   Live Prices: {'✓' if live_data.get('live_prices_available') else '✗'}")
        print(f"   News Sentiment: {'✓' if live_data.get('news_sentiment_available') else '✗'}")
        print(f"   Market Indicators: {'✓' if live_data.get('market_indicators_available') else '✗'}")
        print(f"   Data Quality Score: {live_data.get('data_quality_score', 0):.2f}")
        
        # Optimization results
        print(f"\n🎯 Optimization Results:")
        
        methods = ['classical', 'quantum', 'ai_ml', 'ensemble', 'monte_carlo']
        
        for method in methods:
            if method in self.optimization_results:
                result = self.optimization_results[method]
                
                if isinstance(result, dict) and 'error' not in result:
                    print(f"\n   {method.upper()}:")
                    
                    if method == 'classical':
                        for sub_method, sub_result in result.items():
                            if isinstance(sub_result, dict) and 'sharpe_ratio' in sub_result:
                                print(f"     {sub_method}: Sharpe={sub_result['sharpe_ratio']:.3f}, "
                                     f"Return={sub_result.get('expected_return', 0):.2%}, "
                                     f"Vol={sub_result.get('volatility', 0):.2%}")
                    
                    elif method == 'monte_carlo':
                        print(f"     Simulations: {result.get('num_simulations', 0):,}")
                        print(f"     VaR 95%: {result.get('var_95', 0):.2%}")
                        print(f"     VaR 99%: {result.get('var_99', 0):.2%}")
                        print(f"     Expected Return: {result.get('expected_return', 0):.2%}")
                    
                    elif method == 'ai_ml':
                        print(f"     Training Loss: {result.get('final_loss', 0):.6f}")
                        print(f"     Model Confidence: {result.get('model_confidence', 0):.2f}")
                        print(f"     Forecast Horizon: {result.get('forecast_horizon', 0)} periods")
                    
                    else:
                        if 'sharpe_ratio' in result:
                            print(f"     Sharpe Ratio: {result['sharpe_ratio']:.3f}")
                        if 'expected_return' in result:
                            print(f"     Expected Return: {result['expected_return']:.2%}")
                        if 'volatility' in result:
                            print(f"     Volatility: {result['volatility']:.2%}")
                
                elif 'error' in result:
                    print(f"   {method.upper()}: ❌ {result['error']}")
        
        # Recommendations
        recommendations = self.optimization_results.get('recommendations', {})
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   Best Method: {recommendations.get('best_method', 'N/A')}")
            print(f"   Best Sharpe Ratio: {recommendations.get('best_sharpe_ratio', 0):.3f}")
            print(f"   Risk Assessment: {recommendations.get('risk_assessment', 'N/A').upper()}")
            print(f"   Sentiment Impact: {recommendations.get('sentiment_impact', 'N/A').upper()}")
            
            if recommendations.get('recommended_weights'):
                print(f"   Recommended Allocation:")
                for symbol, weight in recommendations['recommended_weights'].items():
                    print(f"     {symbol}: {weight:.1%}")
        
        print("\n" + "="*100)
    
    def cleanup(self):
        """Cleanup system resources"""
        
        self.logger.info("🧹 Cleaning up system resources...")
        
        try:
            if self.live_data_processor:
                self.live_data_processor.stop_live_feeds()
            
            if self.streaming_optimizer:
                self.streaming_optimizer.stop_streaming_optimization()
            
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
        description='Ultimate Portfolio Optimizer - The Best on GitHub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze your portfolio CSV
  python ultimate_portfolio_optimizer.py --csv portfolio.csv --mode analyze
  
  # Real-time optimization with live data
  python ultimate_portfolio_optimizer.py --csv portfolio.csv --mode realtime --api-keys newsapi=KEY finnhub=KEY
  
  # Launch interactive dashboard
  python ultimate_portfolio_optimizer.py --csv portfolio.csv --mode dashboard --port 8501
  
  # Full system with all features
  python ultimate_portfolio_optimizer.py --csv portfolio.csv --mode full --methods classical quantum ai
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--csv', type=str, required=True,
        help='Path to your portfolio CSV file (required)'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str,
        choices=['analyze', 'realtime', 'dashboard', 'full'],
        default='analyze',
        help='Operation mode (default: analyze)'
    )
    
    # Optimization methods
    parser.add_argument(
        '--methods', nargs='+',
        choices=['classical', 'quantum', 'ai', 'ensemble'],
        default=['classical', 'quantum', 'ai'],
        help='Optimization methods to use'
    )
    
    # API keys
    parser.add_argument(
        '--api-keys', nargs='+', default=[],
        help='API keys in format: newsapi=KEY finnhub=KEY'
    )
    
    # Dashboard options
    parser.add_argument('--port', type=int, default=8501, help='Dashboard port')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    
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
        # Parse API keys
        api_keys = {}
        for key_pair in args.api_keys:
            if '=' in key_pair:
                key, value = key_pair.split('=', 1)
                api_keys[key] = value
        
        # Check if CSV file exists
        if not os.path.exists(args.csv):
            logger.error(f"CSV file not found: {args.csv}")
            print(f"❌ Error: CSV file '{args.csv}' not found!")
            print("\nPlease provide a valid CSV file with your portfolio data.")
            print("Expected format:")
            print("  Option 1 (Wide): Date,AAPL,GOOGL,MSFT")
            print("  Option 2 (Long): Date,Symbol,Price")
            sys.exit(1)
        
        # Initialize Ultimate Portfolio Optimizer
        logger.info("🚀 Starting Ultimate Portfolio Optimizer")
        print("🚀 Ultimate Portfolio Optimizer - The Best on GitHub")
        print("="*60)
        
        optimizer = UltimatePortfolioOptimizer(
            csv_path=args.csv,
            api_keys=api_keys,
            optimization_methods=args.methods
        )
        
        # Execute based on mode
        if args.mode == 'analyze':
            print("📊 Running comprehensive portfolio analysis...")
            results = optimizer.run_comprehensive_analysis()
            optimizer.print_results_summary()
            
            if args.save_results:
                optimizer.save_results(args.output)
        
        elif args.mode == 'realtime':
            print("⚡ Starting real-time optimization...")
            optimizer.start_real_time_optimization()
        
        elif args.mode == 'dashboard':
            print(f"🖥️ Launching dashboard on port {args.port}...")
            print(f"Access at: http://localhost:{args.port}")
            optimizer.launch_dashboard(args.port)
        
        elif args.mode == 'full':
            print("🎯 Running full system analysis...")
            
            # Run analysis
            results = optimizer.run_comprehensive_analysis()
            optimizer.print_results_summary()
            
            # Save results
            if args.save_results:
                optimizer.save_results(args.output)
            
            # Launch dashboard in background
            import threading
            dashboard_thread = threading.Thread(
                target=optimizer.launch_dashboard,
                args=(args.port,)
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            
            print(f"\n🖥️ Dashboard available at: http://localhost:{args.port}")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    import time
                    time.sleep(10)
            except KeyboardInterrupt:
                print("\nShutting down...")
        
        # Cleanup
        optimizer.cleanup()
        print("✅ Ultimate Portfolio Optimizer completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation interrupted by user")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()