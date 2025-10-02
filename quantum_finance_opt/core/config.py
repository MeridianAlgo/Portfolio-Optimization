"""
Configuration management for QuantumFinanceOpt.

This module handles configuration loading, validation, and management
for the portfolio optimization system.
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .exceptions import ConfigurationError


@dataclass
class OptimizationConfig:
    """Configuration class for optimization parameters."""
    
    # Data parameters
    tickers: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    start_date: str = '2020-01-01'
    end_date: str = '2023-12-31'
    csv_path: Optional[str] = None
    
    # Portfolio parameters
    budget: float = 100000.0
    rebalance_frequency: str = 'monthly'  # daily, weekly, monthly, quarterly
    risk_free_rate: float = 0.02
    transaction_costs: float = 0.001
    
    # ESG parameters
    esg_threshold: float = 0.5
    esg_weight: float = 0.1
    
    # Optimization parameters
    optimization_methods: List[str] = field(default_factory=lambda: ['classical', 'ml', 'ensemble'])
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    # ML/DL parameters
    ml_models: List[str] = field(default_factory=lambda: ['linear', 'rf', 'gb'])
    dl_architecture: str = 'lstm'
    sequence_length: int = 60
    forecast_horizon: int = 5
    
    # RL parameters
    rl_episodes: int = 1000
    rl_learning_rate: float = 0.001
    rl_batch_size: int = 32
    
    # Backtesting parameters
    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None
    monte_carlo_runs: int = 1000
    
    # Visualization parameters
    save_plots: bool = True
    plot_format: str = 'png'
    output_dir: str = 'output'
    
    # System parameters
    random_seed: int = 42
    n_jobs: int = -1
    gpu_enabled: bool = True
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        try:
            # Validate dates
            if self.start_date:
                datetime.strptime(self.start_date, '%Y-%m-%d')
            if self.end_date:
                datetime.strptime(self.end_date, '%Y-%m-%d')
            
            # Validate numeric parameters
            if self.budget <= 0:
                raise ConfigurationError("Budget must be positive")
            
            if not 0 <= self.esg_threshold <= 1:
                raise ConfigurationError("ESG threshold must be between 0 and 1")
            
            if not 0 <= self.risk_free_rate <= 1:
                raise ConfigurationError("Risk-free rate must be between 0 and 1")
            
            if self.transaction_costs < 0:
                raise ConfigurationError("Transaction costs cannot be negative")
            
            # Validate lists
            if not self.tickers:
                raise ConfigurationError("At least one ticker must be specified")
            
            # Validate rebalance frequency
            valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
            if self.rebalance_frequency not in valid_frequencies:
                raise ConfigurationError(f"Rebalance frequency must be one of {valid_frequencies}")
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {e}")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'OptimizationConfig':
        """Load configuration from YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    @classmethod
    def from_cli(cls) -> 'OptimizationConfig':
        """Create configuration from command line arguments."""
        parser = argparse.ArgumentParser(
            description='QuantumFinanceOpt - Advanced Portfolio Optimization'
        )
        
        # Data arguments
        parser.add_argument('--csv-path', type=str, help='Path to CSV file with price data')
        parser.add_argument('--tickers', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                          help='List of ticker symbols')
        parser.add_argument('--start-date', type=str, default='2020-01-01',
                          help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, default='2023-12-31',
                          help='End date (YYYY-MM-DD)')
        
        # Portfolio arguments
        parser.add_argument('--budget', type=float, default=100000.0,
                          help='Portfolio budget')
        parser.add_argument('--esg-threshold', type=float, default=0.5,
                          help='Minimum ESG score threshold')
        parser.add_argument('--risk-free-rate', type=float, default=0.02,
                          help='Risk-free rate for calculations')
        
        # Optimization arguments
        parser.add_argument('--methods', nargs='+', 
                          default=['classical', 'ml', 'ensemble'],
                          help='Optimization methods to use')
        parser.add_argument('--monte-carlo-runs', type=int, default=1000,
                          help='Number of Monte Carlo simulation runs')
        
        # System arguments
        parser.add_argument('--output-dir', type=str, default='output',
                          help='Output directory for results')
        parser.add_argument('--log-level', type=str, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')
        parser.add_argument('--config', type=str, help='Path to YAML configuration file')
        
        args = parser.parse_args()
        
        # Load from YAML if specified
        if args.config:
            config = cls.from_yaml(args.config)
            # Override with CLI arguments
            for key, value in vars(args).items():
                if value is not None and key != 'config':
                    setattr(config, key.replace('-', '_'), value)
            return config
        
        # Create from CLI arguments
        return cls(
            csv_path=args.csv_path,
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            budget=args.budget,
            esg_threshold=args.esg_threshold,
            risk_free_rate=args.risk_free_rate,
            optimization_methods=args.methods,
            monte_carlo_runs=args.monte_carlo_runs,
            output_dir=args.output_dir,
            log_level=args.log_level
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")


def setup_logging(config: OptimizationConfig):
    """Set up logging configuration."""
    import logging
    
    # Create logs directory
    log_dir = os.path.join(config.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, f'quantum_finance_opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('QuantumFinanceOpt')