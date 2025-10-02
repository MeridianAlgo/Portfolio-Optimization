# QuantumFinanceOpt - Advanced Portfolio Optimization Tool

A comprehensive portfolio optimization framework that integrates classical methods, machine learning, deep learning, and quantum-inspired techniques for advanced financial portfolio management.

## Features

### Core Optimization Methods
- **Classical Optimization**: Mean-variance, maximum Sharpe ratio, minimum volatility
- **Advanced Classical**: Hierarchical Risk Parity (HRP), Black-Litterman, Critical Line Algorithm
- **Quantum-Inspired Risk Modeling**: Quantum entropy, coherence, and entanglement measures
- **Machine Learning**: Ensemble prediction models with feature engineering
- **Multi-Objective Optimization**: NSGA-II with ESG constraints
- **Ensemble Methods**: Combining multiple optimization approaches

### Key Features
- **Offline Operation**: No external API dependencies, works with local data
- **Comprehensive Data Handling**: CSV input, missing data handling, sample data generation
- **ESG Integration**: Environmental, Social, and Governance scoring and constraints
- **Sentiment Analysis**: News sentiment integration using FinBERT (simulated)
- **Risk Assessment**: Multiple risk models including quantum-inspired measures
- **Visualization**: Interactive plots and comprehensive reporting
- **Backtesting**: Monte Carlo simulations and performance analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install -r requirements.txt
```

### Key Dependencies
- numpy, pandas, scipy (scientific computing)
- scikit-learn (machine learning)
- cvxpy (optimization)
- matplotlib, seaborn (visualization)
- qutip (quantum computing, optional)
- torch, transformers (deep learning, optional)

## Quick Start

### Basic Usage

```bash
# Run with default settings (generates sample data)
python main.py

# Use custom tickers and methods
python main.py --tickers AAPL GOOGL MSFT AMZN --methods classical quantum ensemble

# Load data from CSV file
python main.py --csv-path data/prices.csv --methods classical ml

# Full optimization with custom parameters
python main.py --budget 100000 --esg-threshold 0.6 --methods classical quantum ml ensemble
```

### Configuration File

Create a YAML configuration file for complex setups:

```yaml
# config.yaml
tickers: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
start_date: '2020-01-01'
end_date: '2023-12-31'
budget: 100000.0
esg_threshold: 0.6
optimization_methods: ['classical', 'quantum', 'ml', 'ensemble']
forecast_horizon: 5
monte_carlo_runs: 1000
output_dir: 'results'
```

```bash
python main.py --config config.yaml
```

## Usage Examples

### Example 1: Basic Portfolio Optimization

```python
from quantum_finance_opt.core.config import OptimizationConfig
from quantum_finance_opt.core.optimizer import QuantumFinanceOptimizer

# Create configuration
config = OptimizationConfig(
    tickers=['AAPL', 'GOOGL', 'MSFT'],
    optimization_methods=['classical', 'quantum']
)

# Initialize optimizer
optimizer = QuantumFinanceOptimizer(config)

# Load data (generates sample data if no CSV provided)
optimizer.load_data()

# Run optimization
results = optimizer.run_optimization()

# Get best portfolio
best_method, best_portfolio = optimizer.get_best_portfolio('sharpe_ratio')
print(f"Best portfolio: {best_method}")
print(f"Weights: {best_portfolio['weights']}")
```

### Example 2: Classical Methods Only

```bash
python main.py --methods classical --classical-methods mean_variance max_sharpe min_volatility hrp
```

### Example 3: ML-Enhanced Optimization

```bash
python main.py --methods ml --forecast-horizon 10 --tickers AAPL GOOGL MSFT AMZN TSLA
```

### Example 4: Quantum-Inspired Risk Analysis

```bash
python main.py --methods quantum --tickers AAPL GOOGL MSFT --output-dir quantum_results
```

## Data Format

### CSV Input Format
The CSV file should have the following structure:
```csv
date,AAPL,GOOGL,MSFT,AMZN
2020-01-01,100.0,1000.0,200.0,1500.0
2020-01-02,101.5,1010.0,202.0,1520.0
...
```

- First column: Date (YYYY-MM-DD format)
- Subsequent columns: Asset prices
- Column names should match the tickers specified

## Output

The tool generates several outputs in the specified output directory:

### Files Generated
- `optimization_results_YYYYMMDD_HHMMSS.csv`: Summary of all optimization results
- `portfolio_weights_*.png`: Portfolio allocation visualizations
- `performance_comparison.png`: Performance metrics comparison
- `efficient_frontier.png`: Risk-return efficient frontier
- `correlation_heatmap.png`: Asset correlation analysis
- `price_performance.png`: Historical price performance
- Log files in `logs/` subdirectory

### Results Summary
The tool provides a comprehensive summary including:
- Expected returns, volatility, and Sharpe ratios for each method
- Portfolio weights and top holdings
- Risk metrics and performance analysis
- Best portfolio recommendation

## Advanced Features

### Quantum-Inspired Risk Modeling
The tool implements quantum information theory concepts for risk assessment:
- **Quantum Entropy**: Measures portfolio diversification using quantum entropy
- **Quantum Coherence**: Assesses correlation structure
- **Entanglement Measures**: Evaluates asset interdependencies

### ESG Integration
- Simulated ESG scores for each asset
- ESG threshold constraints in optimization
- ESG-weighted objective functions

### Machine Learning Features
- Technical indicator generation (RSI, MACD, Bollinger Bands)
- Ensemble prediction models (Linear, Random Forest, Gradient Boosting)
- Cross-validation and hyperparameter tuning
- Feature importance analysis

## Command Line Options

### Data Options
- `--csv-path`: Path to CSV file with price data
- `--tickers`: List of ticker symbols
- `--start-date`, `--end-date`: Date range for analysis
- `--generate-data`: Generate sample data instead of loading

### Portfolio Options
- `--budget`: Portfolio budget (default: 100000)
- `--esg-threshold`: Minimum ESG score (0-1, default: 0.5)
- `--risk-free-rate`: Risk-free rate (default: 0.02)
- `--rebalance-frequency`: Rebalancing frequency

### Optimization Options
- `--methods`: Optimization methods to use
- `--classical-methods`: Specific classical methods
- `--forecast-horizon`: ML forecast horizon
- `--monte-carlo-runs`: Number of simulation runs

### System Options
- `--output-dir`: Output directory
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--random-seed`: Random seed for reproducibility
- `--config`: YAML configuration file path

## Architecture

The tool follows a modular architecture:

```
quantum_finance_opt/
├── core/                 # Core system components
│   ├── config.py        # Configuration management
│   ├── optimizer.py     # Main optimizer class
│   └── exceptions.py    # Custom exceptions
├── data/                # Data processing
│   ├── processor.py     # Data loading and preprocessing
│   ├── preprocessing.py # Advanced preprocessing
│   └── simulator.py     # Data simulation
├── models/              # Optimization models
│   ├── classical.py     # Classical optimization
│   ├── advanced_classical.py # Advanced classical methods
│   ├── quantum_risk.py  # Quantum-inspired risk models
│   └── ml_predictor.py  # Machine learning models
├── optimization/        # Optimization engines
├── visualization/       # Visualization and reporting
│   └── plotter.py      # Plotting and visualization
└── utils/              # Utility functions
```

## Error Handling

The tool includes comprehensive error handling:
- Graceful fallbacks when advanced methods fail
- Detailed logging for debugging
- Input validation with clear error messages
- Automatic data quality checks

## Performance Considerations

- Efficient vectorized operations using NumPy/Pandas
- Parallel processing support for ML models
- Memory-efficient handling of large datasets
- GPU acceleration support (when available)

## Limitations

- Some advanced features require optional dependencies (qutip, transformers)
- Quantum features use classical approximations when qutip is unavailable
- ML models require sufficient historical data for training
- Real-time data feeds not supported (designed for offline analysis)

## Contributing

This is a demonstration implementation. For production use, consider:
- Adding more sophisticated risk models
- Implementing real-time data integration
- Adding more ML/DL architectures
- Enhancing the quantum computing components
- Adding more comprehensive backtesting

## License

This project is provided as-is for educational and research purposes.

## Support

For issues and questions:
1. Check the log files in the output directory
2. Verify input data format and parameters
3. Ensure all required dependencies are installed
4. Review the error messages and stack traces

## Examples Directory

See the `examples/` directory for additional usage examples and tutorials.