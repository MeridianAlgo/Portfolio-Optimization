"""
Neural Architecture Search for Portfolio Optimization

Automatically discovers optimal neural network architectures for financial forecasting
using evolutionary algorithms and reinforcement learning.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
import json
from datetime import datetime

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class ArchitectureGene:
    """Genetic representation of neural architecture"""
    layers: List[Dict[str, Any]]
    activation_functions: List[str]
    dropout_rates: List[float]
    learning_rate: float
    batch_size: int
    optimizer_type: str
    fitness_score: float = 0.0


class NeuralArchitectureSearch:
    """
    Evolutionary Neural Architecture Search for Financial Models
    
    Automatically discovers optimal neural network architectures for
    portfolio optimization and financial forecasting.
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.logger = logging.getLogger(__name__)
        
        # Architecture search space
        self.layer_types = ['linear', 'lstm', 'gru', 'conv1d', 'attention']
        self.activations = ['relu', 'tanh', 'sigmoid', 'gelu', 'swish']
        self.optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        
        # Population and evolution tracking
        self.population = []
        self.best_architectures = []
        self.evolution_history = []
        
    def initialize_population(self, input_dim: int, output_dim: int) -> List[ArchitectureGene]:
        """Initialize random population of architectures"""
        
        population = []
        
        for _ in range(self.population_size):
            # Random architecture parameters
            num_layers = random.randint(2, 8)
            layers = []
            
            current_dim = input_dim
            
            for i in range(num_layers):
                layer_type = random.choice(self.layer_types)
                
                if layer_type == 'linear':
                    hidden_dim = random.choice([64, 128, 256, 512, 1024])
                    layers.append({
                        'type': 'linear',
                        'input_dim': current_dim,
                        'output_dim': hidden_dim if i < num_layers - 1 else output_dim
                    })
                    current_dim = hidden_dim if i < num_layers - 1 else output_dim
                
                elif layer_type == 'lstm':
                    hidden_dim = random.choice([64, 128, 256, 512])
                    layers.append({
                        'type': 'lstm',
                        'input_dim': current_dim,
                        'hidden_dim': hidden_dim,
                        'num_layers': random.randint(1, 3),
                        'bidirectional': random.choice([True, False])
                    })
                    current_dim = hidden_dim * (2 if layers[-1]['bidirectional'] else 1)
                
                elif layer_type == 'gru':
                    hidden_dim = random.choice([64, 128, 256, 512])
                    layers.append({
                        'type': 'gru',
                        'input_dim': current_dim,
                        'hidden_dim': hidden_dim,
                        'num_layers': random.randint(1, 3),
                        'bidirectional': random.choice([True, False])
                    })
                    current_dim = hidden_dim * (2 if layers[-1]['bidirectional'] else 1)
                
                elif layer_type == 'conv1d':
                    out_channels = random.choice([32, 64, 128, 256])
                    kernel_size = random.choice([3, 5, 7])
                    layers.append({
                        'type': 'conv1d',
                        'in_channels': current_dim,
                        'out_channels': out_channels,
                        'kernel_size': kernel_size,
                        'stride': 1,
                        'padding': kernel_size // 2
                    })
                    current_dim = out_channels
                
                elif layer_type == 'attention':
                    embed_dim = random.choice([128, 256, 512])
                    num_heads = random.choice([4, 8, 16])
                    layers.append({
                        'type': 'attention',
                        'embed_dim': embed_dim,
                        'num_heads': num_heads,
                        'dropout': random.uniform(0.0, 0.3)
                    })
                    current_dim = embed_dim
            
            # Ensure final layer outputs correct dimension
            if layers and layers[-1]['type'] == 'linear':
                layers[-1]['output_dim'] = output_dim
            else:
                layers.append({
                    'type': 'linear',
                    'input_dim': current_dim,
                    'output_dim': output_dim
                })
            
            # Random hyperparameters
            gene = ArchitectureGene(
                layers=layers,
                activation_functions=[random.choice(self.activations) for _ in range(len(layers))],
                dropout_rates=[random.uniform(0.0, 0.5) for _ in range(len(layers))],
                learning_rate=random.uniform(1e-5, 1e-2),
                batch_size=random.choice([16, 32, 64, 128]),
                optimizer_type=random.choice(self.optimizers)
            )
            
            population.append(gene)
        
        self.population = population
        return population
    
    def build_model_from_gene(self, gene: ArchitectureGene, input_dim: int) -> nn.Module:
        """Build PyTorch model from architecture gene"""
        
        class DynamicModel(nn.Module):
            def __init__(self, gene: ArchitectureGene, input_dim: int):
                super().__init__()
                self.layers = nn.ModuleList()
                self.activations = []
                self.dropouts = nn.ModuleList()
                
                current_dim = input_dim
                
                for i, (layer_config, activation, dropout_rate) in enumerate(
                    zip(gene.layers, gene.activation_functions, gene.dropout_rates)
                ):
                    
                    if layer_config['type'] == 'linear':
                        self.layers.append(nn.Linear(
                            layer_config['input_dim'], 
                            layer_config['output_dim']
                        ))
                        current_dim = layer_config['output_dim']
                    
                    elif layer_config['type'] == 'lstm':
                        self.layers.append(nn.LSTM(
                            input_size=layer_config['input_dim'],
                            hidden_size=layer_config['hidden_dim'],
                            num_layers=layer_config['num_layers'],
                            bidirectional=layer_config['bidirectional'],
                            batch_first=True,
                            dropout=dropout_rate if layer_config['num_layers'] > 1 else 0
                        ))
                        current_dim = layer_config['hidden_dim'] * (2 if layer_config['bidirectional'] else 1)
                    
                    elif layer_config['type'] == 'gru':
                        self.layers.append(nn.GRU(
                            input_size=layer_config['input_dim'],
                            hidden_size=layer_config['hidden_dim'],
                            num_layers=layer_config['num_layers'],
                            bidirectional=layer_config['bidirectional'],
                            batch_first=True,
                            dropout=dropout_rate if layer_config['num_layers'] > 1 else 0
                        ))
                        current_dim = layer_config['hidden_dim'] * (2 if layer_config['bidirectional'] else 1)
                    
                    elif layer_config['type'] == 'conv1d':
                        self.layers.append(nn.Conv1d(
                            in_channels=layer_config['in_channels'],
                            out_channels=layer_config['out_channels'],
                            kernel_size=layer_config['kernel_size'],
                            stride=layer_config['stride'],
                            padding=layer_config['padding']
                        ))
                        current_dim = layer_config['out_channels']
                    
                    elif layer_config['type'] == 'attention':
                        self.layers.append(nn.MultiheadAttention(
                            embed_dim=layer_config['embed_dim'],
                            num_heads=layer_config['num_heads'],
                            dropout=layer_config['dropout'],
                            batch_first=True
                        ))
                        current_dim = layer_config['embed_dim']
                    
                    # Activation function
                    if activation == 'relu':
                        self.activations.append(nn.ReLU())
                    elif activation == 'tanh':
                        self.activations.append(nn.Tanh())
                    elif activation == 'sigmoid':
                        self.activations.append(nn.Sigmoid())
                    elif activation == 'gelu':
                        self.activations.append(nn.GELU())
                    elif activation == 'swish':
                        self.activations.append(nn.SiLU())
                    else:
                        self.activations.append(nn.ReLU())
                    
                    # Dropout
                    self.dropouts.append(nn.Dropout(dropout_rate))
            
            def forward(self, x):
                for i, (layer, activation, dropout) in enumerate(
                    zip(self.layers, self.activations, self.dropouts)
                ):
                    
                    if isinstance(layer, (nn.LSTM, nn.GRU)):
                        x, _ = layer(x)
                        if len(x.shape) == 3:  # (batch, seq, features)
                            x = x[:, -1, :]  # Take last timestep
                    
                    elif isinstance(layer, nn.Conv1d):
                        if len(x.shape) == 2:  # (batch, features)
                            x = x.unsqueeze(1)  # Add sequence dimension
                        x = x.transpose(1, 2)  # (batch, features, seq)
                        x = layer(x)
                        x = x.transpose(1, 2)  # (batch, seq, features)
                        x = x.mean(dim=1)  # Global average pooling
                    
                    elif isinstance(layer, nn.MultiheadAttention):
                        if len(x.shape) == 2:  # (batch, features)
                            x = x.unsqueeze(1)  # Add sequence dimension
                        x, _ = layer(x, x, x)
                        x = x.mean(dim=1)  # Average over sequence
                    
                    else:  # Linear layer
                        x = layer(x)
                    
                    # Apply activation (except for last layer)
                    if i < len(self.layers) - 1:
                        x = activation(x)
                        x = dropout(x)
                
                return x
        
        return DynamicModel(gene, input_dim)
    
    def evaluate_fitness(self, gene: ArchitectureGene, 
                        train_data: Tuple[torch.Tensor, torch.Tensor],
                        val_data: Tuple[torch.Tensor, torch.Tensor],
                        epochs: int = 20) -> float:
        """Evaluate fitness of architecture gene"""
        
        try:
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            # Build model
            model = self.build_model_from_gene(gene, X_train.shape[-1])
            
            # Setup optimizer
            if gene.optimizer_type == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=gene.learning_rate)
            elif gene.optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=gene.learning_rate)
            elif gene.optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=gene.learning_rate)
            else:
                optimizer = torch.optim.RMSprop(model.parameters(), lr=gene.learning_rate)
            
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                # Create batches
                num_samples = len(X_train)
                indices = torch.randperm(num_samples)
                
                for i in range(0, num_samples, gene.batch_size):
                    batch_indices = indices[i:i+gene.batch_size]
                    batch_X = X_train[batch_indices]
                    batch_y = y_train[batch_indices]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                
                # Calculate additional metrics
                mse = val_loss
                mae = torch.mean(torch.abs(val_outputs - y_val)).item()
                
                # Sharpe-like ratio for financial data
                returns = val_outputs.squeeze()
                if len(returns) > 1:
                    sharpe = torch.mean(returns) / (torch.std(returns) + 1e-8)
                    sharpe = sharpe.item()
                else:
                    sharpe = 0.0
                
                # Fitness combines multiple objectives
                fitness = 1.0 / (1.0 + mse) + 0.1 * abs(sharpe) - 0.01 * mae
                
                return max(0.0, fitness)
        
        except Exception as e:
            self.logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def crossover(self, parent1: ArchitectureGene, parent2: ArchitectureGene) -> ArchitectureGene:
        """Create offspring through crossover"""
        
        # Choose layers from both parents
        min_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, min_layers - 1)
        
        child_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        child_activations = parent1.activation_functions[:crossover_point] + parent2.activation_functions[crossover_point:]
        child_dropouts = parent1.dropout_rates[:crossover_point] + parent2.dropout_rates[crossover_point:]
        
        # Inherit hyperparameters
        child = ArchitectureGene(
            layers=child_layers,
            activation_functions=child_activations,
            dropout_rates=child_dropouts,
            learning_rate=random.choice([parent1.learning_rate, parent2.learning_rate]),
            batch_size=random.choice([parent1.batch_size, parent2.batch_size]),
            optimizer_type=random.choice([parent1.optimizer_type, parent2.optimizer_type])
        )
        
        return child
    
    def mutate(self, gene: ArchitectureGene) -> ArchitectureGene:
        """Mutate architecture gene"""
        
        mutated_gene = ArchitectureGene(
            layers=gene.layers.copy(),
            activation_functions=gene.activation_functions.copy(),
            dropout_rates=gene.dropout_rates.copy(),
            learning_rate=gene.learning_rate,
            batch_size=gene.batch_size,
            optimizer_type=gene.optimizer_type
        )
        
        # Mutate layers
        if random.random() < self.mutation_rate:
            if len(mutated_gene.layers) > 2:  # Keep at least 2 layers
                # Remove random layer
                idx = random.randint(0, len(mutated_gene.layers) - 2)  # Don't remove output layer
                mutated_gene.layers.pop(idx)
                mutated_gene.activation_functions.pop(idx)
                mutated_gene.dropout_rates.pop(idx)
            elif random.random() < 0.5:  # Add layer
                idx = random.randint(0, len(mutated_gene.layers) - 1)
                new_layer = {
                    'type': 'linear',
                    'input_dim': 128,
                    'output_dim': random.choice([64, 128, 256])
                }
                mutated_gene.layers.insert(idx, new_layer)
                mutated_gene.activation_functions.insert(idx, random.choice(self.activations))
                mutated_gene.dropout_rates.insert(idx, random.uniform(0.0, 0.5))
        
        # Mutate activations
        for i in range(len(mutated_gene.activation_functions)):
            if random.random() < self.mutation_rate:
                mutated_gene.activation_functions[i] = random.choice(self.activations)
        
        # Mutate dropout rates
        for i in range(len(mutated_gene.dropout_rates)):
            if random.random() < self.mutation_rate:
                mutated_gene.dropout_rates[i] = random.uniform(0.0, 0.5)
        
        # Mutate hyperparameters
        if random.random() < self.mutation_rate:
            mutated_gene.learning_rate *= random.uniform(0.5, 2.0)
            mutated_gene.learning_rate = max(1e-6, min(1e-1, mutated_gene.learning_rate))
        
        if random.random() < self.mutation_rate:
            mutated_gene.batch_size = random.choice([16, 32, 64, 128])
        
        if random.random() < self.mutation_rate:
            mutated_gene.optimizer_type = random.choice(self.optimizers)
        
        return mutated_gene
    
    def evolve(self, 
               train_data: Tuple[torch.Tensor, torch.Tensor],
               val_data: Tuple[torch.Tensor, torch.Tensor],
               input_dim: int,
               output_dim: int) -> ArchitectureGene:
        """Run evolutionary search for optimal architecture"""
        
        self.logger.info(f"Starting Neural Architecture Search: {self.generations} generations")
        
        # Initialize population
        if not self.population:
            self.initialize_population(input_dim, output_dim)
        
        for generation in range(self.generations):
            self.logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate fitness for all individuals
            for gene in self.population:
                if gene.fitness_score == 0.0:  # Not evaluated yet
                    gene.fitness_score = self.evaluate_fitness(gene, train_data, val_data)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Track best architecture
            best_gene = self.population[0]
            self.best_architectures.append(best_gene)
            
            self.logger.info(f"Best fitness: {best_gene.fitness_score:.4f}")
            
            # Selection and reproduction
            elite_size = self.population_size // 4
            elite = self.population[:elite_size]
            
            new_population = elite.copy()
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])
                
                # Mutation
                child = self.mutate(child)
                child.fitness_score = 0.0  # Reset fitness
                
                new_population.append(child)
            
            self.population = new_population
            
            # Track evolution
            avg_fitness = np.mean([gene.fitness_score for gene in self.population if gene.fitness_score > 0])
            self.evolution_history.append({
                'generation': generation + 1,
                'best_fitness': best_gene.fitness_score,
                'avg_fitness': avg_fitness,
                'best_architecture': self._gene_to_dict(best_gene)
            })
        
        # Return best architecture
        best_architecture = max(self.best_architectures, key=lambda x: x.fitness_score)
        
        self.logger.info(f"Evolution complete! Best fitness: {best_architecture.fitness_score:.4f}")
        
        return best_architecture
    
    def _tournament_selection(self, tournament_size: int = 3) -> ArchitectureGene:
        """Tournament selection for parent selection"""
        
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _gene_to_dict(self, gene: ArchitectureGene) -> Dict[str, Any]:
        """Convert gene to dictionary for serialization"""
        
        return {
            'layers': gene.layers,
            'activation_functions': gene.activation_functions,
            'dropout_rates': gene.dropout_rates,
            'learning_rate': gene.learning_rate,
            'batch_size': gene.batch_size,
            'optimizer_type': gene.optimizer_type,
            'fitness_score': gene.fitness_score
        }
    
    def save_evolution_results(self, filepath: str):
        """Save evolution results to file"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            },
            'evolution_history': self.evolution_history,
            'best_architectures': [self._gene_to_dict(gene) for gene in self.best_architectures]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evolution results saved to {filepath}")


class OptunaNAS:
    """
    Optuna-based Neural Architecture Search
    
    Uses Bayesian optimization for more efficient architecture search.
    """
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
        self.logger = logging.getLogger(__name__)
        
        if not OPTUNA_AVAILABLE:
            raise QuantumFinanceOptError("Optuna not available for NAS")
    
    def objective(self, trial, train_data, val_data, input_dim, output_dim):
        """Optuna objective function"""
        
        # Suggest architecture parameters
        n_layers = trial.suggest_int('n_layers', 2, 8)
        
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            layer_type = trial.suggest_categorical(f'layer_{i}_type', ['linear', 'lstm', 'gru'])
            
            if layer_type == 'linear':
                hidden_dim = trial.suggest_categorical(f'layer_{i}_dim', [64, 128, 256, 512, 1024])
                layers.append({
                    'type': 'linear',
                    'input_dim': current_dim,
                    'output_dim': hidden_dim if i < n_layers - 1 else output_dim
                })
                current_dim = hidden_dim if i < n_layers - 1 else output_dim
            
            elif layer_type in ['lstm', 'gru']:
                hidden_dim = trial.suggest_categorical(f'layer_{i}_dim', [64, 128, 256, 512])
                num_layers = trial.suggest_int(f'layer_{i}_num_layers', 1, 3)
                bidirectional = trial.suggest_categorical(f'layer_{i}_bidirectional', [True, False])
                
                layers.append({
                    'type': layer_type,
                    'input_dim': current_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional
                })
                current_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Ensure output layer
        if layers[-1]['type'] != 'linear' or layers[-1].get('output_dim') != output_dim:
            layers.append({
                'type': 'linear',
                'input_dim': current_dim,
                'output_dim': output_dim
            })
        
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        
        activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh', 'gelu']) 
                      for i in range(len(layers))]
        dropout_rates = [trial.suggest_float(f'dropout_{i}', 0.0, 0.5) 
                        for i in range(len(layers))]
        
        # Create gene and evaluate
        gene = ArchitectureGene(
            layers=layers,
            activation_functions=activations,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_type=optimizer_type
        )
        
        nas = NeuralArchitectureSearch()
        fitness = nas.evaluate_fitness(gene, train_data, val_data, epochs=10)
        
        return fitness
    
    def search(self, train_data, val_data, input_dim, output_dim):
        """Run Optuna-based architecture search"""
        
        study = optuna.create_study(direction='maximize')
        
        study.optimize(
            lambda trial: self.objective(trial, train_data, val_data, input_dim, output_dim),
            n_trials=self.n_trials
        )
        
        self.logger.info(f"Best trial: {study.best_trial.value}")
        
        return study.best_params, study.best_value