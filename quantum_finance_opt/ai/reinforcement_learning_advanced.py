"""
Advanced Reinforcement Learning for Portfolio Management

Implements state-of-the-art RL algorithms including PPO, SAC, TD3, and multi-agent
systems for dynamic portfolio optimization and trading strategies.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import gym
from gym import spaces
import random
from collections import deque
import pandas as pd

try:
    from stable_baselines3 import PPO, SAC, TD3, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.rllib.agents import ppo, sac, td3
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class RLConfig:
    """Configuration for RL training"""
    algorithm: str = 'PPO'
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class AdvancedPortfolioEnv(gym.Env):
    """
    Advanced Portfolio Management Environment
    
    Sophisticated environment with transaction costs, market impact,
    sentiment data, and multi-asset support.
    """
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 features_data: pd.DataFrame = None,
                 initial_balance: float = 100000,
                 transaction_cost: float = 0.001,
                 market_impact_coef: float = 0.0001,
                 lookback_window: int = 60,
                 max_position: float = 0.3):
        
        super().__init__()
        
        self.price_data = price_data
        self.features_data = features_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.market_impact_coef = market_impact_coef
        self.lookback_window = lookback_window
        self.max_position = max_position
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.positions = np.zeros(len(price_data.columns))
        self.portfolio_value = initial_balance
        self.previous_portfolio_value = initial_balance
        
        # Action and observation spaces
        n_assets = len(price_data.columns)
        
        # Actions: target weights for each asset (-1 to 1, then normalized)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )
        
        # Observations: price features + portfolio state + market features
        obs_dim = (
            n_assets * lookback_window +  # Price history
            n_assets +  # Current positions
            n_assets +  # Price changes
            10 +  # Portfolio metrics
            (features_data.shape[1] if features_data is not None else 0)  # Additional features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Performance tracking
        self.episode_returns = []
        self.episode_sharpe_ratios = []
        self.episode_max_drawdowns = []
        
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset environment to initial state"""
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = np.zeros(len(self.price_data.columns))
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        
        # Normalize action to valid portfolio weights
        action = np.array(action)
        action = np.clip(action, -self.max_position, self.max_position)
        
        # Convert to absolute weights (sum to 1)
        if np.sum(np.abs(action)) > 0:
            action = action / np.sum(np.abs(action))
        else:
            action = np.zeros_like(action)
        
        # Calculate required trades
        current_prices = self.price_data.iloc[self.current_step].values
        current_weights = self._get_current_weights()
        target_weights = action
        
        # Execute trades with transaction costs and market impact
        trade_amounts = (target_weights - current_weights) * self.portfolio_value
        transaction_costs = np.sum(np.abs(trade_amounts)) * self.transaction_cost
        market_impact = np.sum(np.abs(trade_amounts)) * self.market_impact_coef
        
        # Update positions
        for i, trade_amount in enumerate(trade_amounts):
            if current_prices[i] > 0:
                shares_traded = trade_amount / current_prices[i]
                self.positions[i] += shares_traded
        
        # Update balance
        self.balance -= transaction_costs + market_impact
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < len(self.price_data):
            new_prices = self.price_data.iloc[self.current_step].values
            self.previous_portfolio_value = self.portfolio_value
            self.portfolio_value = self.balance + np.sum(self.positions * new_prices)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = (
            self.current_step >= len(self.price_data) - 1 or
            self.portfolio_value <= self.initial_balance * 0.1  # Stop loss
        )
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'transaction_costs': transaction_costs,
            'market_impact': market_impact,
            'current_weights': self._get_current_weights()
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation"""
        
        obs_components = []
        
        # Price history (normalized)
        if self.current_step >= self.lookback_window:
            price_history = self.price_data.iloc[
                self.current_step - self.lookback_window:self.current_step
            ].values
            
            # Normalize by current prices
            current_prices = self.price_data.iloc[self.current_step].values
            price_history_norm = price_history / current_prices
            obs_components.append(price_history_norm.flatten())
        else:
            # Pad with zeros if not enough history
            obs_components.append(np.zeros(len(self.price_data.columns) * self.lookback_window))
        
        # Current positions (as weights)
        current_weights = self._get_current_weights()
        obs_components.append(current_weights)
        
        # Price changes
        if self.current_step > 0:
            prev_prices = self.price_data.iloc[self.current_step - 1].values
            curr_prices = self.price_data.iloc[self.current_step].values
            price_changes = (curr_prices - prev_prices) / prev_prices
            price_changes = np.nan_to_num(price_changes)
        else:
            price_changes = np.zeros(len(self.price_data.columns))
        
        obs_components.append(price_changes)
        
        # Portfolio metrics
        portfolio_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        portfolio_metrics = np.array([
            portfolio_return,
            total_return,
            self.portfolio_value / self.initial_balance,
            np.sum(np.abs(current_weights)),  # Total exposure
            np.max(current_weights),  # Max position
            np.std(current_weights),  # Position concentration
            len(current_weights[current_weights > 0.01]),  # Number of positions
            self.current_step / len(self.price_data),  # Time progress
            self.balance / self.portfolio_value,  # Cash ratio
            np.sum(self.positions * self.price_data.iloc[self.current_step].values) / self.portfolio_value  # Invested ratio
        ])
        
        obs_components.append(portfolio_metrics)
        
        # Additional features if available
        if self.features_data is not None and self.current_step < len(self.features_data):
            additional_features = self.features_data.iloc[self.current_step].values
            obs_components.append(additional_features)
        
        # Concatenate all components
        observation = np.concatenate(obs_components).astype(np.float32)
        
        return observation
    
    def _get_current_weights(self):
        """Get current portfolio weights"""
        
        if self.portfolio_value <= 0:
            return np.zeros(len(self.price_data.columns))
        
        current_prices = self.price_data.iloc[self.current_step].values
        position_values = self.positions * current_prices
        weights = position_values / self.portfolio_value
        
        return weights
    
    def _calculate_reward(self):
        """Calculate reward for the current step"""
        
        # Portfolio return
        portfolio_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        
        # Benchmark return (equal weight)
        if self.current_step > 0:
            prev_prices = self.price_data.iloc[self.current_step - 1].values
            curr_prices = self.price_data.iloc[self.current_step].values
            benchmark_return = np.mean((curr_prices - prev_prices) / prev_prices)
        else:
            benchmark_return = 0
        
        # Excess return
        excess_return = portfolio_return - benchmark_return
        
        # Risk penalty (volatility of positions)
        current_weights = self._get_current_weights()
        concentration_penalty = np.std(current_weights) * 0.1
        
        # Transaction cost penalty
        if len(self.episode_returns) > 0:
            recent_returns = self.episode_returns[-10:]  # Last 10 returns
            volatility_penalty = np.std(recent_returns) * 0.05 if len(recent_returns) > 1 else 0
        else:
            volatility_penalty = 0
        
        # Combined reward
        reward = excess_return - concentration_penalty - volatility_penalty
        
        # Track episode returns
        self.episode_returns.append(portfolio_return)
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state"""
        
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Total Return: {(self.portfolio_value - self.initial_balance) / self.initial_balance:.2%}")
            print(f"Current Weights: {self._get_current_weights()}")
            print("-" * 50)


class MultiAgentPortfolioEnv(gym.Env):
    """
    Multi-Agent Portfolio Environment
    
    Multiple agents compete and collaborate in portfolio management.
    """
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 n_agents: int = 3,
                 initial_balance: float = 100000):
        
        super().__init__()
        
        self.price_data = price_data
        self.n_agents = n_agents
        self.initial_balance = initial_balance
        
        # Create individual environments for each agent
        self.agent_envs = [
            AdvancedPortfolioEnv(price_data, initial_balance=initial_balance)
            for _ in range(n_agents)
        ]
        
        # Multi-agent action and observation spaces
        single_action_space = self.agent_envs[0].action_space
        single_obs_space = self.agent_envs[0].observation_space
        
        self.action_space = spaces.Tuple([single_action_space] * n_agents)
        self.observation_space = spaces.Tuple([single_obs_space] * n_agents)
        
        # Competition metrics
        self.agent_performances = [[] for _ in range(n_agents)]
        self.collaboration_bonus = 0.1
    
    def reset(self):
        """Reset all agent environments"""
        
        observations = []
        for env in self.agent_envs:
            obs = env.reset()
            observations.append(obs)
        
        return observations
    
    def step(self, actions):
        """Execute step for all agents"""
        
        observations = []
        rewards = []
        dones = []
        infos = []
        
        # Execute actions for each agent
        for i, (env, action) in enumerate(zip(self.agent_envs, actions)):
            obs, reward, done, info = env.step(action)
            
            # Add collaboration bonus/penalty
            collaboration_reward = self._calculate_collaboration_reward(i, actions)
            reward += collaboration_reward
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            # Track performance
            self.agent_performances[i].append(env.portfolio_value)
        
        return observations, rewards, dones, infos
    
    def _calculate_collaboration_reward(self, agent_id: int, actions: List[np.ndarray]) -> float:
        """Calculate collaboration reward based on agent coordination"""
        
        # Reward agents for diversified strategies
        agent_action = actions[agent_id]
        other_actions = [actions[i] for i in range(len(actions)) if i != agent_id]
        
        if not other_actions:
            return 0.0
        
        # Calculate diversity bonus
        diversity_score = 0.0
        for other_action in other_actions:
            # Reward for different strategies
            correlation = np.corrcoef(agent_action, other_action)[0, 1]
            if not np.isnan(correlation):
                diversity_score += max(0, 1 - abs(correlation))
        
        return diversity_score * self.collaboration_bonus / len(other_actions)


class AdvancedRLTrainer:
    """
    Advanced Reinforcement Learning Trainer
    
    Supports multiple RL algorithms and advanced training techniques.
    """
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.training_history = {}
        
        # Multi-processing support
        self.n_envs = 4
    
    def train_single_agent(self, 
                          env: gym.Env,
                          algorithm: str = None,
                          total_timesteps: int = None) -> Any:
        """Train single agent using specified algorithm"""
        
        if not SB3_AVAILABLE:
            raise QuantumFinanceOptError("Stable-Baselines3 not available")
        
        algorithm = algorithm or self.config.algorithm
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        self.logger.info(f"Training {algorithm} agent for {total_timesteps} timesteps")
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize model based on algorithm
        if algorithm == 'PPO':
            model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                verbose=1
            )
        
        elif algorithm == 'SAC':
            model = SAC(
                'MlpPolicy',
                vec_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                verbose=1
            )
        
        elif algorithm == 'TD3':
            model = TD3(
                'MlpPolicy',
                vec_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                verbose=1
            )
        
        elif algorithm == 'A2C':
            model = A2C(
                'MlpPolicy',
                vec_env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                verbose=1
            )
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Custom callback for tracking training progress
        callback = TrainingCallback()
        
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Store model and history
        self.models[algorithm] = model
        self.training_history[algorithm] = callback.training_history
        
        self.logger.info(f"{algorithm} training completed")
        
        return model
    
    def train_multi_agent(self, 
                         env: MultiAgentPortfolioEnv,
                         total_timesteps: int = None) -> Dict[str, Any]:
        """Train multiple agents in competitive/collaborative environment"""
        
        if not RAY_AVAILABLE:
            self.logger.warning("Ray not available, using sequential training")
            return self._train_multi_agent_sequential(env, total_timesteps)
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        self.logger.info(f"Training multi-agent system for {total_timesteps} timesteps")
        
        # Ray configuration
        ray.init(ignore_reinit_error=True)
        
        config = {
            "env": MultiAgentPortfolioEnv,
            "env_config": {
                "price_data": env.price_data,
                "n_agents": env.n_agents,
                "initial_balance": env.initial_balance
            },
            "multiagent": {
                "policies": {
                    f"agent_{i}": (None, env.observation_space[i], env.action_space[i], {})
                    for i in range(env.n_agents)
                },
                "policy_mapping_fn": lambda agent_id: f"agent_{agent_id}",
            },
            "framework": "torch",
            "num_workers": 2,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
        }
        
        # Train using PPO
        trainer = ppo.PPOTrainer(config=config)
        
        training_results = []
        for i in range(total_timesteps // 4000):  # Adjust based on batch size
            result = trainer.train()
            training_results.append(result)
            
            if i % 10 == 0:
                self.logger.info(f"Iteration {i}: Mean reward = {result['episode_reward_mean']:.2f}")
        
        # Save trained policies
        checkpoint = trainer.save()
        
        ray.shutdown()
        
        return {
            'checkpoint': checkpoint,
            'training_results': training_results,
            'final_reward': training_results[-1]['episode_reward_mean'] if training_results else 0
        }
    
    def _train_multi_agent_sequential(self, env: MultiAgentPortfolioEnv, total_timesteps: int) -> Dict[str, Any]:
        """Sequential multi-agent training fallback"""
        
        models = {}
        
        # Train each agent separately
        for i in range(env.n_agents):
            self.logger.info(f"Training agent {i}")
            
            # Create single-agent environment
            single_env = env.agent_envs[i]
            
            # Train agent
            model = self.train_single_agent(
                single_env, 
                algorithm='PPO',
                total_timesteps=total_timesteps // env.n_agents
            )
            
            models[f'agent_{i}'] = model
        
        return {'models': models}
    
    def evaluate_model(self, 
                      model: Any,
                      env: gym.Env,
                      n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model performance"""
        
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_returns = []
            peak_value = env.initial_balance
            max_drawdown = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track portfolio performance
                portfolio_value = info.get('portfolio_value', env.initial_balance)
                portfolio_return = (portfolio_value - env.previous_portfolio_value) / env.previous_portfolio_value
                episode_returns.append(portfolio_return)
                
                # Update peak and drawdown
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                
                drawdown = (peak_value - portfolio_value) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_values.append(info.get('portfolio_value', env.initial_balance))
            
            # Calculate Sharpe ratio
            if len(episode_returns) > 1:
                returns_array = np.array(episode_returns)
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                sharpe_ratios.append(sharpe)
            
            max_drawdowns.append(max_drawdown)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values),
            'mean_total_return': (np.mean(portfolio_values) - env.initial_balance) / env.initial_balance,
            'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'mean_max_drawdown': np.mean(max_drawdowns),
            'success_rate': sum(1 for pv in portfolio_values if pv > env.initial_balance) / len(portfolio_values)
        }
    
    def hyperparameter_optimization(self, 
                                   env: gym.Env,
                                   n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        try:
            import optuna
        except ImportError:
            raise QuantumFinanceOptError("Optuna not available for hyperparameter optimization")
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
            gamma = trial.suggest_float('gamma', 0.9, 0.999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
            
            # Create config with suggested parameters
            config = RLConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                total_timesteps=10000  # Reduced for faster optimization
            )
            
            # Train model with suggested parameters
            trainer = AdvancedRLTrainer(config)
            model = trainer.train_single_agent(env, 'PPO', config.total_timesteps)
            
            # Evaluate performance
            results = trainer.evaluate_model(model, env, n_episodes=5)
            
            return results['mean_sharpe_ratio']
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""
    
    def __init__(self):
        super().__init__()
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def _on_step(self) -> bool:
        # Track training metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                self.training_history['episode_rewards'].append(info['episode']['r'])
                self.training_history['episode_lengths'].append(info['episode']['l'])
        
        return True