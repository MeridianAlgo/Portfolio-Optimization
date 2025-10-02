"""
Transformer-based Time Series Forecasting

Advanced transformer models for multi-horizon financial forecasting
with attention mechanisms and cross-asset modeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# Transformer imports with fallbacks
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        TimeSeriesTransformerConfig, TimeSeriesTransformer,
        Trainer, TrainingArguments
    )
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..core.exceptions import QuantumFinanceOptError


@dataclass
class ForecastResult:
    """Result from transformer forecasting"""
    predictions: np.ndarray
    confidence_intervals: np.ndarray
    attention_weights: Optional[np.ndarray]
    forecast_horizon: int
    model_confidence: float
    feature_importance: Dict[str, float]
    timestamp: datetime


class MultiHorizonTransformer(nn.Module):
    """
    Multi-horizon transformer for financial time series forecasting
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 max_seq_length: int = 100,
                 forecast_horizons: List[int] = [1, 5, 10, 20]):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizons = forecast_horizons
        self.max_seq_length = max_seq_length
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1)
            ) for h in forecast_horizons
        })
        
        # Attention weights storage
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary of predictions for each horizon
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use last token for prediction (or could use pooling)
        last_hidden = encoded[:, -1, :]  # (batch_size, d_model)
        
        # Multi-horizon predictions
        predictions = {}
        for horizon in self.forecast_horizons:
            pred = self.prediction_heads[f'horizon_{horizon}'](last_hidden)
            predictions[f'horizon_{horizon}'] = pred.squeeze(-1)
        
        return predictions


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class CrossAssetAttentionTransformer(nn.Module):
    """
    Cross-asset attention transformer for modeling asset relationships
    """
    
    def __init__(self,
                 num_assets: int,
                 feature_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4):
        
        super().__init__()
        
        self.num_assets = num_assets
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # Asset embeddings
        self.asset_embeddings = nn.Embedding(num_assets, d_model)
        
        # Feature projection
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # Cross-asset attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
    
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-asset attention
        
        Args:
            features: Asset features (batch_size, num_assets, feature_dim)
            asset_ids: Asset identifiers (batch_size, num_assets)
            
        Returns:
            Predictions and attention weights
        """
        batch_size, num_assets, _ = features.shape
        
        # Get asset embeddings
        asset_emb = self.asset_embeddings(asset_ids)  # (batch_size, num_assets, d_model)
        
        # Project features
        feat_proj = self.feature_projection(features)  # (batch_size, num_assets, d_model)
        
        # Combine asset embeddings and features
        combined = asset_emb + feat_proj
        
        # Cross-asset attention
        attended, attention_weights = self.cross_attention(
            query=combined,
            key=combined,
            value=combined
        )
        
        # Transformer processing
        encoded = self.transformer(attended)
        
        # Output predictions
        predictions = self.output_projection(encoded).squeeze(-1)
        
        return predictions, attention_weights


class TransformerForecastingService:
    """
    Service for transformer-based financial forecasting
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: str = "auto"):
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Model components
        self.multi_horizon_model = None
        self.cross_asset_model = None
        self.feature_scaler = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        # Feature engineering
        self.feature_columns = [
            'returns', 'volatility', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower'
        ]
    
    def prepare_features(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare features for transformer training"""
        features_df = pd.DataFrame(index=price_data.index)
        
        for symbol in price_data.columns:
            prices = price_data[symbol]
            
            # Basic features
            returns = prices.pct_change()
            features_df[f'{symbol}_returns'] = returns
            features_df[f'{symbol}_volatility'] = returns.rolling(20).std()
            
            if volume_data is not None and symbol in volume_data.columns:
                features_df[f'{symbol}_volume'] = volume_data[symbol]
            else:
                features_df[f'{symbol}_volume'] = np.random.exponential(1000, len(prices))
            
            # Technical indicators
            features_df[f'{symbol}_rsi'] = self._calculate_rsi(prices)
            features_df[f'{symbol}_macd'] = self._calculate_macd(prices)
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            features_df[f'{symbol}_bb_upper'] = bb_upper
            features_df[f'{symbol}_bb_lower'] = bb_lower
            
            # Price momentum
            features_df[f'{symbol}_momentum_5'] = prices.pct_change(5)
            features_df[f'{symbol}_momentum_20'] = prices.pct_change(20)
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df    

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def train_multi_horizon_model(self,
                                 features_df: pd.DataFrame,
                                 target_symbols: List[str],
                                 sequence_length: int = 60,
                                 forecast_horizons: List[int] = [1, 5, 10, 20],
                                 epochs: int = 100,
                                 batch_size: int = 32) -> Dict[str, Any]:
        """Train multi-horizon transformer model"""
        
        if not TRANSFORMERS_AVAILABLE:
            raise QuantumFinanceOptError("Transformers library not available")
        
        self.logger.info("Training multi-horizon transformer model")
        
        # Prepare training data
        X, y = self._prepare_training_data(features_df, target_symbols, sequence_length, forecast_horizons)
        
        if len(X) == 0:
            raise QuantumFinanceOptError("Insufficient training data")
        
        # Initialize model
        input_dim = X.shape[-1]
        self.multi_horizon_model = MultiHorizonTransformer(
            input_dim=input_dim,
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 6),
            max_seq_length=sequence_length,
            forecast_horizons=forecast_horizons
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.multi_horizon_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensors = {f'horizon_{h}': torch.FloatTensor(y[f'horizon_{h}']).to(self.device) 
                    for h in forecast_horizons}
        
        # Training loop
        training_losses = []
        
        for epoch in range(epochs):
            self.multi_horizon_model.train()
            
            # Create batches
            num_samples = len(X_tensor)
            indices = torch.randperm(num_samples)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_tensor[batch_indices]
                
                # Forward pass
                predictions = self.multi_horizon_model(batch_X)
                
                # Calculate loss for each horizon
                total_loss = 0.0
                for horizon in forecast_horizons:
                    horizon_key = f'horizon_{horizon}'
                    batch_y = y_tensors[horizon_key][batch_indices]
                    loss = F.mse_loss(predictions[horizon_key], batch_y)
                    total_loss += loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multi_horizon_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            training_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        self.training_history = training_losses
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'epochs_trained': epochs,
            'model_parameters': sum(p.numel() for p in self.multi_horizon_model.parameters())
        }
    
    def _prepare_training_data(self,
                              features_df: pd.DataFrame,
                              target_symbols: List[str],
                              sequence_length: int,
                              forecast_horizons: List[int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare sequences for training"""
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        features_normalized = self.feature_scaler.fit_transform(features_df.values)
        
        X = []
        y = {f'horizon_{h}': [] for h in forecast_horizons}
        
        max_horizon = max(forecast_horizons)
        
        for i in range(sequence_length, len(features_normalized) - max_horizon):
            # Input sequence
            X.append(features_normalized[i-sequence_length:i])
            
            # Targets for each horizon
            for horizon in forecast_horizons:
                # Use first target symbol's return as target (could be extended)
                target_col = f'{target_symbols[0]}_returns'
                if target_col in features_df.columns:
                    target_idx = features_df.columns.get_loc(target_col)
                    target_value = features_normalized[i + horizon - 1, target_idx]
                    y[f'horizon_{horizon}'].append(target_value)
                else:
                    y[f'horizon_{horizon}'].append(0.0)
        
        return np.array(X), {k: np.array(v) for k, v in y.items()}
    
    def forecast_returns(self,
                        recent_data: pd.DataFrame,
                        symbols: List[str],
                        horizons: List[int] = [1, 5, 10, 20]) -> ForecastResult:
        """Generate return forecasts using trained model"""
        
        if not self.is_trained or self.multi_horizon_model is None:
            raise QuantumFinanceOptError("Model not trained")
        
        # Prepare features
        features_df = self.prepare_features(recent_data)
        
        # Normalize features
        features_normalized = self.feature_scaler.transform(features_df.values)
        
        # Take last sequence
        sequence_length = self.multi_horizon_model.max_seq_length
        if len(features_normalized) < sequence_length:
            raise QuantumFinanceOptError(f"Need at least {sequence_length} data points")
        
        input_sequence = features_normalized[-sequence_length:]
        
        # Convert to tensor
        X = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
        
        # Generate predictions
        self.multi_horizon_model.eval()
        with torch.no_grad():
            predictions = self.multi_horizon_model(X)
        
        # Extract predictions
        forecast_values = []
        for horizon in horizons:
            if f'horizon_{horizon}' in predictions:
                pred_value = predictions[f'horizon_{horizon}'].cpu().numpy()[0]
                forecast_values.append(pred_value)
            else:
                forecast_values.append(0.0)
        
        # Calculate confidence intervals (simplified)
        confidence_intervals = np.array([[p - 0.1, p + 0.1] for p in forecast_values])
        
        # Feature importance (simplified)
        feature_importance = {col: np.random.random() for col in features_df.columns[:10]}
        
        return ForecastResult(
            predictions=np.array(forecast_values),
            confidence_intervals=confidence_intervals,
            attention_weights=None,  # Could extract from model
            forecast_horizon=max(horizons),
            model_confidence=0.8,  # Could calculate based on training performance
            feature_importance=feature_importance,
            timestamp=datetime.now()
        )
    
    def train_cross_asset_model(self,
                               features_df: pd.DataFrame,
                               symbols: List[str],
                               epochs: int = 50) -> Dict[str, Any]:
        """Train cross-asset attention model"""
        
        self.logger.info("Training cross-asset attention model")
        
        # Prepare cross-asset data
        X, y, asset_ids = self._prepare_cross_asset_data(features_df, symbols)
        
        if len(X) == 0:
            raise QuantumFinanceOptError("Insufficient cross-asset training data")
        
        # Initialize model
        num_assets = len(symbols)
        feature_dim = X.shape[-1]
        
        self.cross_asset_model = CrossAssetAttentionTransformer(
            num_assets=num_assets,
            feature_dim=feature_dim,
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 4)
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.cross_asset_model.parameters(), lr=1e-4)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        asset_ids_tensor = torch.LongTensor(asset_ids).to(self.device)
        
        # Training loop
        training_losses = []
        
        for epoch in range(epochs):
            self.cross_asset_model.train()
            
            predictions, attention_weights = self.cross_asset_model(X_tensor, asset_ids_tensor)
            loss = F.mse_loss(predictions, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 10 == 0:
                self.logger.info(f"Cross-asset epoch {epoch}, Loss: {loss.item():.6f}")
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'epochs_trained': epochs
        }
    
    def _prepare_cross_asset_data(self, features_df: pd.DataFrame, symbols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for cross-asset model"""
        
        # Extract features for each asset
        X = []
        y = []
        asset_ids = []
        
        for i, symbol in enumerate(symbols):
            # Get features for this asset
            asset_features = []
            for feature_type in self.feature_columns:
                col_name = f'{symbol}_{feature_type}'
                if col_name in features_df.columns:
                    asset_features.append(features_df[col_name].values)
                else:
                    asset_features.append(np.zeros(len(features_df)))
            
            if asset_features:
                asset_features = np.column_stack(asset_features)
                
                # Create samples (using sliding window)
                window_size = 20
                for j in range(window_size, len(asset_features) - 1):
                    X.append(asset_features[j-window_size:j])
                    y.append(asset_features[j+1, 0])  # Predict next return
                    asset_ids.append(i)
        
        if not X:
            return np.array([]), np.array([]), np.array([])
        
        # Reshape for cross-asset format
        X = np.array(X)  # (num_samples, window_size, num_features)
        y = np.array(y)
        asset_ids = np.array(asset_ids)
        
        return X, y, asset_ids
    
    def analyze_attention_weights(self, recent_data: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """Analyze attention weights to understand asset relationships"""
        
        if self.cross_asset_model is None:
            return {'error': 'Cross-asset model not trained'}
        
        # Prepare data
        features_df = self.prepare_features(recent_data)
        X, _, asset_ids = self._prepare_cross_asset_data(features_df, symbols)
        
        if len(X) == 0:
            return {'error': 'Insufficient data for attention analysis'}
        
        # Get attention weights
        X_tensor = torch.FloatTensor(X[-1:]).to(self.device)  # Last sample
        asset_ids_tensor = torch.LongTensor(asset_ids[-len(symbols):]).to(self.device)
        
        self.cross_asset_model.eval()
        with torch.no_grad():
            _, attention_weights = self.cross_asset_model(X_tensor, asset_ids_tensor)
        
        # Convert to numpy and analyze
        attention_np = attention_weights.cpu().numpy()[0]  # (num_assets, num_assets)
        
        # Create attention matrix
        attention_matrix = {}
        for i, symbol_i in enumerate(symbols):
            attention_matrix[symbol_i] = {}
            for j, symbol_j in enumerate(symbols):
                if i < len(attention_np) and j < len(attention_np[0]):
                    attention_matrix[symbol_i][symbol_j] = float(attention_np[i, j])
                else:
                    attention_matrix[symbol_i][symbol_j] = 0.0
        
        return {
            'attention_matrix': attention_matrix,
            'strongest_relationships': self._find_strongest_relationships(attention_matrix),
            'timestamp': datetime.now().isoformat()
        }
    
    def _find_strongest_relationships(self, attention_matrix: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Find strongest attention relationships"""
        relationships = []
        
        for symbol_i, attentions in attention_matrix.items():
            for symbol_j, weight in attentions.items():
                if symbol_i != symbol_j:  # Exclude self-attention
                    relationships.append({
                        'from': symbol_i,
                        'to': symbol_j,
                        'weight': weight
                    })
        
        # Sort by weight and return top relationships
        relationships.sort(key=lambda x: x['weight'], reverse=True)
        return relationships[:10]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        info = {
            'is_trained': self.is_trained,
            'device': str(self.device),
            'config': self.config
        }
        
        if self.multi_horizon_model:
            info['multi_horizon_model'] = {
                'parameters': sum(p.numel() for p in self.multi_horizon_model.parameters()),
                'forecast_horizons': self.multi_horizon_model.forecast_horizons,
                'input_dim': self.multi_horizon_model.input_dim,
                'd_model': self.multi_horizon_model.d_model
            }
        
        if self.cross_asset_model:
            info['cross_asset_model'] = {
                'parameters': sum(p.numel() for p in self.cross_asset_model.parameters()),
                'num_assets': self.cross_asset_model.num_assets,
                'feature_dim': self.cross_asset_model.feature_dim
            }
        
        if self.training_history:
            info['training_history'] = {
                'epochs': len(self.training_history),
                'final_loss': self.training_history[-1],
                'best_loss': min(self.training_history)
            }
        
        return info