"""
Advanced Hybrid LSTM for Memecoin Price Forecasting

This model adapts the Advanced Hybrid LSTM architecture for price prediction (regression)
instead of directional classification. It uses:

1. Multi-scale fixed window processing (15m, 1h, 4h)
2. Expanding window with attention mechanisms
3. Cross-attention between different time scales
4. Multi-horizon price prediction

Key differences from directional model:
- Regression output instead of classification
- MSE loss instead of BCE/Focal loss
- Price prediction heads instead of binary classification heads
- Continuous target values instead of binary labels
"""

import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from ML.utils.winsorizer import Winsorizer
from ML.utils.training_plots import plot_training_curves, create_learning_summary
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.subplots as sp
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
CONFIG = {
    'base_dir': Path("data/features"),
    'results_dir': Path("ML/results/advanced_hybrid_lstm_forecasting"),
    'categories': [
        "normal_behavior_tokens",
        "tokens_with_extremes",
        "dead_tokens",
    ],
    # Multi-scale windows for feature extraction
    'fixed_windows': [15, 60, 240],  # 15m, 1h, 4h
    'expanding_window_min': 60,       # Minimum 1 hour
    'expanding_window_max': 720,      # Maximum 12 hours
    'forecast_horizons': [15, 30, 60, 120, 240],  # Multiple forecasting horizons
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0005,
    'early_stopping_patience': 15,
    'hidden_size': 128,
    'attention_heads': 8,
    'dropout': 0.3,
}


# --- Advanced Forecasting Dataset ---
class AdvancedHybridForecastingDataset(Dataset):
    """Dataset for multi-scale forecasting with fixed + expanding windows"""
    
    def __init__(self, 
                 data_paths: List[Path],
                 fixed_windows: List[int],
                 expanding_min: int,
                 expanding_max: int,
                 forecast_horizons: List[int]):
        
        self.fixed_windows = sorted(fixed_windows)
        self.expanding_min = expanding_min
        self.expanding_max = expanding_max
        self.forecast_horizons = sorted(forecast_horizons)
        
        self.samples = []
        self.token_winsorizers = {}
        self.global_price_scaler = None
        
        self._load_data(data_paths)
    
    def _load_data(self, data_paths: List[Path]):
        """Load data and create multi-scale forecasting samples"""
        
        print(f"Loading {len(data_paths)} files for forecasting dataset...")
        
        # First pass: collect all prices for global scaling
        all_prices = []
        valid_files = []
        
        for path in tqdm(data_paths, desc="1/3 Collecting prices"):
            try:
                # Load pre-engineered features
                features_df = pl.read_parquet(path)
                
                if len(features_df) < self.expanding_min + max(self.forecast_horizons):
                    continue
                
                if 'price' not in features_df.columns:
                    continue
                
                prices = features_df['price'].to_numpy()
                
                # Skip if too many NaNs
                if np.isnan(prices).sum() / len(prices) > 0.1:
                    continue
                
                all_prices.extend(prices[~np.isnan(prices)])
                valid_files.append(path)
                
            except Exception as e:
                print(f"Error loading {path.name}: {e}")
                continue
        
        if not all_prices:
            raise ValueError("No valid price data found!")
        
        # Fit global price scaler
        self.global_price_scaler = Winsorizer(
            lower_percentile=0.005, 
            upper_percentile=0.995
        )
        self.global_price_scaler.fit(np.array(all_prices).reshape(-1, 1))
        print(f"Fitted global price scaler on {len(all_prices)} price points")
        
        # Second pass: process features and create samples
        for path in tqdm(valid_files, desc="2/3 Processing features"):
            try:
                features_df = pl.read_parquet(path)
                
                # Get feature columns
                exclude_cols = ['datetime', 'token_id', 'price']
                feature_cols = [col for col in features_df.columns 
                              if col not in exclude_cols and 
                              features_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                
                if len(feature_cols) < 10:
                    continue
                
                # Extract data
                feature_matrix = features_df[feature_cols].to_numpy()
                prices = features_df['price'].to_numpy()
                token_id = path.stem
                
                # Fit per-token winsorizer for features
                token_winsorizer = Winsorizer(
                    lower_percentile=0.005, 
                    upper_percentile=0.995
                )
                token_winsorizer.fit(feature_matrix)
                self.token_winsorizers[token_id] = token_winsorizer
                
                # Normalize features
                features_norm = token_winsorizer.transform(feature_matrix)
                
                # Create samples
                self._create_forecasting_samples(features_norm, prices, token_id)
                
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue
        
        print(f"Created {len(self.samples)} forecasting samples")
    
    def _create_forecasting_samples(self, features: np.ndarray, prices: np.ndarray, token_id: str):
        """Create samples for price forecasting"""
        
        max_horizon = max(self.forecast_horizons)
        max_window = max(self.fixed_windows)
        
        # Start from the point where we have enough history for largest fixed window
        for current_idx in range(max_window, len(features) - max_horizon):
            # Skip if expanding window would be too large
            expanding_length = min(current_idx, self.expanding_max)
            if expanding_length < self.expanding_min:
                continue
            
            sample = {
                'token_id': token_id,
                'fixed_sequences': {},
                'expanding_sequence': None,
                'current_idx': current_idx,
                'price_targets': []
            }
            
            # Extract fixed window sequences
            valid_sample = True
            for window_size in self.fixed_windows:
                start_idx = current_idx - window_size
                if start_idx >= 0:
                    seq = features[start_idx:current_idx]
                    if np.isnan(seq).sum() / seq.size > 0.1:  # Too many NaNs
                        valid_sample = False
                        break
                    sample['fixed_sequences'][window_size] = seq
                else:
                    valid_sample = False
                    break
            
            if not valid_sample:
                continue
            
            # Extract expanding window sequence
            expanding_start = max(0, current_idx - expanding_length)
            expanding_seq = features[expanding_start:current_idx]
            if np.isnan(expanding_seq).sum() / expanding_seq.size > 0.1:
                continue
            sample['expanding_sequence'] = expanding_seq
            sample['expanding_length'] = len(expanding_seq)
            
            # Create price targets for all horizons (Option 1: Per-horizon validation)
            current_price = prices[current_idx]
            if np.isnan(current_price):
                continue
            
            # Normalize current price
            current_price_norm = self.global_price_scaler.transform([[current_price]])[0, 0]
            
            sample['price_targets'] = []
            sample['valid_horizons_mask'] = []
            any_valid_horizon = False
            
            for h in self.forecast_horizons:
                future_idx = current_idx + h
                if future_idx < len(prices):
                    future_price = prices[future_idx]
                    if not np.isnan(future_price):
                        # Valid horizon
                        future_price_norm = self.global_price_scaler.transform([[future_price]])[0, 0]
                        sample['price_targets'].append(future_price_norm)
                        sample['valid_horizons_mask'].append(True)
                        any_valid_horizon = True
                    else:
                        # Invalid horizon (NaN price)
                        sample['price_targets'].append(current_price_norm)  # Placeholder (will be masked)
                        sample['valid_horizons_mask'].append(False)
                else:
                    # Invalid horizon (beyond token data)
                    sample['price_targets'].append(current_price_norm)  # Placeholder (will be masked)
                    sample['valid_horizons_mask'].append(False)
            
            # Keep sample if at least one horizon is valid
            if any_valid_horizon:
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample['fixed_sequences'],
            sample['expanding_sequence'],
            torch.FloatTensor(sample['price_targets']),
            torch.BoolTensor(sample['valid_horizons_mask']),
            sample['expanding_length']
        )


def collate_forecasting_batch(batch):
    """Custom collate function for forecasting dataset"""
    fixed_seqs_batch = {window: [] for window in CONFIG['fixed_windows']}
    expanding_seqs = []
    targets = []
    valid_horizons_masks = []
    expanding_lengths = []
    
    for fixed_seqs, expanding_seq, target, valid_mask, exp_len in batch:
        # Collect fixed sequences
        for window, seq in fixed_seqs.items():
            fixed_seqs_batch[window].append(torch.FloatTensor(seq))
        
        # Collect expanding sequences
        expanding_seqs.append(torch.FloatTensor(expanding_seq))
        targets.append(target)
        valid_horizons_masks.append(valid_mask)
        expanding_lengths.append(exp_len)
    
    # Stack fixed sequences (all same length per window)
    fixed_tensors = {}
    for window, seqs in fixed_seqs_batch.items():
        fixed_tensors[window] = torch.stack(seqs)
    
    # Pad expanding sequences
    expanding_padded = pad_sequence(expanding_seqs, batch_first=True, padding_value=0)
    
    return (
        fixed_tensors,
        expanding_padded,
        torch.stack(targets),
        torch.stack(valid_horizons_masks),
        torch.LongTensor(expanding_lengths)
    )


# --- Advanced Hybrid LSTM for Forecasting ---
class AdvancedHybridLSTMForecasting(nn.Module):
    """
    Advanced LSTM for price forecasting with:
    1. Multi-scale fixed window processing
    2. Expanding window with attention
    3. Cross-attention between scales
    4. Multi-horizon price prediction heads
    5. Batch normalization for improved training
    """
    
    def __init__(self,
                 input_size: int,
                 fixed_windows: List[int],
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 attention_heads: int = 8,
                 dropout: float = 0.3,
                 forecast_horizons: List[int] = [15, 30, 60, 120, 240]):
        super().__init__()
        
        self.fixed_windows = sorted(fixed_windows)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizons = forecast_horizons
        
        # Multi-scale fixed window LSTMs with batch norm
        self.fixed_lstms = nn.ModuleDict({
            str(window): nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size // len(fixed_windows),
                num_layers=1,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            ) for window in fixed_windows
        })
        
        # Batch normalization for fixed window features
        self.fixed_bns = nn.ModuleDict({
            str(window): nn.BatchNorm1d(hidden_size // len(fixed_windows))
            for window in fixed_windows
        })
        
        # Expanding window LSTM (larger capacity) with batch norm
        self.expanding_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.expanding_bn = nn.BatchNorm1d(hidden_size)
        
        # Self-attention for expanding window with batch norm
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attention_bn = nn.BatchNorm1d(hidden_size)
        
        # Cross-attention between fixed and expanding features with batch norm
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attention_bn = nn.BatchNorm1d(hidden_size)
        
        # Feature fusion layer with batch norm
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final feature extraction with batch norm
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-horizon prediction heads (regression) with batch norm
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)
            ) for _ in forecast_horizons
        ])
    
    def forward(self, fixed_sequences, expanding_sequence, expanding_lengths):
        batch_size = expanding_sequence.shape[0]
        
        # Process fixed window sequences with batch norm
        fixed_features = []
        for window in self.fixed_windows:
            seq = fixed_sequences[window]
            lstm_out, (h_n, _) = self.fixed_lstms[str(window)](seq)
            # Use last hidden state and apply batch norm
            h_n_last = h_n[-1]
            h_n_norm = self.fixed_bns[str(window)](h_n_last)
            fixed_features.append(h_n_norm)
        
        # Concatenate fixed features
        fixed_combined = torch.cat(fixed_features, dim=-1)
        
        # Process expanding window with packing for efficiency
        packed_expanding = pack_padded_sequence(
            expanding_sequence, 
            expanding_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        packed_lstm_out, (h_n_exp, _) = self.expanding_lstm(packed_expanding)
        lstm_out_exp, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        # Apply self-attention to expanding window outputs
        # Create attention mask for padded positions
        max_len = lstm_out_exp.shape[1]
        attention_mask = torch.arange(max_len).expand(
            batch_size, max_len
        ).to(expanding_lengths.device) >= expanding_lengths.unsqueeze(1)
        
        attended_exp, attention_weights = self.self_attention(
            lstm_out_exp, lstm_out_exp, lstm_out_exp,
            key_padding_mask=attention_mask
        )
        
        # Get final expanding representation (attended last valid position)
        expanding_final = []
        for i, length in enumerate(expanding_lengths):
            expanding_final.append(attended_exp[i, length-1])
        expanding_final = torch.stack(expanding_final)
        
        # Apply batch norm to self-attention output
        expanding_final = self.self_attention_bn(expanding_final)
        
        # Cross-attention: expanding queries, fixed keys/values
        fixed_expanded = fixed_combined.unsqueeze(1)  # (batch, 1, hidden)
        expanding_query = expanding_final.unsqueeze(1)
        
        cross_attended, _ = self.cross_attention(
            expanding_query,
            fixed_expanded,
            fixed_expanded
        )
        cross_attended = cross_attended.squeeze(1)
        
        # Apply batch norm to cross-attention output
        cross_attended = self.cross_attention_bn(cross_attended)
        
        # Fusion of all features
        fused_features = self.fusion_layer(
            torch.cat([expanding_final, cross_attended], dim=-1)
        )
        
        # Final feature extraction
        final_features = self.feature_extractor(fused_features)
        
        # Generate price predictions for each horizon
        predictions = []
        for head in self.horizon_heads:
            pred = head(final_features)
            predictions.append(pred)
        
        output = torch.cat(predictions, dim=1)
        return output, attention_weights


# --- Masked Loss Function ---
def masked_mse_loss(predictions, targets, valid_masks):
    """
    Compute MSE loss only for valid horizons.
    
    Args:
        predictions: (batch_size, num_horizons) - model predictions
        targets: (batch_size, num_horizons) - true targets
        valid_masks: (batch_size, num_horizons) - True for valid horizons
    """
    # Apply mask
    valid_predictions = predictions[valid_masks]
    valid_targets = targets[valid_masks]
    
    if len(valid_predictions) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return nn.functional.mse_loss(valid_predictions, valid_targets)


# --- Training Functions ---
def train_forecasting_model(model, train_loader, val_loader, config):
    """Train the advanced hybrid forecasting model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    
    model = model.to(device)
    
    # Optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.fixed_lstms.parameters(), 'lr': config['learning_rate']},
        {'params': model.expanding_lstm.parameters(), 'lr': config['learning_rate']},
        {'params': model.self_attention.parameters(), 'lr': config['learning_rate'] * 0.5},
        {'params': model.cross_attention.parameters(), 'lr': config['learning_rate'] * 0.5},
        {'params': model.fusion_layer.parameters(), 'lr': config['learning_rate'] * 2},
        {'params': model.feature_extractor.parameters(), 'lr': config['learning_rate'] * 2},
        {'params': model.horizon_heads.parameters(), 'lr': config['learning_rate'] * 2}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"\nTraining Advanced Hybrid LSTM Forecasting on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            fixed_seqs, expanding_seq, targets, valid_masks, exp_lengths = batch_data
            
            # Move to device
            fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
            expanding_seq = expanding_seq.to(device)
            targets = targets.to(device)
            valid_masks = valid_masks.to(device)
            exp_lengths = exp_lengths.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
            loss = masked_mse_loss(outputs, targets, valid_masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                fixed_seqs, expanding_seq, targets, valid_masks, exp_lengths = batch_data
                
                fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
                expanding_seq = expanding_seq.to(device)
                targets = targets.to(device)
                valid_masks = valid_masks.to(device)
                exp_lengths = exp_lengths.to(device)
                
                outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
                loss = masked_mse_loss(outputs, targets, valid_masks)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_forecasting_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Logging
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nEpoch [{epoch+1}/{config["epochs"]}]')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            print(f'  Best Val Loss: {best_val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_forecasting_model.pth'))
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': val_losses.index(best_val_loss) + 1,
        'best_val_loss': best_val_loss
    }
    
    return model, training_history


def evaluate_forecasting_model(model, test_loader, price_scaler, horizons, device=None):
    """Evaluate the forecasting model with regression metrics"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
    
    model.eval()
    all_preds = []
    all_targets = []
    all_valid_masks = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            fixed_seqs, expanding_seq, targets, valid_masks, exp_lengths = batch_data
            
            fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
            expanding_seq = expanding_seq.to(device)
            exp_lengths = exp_lengths.to(device)
            
            outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_valid_masks.append(valid_masks.numpy())
    
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    valid_masks = np.vstack(all_valid_masks)
    
    # Denormalize predictions and targets
    pred_prices = price_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    target_prices = price_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    # Calculate metrics for each horizon (only on valid samples)
    metrics = {}
    for i, h in enumerate(horizons):
        # Get valid samples for this horizon
        horizon_mask = valid_masks[:, i]
        valid_preds = pred_prices[horizon_mask, i]
        valid_targets = target_prices[horizon_mask, i]
        
        if len(valid_preds) == 0:
            # No valid samples for this horizon
            horizon_name = f'{h}m' if h < 60 else f'{h//60}h'
            metrics[horizon_name] = {
                'mse': 0.0, 'mae': 0.0, 'r2': 0.0, 
                'direction_accuracy': 0.0, 'mean_price_error_pct': 0.0, 'valid_samples': 0
            }
            continue
        
        horizon_name = f'{h}m' if h < 60 else f'{h//60}h'
        
        # Calculate direction accuracy (compare with first target price for baseline)
        baseline_prices = target_prices[horizon_mask, 0]  # Current prices for comparison
        pred_direction = np.sign(valid_preds - baseline_prices)
        true_direction = np.sign(valid_targets - baseline_prices)
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        metrics[horizon_name] = {
            'mse': mean_squared_error(valid_targets, valid_preds),
            'mae': mean_absolute_error(valid_targets, valid_preds),
            'r2': r2_score(valid_targets, valid_preds),
            'direction_accuracy': direction_accuracy,
            'mean_price_error_pct': np.mean(np.abs(valid_preds - valid_targets) / valid_targets) * 100,
            'valid_samples': len(valid_preds)
        }
    
    return metrics


# --- Main Training Pipeline ---
def main():
    """Main training pipeline for advanced hybrid LSTM forecasting with walk-forward validation"""
    
    print("="*60)
    print("🚀 Advanced Hybrid LSTM Forecasting Training")
    print("Features: Multi-scale extraction, Attention, Hybrid windows")
    print("WITH WALK-FORWARD VALIDATION")
    print("="*60)
    
    # Create results directory
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load data paths
    all_paths = []
    for category in CONFIG['categories']:
        cat_dir = CONFIG['base_dir'] / category
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.parquet"))
            all_paths.extend(paths)
            print(f"Found {len(paths)} files in {category}")
    
    if not all_paths:
        print("ERROR: No data files found!")
        return
    
    print(f"\nTotal files: {len(all_paths)}")
    
    # Import walk-forward splitter
    from ML.utils.walk_forward_splitter import WalkForwardSplitter
    
    print("\n🔄 Using Walk-Forward Validation for Advanced Hybrid LSTM Forecasting")
    
    # Load all data first to prepare for walk-forward splitting
    print("\n📊 Loading all feature data for walk-forward validation...")
    
    all_data_frames = []
    for path in tqdm(all_paths, desc="Loading feature files"):
        try:
            df = pl.read_parquet(path)
            if len(df) < 400:  # Minimum token length
                continue
            
            # Add token identifier
            df = df.with_columns(pl.lit(path.stem).alias('token_id'))
            all_data_frames.append(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if not all_data_frames:
        print("ERROR: No valid feature files found!")
        return
    
    # Combine all data
    combined_data = pl.concat(all_data_frames)
    print(f"Loaded {len(combined_data):,} rows from {len(all_data_frames)} tokens")
    
    # Setup walk-forward splitter
    splitter = WalkForwardSplitter(config='medium')  # Good for 400-2000 minute tokens
    
    # For LSTM forecasting, use per-token splits (better for sequence learning)
    print("\n🔀 Creating per-token walk-forward splits...")
    token_splits = splitter.split_by_token(
        combined_data, 
        token_column='token_id',
        time_column='datetime',
        min_token_length=400
    )
    
    print(f"Created walk-forward splits for {len(token_splits)} tokens")
    
    # Collect all folds for training
    all_train_data = []
    all_val_data = []  
    all_test_data = []
    
    # Use first N-1 folds for training/validation, last fold for testing
    for token_id, folds in token_splits.items():
        if len(folds) < 2:
            continue  # Need at least 2 folds
            
        # Use last fold as test, split remaining into train/val
        *train_val_folds, test_fold = folds
        test_train_df, test_test_df = test_fold
        all_test_data.append(test_test_df)
        
        # Split train_val_folds into train and validation
        n_train_folds = max(1, int(len(train_val_folds) * 0.8))
        
        for i, (fold_train_df, fold_test_df) in enumerate(train_val_folds):
            if i < n_train_folds:
                all_train_data.append(fold_test_df)  # Use 'test' part of fold for training
            else:
                all_val_data.append(fold_test_df)    # Use for validation
    
    print(f"\n📈 Walk-forward data split:")
    print(f"  Train folds: {len(all_train_data)} DataFrames")
    print(f"  Val folds: {len(all_val_data)} DataFrames")  
    print(f"  Test folds: {len(all_test_data)} DataFrames")
    
    # Create datasets from walk-forward splits
    print("\nCreating datasets with walk-forward splits...")
    
    # Convert DataFrames back to paths for compatibility with existing dataset code
    # Save temporary files for each split
    temp_dir = CONFIG['results_dir'] / 'temp_splits'
    temp_dir.mkdir(exist_ok=True)
    
    train_paths = []
    val_paths = []
    test_paths = []
    
    # Save train splits
    for i, df in enumerate(all_train_data):
        temp_path = temp_dir / f'train_fold_{i}.parquet'
        df.write_parquet(temp_path)
        train_paths.append(temp_path)
    
    # Save val splits
    for i, df in enumerate(all_val_data):
        temp_path = temp_dir / f'val_fold_{i}.parquet'
        df.write_parquet(temp_path)
        val_paths.append(temp_path)
        
    # Save test splits
    for i, df in enumerate(all_test_data):
        temp_path = temp_dir / f'test_fold_{i}.parquet'
        df.write_parquet(temp_path)
        test_paths.append(temp_path)

    # Create datasets
    print("\nCreating advanced hybrid forecasting datasets...")
    train_dataset = AdvancedHybridForecastingDataset(
        train_paths,
        CONFIG['fixed_windows'],
        CONFIG['expanding_window_min'],
        CONFIG['expanding_window_max'],
        CONFIG['forecast_horizons']
    )
    
    val_dataset = AdvancedHybridForecastingDataset(
        val_paths,
        CONFIG['fixed_windows'],
        CONFIG['expanding_window_min'],
        CONFIG['expanding_window_max'],
        CONFIG['forecast_horizons']
    )
    
    test_dataset = AdvancedHybridForecastingDataset(
        test_paths,
        CONFIG['fixed_windows'],
        CONFIG['expanding_window_min'],
        CONFIG['expanding_window_max'],
        CONFIG['forecast_horizons']
    )
    
    print(f"\nDataset sizes (walk-forward):")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples created!")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_forecasting_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_forecasting_batch,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_forecasting_batch,
        num_workers=0
    )
    
    # Get feature size from first sample
    sample = train_dataset[0]
    input_size = sample[0][CONFIG['fixed_windows'][0]].shape[1]
    
    print(f"\nInput feature size: {input_size}")
    
    # Initialize model
    model = AdvancedHybridLSTMForecasting(
        input_size=input_size,
        fixed_windows=CONFIG['fixed_windows'],
        hidden_size=CONFIG['hidden_size'],
        attention_heads=CONFIG['attention_heads'],
        dropout=CONFIG['dropout'],
        forecast_horizons=CONFIG['forecast_horizons']
    )
    
    # Train model
    model, training_history, price_scaler = train_forecasting_model(
        model, train_loader, val_loader, CONFIG
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set (walk-forward)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    model = model.to(device)
    
    metrics = evaluate_forecasting_model(
        model, test_loader, price_scaler, CONFIG['forecast_horizons'], device
    )
    
    # Print results
    print("\n" + "="*60)
    print("FORECASTING RESULTS (Walk-Forward Validation)")
    print("="*60)
    
    for horizon, m in metrics.items():
        print(f"\nHorizon {horizon}:")
        print(f"  RMSE: {m.get('rmse', 0):.6f}")
        print(f"  MAE: {m.get('mae', 0):.6f}")
        print(f"  MAPE: {m.get('mape', 0):.2f}%")
        print(f"  R²: {m.get('r2', 0):.4f}")
        if 'directional_accuracy' in m:
            print(f"  Directional Accuracy: {m['directional_accuracy']:.2%}")
    
    # Save model and results
    model_path = CONFIG['results_dir'] / 'advanced_hybrid_lstm_forecasting_walkforward.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'price_scaler': price_scaler,
        'token_winsorizers': train_dataset.token_winsorizers,
        'metrics': metrics,
        'input_size': input_size,
        'training_history': training_history,
        'validation_method': 'walk_forward'
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Create visualizations
    # 1. Training curves
    training_fig = plot_training_curves(
        training_history['train_losses'],
        training_history['val_losses'],
        title="Advanced Hybrid LSTM Forecasting Progress (Walk-Forward)"
    )
    training_path = CONFIG['results_dir'] / 'training_curves_walkforward.html'
    training_fig.write_html(training_path)
    
    # 2. Save metrics JSON
    metrics_json_path = CONFIG['results_dir'] / 'forecasting_metrics_walkforward.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary split files")
    
    print(f"\nResults saved to: {CONFIG['results_dir']}")
    print("\n✅ Advanced Hybrid LSTM forecasting training complete with walk-forward validation!")
    print(f"   More realistic forecasting metrics due to temporal validation")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()