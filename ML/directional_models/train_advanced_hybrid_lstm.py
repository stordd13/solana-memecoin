"""
Advanced Hybrid LSTM Model with Attention for Memecoin Directional Prediction

This model incorporates three key innovations:
1. Attention mechanisms for focusing on important timepoints
2. Multi-scale feature extraction (different time windows)
3. Hybrid fixed + expanding window approach

Designed to outperform basic LSTM by capturing both short-term momentum 
and long-term lifecycle patterns with attention-based feature selection.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    'results_dir': Path("ML/results/advanced_hybrid_lstm"),
    'categories': [
        "normal_behavior_tokens",
        "tokens_with_extremes",
        "dead_tokens",
    ],
    # Multi-scale windows for feature extraction
    'fixed_windows': [15, 60, 240],  # 15m, 1h, 4h
    'expanding_window_min': 60,       # Minimum 1 hour
    'expanding_window_max': 720,      # Maximum 12 hours
    'horizons': [15, 30, 60, 120, 240, 360, 720],
    'batch_size': 64,  # Smaller due to more complex model
    'epochs': 100,
    'learning_rate': 0.0005,
    'early_stopping_patience': 15,
    'hidden_size': 128,
    'attention_heads': 8,
    'dropout': 0.3,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
}


# --- Advanced Dataset with Multi-Scale Features ---
class AdvancedHybridDataset(Dataset):
    """Dataset that creates multi-scale fixed windows + expanding window features"""
    
    def __init__(self, 
                 data_paths: List[Path],
                 fixed_windows: List[int],
                 expanding_min: int,
                 expanding_max: int,
                 horizons: List[int]):
        
        self.fixed_windows = sorted(fixed_windows)
        self.expanding_min = expanding_min
        self.expanding_max = expanding_max
        self.horizons = sorted(horizons)
        
        self.samples = []
        self.token_winsorizers = {}
        
        self._load_data(data_paths)
    
    def _load_data(self, data_paths: List[Path]):
        """Load data and create multi-scale samples"""
        
        print(f"Loading {len(data_paths)} files for advanced hybrid dataset...")
        
        for path in tqdm(data_paths, desc="Processing tokens"):
            try:
                # Load pre-engineered features
                features_df = pl.read_parquet(path)
                
                if len(features_df) < self.expanding_min + max(self.horizons):
                    continue
                
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
                
                # Fit winsorizer
                winsorizer = Winsorizer(lower_percentile=0.005, upper_percentile=0.995)
                winsorizer.fit(feature_matrix)
                self.token_winsorizers[token_id] = winsorizer
                
                # Normalize features
                features_norm = winsorizer.transform(feature_matrix)
                
                # Create samples
                self._create_multi_scale_samples(features_norm, prices, token_id)
                
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue
        
        print(f"Created {len(self.samples)} multi-scale samples")
    
    def _create_multi_scale_samples(self, features: np.ndarray, prices: np.ndarray, token_id: str):
        """Create samples with multiple fixed windows and expanding window"""
        
        # Apply winsorization
        winsorizer = self.token_winsorizers[token_id]
        features_norm = winsorizer.transform(features)
        
        # Calculate the minimum start position
        min_fixed_start = max(self.fixed_windows)
        
        # Process expanding window range
        for exp_end in range(self.expanding_min, min(self.expanding_max + 1, len(features_norm))):
            # Skip if we don't have enough data after expanding window
            if exp_end + max(self.horizons) >= len(features_norm):
                continue
            
            # Create fixed window sequences
            fixed_seqs = {}
            valid = True
            
            for window in self.fixed_windows:
                start_idx = exp_end - window
                if start_idx < 0:
                    valid = False
                    break
                fixed_seq = features_norm[start_idx:exp_end]
                
                # Skip if too many NaNs
                if np.isnan(fixed_seq).sum() / fixed_seq.size > 0.1:
                    valid = False
                    break
                fixed_seqs[window] = fixed_seq
            
            if not valid:
                continue
            
            # Create expanding window sequence
            expanding_seq = features_norm[:exp_end]
            
            # Create labels and returns for all horizons
            current_price = prices[exp_end]
            if np.isnan(current_price):
                continue
            
            labels = []
            returns = []
            valid_mask = []
            
            for h in self.horizons:
                future_idx = exp_end + h
                if future_idx < len(prices):
                    future_price = prices[future_idx]
                    if not np.isnan(future_price):
                        label = 1.0 if future_price > current_price else 0.0
                        ret = (future_price - current_price) / current_price
                        labels.append(label)
                        returns.append(ret)
                        valid_mask.append(True)
                    else:
                        labels.append(0.0)
                        returns.append(0.0)
                        valid_mask.append(False)
                else:
                    labels.append(0.0)
                    returns.append(0.0)
                    valid_mask.append(False)
            
            # Only add if we have at least one valid horizon
            if any(valid_mask):
                self.samples.append({
                    'fixed_sequences': fixed_seqs,
                    'expanding_sequence': expanding_seq,
                    'labels': np.array(labels, dtype=np.float32),
                    'returns': np.array(returns, dtype=np.float32),
                    'valid_mask': np.array(valid_mask, dtype=bool),
                    'expanding_length': len(expanding_seq),
                    'token_id': token_id
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        fixed_tensors = {
            window: torch.FloatTensor(seq) 
            for window, seq in sample['fixed_sequences'].items()
        }
        
        return (
            fixed_tensors,
            torch.FloatTensor(sample['expanding_sequence']),
            torch.FloatTensor(sample['labels']),
            torch.FloatTensor(sample['returns']),
            torch.BoolTensor(sample['valid_mask']),
            sample['expanding_length']
        )


def collate_hybrid_batch(batch):
    """Custom collate function for hybrid dataset"""
    fixed_seqs_batch = {window: [] for window in CONFIG['fixed_windows']}
    expanding_seqs = []
    labels = []
    returns = []
    valid_masks = []
    expanding_lengths = []
    
    for fixed_seqs, expanding_seq, label, ret, valid_mask, exp_len in batch:
        # Collect fixed sequences
        for window, seq in fixed_seqs.items():
            fixed_seqs_batch[window].append(torch.FloatTensor(seq))
        
        # Collect expanding sequences
        expanding_seqs.append(torch.FloatTensor(expanding_seq))
        labels.append(label)
        returns.append(ret)
        valid_masks.append(valid_mask)
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
        torch.stack(labels),
        torch.stack(returns),
        torch.stack(valid_masks),
        torch.LongTensor(expanding_lengths)
    )


# --- Advanced Hybrid LSTM Architecture ---
class AdvancedHybridLSTM(nn.Module):
    """
    Advanced LSTM with:
    - Multi-scale fixed window processing
    - Expanding window with attention
    - Cross-modal attention between fixed and expanding
    - Batch normalization for improved training
    """
    
    def __init__(self,
                 input_size: int,
                 fixed_windows: List[int],
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 attention_heads: int = 8,
                 dropout: float = 0.3,
                 horizons: List[int] = [15, 30, 60, 120, 240, 360, 720]):
        super().__init__()
        
        self.fixed_windows = fixed_windows
        self.hidden_size = hidden_size
        
        # Fixed window LSTMs with batch norm
        self.fixed_lstms = nn.ModuleDict()
        self.fixed_bns = nn.ModuleDict()
        for window in fixed_windows:
            self.fixed_lstms[str(window)] = nn.LSTM(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.fixed_bns[str(window)] = nn.BatchNorm1d(hidden_size)
        
        # Expanding window LSTM with batch norm
        self.expanding_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.expanding_bn = nn.BatchNorm1d(hidden_size)
        
        # Self-attention for expanding window
        self.self_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_bn = nn.BatchNorm1d(hidden_size)
        
        # Cross-attention (expanding queries, fixed keys/values)
        combined_fixed_size = hidden_size * len(fixed_windows)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_bn = nn.BatchNorm1d(hidden_size)
        
        # Fusion layer with batch norm
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
        
        # Multi-horizon prediction heads with batch norm
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.BatchNorm1d(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)
            ) for _ in horizons
        ])
    
    def forward(self, fixed_sequences, expanding_sequence, expanding_lengths):
        batch_size = expanding_sequence.shape[0]
        
        # Process fixed window sequences
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
        
        # Apply batch norm to attention output
        expanding_final = self.self_attention_bn(expanding_final)
        
        # Cross-attention: expanding queries, fixed keys/values
        # Expand fixed features to sequence format for attention
        fixed_expanded = fixed_combined.unsqueeze(1)  # (batch, 1, hidden)
        expanding_query = expanding_final.unsqueeze(1)  # (batch, 1, hidden)
        
        cross_attended, _ = self.cross_attention(
            expanding_query,
            fixed_expanded,
            fixed_expanded
        )
        cross_attended = cross_attended.squeeze(1)
        cross_attended = self.cross_attention_bn(cross_attended)
        
        # Fusion of all features
        fused_features = self.fusion_layer(
            torch.cat([expanding_final, cross_attended], dim=-1)
        )
        
        # Final feature extraction
        final_features = self.feature_extractor(fused_features)
        
        # Generate predictions for each horizon
        predictions = []
        for head in self.horizon_heads:
            pred = head(final_features)
            predictions.append(pred)
        
        output = torch.cat(predictions, dim=1)
        return torch.sigmoid(output), attention_weights


# --- Masked Loss Function ---
def masked_bce_loss(predictions, labels, valid_masks):
    """
    Compute BCE loss only for valid horizons.
    
    Args:
        predictions: (batch_size, num_horizons) - model predictions
        labels: (batch_size, num_horizons) - true labels
        valid_masks: (batch_size, num_horizons) - True for valid horizons
    """
    # Apply mask
    valid_predictions = predictions[valid_masks]
    valid_labels = labels[valid_masks]
    
    if len(valid_predictions) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return nn.functional.binary_cross_entropy(valid_predictions, valid_labels)


# --- Training Functions ---
def train_advanced_model(model, train_loader, val_loader, config):
    """Train the advanced hybrid model with detailed logging"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    
    model = model.to(device)
    
    # Use focal loss for class imbalance
    criterion = nn.BCELoss()  # Using BCE since we apply sigmoid in model
    
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
    
    print(f"\nTraining Advanced Hybrid LSTM on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            fixed_seqs, expanding_seq, labels, returns, valid_masks, exp_lengths = batch_data
            
            # Move to device
            fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
            expanding_seq = expanding_seq.to(device)
            labels = labels.to(device)
            returns = returns.to(device)
            valid_masks = valid_masks.to(device)
            exp_lengths = exp_lengths.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
            
            # Masked loss: only compute loss for valid horizons
            loss = masked_bce_loss(outputs, labels, valid_masks)
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
                fixed_seqs, expanding_seq, labels, returns, valid_masks, exp_lengths = batch_data
                
                fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
                expanding_seq = expanding_seq.to(device)
                labels = labels.to(device)
                returns = returns.to(device)
                valid_masks = valid_masks.to(device)
                exp_lengths = exp_lengths.to(device)
                
                outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
                loss = masked_bce_loss(outputs, labels, valid_masks)
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
            torch.save(model.state_dict(), 'best_advanced_model.pth')
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
    model.load_state_dict(torch.load('best_advanced_model.pth'))
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': val_losses.index(best_val_loss) + 1,
        'best_val_loss': best_val_loss
    }
    
    return model, training_history


def evaluate_advanced_model(model, test_loader, horizons, device=None):
    """Evaluate the advanced model with detailed metrics including financial metrics"""
    from ML.utils.metrics_helpers import financial_classification_metrics
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
    
    model.eval()
    all_preds = []
    all_labels = []
    all_returns = []
    all_valid_masks = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            fixed_seqs, expanding_seq, labels, returns, valid_masks, exp_lengths = batch_data
            
            fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
            expanding_seq = expanding_seq.to(device)
            exp_lengths = exp_lengths.to(device)
            
            outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_returns.append(returns.numpy())
            all_valid_masks.append(valid_masks.numpy())
    
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_labels)
    returns = np.vstack(all_returns)
    valid_masks = np.vstack(all_valid_masks)
    
    # Calculate metrics for each horizon (only on valid samples)
    metrics = {}
    for i, h in enumerate(horizons):
        # Get valid samples for this horizon
        horizon_mask = valid_masks[:, i]
        valid_preds = predictions[horizon_mask, i]
        valid_targets = targets[horizon_mask, i]
        valid_returns = returns[horizon_mask, i]
        
        if len(valid_preds) == 0:
            # No valid samples for this horizon
            horizon_name = f'{h}min'
            metrics[horizon_name] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1_score': 0.0, 'roc_auc': 0.5, 'valid_samples': 0
            }
            continue
        
        binary_preds = (valid_preds > 0.5).astype(int)
        horizon_name = f'{h}min'
        
        # Get comprehensive financial metrics
        horizon_metrics = financial_classification_metrics(
            y_true=valid_targets,
            y_pred=binary_preds,
            returns=valid_returns,
            y_prob=valid_preds
        )
        
        # Add additional metrics
        horizon_metrics['valid_samples'] = len(valid_preds)
        horizon_metrics['positive_class_ratio'] = np.mean(valid_targets)
        horizon_metrics['predictions_positive_ratio'] = np.mean(binary_preds)
        
        # Store with horizon prefix
        metrics[horizon_name] = {f"{horizon_name}_{k}": v for k, v in horizon_metrics.items()}
    
    return metrics


# --- Main Training Pipeline ---
def main():
    """Main training pipeline for advanced hybrid LSTM"""
    
    print("="*60)
    print("🚀 Advanced Hybrid LSTM Training")
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
    
    print("\n🔄 Using Walk-Forward Validation for Advanced Hybrid LSTM")
    
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
    
    # Split by token (per-token walk-forward)
    print("\n🔀 Creating per-token walk-forward splits...")
    token_splits = splitter.split_by_token(
        combined_data, 
        token_column='token_id',
        time_column='datetime',  # Assuming datetime column exists
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
    print("\nCreating advanced hybrid datasets...")
    train_dataset = AdvancedHybridDataset(
        train_paths,
        CONFIG['fixed_windows'],
        CONFIG['expanding_window_min'],
        CONFIG['expanding_window_max'],
        CONFIG['horizons']
    )
    
    val_dataset = AdvancedHybridDataset(
        val_paths,
        CONFIG['fixed_windows'],
        CONFIG['expanding_window_min'],
        CONFIG['expanding_window_max'],
        CONFIG['horizons']
    )
    
    test_dataset = AdvancedHybridDataset(
        test_paths,
        CONFIG['fixed_windows'],
        CONFIG['expanding_window_min'],
        CONFIG['expanding_window_max'],
        CONFIG['horizons']
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
        collate_fn=collate_hybrid_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_hybrid_batch,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_hybrid_batch,
        num_workers=0
    )
    
    # Get feature size from first sample
    sample = train_dataset[0]
    input_size = sample[0][CONFIG['fixed_windows'][0]].shape[1]
    
    print(f"\nInput feature size: {input_size}")
    
    # Initialize model
    model = AdvancedHybridLSTM(
        input_size=input_size,
        fixed_windows=CONFIG['fixed_windows'],
        hidden_size=CONFIG['hidden_size'],
        attention_heads=CONFIG['attention_heads'],
        dropout=CONFIG['dropout'],
        horizons=CONFIG['horizons']
    )
    
    # Train model
    model, training_history = train_advanced_model(
        model, train_loader, val_loader, CONFIG
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set (walk-forward)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    model = model.to(device)
    
    metrics = evaluate_advanced_model(model, test_loader, CONFIG['horizons'], device)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS (Walk-Forward Validation)")
    print("="*60)
    
    for horizon, m in metrics.items():
        print(f"\nHorizon {horizon}:")
        for metric_name, value in m.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.2%}")
    
    # Save model and results
    model_path = CONFIG['results_dir'] / 'advanced_hybrid_lstm_model_walkforward.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
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
        title="Advanced Hybrid LSTM Training Progress (Walk-Forward)"
    )
    training_path = CONFIG['results_dir'] / 'training_curves_walkforward.html'
    training_fig.write_html(training_path)
    
    # 2. Performance comparison plot
    from ML.directional_models.train_unified_lstm_model import plot_metrics
    metrics_fig = plot_metrics(metrics)
    metrics_fig.update_layout(title="Advanced Hybrid LSTM Performance (Walk-Forward)")
    metrics_path = CONFIG['results_dir'] / 'performance_metrics_walkforward.html'
    metrics_fig.write_html(metrics_path)
    
    # 3. Save metrics JSON
    metrics_json_path = CONFIG['results_dir'] / 'metrics_walkforward.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary split files")
    
    print(f"\nResults saved to: {CONFIG['results_dir']}")
    print("\n✅ Advanced Hybrid LSTM training complete with walk-forward validation!")
    print(f"   More realistic metrics due to temporal validation")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main() 