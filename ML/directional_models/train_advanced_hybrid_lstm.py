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
        """Create samples with multiple fixed windows + expanding window"""
        
        max_horizon = max(self.horizons)
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
                'labels': []
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
            
            # Create labels for all horizons (Option 1: Per-horizon validation)
            current_price = prices[current_idx]
            if np.isnan(current_price):
                continue
            
            sample['labels'] = []
            sample['valid_horizons_mask'] = []
            any_valid_horizon = False
            
            for h in self.horizons:
                future_idx = current_idx + h
                if future_idx < len(prices):
                    future_price = prices[future_idx]
                    if not np.isnan(future_price):
                        # Valid horizon
                        label = 1.0 if future_price > current_price else 0.0
                        sample['labels'].append(label)
                        sample['valid_horizons_mask'].append(True)
                        any_valid_horizon = True
                    else:
                        # Invalid horizon (NaN price)
                        sample['labels'].append(0.0)  # Placeholder (will be masked)
                        sample['valid_horizons_mask'].append(False)
                else:
                    # Invalid horizon (beyond token data)
                    sample['labels'].append(0.0)  # Placeholder (will be masked)
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
            torch.FloatTensor(sample['labels']),
            torch.BoolTensor(sample['valid_horizons_mask']),
            sample['expanding_length']
        )


def collate_hybrid_batch(batch):
    """Custom collate function for hybrid dataset"""
    fixed_seqs_batch = {window: [] for window in CONFIG['fixed_windows']}
    expanding_seqs = []
    labels = []
    valid_horizons_masks = []
    expanding_lengths = []
    
    for fixed_seqs, expanding_seq, label, valid_mask, exp_len in batch:
        # Collect fixed sequences
        for window, seq in fixed_seqs.items():
            fixed_seqs_batch[window].append(torch.FloatTensor(seq))
        
        # Collect expanding sequences
        expanding_seqs.append(torch.FloatTensor(expanding_seq))
        labels.append(label)
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
        torch.stack(labels),
        torch.stack(valid_horizons_masks),
        torch.LongTensor(expanding_lengths)
    )


# --- Advanced Hybrid LSTM Architecture ---
class AdvancedHybridLSTM(nn.Module):
    """
    Advanced LSTM with:
    1. Multi-scale fixed window processing
    2. Expanding window with attention
    3. Cross-attention between scales
    4. Multi-horizon prediction heads
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
        
        self.fixed_windows = sorted(fixed_windows)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-scale fixed window LSTMs
        self.fixed_lstms = nn.ModuleDict({
            str(window): nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size // len(fixed_windows),
                num_layers=1,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            ) for window in fixed_windows
        })
        
        # Expanding window LSTM (larger capacity)
        self.expanding_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention for expanding window
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between fixed and expanding features
        combined_fixed_size = hidden_size  # Sum of all fixed LSTM outputs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-horizon prediction heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
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
            # Use last hidden state
            fixed_features.append(h_n[-1])
        
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
            fixed_seqs, expanding_seq, labels, valid_masks, exp_lengths = batch_data
            
            # Move to device
            fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
            expanding_seq = expanding_seq.to(device)
            labels = labels.to(device)
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
                fixed_seqs, expanding_seq, labels, valid_masks, exp_lengths = batch_data
                
                fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
                expanding_seq = expanding_seq.to(device)
                labels = labels.to(device)
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
    """Evaluate the advanced model with detailed metrics"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
    
    model.eval()
    all_preds = []
    all_labels = []
    all_valid_masks = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            fixed_seqs, expanding_seq, labels, valid_masks, exp_lengths = batch_data
            
            fixed_seqs = {k: v.to(device) for k, v in fixed_seqs.items()}
            expanding_seq = expanding_seq.to(device)
            exp_lengths = exp_lengths.to(device)
            
            outputs, _ = model(fixed_seqs, expanding_seq, exp_lengths)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_valid_masks.append(valid_masks.numpy())
    
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_labels)
    valid_masks = np.vstack(all_valid_masks)
    
    # Calculate metrics for each horizon (only on valid samples)
    metrics = {}
    for i, h in enumerate(horizons):
        # Get valid samples for this horizon
        horizon_mask = valid_masks[:, i]
        valid_preds = predictions[horizon_mask, i]
        valid_targets = targets[horizon_mask, i]
        
        if len(valid_preds) == 0:
            # No valid samples for this horizon
            horizon_name = f'{h}m' if h < 60 else f'{h//60}h'
            metrics[horizon_name] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1_score': 0.0, 'roc_auc': 0.5, 'valid_samples': 0
            }
            continue
        
        binary_preds = (valid_preds > 0.5).astype(int)
        horizon_name = f'{h}m' if h < 60 else f'{h//60}h'
        
        metrics[horizon_name] = {
            'accuracy': accuracy_score(valid_targets, binary_preds),
            'precision': precision_score(valid_targets, binary_preds, zero_division=0),
            'recall': recall_score(valid_targets, binary_preds, zero_division=0),
            'f1_score': f1_score(valid_targets, binary_preds, zero_division=0),
            'roc_auc': roc_auc_score(valid_targets, valid_preds) if len(np.unique(valid_targets)) > 1 else 0.5,
            'valid_samples': len(valid_preds)
        }
        
        # Add class distribution info
        pos_ratio = np.mean(valid_targets)
        metrics[horizon_name]['positive_class_ratio'] = pos_ratio
        metrics[horizon_name]['predictions_positive_ratio'] = np.mean(binary_preds)
    
    return metrics


# --- Main Training Pipeline ---
def main():
    """Main training pipeline for advanced hybrid LSTM"""
    
    print("="*60)
    print("🚀 Advanced Hybrid LSTM Training")
    print("Features: Multi-scale extraction, Attention, Hybrid windows")
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
    
    # Smart data split
    from ML.directional_models.train_unified_lstm_model import smart_data_split
    train_paths, val_paths, test_paths = smart_data_split(all_paths)
    
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
    
    print(f"\nDataset sizes:")
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
    print("\nEvaluating on test set...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    model = model.to(device)
    
    metrics = evaluate_advanced_model(model, test_loader, CONFIG['horizons'], device)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    for horizon, m in metrics.items():
        print(f"\nHorizon {horizon}:")
        print(f"  Accuracy: {m['accuracy']:.2%}")
        print(f"  Precision: {m['precision']:.2%}")
        print(f"  Recall: {m['recall']:.2%}")
        print(f"  F1 Score: {m['f1_score']:.2%}")
        print(f"  ROC AUC: {m['roc_auc']:.2%}")
        print(f"  Positive class ratio: {m['positive_class_ratio']:.2%}")
    
    # Save model and results
    model_path = CONFIG['results_dir'] / 'advanced_hybrid_lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'token_winsorizers': train_dataset.token_winsorizers,
        'metrics': metrics,
        'input_size': input_size,
        'training_history': training_history
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Create visualizations
    # 1. Training curves
    training_fig = plot_training_curves(
        training_history['train_losses'],
        training_history['val_losses'],
        title="Advanced Hybrid LSTM Training Progress"
    )
    training_path = CONFIG['results_dir'] / 'training_curves.html'
    training_fig.write_html(training_path)
    
    # 2. Performance comparison plot
    from ML.directional_models.train_unified_lstm_model import plot_metrics
    metrics_fig = plot_metrics(metrics)
    metrics_fig.update_layout(title="Advanced Hybrid LSTM Performance")
    metrics_path = CONFIG['results_dir'] / 'performance_metrics.html'
    metrics_fig.write_html(metrics_path)
    
    # 3. Save metrics JSON
    metrics_json_path = CONFIG['results_dir'] / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up
    if os.path.exists('best_advanced_model.pth'):
        os.remove('best_advanced_model.pth')
    
    print(f"\nResults saved to: {CONFIG['results_dir']}")
    print("\n✅ Advanced Hybrid LSTM training complete!")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main() 