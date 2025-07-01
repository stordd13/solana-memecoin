"""
Unified LSTM Model for All Horizon Directional Prediction
This model loads pre-engineered features from the feature_engineering module.

IMPORTANT WORKFLOW:
1. Run feature_engineering/advanced_feature_engineering.py FIRST
2. Then run this script to train on pre-engineered features

Improved with per-token robust scaling instead of global scaling.
"""

import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ML.utils.winsorizer import Winsorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ML.utils.metrics_helpers import financial_classification_metrics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.subplots as sp
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
from ML.utils.training_plots import plot_training_curves, create_learning_summary

# --- Configuration ---
CONFIG = {
    'base_dir': Path("data/features"),  # CHANGED: Read from features dir instead of cleaned
    'features_dir': Path("data/features"),  # Pre-engineered features directory
    'results_dir': Path("ML/results/unified_lstm"),
    'categories': [
        "normal_behavior_tokens",      # Highest quality for training
        "tokens_with_extremes",        # Valuable volatile patterns  
        "dead_tokens",                 # Complete token lifecycles
        # "tokens_with_gaps",          # EXCLUDED: Data quality issues
    ],
    'sequence_length': 60,  # 1 hour lookback
    'horizons': [15, 30, 60, 120, 240, 360, 720],  # All prediction horizons
    'batch_size': 128,
    'epochs': 50,
    'learning_rate': 0.001,
    'random_state': 42,
    'early_stopping_patience': 10,
    'val_size': 0.2,
    'test_size': 0.2,
    'hidden_size': 32,   # Larger due to more horizons
    'num_layers': 2,     # Deeper network
    'dropout': 0.2,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    # Note: deduplicate_tokens removed - handled upstream in data_analysis
}

# --- Dataset Class with Pre-Engineered Features ---
class UnifiedDirectionalDataset(Dataset):
    """Dataset that loads pre-engineered features instead of raw prices."""
    
    def __init__(self, 
                 data_paths: List[Path], 
                 sequence_length: int,
                 horizons: List[int]):
        
        self.sequence_length = sequence_length
        self.horizons = sorted(horizons)
        
        self.sequences = []
        self.labels = []
        self.returns = []
        self.token_scalers = {}  # Per-token scalers for features
        
        self._load_data(data_paths)
    
    def _load_data(self, data_paths: List[Path]):
        """Load pre-engineered features and create sequences."""
        valid_files = []
        
        print(f"Loading pre-engineered features from {len(data_paths)} files...")
        
        # First pass: check valid files
        for path in tqdm(data_paths, desc="1/2 Validating feature files"):
            try:
                token_name = path.stem
                features_path = CONFIG['features_dir'] / f"{token_name}_features.parquet"
                
                if not features_path.exists():
                    continue
                
                features_df = pl.read_parquet(features_path)
                
                # Check if we have enough data for sequences
                if len(features_df) < self.sequence_length + max(self.horizons):
                    continue
                
                # Check for required columns
                required_cols = ['price', 'datetime']
                if not all(col in features_df.columns for col in required_cols):
                    continue
                    
                valid_files.append((path, features_df))
                
            except Exception as e:
                print(f"Error loading features for {path.name}: {e}")
                continue

        print(f"Found {len(valid_files)} tokens with valid pre-engineered features")
        
        # Second pass: create sequences with per-token scaling
        for path, features_df in tqdm(valid_files, desc="2/2 Creating feature sequences"):
            try:
                token_id = path.stem
                
                # Select numeric feature columns (exclude metadata)
                exclude_cols = ['datetime', 'token_id', 'price']
                feature_cols = [col for col in features_df.columns 
                              if col not in exclude_cols and features_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                
                if len(feature_cols) < 10:  # Need at least 10 features
                    print(f"Not enough features for {token_id}: {len(feature_cols)}")
                    continue
                
                # Extract feature matrix
                feature_matrix = features_df[feature_cols].to_numpy()
                prices = features_df['price'].to_numpy()
                
                # Create per-token winsorizer for features - better for crypto!
                winsorizer = Winsorizer(lower_percentile=0.005, upper_percentile=0.995)
                winsorizer.fit(feature_matrix)
                self.token_scalers[token_id] = winsorizer
                
                self._create_sequences(feature_matrix, prices, token_id, winsorizer)
                
            except Exception as e:
                print(f"Error processing features for {path.name}: {e}")
                continue

        print(f"Created {len(self.sequences)} feature sequences total")
        
        if len(self.sequences) > 0:
            self.sequences = torch.FloatTensor(self.sequences)
            self.labels = torch.FloatTensor(self.labels)
            self.returns = torch.FloatTensor(self.returns)
        else:
            print("WARNING: No feature sequences were created! Check feature engineering output.")
    
    def _create_sequences(self, feature_matrix: np.ndarray, prices: np.ndarray, token_id: str, winsorizer: Winsorizer):
        """Create sequences from normalized features with proper handling of NaN values"""
        
        # Apply winsorization to features
        features_norm = winsorizer.transform(feature_matrix)
        
        for i in range(self.sequence_length, len(features_norm)):
            # Extract sequence
            seq = features_norm[i-self.sequence_length:i]
            
            # Check if we have valid prices for all required horizons
            valid_sequence = True
            horizon_labels = []
            horizon_returns = []
            
            current_price = prices[i]
            if np.isnan(current_price):
                continue
                
            for h in self.horizons:
                if i + h < len(prices):
                    future_price = prices[i + h]
                    if not np.isnan(future_price):
                        label = 1.0 if future_price > current_price else 0.0
                        horizon_labels.append(label)
                        # Calculate return for financial metrics
                        horizon_returns.append((future_price - current_price) / current_price)
                    else:
                        valid_sequence = False
                        break
                else:
                    valid_sequence = False
                    break

            if not valid_sequence:
                continue

            # Interpolate any remaining NaNs in the sequence
            if np.isnan(seq).any():
                for col in range(seq.shape[1]):
                    col_data = seq[:, col]
                    mask = np.isnan(col_data)
                    if mask.any() and not mask.all():
                        seq[mask, col] = np.interp(np.where(mask)[0], np.where(~mask)[0], col_data[~mask])

            self.sequences.append(seq)
            self.labels.append(horizon_labels)
            self.returns.append(horizon_returns)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.returns[idx]


# --- Unified Model Architecture ---
class UnifiedLSTMPredictor(nn.Module):
    """LSTM model for multi-horizon directional prediction using pre-engineered features."""
    
    def __init__(self, 
                 input_size: int,  # Number of features per timestep
                 hidden_size: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 horizons: List[int] = [15, 30, 60, 120, 240, 360, 720]):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Add batch normalization after LSTM
        self.lstm_bn = nn.BatchNorm1d(hidden_size)
        
        # Shared feature extractor with batch normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Create a separate output head for each prediction horizon with batch norm
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in horizons
        ])
    
    def forward(self, x):
        # x is already feature sequences, not raw prices
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Apply batch normalization to LSTM output
        last_hidden = self.lstm_bn(last_hidden)
        
        # Extract shared features
        features = self.feature_extractor(last_hidden)
        
        # Generate predictions for each horizon
        outputs = [head(features) for head in self.output_heads]
        outputs = torch.cat(outputs, dim=1)
        
        return torch.sigmoid(outputs)


# --- Focal Loss for Class Imbalance ---
class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance better than standard BCE"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, config):
    """Train the unified directional prediction model with loss tracking."""
    # Use MPS (Apple Silicon GPU) if available, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)
    
    # Use focal loss instead of BCE
    criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\nTraining Unified LSTM on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        
        # Add progress bar for training batches
        from tqdm import tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training")
        for batch_x, batch_y, batch_r in train_pbar:
            batch_x, batch_y, batch_r = batch_x.to(device), batch_y.to(device), batch_r.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Validation", leave=False)
            for batch_x, batch_y, batch_r in val_pbar:
                batch_x, batch_y, batch_r = batch_x.to(device), batch_y.to(device), batch_r.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')
    
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Return model and training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    return model, training_history

def evaluate_model(model, test_loader, horizons, device='cpu'):
    """Evaluate the model on the test set with both standard and financial metrics."""
    from ML.utils.metrics_helpers import financial_classification_metrics
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU.")
        device = 'cpu'
    model.eval()
    all_preds, all_targets, all_returns = [], [], []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating on test set")
        for batch_x, batch_y, batch_r in test_pbar:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.numpy())
            all_returns.append(batch_r.numpy())
            
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    returns = np.vstack(all_returns)
    
    metrics = {}
    for i, h in enumerate(horizons):
        preds_h = predictions[:, i]
        targets_h = targets[:, i]
        returns_h = returns[:, i]
        binary_preds_h = (preds_h >= 0.5).astype(int)
        
        # Get comprehensive financial metrics
        horizon_metrics = financial_classification_metrics(
            y_true=targets_h,
            y_pred=binary_preds_h,
            returns=returns_h,
            y_prob=preds_h
        )
        
        # Add horizon prefix to all metrics
        metrics[f"{h}min"] = {f"{h}min_{k}": v for k, v in horizon_metrics.items()}
        
    return metrics

def plot_metrics(metrics: Dict):
    """Plot key metrics for all horizons."""
    horizons = list(metrics.keys())
    
    # Create subplots for better organization
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'Performance Summary'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Individual metric plots
    metric_names = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    
    for idx, (metric, color) in enumerate(zip(metric_names, colors)):
        row, col = positions[idx]
        values = [metrics[h][metric] for h in horizons]
        
        fig.add_trace(go.Bar(
            x=horizons,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            marker_color=color,
            showlegend=False
        ), row=row, col=col)
        
        # Add baseline line at 50% for reference
        if metric in ['accuracy', 'f1_score']:
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                         annotation_text="50% Random", row=row, col=col)
    
    # Summary plot - all metrics together
    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        values = [metrics[h][metric] for h in horizons]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=horizons,
            y=values,
            marker_color=color,
            showlegend=True
        ), row=2, col=3)
    
    fig.update_layout(
        height=800,
        title_text='Unified LSTM: Performance with Pre-Engineered Features',
        barmode='group'
    )
    
    return fig


def main():
    """Main training pipeline for the unified directional model."""
    print("="*50)
    print("Training Unified LSTM Directional Model")
    print("USING PRE-ENGINEERED FEATURES")
    print("All Horizons: 15min, 30min, 1h, 2h, 4h, 6h, 12h")
    print("WITH WALK-FORWARD VALIDATION")
    print("="*50)
    
    # Check if feature engineering has been run
    if not CONFIG['features_dir'].exists():
        print(f"\n❌ ERROR: Features directory not found: {CONFIG['features_dir']}")
        print("\n🔧 REQUIRED STEP: Run feature engineering first!")
        print("   python feature_engineering/advanced_feature_engineering.py")
        print("\nThis will create the pre-engineered features needed for training.")
        return
    
    # Create results directory
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {CONFIG['results_dir']}")
    
    # Load paths and split data
    all_paths = []
    for category in CONFIG['categories']:
        cat_dir = CONFIG['base_dir'] / category
        if cat_dir.exists():
            all_paths.extend(list(cat_dir.glob("*.parquet")))
    
    if not all_paths:
        print(f"ERROR: No files found in {CONFIG['base_dir']}. Exiting.")
        return

    # Import walk-forward splitter
    from ML.utils.walk_forward_splitter import WalkForwardSplitter
    
    print("\n🔄 Using Walk-Forward Validation instead of temporal split")
    
    # For LSTM, we'll use per-token walk-forward validation
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
    
    # Create datasets using the saved splits
    train_dataset = UnifiedDirectionalDataset(train_paths, CONFIG['sequence_length'], CONFIG['horizons'])
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Check feature engineering output.")
        return

    val_dataset = UnifiedDirectionalDataset(val_paths, CONFIG['sequence_length'], CONFIG['horizons'])
    test_dataset = UnifiedDirectionalDataset(test_paths, CONFIG['sequence_length'], CONFIG['horizons'])
    
    print(f"\n📊 Dataset sizes (walk-forward):")
    print(f"  Training: {len(train_dataset):,} sequences")
    print(f"  Validation: {len(val_dataset):,} sequences") 
    print(f"  Test: {len(test_dataset):,} sequences")
    
    # Determine input size from first batch
    sample_sequence, _, _ = train_dataset[0]
    input_size = sample_sequence.shape[1]  # Number of features per timestep
    print(f"Using {input_size} features per timestep for LSTM input")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize and train model
    model_params = {
        'input_size': input_size,
        'hidden_size': CONFIG['hidden_size'],
        'num_layers': CONFIG['num_layers'],
        'dropout': CONFIG['dropout'],
        'horizons': CONFIG['horizons']
    }
    model = UnifiedLSTMPredictor(**model_params)
    model, training_history = train_model(model, train_loader, val_loader, CONFIG)
    
    # Evaluate
    print("\nEvaluating on test set (walk-forward)...")
    metrics = evaluate_model(model, test_loader, CONFIG['horizons'])
    
    print("\nTest Set Metrics (Walk-Forward Validation):")
    for horizon, m in metrics.items():
        print(f"  Horizon {horizon}:")
        for name, val in m.items():
            print(f"    {name.title()}: {val:.2%}")

    # Save model and results
    model_path = CONFIG['results_dir'] / 'unified_lstm_model_walkforward.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'token_scalers': train_dataset.token_scalers,
        'metrics': metrics,
        'input_size': input_size,
        'training_history': training_history,
        'validation_method': 'walk_forward'
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Create and save performance metrics plot
    fig = plot_metrics(metrics)
    fig.update_layout(title="Unified LSTM Performance (Walk-Forward Validation)")
    metrics_path = CONFIG['results_dir'] / 'unified_lstm_metrics_walkforward.html'
    fig.write_html(metrics_path)
    print(f"Metrics plot saved to: {metrics_path}")
    
    # Create and save training curves
    training_fig = plot_training_curves(
        training_history['train_losses'],
        training_history['val_losses'],
        title="Unified LSTM Training Progress (Walk-Forward)"
    )
    training_path = CONFIG['results_dir'] / 'training_curves_walkforward.html'
    training_fig.write_html(training_path)
    print(f"Training curves saved to: {training_path}")
    
    # Save metrics as JSON for easy analysis
    metrics_json_path = CONFIG['results_dir'] / 'metrics_walkforward.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_json_path}")
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary split files")
    
    print(f"\n✅ Walk-forward validation training complete!")
    print(f"   More realistic metrics due to temporal validation")
    print(f"   Results saved to: {CONFIG['results_dir']}")

# --- Data Preparation ---
def prepare_data_fixed(data_paths: List[Path], 
                      horizons: List[int], 
                      sequence_length: int,
                      split_type: str = 'all') -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    FIXED: Load pre-engineered features and create sequences
    Now reads from features/[category]/ structure instead of flat directory
    """
    all_sequences = []
    all_labels = []
    all_tokens = []
    processed_tokens = 0
    
    print(f"Loading pre-engineered features for {len(data_paths)} tokens ({split_type} split)...")
    
    for path in tqdm(data_paths, desc=f"Loading {split_type} features"):
        try:
            # The path is now pointing to features/[category]/[token].parquet
            # which is the actual feature file - no need to reconstruct the path
            token_name = path.stem
            
            # Since we're already reading from features dir, just load directly
            if path.exists() and path.suffix == '.parquet':
                features_df = load_features_from_file(path)
            else:
                print(f"Feature file not found: {path}")
                continue
            
            if features_df is None or len(features_df) == 0:
                continue
            
            # Create directional labels
            features_df = create_labels_for_horizons(features_df, horizons)
            
            # CRITICAL FIX: Split FIRST, then create sequences
            n_rows = len(features_df) if hasattr(features_df, '__len__') else features_df.height
            
            if split_type == 'train':
                start_idx = 0
                end_idx = int(n_rows * 0.6)
            elif split_type == 'val':
                start_idx = int(n_rows * 0.6)
                end_idx = int(n_rows * 0.8)
            elif split_type == 'test':
                start_idx = int(n_rows * 0.8)
                end_idx = n_rows
            else:  # 'all' for backwards compatibility
                start_idx = 0
                end_idx = n_rows
            
            # Only include if we have enough samples for sequences
            if end_idx - start_idx < sequence_length + 50:  # Need extra samples for sequences
                continue
                
            # Split the features temporally
            token_split = features_df.slice(start_idx, end_idx - start_idx)
            
            # Get the features (assuming feature columns start after basic columns)
            feature_cols = [col for col in token_split.columns 
                          if col not in ['datetime', 'price', 'timestamp', 'token_id'] 
                          and not col.startswith('label_')]
            
            # Extract features and labels
            features = token_split.select(feature_cols).to_numpy()
            label_cols = [f'label_{h}m' for h in horizons]
            labels = token_split.select(label_cols).to_numpy()
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                seq = features[i:i+sequence_length]
                label = labels[i+sequence_length-1]  # Labels at the end of sequence
                
                # Skip if any label is null
                if not np.isnan(label).any():
                    all_sequences.append(seq)
                    all_labels.append(label)
                    all_tokens.append(token_name)
            
            processed_tokens += 1
                
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            continue
    
    print(f"Successfully processed {processed_tokens} tokens")
    
    if not all_sequences:
        print("ERROR: No valid sequences created!")
        return torch.empty(0), torch.empty(0), []
    
    # Convert to tensors
    X = torch.FloatTensor(np.array(all_sequences))
    y = torch.FloatTensor(np.array(all_labels))
    
    print(f"{split_type.upper()} set: {X.shape[0]:,} sequences from {processed_tokens} tokens")
    print(f"Sequence shape: {X.shape}, Labels shape: {y.shape}")
    
    return X, y, all_tokens


if __name__ == "__main__":
    main()
