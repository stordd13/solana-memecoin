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
from sklearn.preprocessing import RobustScaler
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
    'base_dir': Path("data/cleaned"),
    'features_dir': Path("data/features"),  # NEW: Pre-engineered features directory
    'results_dir': Path("ML/results/unified_lstm_directional"),
    'categories': [
        "normal_behavior_tokens",      # Highest quality for training
        "tokens_with_extremes",        # Valuable volatile patterns  
        "dead_tokens",                 # Complete token lifecycles
        # "tokens_with_gaps",          # EXCLUDED: Data quality issues
    ],
    'sequence_length': 60,  # Use 60 features as sequence (instead of raw prices)
    'horizons': [15, 30, 60, 120, 240, 360, 720],  # ALL horizons: 15min, 30min, 1h, 2h, 4h, 6h, 12h
    'batch_size': 32,    # Smaller batch due to longer sequences
    'num_epochs': 30,
    'learning_rate': 0.001,
    'hidden_size': 32,   # Larger due to more horizons
    'num_layers': 2,     # Deeper network
    'dropout': 0.2,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    # Note: deduplicate_tokens removed - handled upstream in data_analysis
}

def smart_data_split(all_paths: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split data intelligently with stratification by category"""
    print("\nAnalyzing data distribution...")
    
    # Note: Token deduplication is now handled upstream in data_analysis
    # Each token should appear in exactly one category folder
    
    # Group paths by category
    category_paths = {}
    for path in all_paths:
        category = path.parent.name
        if category not in category_paths:
            category_paths[category] = []
        category_paths[category].append(path)
    
    # Display distribution
    print("\nFiles per category:")
    for cat, paths in category_paths.items():
        print(f"  {cat}: {len(paths)} files")
    
    # Stratified split by category
    train_paths, val_paths, test_paths = [], [], []
    
    for category, paths in category_paths.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:n_train + n_val])
        test_paths.extend(paths[n_train + n_val:])
        
        print(f"\n{category}:")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")
    
    np.random.shuffle(train_paths)
    np.random.shuffle(val_paths)
    np.random.shuffle(test_paths)
    
    print(f"\nTotal split:")
    print(f"  Train: {len(train_paths)} files")
    print(f"  Val: {len(val_paths)} files")
    print(f"  Test: {len(test_paths)} files")
    
    return train_paths, val_paths, test_paths

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
                
                # Create per-token scaler for features - key improvement!
                scaler = RobustScaler()
                scaler.fit(feature_matrix)
                self.token_scalers[token_id] = scaler
                
                self._create_sequences(feature_matrix, prices, token_id, scaler)
                
            except Exception as e:
                print(f"Error processing features for {path.name}: {e}")
                continue

        print(f"Created {len(self.sequences)} feature sequences total")
        
        if len(self.sequences) > 0:
            self.sequences = torch.FloatTensor(self.sequences)
            self.labels = torch.FloatTensor(self.labels)
        else:
            print("WARNING: No feature sequences were created! Check feature engineering output.")
    
    def _create_sequences(self, feature_matrix: np.ndarray, prices: np.ndarray, token_id: str, scaler: RobustScaler):
        """Create sequences from pre-engineered features."""
        features_norm = scaler.transform(feature_matrix)
        
        # Create overlapping sequences
        max_horizon = max(self.horizons)
        for i in range(len(features_norm) - self.sequence_length - max_horizon + 1):
            # Use sequence of features instead of raw prices
            seq = features_norm[i:i + self.sequence_length]
            
            # Skip if too many NaNs in inputs
            nan_ratio = np.isnan(seq).sum() / seq.size
            if nan_ratio > 0.1:
                continue

            current_price = prices[i + self.sequence_length - 1]
            if np.isnan(current_price):
                continue
                
            horizon_labels = []
            valid_sequence = True
            for h in self.horizons:
                future_idx = i + self.sequence_length + h - 1
                if future_idx < len(prices):
                    future_price = prices[future_idx]
                    if np.isnan(future_price):
                        valid_sequence = False
                        break
                    label = 1.0 if future_price > current_price else 0.0
                    horizon_labels.append(label)
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
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


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
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Create a separate output head for each prediction horizon
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in horizons
        ])
    
    def forward(self, x):
        # x is already feature sequences, not raw prices
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
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
    """Train the unified directional prediction model."""
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
    
    print(f"\nTraining on {device}...")
    for epoch in range(config['num_epochs']):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')
    return model

def evaluate_model(model, test_loader, horizons):
    """Evaluate using classification metrics for each horizon."""
    # Use MPS (Apple Silicon GPU) if available, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            all_preds.append(model(batch_x).cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    metrics = {}
    for i, h in enumerate(horizons):
        preds_h = predictions[:, i]
        targets_h = targets[:, i]
        binary_preds_h = (preds_h > 0.5).astype(int)
        
        if h >= 60:
            horizon_name = f'{h//60}h'
        else:
            horizon_name = f'{h}m'
        
        metrics[horizon_name] = {
            'accuracy': accuracy_score(targets_h, binary_preds_h),
            'precision': precision_score(targets_h, binary_preds_h, zero_division=0),
            'recall': recall_score(targets_h, binary_preds_h, zero_division=0),
            'f1_score': f1_score(targets_h, binary_preds_h, zero_division=0),
            'roc_auc': roc_auc_score(targets_h, preds_h)
        }
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

    train_paths, val_paths, test_paths = smart_data_split(all_paths)
    
    # Create datasets
    print("\nCreating datasets with pre-engineered features...")
    train_dataset = UnifiedDirectionalDataset(train_paths, CONFIG['sequence_length'], CONFIG['horizons'])
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Check feature engineering output.")
        return

    val_dataset = UnifiedDirectionalDataset(val_paths, CONFIG['sequence_length'], CONFIG['horizons'])
    test_dataset = UnifiedDirectionalDataset(test_paths, CONFIG['sequence_length'], CONFIG['horizons'])
    
    # Determine input size from first batch
    sample_sequence, _ = train_dataset[0]
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
    model = train_model(model, train_loader, val_loader, CONFIG)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, CONFIG['horizons'])
    
    print("\nTest Set Metrics:")
    for horizon, m in metrics.items():
        print(f"  Horizon {horizon}:")
        for name, val in m.items():
            print(f"    {name.title()}: {val:.2%}")

    # Save model and results
    model_path = CONFIG['results_dir'] / 'unified_lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'token_scalers': train_dataset.token_scalers,
        'metrics': metrics,
        'input_size': input_size
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Create and save plot
    fig = plot_metrics(metrics)
    metrics_path = CONFIG['results_dir'] / 'unified_lstm_metrics.html'
    fig.write_html(metrics_path)
    print(f"Metrics plot saved to: {metrics_path}")
    
    # Save metrics as JSON for easy analysis
    metrics_json_path = CONFIG['results_dir'] / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_json_path}")

if __name__ == "__main__":
    main()
