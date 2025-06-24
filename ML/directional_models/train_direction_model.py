"""
Directional Prediction LSTM Model for Memecoin Price Movement
This model predicts UP/DOWN price movement over multiple time horizons.
"""

import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.subplots as sp
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Data splitting function (copied from train_lstm_model.py for consistency)
def smart_data_split(all_paths: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split data intelligently - since most tokens are from 2025,
    use random split with stratification by category
    """
    
    print("\nAnalyzing data distribution...")
    
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
        # Shuffle paths
        np.random.shuffle(paths)
        
        # Calculate splits
        n = len(paths)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        # Assign to splits
        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:n_train + n_val])
        test_paths.extend(paths[n_train + n_val:])
        
        print(f"\n{category}:")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")
    
    # Shuffle final lists
    np.random.shuffle(train_paths)
    np.random.shuffle(val_paths)
    np.random.shuffle(test_paths)
    
    print(f"\nTotal split:")
    print(f"  Train: {len(train_paths)} files")
    print(f"  Val: {len(val_paths)} files")
    print(f"  Test: {len(test_paths)} files")
    
    return train_paths, val_paths, test_paths

# --- Configuration ---
CONFIG = {
    'base_dir': Path("data/cleaned"),
    'results_dir': Path("ML/results/direction_lstm_short_term"),
    'categories': [
        "normal_behavior_tokens",
        "tokens_with_gaps",
        "tokens_with_extremes",
    ],
    'lookback': 60,  # 1 hour lookback for short-term patterns
    'horizons': [15, 30, 60],  # 15min, 30min, 1h - short-term trading
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3
}


# --- Dataset Class ---
class DirectionalMemecoinDataset(Dataset):
    """Dataset for loading memecoin data for directional prediction."""
    
    def __init__(self, 
                 data_paths: List[Path], 
                 lookback: int, 
                 horizons: List[int],
                 scaler: Optional[StandardScaler] = None):
        
        self.lookback = lookback
        self.horizons = sorted(horizons)
        self.max_horizon = self.horizons[-1]
        
        self.sequences = []
        self.labels = []
        
        self.scaler = scaler if scaler else StandardScaler()
        self.fit_scaler = scaler is None
        
        self._load_data(data_paths)
    
    def _load_data(self, data_paths: List[Path]):
        """Load all parquet files and create sequences and labels."""
        all_prices = []
        valid_files = []
        
        print(f"Processing {len(data_paths)} files for dataset creation...")
        
        if self.fit_scaler:
            for path in tqdm(data_paths, desc="1/2 Collecting prices for scaler"):
                try:
                    df = pl.read_parquet(path)
                    if 'price' not in df.columns or len(df) < self.lookback + self.max_horizon:
                        continue
                    
                    prices = df['price'].to_numpy()
                    
                    # Drop tokens with NaN values for cleaner baseline
                    if np.isnan(prices).any():
                        print(f"Dropping {path.name}: has {np.isnan(prices).sum()} NaN values")
                        continue
                        
                    all_prices.extend(prices)
                    valid_files.append(path)
                except Exception as e:
                    print(f"Error loading {path.name}: {e}")
                    continue
                    
            if all_prices:
                self.scaler.fit(np.array(all_prices).reshape(-1, 1))
                # Prevent division-by-zero scaling (flat price series)
                if hasattr(self.scaler, 'scale_'):
                    self.scaler.scale_[self.scaler.scale_ == 0] = 1.0
            else:
                print("WARNING: No valid price data found for scaler fitting!")
                return
        else:
            valid_files = data_paths

        print(f"Found {len(valid_files)} valid files (need at least {self.lookback + self.max_horizon} data points)")
        
        for path in tqdm(valid_files, desc="2/2 Creating sequences"):
            try:
                df = pl.read_parquet(path)
                prices = df['price'].to_numpy()
                
                # Drop tokens with NaN values for cleaner baseline
                if np.isnan(prices).any():
                    print(f"Dropping {path.name}: has {np.isnan(prices).sum()} NaN values")
                    continue
                    
                self._create_sequences(prices)
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue

        print(f"Created {len(self.sequences)} sequences total")
        
        if len(self.sequences) > 0:
            self.sequences = torch.FloatTensor(self.sequences)
            self.labels = torch.FloatTensor(self.labels)
        else:
            print("WARNING: No sequences were created! Check data requirements.")
    
    def _create_sequences(self, prices: np.ndarray):
        """Create sequences and corresponding directional labels."""
        prices_norm = self.scaler.transform(prices.reshape(-1, 1)).flatten()
        stride = 5  # Use a stride to generate more diverse samples
        
        for i in range(0, len(prices_norm) - self.lookback - self.max_horizon + 1, stride):
            seq = prices_norm[i : i + self.lookback]

            # Skip if too many NaNs in inputs (allow up to 20% NaN)
            nan_ratio = np.isnan(seq).sum() / len(seq)
            if nan_ratio > 0.2:
                continue

            current_price = prices[i + self.lookback - 1]
            if np.isnan(current_price):
                continue
                
            horizon_labels = []
            valid_sequence = True
            for h in self.horizons:
                future_price = prices[i + self.lookback + h - 1]
                # Skip if future price is NaN
                if np.isnan(future_price):
                    valid_sequence = False
                    break
                label = 1.0 if future_price > current_price else 0.0
                horizon_labels.append(label)

            if not valid_sequence:
                continue

            # Interpolate any remaining NaNs in the sequence
            if np.isnan(seq).any():
                mask = np.isnan(seq)
                seq[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], seq[~mask])

            self.sequences.append(seq)
            self.labels.append(horizon_labels)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def _load_and_normalize_data(self, file_path: Path) -> np.ndarray:
        """Load price data and normalize it"""
        df = pl.read_parquet(file_path)
        
        # Handle any remaining NaN values from cleaning process
        # Some tokens_with_gaps files have intentionally unfilled large gaps
        prices = df['price'].to_numpy()
        
        # Forward fill NaN values first
        mask = np.isnan(prices)
        if mask.any():
            # Forward fill
            indices = np.where(~mask)[0]
            if len(indices) > 0:
                for i in range(len(prices)):
                    if mask[i]:
                        # Find the last valid value before this point
                        prev_valid = indices[indices < i]
                        if len(prev_valid) > 0:
                            prices[i] = prices[prev_valid[-1]]
                        else:
                            # If no previous valid value, use next valid value
                            next_valid = indices[indices > i]
                            if len(next_valid) > 0:
                                prices[i] = prices[next_valid[0]]
        
        # If still have NaN values, this file is too corrupted
        if np.isnan(prices).any():
            print(f"Warning: {file_path.name} still has NaN after forward fill, skipping")
            return None
        
        # Normalize prices using min-max scaling (avoid issues with very small prices)
        price_min = np.min(prices)
        price_max = np.max(prices)
        
        if price_max == price_min:
            print(f"Warning: {file_path.name} has constant prices, skipping")
            return None
            
        prices_norm = (prices - price_min) / (price_max - price_min)
        return prices_norm


# --- Model Architecture ---
class DirectionalLSTMPredictor(nn.Module):
    """LSTM model for multi-horizon directional prediction."""
    
    def __init__(self, 
                 input_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 horizons: List[int] = [15, 60, 240]):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
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
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        outputs = [head(last_hidden) for head in self.output_heads]
        outputs = torch.cat(outputs, dim=1)
        
        return torch.sigmoid(outputs)


# --- Training and Evaluation ---
def train_model(model, train_loader, val_loader, config):
    """Train the directional prediction model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
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
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Val Loss: {val_loss/len(val_loader):.4f}')
    return model

def evaluate_model(model, test_loader, horizons):
    """Evaluate using classification metrics for each horizon."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        
        metrics[f'{h}m'] = {
            'accuracy': accuracy_score(targets_h, binary_preds_h),
            'precision': precision_score(targets_h, binary_preds_h, zero_division=0),
            'recall': recall_score(targets_h, binary_preds_h, zero_division=0),
            'f1_score': f1_score(targets_h, binary_preds_h, zero_division=0),
            'roc_auc': roc_auc_score(targets_h, preds_h)
        }
    return metrics

def plot_metrics(metrics: Dict):
    """Plot evaluation metrics as a grouped bar chart."""
    horizons = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig = go.Figure()
    for name in metric_names:
        fig.add_trace(go.Bar(
            name=name.replace('_', ' ').title(),
            x=horizons,
            y=[metrics[h][name] for h in horizons],
            text=[f"{metrics[h][name]:.2f}" for h in horizons],
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title='Directional Model Performance by Horizon',
        xaxis_title='Prediction Horizon',
        yaxis_title='Score',
        yaxis_range=[0,1],
        legend_title='Metric'
    )
    return fig


# --- Main Pipeline ---
def main():
    """Main training pipeline for the short-term directional model."""
    print("="*50)
    print("Training Short-Term Directional LSTM Model")
    print("Horizons: 15min, 30min, 1h")
    print("="*50)
    
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
    print("\nCreating datasets...")
    train_dataset = DirectionalMemecoinDataset(train_paths, CONFIG['lookback'], CONFIG['horizons'])
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty. Check data paths and file content.")
        return

    val_dataset = DirectionalMemecoinDataset(val_paths, CONFIG['lookback'], CONFIG['horizons'], scaler=train_dataset.scaler)
    test_dataset = DirectionalMemecoinDataset(test_paths, CONFIG['lookback'], CONFIG['horizons'], scaler=train_dataset.scaler)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize and train model
    model_params = {
        'hidden_size': CONFIG['hidden_size'],
        'num_layers': CONFIG['num_layers'],
        'dropout': CONFIG['dropout'],
        'horizons': CONFIG['horizons']
    }
    model = DirectionalLSTMPredictor(**model_params)
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
    model_path = CONFIG['results_dir'] / 'short_term_lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'scaler': train_dataset.scaler,
        'metrics': metrics
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Create and save plot
    fig = plot_metrics(metrics)
    metrics_path = CONFIG['results_dir'] / 'short_term_lstm_metrics.html'
    fig.write_html(metrics_path)
    print(f"Metrics plot saved to: {metrics_path}")
    
    # Save metrics as JSON for easy analysis
    metrics_json_path = CONFIG['results_dir'] / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_json_path}")

if __name__ == "__main__":
    main() 