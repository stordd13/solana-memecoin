"""
LSTM Training Script for Memecoin Price Prediction
Designed for category-aware cleaned data using Polars
"""

import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.subplots as sp
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


# ================== DATASET CLASS ==================
class MemecoinDataset(Dataset):
    """Dataset for loading category-aware cleaned memecoin data"""
    
    def __init__(self, 
                 data_paths: List[Path], 
                 lookback: int = 60, 
                 forecast_horizon: int = 15,
                 scaler: Optional[StandardScaler] = None):
        
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.sequences = []
        self.labels = []
        self.metadata = []
        
        self.scaler = scaler if scaler else StandardScaler()
        self.fit_scaler = scaler is None
        
        self._load_data(data_paths)
    
    def _load_data(self, data_paths: List[Path]):
        """Load all parquet files and create sequences"""
        all_prices = []
        valid_files = []
        
        print(f"Loading {len(data_paths)} files...")
        
        # First pass: collect all prices for scaler fitting
        if self.fit_scaler:
            for path in tqdm(data_paths, desc="Collecting prices for scaler"):
                try:
                    df = pl.read_parquet(path)
                    
                    if 'price' not in df.columns:
                        continue
                    
                    prices = df['price'].to_numpy()
                    
                    # Drop tokens with NaN values for cleaner baseline
                    if np.isnan(prices).any():
                        print(f"Dropping {path.name}: has {np.isnan(prices).sum()} NaN values")
                        continue
                    
                    if len(prices) < self.lookback + self.forecast_horizon:
                        continue
                    
                    all_prices.extend(prices)
                    valid_files.append(path)
                    
                except Exception as e:
                    print(f"Error loading {path.name}: {e}")
            
            # Fit scaler on all collected prices
            if all_prices:
                print(f"Fitting scaler on {len(all_prices)} price points...")
                self.scaler.fit(np.array(all_prices).reshape(-1, 1))
            else:
                raise ValueError("No valid price data found for scaler fitting!")
        else:
            # If scaler is provided, all files are potentially valid
            valid_files = data_paths
        
        # Second pass: create sequences with fitted scaler
        for path in tqdm(valid_files, desc="Creating sequences"):
            try:
                df = pl.read_parquet(path)
                
                if 'datetime' in df.columns:
                    df = df.sort('datetime')
                
                prices = df['price'].to_numpy()
                
                # Drop tokens with NaN values for cleaner baseline
                if np.isnan(prices).any():
                    print(f"Dropping {path.name}: has {np.isnan(prices).sum()} NaN values")
                    continue
                
                if len(prices) < self.lookback + self.forecast_horizon:
                    continue
                
                # Extract metadata
                token_name = path.stem
                category = path.parent.name
                
                # Create sequences
                self._create_sequences(prices, token_name, category)
                
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
        
        # Convert to tensors
        if len(self.sequences) > 0:
            self.sequences = torch.FloatTensor(self.sequences)
            self.labels = torch.FloatTensor(self.labels)
            print(f"Created {len(self.sequences)} training sequences")
        else:
            raise ValueError("No sequences created! Check your data.")
    
    def _create_sequences(self, prices: np.ndarray, token_name: str, category: str):
        """Create overlapping sequences from price data"""
        prices_norm = self.scaler.transform(prices.reshape(-1, 1)).flatten()
        
        # Less overlap for longer horizons
        stride = max(1, self.forecast_horizon // 3)
        
        for i in range(0, len(prices_norm) - self.lookback - self.forecast_horizon + 1, stride):
            seq = prices_norm[i:i + self.lookback]
            target = prices_norm[i + self.lookback:i + self.lookback + self.forecast_horizon]
            
            self.sequences.append(seq)
            self.labels.append(target)
            self.metadata.append({
                'token': token_name,
                'category': category,
                'start_idx': i
            })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ================== MODEL ARCHITECTURE ==================
class LSTMPredictor(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, 
                 input_size: int = 1,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 forecast_horizon: int = 15):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon)
        )
    
    def forward(self, x):
        # Add feature dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Generate predictions
        predictions = self.fc(last_hidden)
        
        return predictions


# ================== DATA SPLITTING ==================
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


# ================== DEVICE SELECTION ==================
def get_device():
    """Get best available device: MPS (Apple Silicon) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# ================== TRAINING FUNCTIONS ==================
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 50,
                learning_rate: float = 0.001,
                device: str = None):
    """Train the LSTM model"""
    
    if device is None:
        device = get_device()
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining on {device}...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model_checkpoint.pth'))
    
    return train_losses, val_losses


# ================== EVALUATION METRICS ==================
def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader,
                   scaler: StandardScaler,
                   device: str = None):
    """Calculate trading-specific metrics"""
    
    if device is None:
        device = get_device()
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Inverse transform to get actual prices
    pred_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    target_prices = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = np.mean((pred_prices - target_prices) ** 2)
    metrics['mae'] = np.mean(np.abs(pred_prices - target_prices))
    
    # Direction accuracy
    pred_direction = np.sign(predictions[:, -1] - predictions[:, 0])
    true_direction = np.sign(targets[:, -1] - targets[:, 0])
    metrics['direction_accuracy'] = np.mean(pred_direction == true_direction)
    
    # Pump detection (2x = 100% gain)
    entry_prices = scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
    pred_returns = (pred_prices[:, -1] - entry_prices) / entry_prices
    true_returns = (target_prices[:, -1] - entry_prices) / entry_prices
    
    # Multiple pump thresholds
    for threshold, name in [(0.5, '50%'), (1.0, '2x'), (4.0, '5x')]:
        pred_pumps = pred_returns > threshold
        true_pumps = true_returns > threshold
        
        if np.sum(pred_pumps) > 0:
            metrics[f'pump_{name}_precision'] = np.sum(pred_pumps & true_pumps) / np.sum(pred_pumps)
        else:
            metrics[f'pump_{name}_precision'] = 0
            
        if np.sum(true_pumps) > 0:
            metrics[f'pump_{name}_recall'] = np.sum(pred_pumps & true_pumps) / np.sum(true_pumps)
        else:
            metrics[f'pump_{name}_recall'] = 0
    
    return metrics, predictions, targets


# ================== PLOTTING FUNCTIONS ==================
def plot_training_history(train_losses: List[float], val_losses: List[float]) -> go.Figure:
    """Plot training and validation loss"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(train_losses) + 1)),
        y=train_losses,
        mode='lines',
        name='Train Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(val_losses) + 1)),
        y=val_losses,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_evaluation_metrics(metrics: Dict) -> go.Figure:
    """Plot evaluation metrics as grouped bar chart"""
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=('Detection Performance', 'Pump Detection by Threshold'),
        column_widths=[0.4, 0.6]
    )
    
    # Basic metrics
    basic_metrics = ['direction_accuracy']
    basic_values = [metrics.get(m, 0) for m in basic_metrics]
    
    fig.add_trace(
        go.Bar(
            x=['Direction\nAccuracy'],
            y=basic_values,
            text=[f'{v:.1%}' for v in basic_values],
            textposition='auto',
            marker_color='green' if basic_values[0] > 0.5 else 'orange'
        ),
        row=1, col=1
    )
    
    # Pump detection metrics
    thresholds = ['50%', '2x', '5x']
    precisions = [metrics.get(f'pump_{t}_precision', 0) for t in thresholds]
    recalls = [metrics.get(f'pump_{t}_recall', 0) for t in thresholds]
    
    fig.add_trace(
        go.Bar(
            name='Precision',
            x=thresholds,
            y=precisions,
            text=[f'{v:.1%}' for v in precisions],
            textposition='auto',
            marker_color='blue'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Recall',
            x=thresholds,
            y=recalls,
            text=[f'{v:.1%}' for v in recalls],
            textposition='auto',
            marker_color='red'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Model Performance Metrics',
        showlegend=True,
        template='plotly_white',
        barmode='group'
    )
    
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    
    return fig


# ================== MAIN TRAINING PIPELINE ==================
def main():
    """Main training pipeline"""
    
    # Configuration
    CONFIG = {
        'base_dir': Path("data/features"),  # CHANGED: Read from features dir instead of cleaned
        'results_dir': Path("ML/results/lstm"),
        'categories': [
            "normal_behavior_tokens",      # Best quality
            "tokens_with_gaps",           # Gaps filled during cleaning
            "tokens_with_extremes",       # Extremes preserved!
            # "dead_tokens",              # See smart sampling below
        ],
        'include_smart_dead_tokens': True,  # Smart sampling of dead tokens
        'scaler_type': 'robust',  # 'standard', 'robust', 'log', 'percentile'
        'lookback': 60,
        'forecast_horizon': 15,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3
    }
    
    print("="*50)
    print("LSTM Training for Memecoin Price Prediction")
    print("Using Category-Aware Cleaned Data")
    print("="*50)
    
    # Create results directory
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {CONFIG['results_dir']}")
    
    # 1. Load data paths
    all_paths = []
    for category in CONFIG['categories']:
        cat_dir = CONFIG['base_dir'] / category
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.parquet"))
            all_paths.extend(paths)
            print(f"Found {len(paths)} files in {category}")
        else:
            print(f"Warning: Directory not found: {cat_dir}")
    
    if len(all_paths) == 0:
        print("\nERROR: No files found!")
        print(f"Please check that cleaned data exists in: {CONFIG['base_dir']}")
        return None, None, None
    
    print(f"\nTotal files: {len(all_paths)}")
    
    # 2. Split data
    train_paths, val_paths, test_paths = smart_data_split(all_paths)
    
    # 3. Create datasets
    print("\nCreating datasets...")
    print(f"Using scaler type: {CONFIG['scaler_type']}")
    
    # IMPORTANT: Create train dataset first to fit the scaler
    train_dataset = MemecoinDataset(
        train_paths, 
        CONFIG['lookback'], 
        CONFIG['forecast_horizon']
    )
    
    # Check if train dataset is valid
    if len(train_dataset) == 0:
        print("\nERROR: Train dataset is empty!")
        print("Possible causes:")
        print("1. Files don't have enough data points (need at least 75 minutes)")
        print("2. Files are missing 'price' column")
        print("3. Data loading errors")
        return None, None, None
    
    # Now create val and test datasets with the fitted scaler
    val_dataset = MemecoinDataset(val_paths, CONFIG['lookback'], CONFIG['forecast_horizon'], 
                                  scaler=train_dataset.scaler)
    test_dataset = MemecoinDataset(test_paths, CONFIG['lookback'], CONFIG['forecast_horizon'], 
                                   scaler=train_dataset.scaler)
    
    # 4. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")
    
    # 5. Initialize model
    model = LSTMPredictor(
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        forecast_horizon=CONFIG['forecast_horizon']
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # 7. Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, predictions, targets = evaluate_model(
        model, test_loader, train_dataset.scaler,
        device=get_device()
    )
    
    print("\nTest Set Metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"  50% Pump Detection - Precision: {metrics['pump_50%_precision']:.2%}, Recall: {metrics['pump_50%_recall']:.2%}")
    print(f"  2x Pump Detection - Precision: {metrics['pump_2x_precision']:.2%}, Recall: {metrics['pump_2x_recall']:.2%}")
    print(f"  5x Pump Detection - Precision: {metrics['pump_5x_precision']:.2%}, Recall: {metrics['pump_5x_recall']:.2%}")
    
    # 8. Save model
    model_path = CONFIG['results_dir'] / 'lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'scaler': train_dataset.scaler,
        'metrics': metrics
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # 9. Create plots
    fig1 = plot_training_history(train_losses, val_losses)
    training_plot_path = CONFIG['results_dir'] / 'training_history.html'
    fig1.write_html(training_plot_path)
    
    fig2 = plot_evaluation_metrics(metrics)
    metrics_plot_path = CONFIG['results_dir'] / 'evaluation_metrics.html'
    fig2.write_html(metrics_plot_path)
    
    # Save metrics as JSON
    metrics_json_path = CONFIG['results_dir'] / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  - Model: {model_path}")
    print(f"  - Training history: {training_plot_path}")
    print(f"  - Evaluation metrics: {metrics_plot_path}")
    print(f"  - Metrics JSON: {metrics_json_path}")
    
    # Clean up checkpoint
    if os.path.exists('best_model_checkpoint.pth'):
        os.remove('best_model_checkpoint.pth')
    
    return model, train_dataset.scaler, metrics


if __name__ == "__main__":
    model, scaler, metrics = main()