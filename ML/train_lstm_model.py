"""
LSTM Training Script for Memecoin Price Prediction
This is the MAIN script you run to train your model
"""

import os
import numpy as np
import pandas as pd
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
    """Dataset for loading and preprocessing memecoin price data"""
    
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
        
        print(f"Loading {len(data_paths)} files...")
        for path in tqdm(data_paths):
            try:
                df = pl.read_parquet(path)
                prices = df['price'].to_numpy()
                
                if len(prices) < self.lookback + self.forecast_horizon:
                    continue
                
                if self.fit_scaler:
                    all_prices.extend(prices)
                
                # Extract metadata
                token_name = path.stem
                launch_date = self._get_launch_date(path)
                
                # Create sequences with sliding window
                self._create_sequences(prices, token_name, launch_date)
                
            except Exception as e:
                print(f"Error loading {path.name}: {e}")
        
        # Fit scaler on all data
        if self.fit_scaler and all_prices:
            self.scaler.fit(np.array(all_prices).reshape(-1, 1))
        
        # Convert to tensors
        self.sequences = torch.FloatTensor(self.sequences)
        self.labels = torch.FloatTensor(self.labels)
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def _create_sequences(self, prices: np.ndarray, token_name: str, launch_date: datetime):
        """Create overlapping sequences from price data"""
        prices_norm = self.scaler.transform(prices.reshape(-1, 1)).flatten()
        
        stride = max(1, self.forecast_horizon // 3)  # Reduce overlap
        
        for i in range(0, len(prices_norm) - self.lookback - self.forecast_horizon + 1, stride):
            seq = prices_norm[i:i + self.lookback]
            target = prices_norm[i + self.lookback:i + self.lookback + self.forecast_horizon]
            
            self.sequences.append(seq)
            self.labels.append(target)
            self.metadata.append({
                'token': token_name,
                'date': launch_date,
                'start_idx': i
            })
    
    def _get_launch_date(self, path: Path) -> datetime:
        """Extract launch date from file or metadata"""
        # Try JSON metadata first
        json_path = path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    if 'launch_date' in meta:
                        return datetime.fromisoformat(meta['launch_date'])
            except:
                pass
        
        # Try filename
        parts = path.stem.split('_')
        for part in parts:
            if len(part) == 8 and part.isdigit():
                try:
                    return datetime.strptime(part, '%Y%m%d')
                except:
                    pass
        
        # Fallback to file modification time
        return datetime.fromtimestamp(path.stat().st_mtime)
    
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
    Split data intelligently based on temporal distribution
    Since 80-90% of tokens are from 2025, use stratified weekly split
    """
    
    print("\nAnalyzing data distribution...")
    
    # Extract dates from all files
    file_dates = []
    for path in all_paths:
        try:
            # Try to get date from filename
            parts = path.stem.split('_')
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    date = datetime.strptime(part, '%Y%m%d')
                    file_dates.append((path, date))
                    break
            else:
                # Use file modification time as fallback
                date = datetime.fromtimestamp(path.stat().st_mtime)
                file_dates.append((path, date))
        except:
            continue
    
    # Create DataFrame for analysis
    df = pd.DataFrame(file_dates, columns=['path', 'date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week
    df['year_week'] = df['year'].astype(str) + '_W' + df['week'].astype(str).str.zfill(2)
    
    # Check distribution
    year_counts = df['year'].value_counts()
    print(f"\nYear distribution:")
    for year, count in year_counts.items():
        print(f"  {year}: {count} files ({count/len(df)*100:.1f}%)")
    
    # Stratified split by week
    train_paths, val_paths, test_paths = [], [], []
    
    for year_week, group in df.groupby('year_week'):
        paths = group['path'].tolist()
        np.random.shuffle(paths)
        
        n = len(paths)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:n_train + n_val])
        test_paths.extend(paths[n_train + n_val:])
    
    print(f"\nData split (stratified by week):")
    print(f"  Train: {len(train_paths)} files")
    print(f"  Val: {len(val_paths)} files")
    print(f"  Test: {len(test_paths)} files")
    
    return train_paths, val_paths, test_paths


# ================== TRAINING FUNCTIONS ==================
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 50,
                learning_rate: float = 0.001,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the LSTM model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
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
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses


# ================== EVALUATION METRICS ==================
def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader,
                   scaler: StandardScaler,
                   device: str = 'cuda'):
    """Calculate trading-specific metrics"""
    
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
    
    pump_threshold = 1.0  # 100% gain
    pred_pumps = pred_returns > pump_threshold
    true_pumps = true_returns > pump_threshold
    
    metrics['pump_precision'] = np.sum(pred_pumps & true_pumps) / np.sum(pred_pumps) if np.sum(pred_pumps) > 0 else 0
    metrics['pump_recall'] = np.sum(pred_pumps & true_pumps) / np.sum(true_pumps) if np.sum(true_pumps) > 0 else 0
    
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
    """Plot evaluation metrics as bar chart"""
    
    fig = go.Figure()
    
    # Prepare data
    metric_names = ['Direction\nAccuracy', 'Pump\nPrecision', 'Pump\nRecall']
    metric_values = [
        metrics['direction_accuracy'],
        metrics['pump_precision'],
        metrics['pump_recall']
    ]
    
    colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in metric_values]
    
    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        text=[f'{v:.1%}' for v in metric_values],
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        showlegend=False
    )
    
    return fig


# ================== MAIN TRAINING PIPELINE ==================
def main():
    """Main training pipeline"""
    
    # Configuration
    CONFIG = {
        'base_dir': Path("data/processed"),
        'categories': ["cleaned_normal_behavior_tokens", "cleaned_tokens_with_gaps"],
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
    print("="*50)
    
    # 1. Load data paths
    all_paths = []
    for category in CONFIG['categories']:
        cat_dir = CONFIG['base_dir'] / category
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.parquet"))
            all_paths.extend(paths)
            print(f"Found {len(paths)} files in {category}")
    
    print(f"\nTotal files: {len(all_paths)}")
    
    # 2. Split data
    train_paths, val_paths, test_paths = smart_data_split(all_paths)
    
    # 3. Create datasets
    print("\nCreating datasets...")
    train_dataset = MemecoinDataset(train_paths, CONFIG['lookback'], CONFIG['forecast_horizon'])
    val_dataset = MemecoinDataset(val_paths, CONFIG['lookback'], CONFIG['forecast_horizon'], 
                                  scaler=train_dataset.scaler)
    test_dataset = MemecoinDataset(test_paths, CONFIG['lookback'], CONFIG['forecast_horizon'], 
                                   scaler=train_dataset.scaler)
    
    # 4. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
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
    metrics, predictions, targets = evaluate_model(model, test_loader, train_dataset.scaler)
    
    print("\nTest Set Metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"  Pump Detection Precision: {metrics['pump_precision']:.2%}")
    print(f"  Pump Detection Recall: {metrics['pump_recall']:.2%}")
    
    # 8. Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'scaler': train_dataset.scaler,
        'metrics': metrics
    }, 'lstm_model.pth')
    
    print("\nModel saved to 'lstm_model.pth'")
    
    # 9. Create plots
    fig1 = plot_training_history(train_losses, val_losses)
    fig1.write_html('training_history.html')
    
    fig2 = plot_evaluation_metrics(metrics)
    fig2.write_html('evaluation_metrics.html')
    
    print("\nPlots saved:")
    print("  - training_history.html")
    print("  - evaluation_metrics.html")
    
    return model, train_dataset.scaler, metrics


if __name__ == "__main__":
    model, scaler, metrics = main()