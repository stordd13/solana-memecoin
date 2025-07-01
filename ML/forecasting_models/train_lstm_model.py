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
# Add option to use Winsorizer
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from ML.utils.winsorizer import Winsorizer
    from ML.utils.training_plots import plot_training_curves, create_learning_summary
    WINSORIZER_AVAILABLE = True
except ImportError:
    WINSORIZER_AVAILABLE = False
    print("Warning: Winsorizer not available, using StandardScaler")


# ================== DATASET CLASS ==================
class MemecoinDataset(Dataset):
    """Dataset for loading category-aware cleaned memecoin data"""
    
    def __init__(self, 
                 data_paths: List[Path], 
                 lookback: int = 60, 
                 forecast_horizon: int = 15,
                 scaler = None,
                 use_winsorizer: bool = True):
        
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.sequences = []
        self.labels = []
        self.metadata = []
        
        # Use Winsorizer if available and requested
        if scaler is None:
            if use_winsorizer and WINSORIZER_AVAILABLE:
                self.scaler = Winsorizer(lower_percentile=0.005, upper_percentile=0.995)
                print("Using Winsorizer for scaling (better for crypto data)")
            else:
                self.scaler = StandardScaler()
                print("Using StandardScaler")
            self.fit_scaler = True
        else:
            self.scaler = scaler
            self.fit_scaler = False
        
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
        
        # Add batch normalization after LSTM
        self.lstm_bn = nn.BatchNorm1d(hidden_size)
        
        # Output layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
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
        
        # Use last hidden state and apply batch norm
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.lstm_bn(last_hidden)
        
        # Generate predictions
        predictions = self.fc(last_hidden)
        
        return predictions


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
    
    print(f"\nTraining LSTM Forecasting on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Add progress bar for training
        from tqdm import tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
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
    
    # Find best epoch
    best_epoch = val_losses.index(best_val_loss) + 1
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    return model, training_history


# ================== EVALUATION METRICS ==================
def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader,
                   scaler: StandardScaler,
                   device: str = None):
    """Calculate trading-specific metrics including directional accuracy and financial metrics"""
    
    if device is None:
        device = get_device()
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        from tqdm import tqdm
        test_pbar = tqdm(test_loader, desc="Evaluating forecasting model")
        for batch_x, batch_y in test_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Inverse transform to get actual prices
    pred_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    target_prices = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    
    # Calculate basic regression metrics
    metrics = {}
    metrics['mse'] = np.mean((pred_prices - target_prices) ** 2)
    metrics['mae'] = np.mean(np.abs(pred_prices - target_prices))
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Calculate R² for final horizon (most important)
    from sklearn.metrics import r2_score
    metrics['r2_final_horizon'] = r2_score(target_prices[:, -1], pred_prices[:, -1])
    
    # Financial and directional metrics
    entry_prices = target_prices[:, 0]  # Current prices at prediction time
    final_pred_prices = pred_prices[:, -1]  # Predicted final prices
    final_true_prices = target_prices[:, -1]  # True final prices
    
    # Directional accuracy (key trading metric)
    pred_direction = np.sign(final_pred_prices - entry_prices)
    true_direction = np.sign(final_true_prices - entry_prices)
    metrics['direction_accuracy'] = np.mean(pred_direction == true_direction)
    
    # Calculate returns
    pred_returns = (final_pred_prices - entry_prices) / entry_prices
    true_returns = (final_true_prices - entry_prices) / entry_prices
    
    # Financial metrics
    metrics['mean_pred_return'] = np.mean(pred_returns)
    metrics['mean_true_return'] = np.mean(true_returns)
    metrics['return_correlation'] = np.corrcoef(pred_returns, true_returns)[0, 1]
    
    # Trading simulation metrics
    # Simple strategy: Buy if predicted return > 0
    buy_signals = pred_returns > 0
    if np.sum(buy_signals) > 0:
        strategy_returns = true_returns[buy_signals]
        metrics['strategy_win_rate'] = np.mean(strategy_returns > 0)
        metrics['strategy_avg_return'] = np.mean(strategy_returns)
        metrics['strategy_sharpe'] = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
    else:
        metrics['strategy_win_rate'] = 0.0
        metrics['strategy_avg_return'] = 0.0
        metrics['strategy_sharpe'] = 0.0
    
    # Pump detection (2x = 100% gain)
    pred_pumps = pred_returns > 1.0  # Predicted 2x
    true_pumps = true_returns > 1.0  # Actual 2x
    
    if np.sum(true_pumps) > 0:
        metrics['pump_detection_recall'] = np.sum(pred_pumps & true_pumps) / np.sum(true_pumps)
    else:
        metrics['pump_detection_recall'] = 0.0
        
    if np.sum(pred_pumps) > 0:
        metrics['pump_detection_precision'] = np.sum(pred_pumps & true_pumps) / np.sum(pred_pumps)
    else:
        metrics['pump_detection_precision'] = 0.0
    
    # Price error as percentage
    metrics['mean_price_error_pct'] = np.mean(np.abs(pred_prices - target_prices) / target_prices) * 100
    
    # Volatility metrics
    metrics['price_volatility'] = np.std(target_prices[:, -1] / target_prices[:, 0])
    metrics['prediction_volatility'] = np.std(pred_prices[:, -1] / target_prices[:, 0])
    
    return metrics


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
    """Main training pipeline for LSTM forecasting model with walk-forward validation"""
    
    print("="*60)
    print("🔮 LSTM Price Forecasting Training")  
    print("Features: Price prediction, Enhanced plotting, Financial metrics")
    print("WITH WALK-FORWARD VALIDATION")
    print("="*60)
    
    # Device setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Data loading
    data_base = Path("data/features")
    categories = [
        "normal_behavior_tokens",
        "tokens_with_extremes", 
        "dead_tokens",
    ]
    
    all_paths = []
    for category in categories:
        cat_dir = data_base / category
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.parquet"))
            all_paths.extend(paths)
            print(f"Found {len(paths)} files in {category}")
    
    if not all_paths:
        print("ERROR: No feature files found!")
        return
    
    print(f"\nTotal files: {len(all_paths)}")
    
    # Import walk-forward splitter
    from ML.utils.walk_forward_splitter import WalkForwardSplitter
    
    print("\n🔄 Using Walk-Forward Validation for LSTM Forecasting")
    
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
    temp_dir = Path("ML/results/lstm_forecasting/temp_splits")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
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

    # Walk-forward validation is now used
    # train_paths, val_paths, test_paths = smart_data_split(all_paths)
    
    # Create datasets with walk-forward splits
    # Temporary file-based approach for compatibility
    
    # Create datasets
    print("\nCreating LSTM forecasting datasets...")
    
    # Shared scaler for consistent normalization
    scaler = StandardScaler()
    
    train_dataset = MemecoinDataset(
        train_paths, 
        lookback=60, 
        forecast_horizon=15,
        scaler=scaler,
        use_winsorizer=True
    )
    
    val_dataset = MemecoinDataset(
        val_paths, 
        lookback=60, 
        forecast_horizon=15,
        scaler=scaler,
        use_winsorizer=True
    )
    
    test_dataset = MemecoinDataset(
        test_paths, 
        lookback=60, 
        forecast_horizon=15,
        scaler=scaler,
        use_winsorizer=True
    )
    
    print(f"\nDataset sizes (walk-forward):")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")
    
    if len(train_dataset) == 0:
        print("ERROR: No training sequences created!")
        return
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = LSTMPredictor(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        forecast_horizon=15
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\nTraining LSTM forecasting model with walk-forward validation...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=100, learning_rate=0.001, device=device
    )
    
    # Evaluation
    print("\nEvaluating on test set (walk-forward)...")
    metrics = evaluate_model(model, test_loader, scaler, device)
    
    # Print results
    print("\n" + "="*60)
    print("FORECASTING RESULTS (Walk-Forward Validation)")
    print("="*60)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r2']:.4f}")
    
    if 'financial_metrics' in metrics:
        fin_metrics = metrics['financial_metrics']
        print(f"\nFinancial Performance:")
        print(f"  Directional Accuracy: {fin_metrics.get('directional_accuracy', 0):.2%}")
        print(f"  Strategy Return: {fin_metrics.get('strategy_return', 0):.2%}")
        print(f"  Win Rate: {fin_metrics.get('win_rate', 0):.2%}")
        print(f"  Sharpe Ratio: {fin_metrics.get('sharpe_ratio', 0):.2f}")
    
    # Save model and results
    results_dir = Path("ML/results/lstm_forecasting")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = results_dir / 'lstm_model_walkforward.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_config': {
            'input_size': 1,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'forecast_horizon': 15
        },
        'metrics': metrics,
        'validation_method': 'walk_forward'
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Create visualizations
    # 1. Training curves
    training_fig = plot_training_history(train_losses, val_losses)
    training_fig.update_layout(title="LSTM Forecasting Training Progress (Walk-Forward)")
    training_path = results_dir / 'training_curves_walkforward.html'
    training_fig.write_html(training_path)
    
    # 2. Evaluation metrics
    eval_fig = plot_evaluation_metrics(metrics)
    eval_fig.update_layout(title="LSTM Forecasting Performance (Walk-Forward)")
    eval_path = results_dir / 'evaluation_metrics_walkforward.html'
    eval_fig.write_html(eval_path)
    
    # 3. Save metrics JSON
    metrics_json_path = results_dir / 'metrics_walkforward.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary split files")
    
    print(f"\nResults saved to: {results_dir}")
    print("\n✅ LSTM forecasting training complete with walk-forward validation!")
    print(f"   More realistic forecasting metrics due to temporal validation")
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()