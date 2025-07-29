# ml/transformer_forecast.py - Unified transformer for cross-token pump prediction (no archetypes)
# Rolling window sequences: 5min, 10min, 15min, etc. for realistic trading simulation
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import numpy as np
import time
import os
import joblib
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import setup_logger
import config

logger = setup_logger(__name__)

class UnifiedPumpTransformer(nn.Module):
    """Transformer for unified cross-token pump prediction with variable sequence lengths"""
    
    def __init__(self, input_dim: int, max_seq_len: int = 60, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection and positional encoding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-horizon price forecasting head (not classification)
        self.forecaster = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 5)  # Forecast 1,2,3,4,5 minutes ahead
        )
        
    def forward(self, x, attention_mask=None):
        """Forward pass with optional attention masking for variable lengths"""
        batch_size, seq_len, _ = x.shape
        
        # Project input and add positional encoding
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.dropout(x)
        
        # Apply transformer with masking
        if attention_mask is not None:
            # Create causal mask for transformer
            src_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
            transformer_out = self.transformer(x, mask=src_mask, src_key_padding_mask=~attention_mask)
        else:
            transformer_out = self.transformer(x)
        
        # Use last token for classification (or last valid token if masked)
        if attention_mask is not None:
            # Get last valid position for each sequence
            last_positions = attention_mask.sum(dim=1) - 1
            output = transformer_out[range(batch_size), last_positions]
        else:
            output = transformer_out[:, -1, :]
        
        return self.forecaster(output)  # Output: [batch_size, 5] for 5 horizons

def prepare_rolling_sequences(df: pl.DataFrame, window_sizes: list = [5, 10, 15, 20, 30], features: list = None) -> dict:
    """Prepare rolling window sequences for multi-horizon price forecasting"""
    if features is None:
        # Use top features from baseline model results
        features = ["scaled_returns", "ma_5", "rsi_14", "max_returns", 
                   "vol_std_5", "momentum_lag1", "vol_return_ratio", "initial_price",
                   "minutes_since_start", "current_total_return", "recent_avg_volatility"]
    
    logger.info(f"Preparing rolling sequences with windows: {window_sizes}")
    logger.info(f"Using features: {features}")
    
    all_sequences = {}
    
    for window_size in window_sizes:
        sequences = []
        price_targets = []  # Multi-horizon price targets: [1min, 2min, 3min, 4min, 5min] ahead
        token_ids = []
        sequence_positions = []  # Track position within token for validation
        
        tokens = df["token_id"].unique().to_list()
        logger.info(f"Processing {len(tokens)} tokens for window size {window_size} (price forecasting)")
        
        for token_idx, token in enumerate(tokens):
            if token_idx % 1000 == 0:
                logger.info(f"Processed {token_idx}/{len(tokens)} tokens")
                
            token_df = df.filter(pl.col("token_id") == token).sort("datetime")
            
            # Skip tokens with insufficient data (need 5 future minutes for targets)
            if token_df.height < window_size + 6:
                continue
                
            # Get feature data and price data
            try:
                feature_data = token_df.select(features).fill_null(0).to_numpy()
                prices = token_df["avg_price"].to_numpy() if "avg_price" in token_df.columns else token_df["price"].to_numpy()
                
                # Create rolling windows with future price targets
                for i in range(window_size, len(feature_data) - 5):  # Leave 5 minutes for future targets
                    # Use past window_size minutes to predict future prices
                    sequence = feature_data[max(0, i-window_size):i]
                    
                    # Pad if sequence is shorter than window_size (early in token's life)
                    if len(sequence) < window_size:
                        padding = np.zeros((window_size - len(sequence), len(features)))
                        sequence = np.vstack([padding, sequence])
                    
                    # Extract future price targets (1-5 minutes ahead)
                    current_price = prices[i]
                    future_prices = prices[i+1:i+6]  # Next 5 minutes
                    
                    # Calculate price change ratios for each horizon
                    price_changes = (future_prices - current_price) / current_price
                    
                    sequences.append(sequence)
                    price_targets.append(price_changes)
                    token_ids.append(token)
                    sequence_positions.append(i)
                    
            except Exception as e:
                logger.warning(f"Error processing token {token}: {e}")
                continue
        
        all_sequences[window_size] = {
            'sequences': np.array(sequences),
            'price_targets': np.array(price_targets),  # Shape: (n_samples, 5) for 5 horizons
            'token_ids': np.array(token_ids),
            'positions': np.array(sequence_positions)
        }
        
        avg_target = np.mean(price_targets, axis=0) if len(price_targets) > 0 else [0] * 5
        logger.info(f"Window {window_size}: {len(sequences)} sequences, avg targets: {avg_target}")
    
    return all_sequences

def create_train_test_split(sequences_data: dict, test_token_ratio: float = 0.2) -> dict:
    """Split by tokens (not sequences) to prevent data leakage"""
    split_data = {}
    
    for window_size, data in sequences_data.items():
        unique_tokens = np.unique(data['token_ids'])
        n_test_tokens = int(len(unique_tokens) * test_token_ratio)
        
        # Use last tokens for testing (temporal split)
        test_tokens = set(unique_tokens[-n_test_tokens:])
        
        train_mask = ~np.isin(data['token_ids'], list(test_tokens))
        test_mask = np.isin(data['token_ids'], list(test_tokens))
        
        split_data[window_size] = {
            'X_train': data['sequences'][train_mask],
            'y_train': data['price_targets'][train_mask],  # Price targets instead of labels
            'tokens_train': data['token_ids'][train_mask],
            'X_test': data['sequences'][test_mask],
            'y_test': data['price_targets'][test_mask],   # Price targets instead of labels
            'tokens_test': data['token_ids'][test_mask]
        }
        
        # Calculate mean absolute target values for each horizon
        train_targets_abs = np.mean(np.abs(split_data[window_size]['y_train']), axis=0)
        test_targets_abs = np.mean(np.abs(split_data[window_size]['y_test']), axis=0)
        
        logger.info(f"Window {window_size}: Train {len(split_data[window_size]['X_train'])} seqs (avg abs targets: {train_targets_abs}), "
                   f"Test {len(split_data[window_size]['X_test'])} seqs (avg abs targets: {test_targets_abs})")
    
    return split_data

def train_unified_transformer(df: pl.DataFrame, window_size: int = 10, epochs: int = 15, lr: float = 0.001, 
                             batch_size: int = None, device: str = 'auto') -> dict:
    """Train unified transformer across all tokens with specified window size (M4 Max optimized)"""
    logger.info(f"Training unified transformer with window size {window_size}")
    
    # Auto-optimize batch size for M4 Max hardware
    if batch_size is None:
        if device == 'mps':
            batch_size = 256  # Optimized for Apple Silicon + 64GB RAM
            logger.info("🚀 M4 Max detected: Using batch_size=256 for optimal GPU utilization")
        elif device == 'cuda':
            batch_size = 128  # Conservative for unknown GPU memory
            logger.info("⚡ CUDA detected: Using batch_size=128")
        else:
            batch_size = 64   # CPU fallback
            logger.info("💻 CPU detected: Using batch_size=64")
    
    logger.info(f"Batch size: {batch_size}")
    
    # Prepare sequences for this window size
    all_sequences = prepare_rolling_sequences(df, window_sizes=[window_size])
    split_data = create_train_test_split(all_sequences)
    
    data = split_data[window_size]
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    if len(X_train) == 0:
        logger.error("No training sequences generated")
        return None
    
    logger.info(f"Training data: {X_train.shape[0]} sequences, {X_train.shape[1]} timesteps, {X_train.shape[2]} features")
    
    # Price forecasting statistics
    target_stats = np.mean(np.abs(y_train), axis=0)
    logger.info(f"Price target statistics (mean abs change): {target_stats}")
    logger.info(f"Target range: min={np.min(y_train):.4f}, max={np.max(y_train):.4f}")
    
    # Create model
    input_dim = X_train.shape[2]
    model = UnifiedPumpTransformer(input_dim=input_dim, max_seq_len=window_size).to(device)
    
    # Loss for regression (Mean Squared Error)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Optimized data loaders for M4 Max
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    
    # Multi-core data loading for M4 Max (14 cores)
    num_workers = 8 if device == 'mps' else 4  # Use 8 cores for Apple Silicon
    pin_memory = device == 'cuda'  # Only use pin_memory for CUDA (not MPS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"DataLoader: {num_workers} workers, pin_memory={pin_memory}")
    
    # Memory monitoring for M4 Max optimization
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    memory_available = psutil.virtual_memory().available / (1024**3)
    logger.info(f"💾 System RAM: {memory_gb:.1f}GB total, {memory_available:.1f}GB available")
    
    if device == 'mps':
        logger.info("🧠 M4 Max tip: If you see memory issues, reduce batch_size from 256 to 128")
    
    # Training loop
    train_losses = []
    best_f1 = 0
    best_model_state = None
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Progress: [{'='*20}] 0/{epochs} epochs (0%)")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)  # Shape: [batch_size, 5] for 5 horizons
            loss = criterion(outputs, batch_y)  # MSE loss for regression
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Progress indicator every epoch
        progress = (epoch + 1) / epochs
        filled = int(20 * progress)
        bar = '=' * filled + '-' * (20 - filled)
        percent = int(100 * progress)
        logger.info(f"Progress: [{bar}] {epoch+1}/{epochs} epochs ({percent}%) - Loss: {avg_loss:.4f}")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
                test_forecasts = test_outputs.cpu().numpy()
                
                # Calculate regression metrics
                mse = np.mean((test_forecasts - y_test) ** 2)
                mae = np.mean(np.abs(test_forecasts - y_test))
                
                # Calculate R² for each horizon
                r2_scores = []
                for h in range(5):  # 5 horizons
                    y_true_h = y_test[:, h]
                    y_pred_h = test_forecasts[:, h]
                    ss_res = np.sum((y_true_h - y_pred_h) ** 2)
                    ss_tot = np.sum((y_true_h - np.mean(y_true_h)) ** 2)
                    r2_h = 1 - (ss_res / (ss_tot + 1e-8))
                    r2_scores.append(r2_h)
                
                avg_r2 = np.mean(r2_scores)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, MSE={mse:.4f}, "
                           f"MAE={mae:.4f}, Avg R²={avg_r2:.3f}")
                
                # Use negative MSE as "score" to maximize (minimize MSE)
                current_score = -mse
                if current_score > best_f1:  # Reusing best_f1 variable as best_score
                    best_f1 = current_score
                    best_model_state = model.state_dict().copy()
        
        scheduler.step(avg_loss)
    
    # Final evaluation with best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_forecasts = test_outputs.cpu().numpy()
        
        # Calculate final regression metrics
        final_mse = np.mean((test_forecasts - y_test) ** 2)
        final_mae = np.mean(np.abs(test_forecasts - y_test))
        
        # R² for each horizon
        horizon_r2 = []
        horizon_mae = []
        for h in range(5):
            y_true_h = y_test[:, h]
            y_pred_h = test_forecasts[:, h]
            
            # R²
            ss_res = np.sum((y_true_h - y_pred_h) ** 2)
            ss_tot = np.sum((y_true_h - np.mean(y_true_h)) ** 2)
            r2_h = 1 - (ss_res / (ss_tot + 1e-8))
            horizon_r2.append(r2_h)
            
            # MAE for this horizon
            mae_h = np.mean(np.abs(y_true_h - y_pred_h))
            horizon_mae.append(mae_h)
        
        final_metrics = {
            'mse': final_mse,
            'mae': final_mae,
            'avg_r2': np.mean(horizon_r2),
            'horizon_r2': horizon_r2,
            'horizon_mae': horizon_mae,
            'train_sequences': len(X_train),
            'test_sequences': len(X_test),
            'window_size': window_size,
            'forecast_horizons': 5
        }
    
    logger.info(f"Final forecasting metrics for window {window_size}:")
    logger.info(f"  MSE: {final_metrics['mse']:.6f}")
    logger.info(f"  MAE: {final_metrics['mae']:.6f}")
    logger.info(f"  Avg R²: {final_metrics['avg_r2']:.4f}")
    logger.info(f"  Horizon R²: {[f'{r:.3f}' for r in final_metrics['horizon_r2']]}")
    
    return {
        'model': model,
        'metrics': final_metrics,
        'train_losses': train_losses,
        'features': ["scaled_returns", "ma_5", "rsi_14", "max_returns", 
                    "vol_std_5", "momentum_lag1", "vol_return_ratio", "initial_price",
                    "minutes_since_start", "current_total_return", "recent_avg_volatility"]
    }

def main():
    """Main training function for unified transformer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train unified transformer for pump prediction')
    parser.add_argument('--interval', default='5m', choices=['1m', '5m'], help='Data interval')
    parser.add_argument('--window-sizes', nargs='+', type=int, default=[10, 20, 30], 
                       help='Window sizes to train (e.g., 10 20 30)')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    # Load unified data (no archetypes)
    data_file = f"processed_features_{args.interval}_unified.parquet"
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_file)
    
    if not os.path.exists(data_path):
        logger.error(f"Unified data file not found: {data_path}")
        logger.error("Please run: python scripts/run_pipeline3.py")
        return
    
    logger.info(f"Loading unified data from {data_path}")
    df = pl.read_parquet(data_path)
    logger.info(f"Data shape: {df.shape}")
    if 'pump_label' in df.columns:
        logger.info(f"Pump rate: {df['pump_label'].sum() / df.height:.3%}")
    else:
        logger.info("Price forecasting mode - no pump labels needed")
    
    # Train models for different window sizes
    all_results = {}
    # Optimize device selection for M4 Max MacBook Pro
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
        logger.info("🚀 Using Apple Silicon GPU (MPS) - Optimized for M4 Max!")
    elif torch.cuda.is_available():
        device = 'cuda'  # NVIDIA GPU
        logger.info("🚀 Using CUDA GPU")
    else:
        device = 'cpu'
        logger.info("⚠️  Using CPU only - Consider GPU acceleration")
    
    logger.info(f"Device: {device}")
    
    # Log system capabilities
    if device == 'mps':
        logger.info(f"🧠 Apple Silicon detected - Using optimized batch sizes for 64GB RAM")
    
    for window_size in args.window_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training transformer with window size: {window_size}")
        logger.info(f"{'='*60}")
        
        results = train_unified_transformer(
            df, window_size=window_size, epochs=args.epochs, 
            lr=args.lr, batch_size=args.batch_size, device=device
        )
        
        if results:
            all_results[window_size] = results
            
            # Save model in unified format
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save as PyTorch checkpoint (for unified loader compatibility)
            model_path = os.path.join(model_dir, f'transformer_{args.interval}_unified_window{window_size}.pth')
            
            # Save model with metadata for unified loader
            save_data = {
                'model_state_dict': results['model'].state_dict(),
                'input_dim': results['model'].input_projection.in_features,
                'max_seq_len': results['model'].max_seq_len,
                'd_model': results['model'].d_model,
                'metrics': results['metrics'],
                'features': results['features'],
                'window_size': window_size,
                'interval': args.interval,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(save_data, model_path)
            logger.info(f"✅ Saved model to {model_path}")
            
            # Also save results to analysis folder
            analysis_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis', args.interval)
            os.makedirs(analysis_dir, exist_ok=True)
            
            results_data = {
                'model_type': 'transformer',
                'window_size': window_size,
                'interval': args.interval,
                'metrics': results['metrics'],
                'features': results['features'],
                'training_config': {
                    'epochs': args.epochs,
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size,
                    'device': device
                },
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = os.path.join(analysis_dir, f'transformer_window{window_size}_results.json')
            import json
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"📊 Saved results to {results_path}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    for window_size, results in all_results.items():
        metrics = results['metrics']
        logger.info(f"Window {window_size}: MSE={metrics['mse']:.6f}, "
                   f"MAE={metrics['mae']:.6f}, Avg R²={metrics['avg_r2']:.3f}")
    
    if all_results:
        best_window = max(all_results.keys(), key=lambda w: all_results[w]['metrics']['avg_r2'])
        logger.info(f"🏆 Best performing window size: {best_window} "
                   f"(Avg R²={all_results[best_window]['metrics']['avg_r2']:.3f})")

if __name__ == "__main__":
    main()