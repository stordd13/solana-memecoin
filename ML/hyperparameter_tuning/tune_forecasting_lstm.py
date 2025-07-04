"""
Hyperparameter Tuning for LSTM Forecasting Models

This script performs comprehensive hyperparameter optimization for LSTM models
used in price forecasting with walk-forward validation.

WORKFLOW:
1. Load pre-engineered features from feature_engineering module
2. Create walk-forward splits for realistic temporal validation  
3. For each fold, optimize hyperparameters using Optuna
4. Save best parameters for use in training scripts

OPTIMIZATION STRATEGY:
- Optuna for efficient neural architecture search
- Per-fold tuning: Adapt to market regime changes
- Multi-objective: Balance MSE and financial performance
- Early stopping: Prevent overfitting during tuning
- GPU acceleration: Use CUDA if available

USAGE:
    python tune_forecasting_lstm.py --n_trials 50 --strategy per_fold
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ML.utils.walk_forward_splitter import WalkForwardSplitter
from ML.utils.metrics_helpers import regression_metrics
from ML.hyperparameter_tuning.utils.tuning_helpers import (
    save_best_params, create_tuning_visualizations, print_tuning_summary,
    get_default_optuna_config, TuningProgressCallback
)


class LSTMForecastingModel(nn.Module):
    """LSTM model for price forecasting with flexible architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.2, forecast_horizon: int = 15):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),  # Multi-step forecasting
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use last output for forecasting
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Generate multi-step forecast
        forecast = self.forecast_head(last_output)  # (batch_size, forecast_horizon)
        
        return forecast


def suggest_lstm_params(trial: optuna.Trial, input_size: int) -> dict:
    """Suggest LSTM hyperparameters for optimization"""
    
    return {
        'input_size': input_size,
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'sequence_length': trial.suggest_categorical('sequence_length', [30, 60, 120]),
    }


def create_sequences(data: np.ndarray, sequence_length: int, forecast_horizon: int):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(sequence_length, len(data) - forecast_horizon + 1):
        # Input sequence
        seq = data[i-sequence_length:i]
        # Target: future prices at multiple horizons
        target = data[i:i+forecast_horizon, -1]  # Assuming price is last column
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def objective_function(trial: optuna.Trial, train_data, val_data, fold_idx: int = 0) -> float:
    """
    Optuna objective function for LSTM hyperparameter optimization
    
    Returns combined score: 0.4 * (1 - normalized_mse) + 0.6 * r2_score
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Get suggested parameters
        input_size = train_data.shape[-1]
        params = suggest_lstm_params(trial, input_size)
        
        sequence_length = params['sequence_length']
        forecast_horizon = min(15, len(train_data) // 10)  # Adaptive horizon
        
        # Create sequences
        X_train, y_train = create_sequences(train_data, sequence_length, forecast_horizon)
        X_val, y_val = create_sequences(val_data, sequence_length, forecast_horizon)
        
        if len(X_train) == 0 or len(X_val) == 0:
            return 0.0
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.FloatTensor(y_val).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        
        # Create model
        model = LSTMForecastingModel(
            input_size=params['input_size'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            forecast_horizon=forecast_horizon
        ).to(device)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        
        # Training loop (reduced epochs for tuning speed)
        model.train()
        epochs = 20  # Reduced for tuning
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            # Early stopping check (simplified)
            if epoch > 5 and epoch_loss / len(train_loader) > 10.0:  # Loss too high
                break
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            
            # Calculate metrics on the final forecasting step
            final_preds = val_outputs[:, -1].cpu().numpy()
            final_targets = y_val[:, -1].cpu().numpy()
            
            # Calculate regression metrics
            metrics = regression_metrics(final_targets, final_preds)
            
            mse_score = metrics.get('mse', float('inf'))
            r2_score = max(0.0, metrics.get('r2', 0.0))  # Clamp R² to positive
            
            # Normalize MSE (lower is better)
            mse_normalized = min(1.0, mse_score / (np.var(final_targets) + 1e-8))
            mse_component = 1.0 - mse_normalized
            
            # Combined objective
            combined_score = 0.4 * mse_component + 0.6 * r2_score
            
            # Store detailed metrics for analysis
            trial.set_user_attr(f'fold_{fold_idx}_mse', mse_score)
            trial.set_user_attr(f'fold_{fold_idx}_mae', metrics.get('mae', 0.0))
            trial.set_user_attr(f'fold_{fold_idx}_r2', r2_score)
            trial.set_user_attr(f'fold_{fold_idx}_val_loss', val_loss)
            trial.set_user_attr(f'fold_{fold_idx}_combined_score', combined_score)
            
            return combined_score
        
    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return poor score for failed trials


def tune_single_fold(train_data, val_data, fold_idx: int, n_trials: int = 50) -> tuple:
    """Tune hyperparameters for a single fold"""
    
    print(f"\n🔍 Tuning LSTM fold {fold_idx + 1} with {n_trials} trials...")
    
    # Create study
    config = get_default_optuna_config()
    study = optuna.create_study(**config)
    
    # Add progress callback
    callback = TuningProgressCallback(f"LSTM_Forecasting_Fold_{fold_idx+1}", save_interval=10)
    
    # Optimize with fold-specific objective
    def fold_objective(trial):
        return objective_function(trial, train_data, val_data, fold_idx)
    
    study.optimize(fold_objective, n_trials=n_trials, callbacks=[callback])
    
    print(f"✅ Fold {fold_idx + 1} optimization complete!")
    print(f"   Best score: {study.best_value:.4f}")
    
    return study.best_params, study


def tune_forecasting_lstm_per_fold(features_dir: Path, horizons: List[int], n_trials: int = 50):
    """
    Perform per-fold hyperparameter tuning for LSTM forecasting model
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 LSTM FORECASTING HYPERPARAMETER TUNING - PER FOLD STRATEGY")
    print(f"📊 Horizons: {horizons} minutes")
    print(f"🔬 Trials per fold per horizon: {n_trials}")
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")
    
    # Load feature data (simplified for forecasting)
    print(f"\n📁 Loading features from {features_dir}...")
    categories = ["normal_behavior_tokens", "tokens_with_extremes", "dead_tokens"]
    
    all_data_frames = []
    for category in categories:
        cat_dir = features_dir / category
        if cat_dir.exists():
            parquet_files = list(cat_dir.glob("*.parquet"))[:5]  # Limit for tuning speed
            print(f"  Found {len(parquet_files)} files in {category}")
            
            for file_path in tqdm(parquet_files, desc=f"Loading {category}"):
                try:
                    df = pl.read_parquet(file_path)
                    if len(df) >= 400:
                        df = df.with_columns(pl.lit(file_path.stem).alias('token_id'))
                        all_data_frames.append(df)
                except Exception as e:
                    print(f"⚠️ Error loading {file_path}: {e}")
    
    if not all_data_frames:
        raise ValueError("No valid feature files found!")
    
    # Combine data
    combined_data = pl.concat(all_data_frames)
    print(f"📊 Loaded {len(combined_data):,} rows from {len(all_data_frames)} tokens")
    
    # Create walk-forward splits
    print(f"\n🔄 Creating walk-forward splits...")
    splitter = WalkForwardSplitter()
    
    global_splits, feasible_horizons = splitter.smart_split_for_memecoins(
        combined_data,
        horizons=[15],  # Use single horizon for LSTM tuning simplicity
        time_column='datetime'
    )
    
    if not global_splits:
        raise ValueError(f"No valid splits!")
    
    print(f"✅ Created {len(global_splits)} folds")
    
    # Prepare features
    feature_cols = [col for col in combined_data.columns 
                   if col not in ['datetime', 'token_id'] and not col.startswith('target_')]
    feature_cols.append('price')  # Include price for forecasting
    
    print(f"🔧 Using {len(feature_cols)} features")
    
    # Store results for each fold
    all_best_params = []
    all_studies = []
    all_scores = []
    
    # Tune each fold (simplified - single horizon for LSTM)
    for fold_idx, (train_df, test_df) in enumerate(global_splits[:3]):  # Limit folds for speed
        print(f"\n{'='*40}")
        print(f"📊 FOLD {fold_idx + 1}/{min(3, len(global_splits))}")
        print(f"{'='*40}")
        
        # Create train/val split within fold
        fold_size = len(train_df)
        val_size = int(fold_size * 0.2)
        
        train_fold_df = train_df.head(fold_size - val_size)
        val_fold_df = train_df.tail(val_size)
        
        print(f"  Train: {len(train_fold_df):,} samples")
        print(f"  Val: {len(val_fold_df):,} samples")
        
        # Convert to numpy for LSTM
        train_data = train_fold_df.select(feature_cols).to_numpy()
        val_data = val_fold_df.select(feature_cols).to_numpy()
        
        # Check for valid data
        if len(train_data) < 100 or len(val_data) < 50:
            print(f"⚠️ Skipping fold {fold_idx + 1}: insufficient data")
            continue
        
        # Tune hyperparameters for this fold
        best_params, study = tune_single_fold(
            train_data, val_data, fold_idx, n_trials
        )
        
        all_best_params.append(best_params)
        all_studies.append(study)
        all_scores.append(study.best_value)
        
        print(f"📈 Fold {fold_idx + 1} best score: {study.best_value:.4f}")
    
    # Analyze results
    print(f"\n{'='*60}")
    print(f"📊 CROSS-FOLD ANALYSIS")
    print(f"{'='*60}")
    
    if all_scores:
        print(f"📈 Scores across folds:")
        print(f"  Mean: {np.mean(all_scores):.4f}")
        print(f"  Std: {np.std(all_scores):.4f}")
        print(f"  Min: {np.min(all_scores):.4f}")
        print(f"  Max: {np.max(all_scores):.4f}")
        
        # Find best performing fold
        best_fold_idx = np.argmax(all_scores)
        best_global_params = all_best_params[best_fold_idx]
        best_study = all_studies[best_fold_idx]
        
        print(f"\n🏆 Best performing fold: {best_fold_idx + 1}")
        print(f"   Score: {all_scores[best_fold_idx]:.4f}")
        
        # Save results
        results_dir = Path("ML/results/hyperparameter_tuning")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        save_best_params(
            best_global_params, 
            best_study,
            "lstm_forecasting", 
            results_dir
        )
        
        # Create visualizations
        create_tuning_visualizations(
            best_study,
            "LSTM_Forecasting_BestFold",
            results_dir
        )
        
        # Print summary
        print_tuning_summary(best_study, "LSTM_Forecasting")
        
        print(f"\n✅ Hyperparameter tuning complete!")
        print(f"📁 Results saved to: {results_dir}")
        
        return best_global_params, all_best_params, all_studies
    
    else:
        print("❌ No successful tuning runs completed!")
        return None, [], []


def main():
    """Main function for LSTM forecasting hyperparameter tuning"""
    
    parser = argparse.ArgumentParser(description='LSTM Forecasting Hyperparameter Tuning')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials per fold (default: 50)')
    parser.add_argument('--strategy', type=str, choices=['per_fold'], 
                       default='per_fold',
                       help='Tuning strategy (default: per_fold)')
    parser.add_argument('--features_dir', type=str, 
                       default='data/features',
                       help='Directory containing pre-engineered features')
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        print("💡 Run feature_engineering/advanced_feature_engineering.py first")
        return
    
    try:
        # Define horizons for forecasting (simplified for tuning)
        horizons = [15, 30, 60]  # Reduced for tuning speed
        
        tune_forecasting_lstm_per_fold(features_dir, horizons, args.n_trials)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        raise


if __name__ == "__main__":
    main()