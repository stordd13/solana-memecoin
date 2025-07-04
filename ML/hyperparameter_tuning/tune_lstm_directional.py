"""
Hyperparameter Tuning for LSTM Directional Models

This script performs comprehensive hyperparameter optimization for LSTM models
used in directional prediction (UP/DOWN movement classification) with walk-forward validation.

WORKFLOW:
1. Load pre-engineered features from feature_engineering module
2. Create walk-forward splits for realistic temporal validation  
3. For each fold, optimize hyperparameters using Optuna
4. Save best parameters for use in training scripts

OPTIMIZATION STRATEGY:
- Optuna for efficient neural architecture search
- Per-fold tuning: Adapt to market regime changes
- Multi-objective: Balance ROC AUC and financial return capture
- Early stopping: Prevent overfitting during tuning
- GPU acceleration: Use CUDA if available

USAGE:
    python tune_lstm_directional.py --n_trials 100 --strategy per_fold
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
from ML.utils.metrics_helpers import financial_classification_metrics
from ML.hyperparameter_tuning.utils.tuning_helpers import (
    save_best_params, create_tuning_visualizations, print_tuning_summary,
    get_default_optuna_config, TuningProgressCallback
)


class LSTMDirectionalModel(nn.Module):
    """LSTM model for directional prediction with flexible architecture"""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, 
                 bidirectional=False, use_attention=False):
        super(LSTMDirectionalModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Take the last output
            lstm_out = attn_out[:, -1, :]
        else:
            # Take the last output
            lstm_out = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(lstm_out)
        
        return output.squeeze()


def suggest_lstm_params(trial: optuna.Trial) -> dict:
    """Suggest LSTM hyperparameters for optimization"""
    
    return {
        # Architecture parameters
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        
        # Training parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'sequence_length': trial.suggest_categorical('sequence_length', [20, 30, 40, 50]),
        
        # Optimization parameters
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop']),
        'scheduler': trial.suggest_categorical('scheduler', ['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR']),
        'patience': trial.suggest_int('patience', 5, 20),
        'max_epochs': trial.suggest_int('max_epochs', 50, 200),
        
        # Loss function parameters
        'pos_weight': trial.suggest_float('pos_weight', 0.5, 2.0),
        'focal_alpha': trial.suggest_float('focal_alpha', 0.25, 0.75),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False])
    }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def create_sequences(data, target, sequence_length):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        tgt = target[i + sequence_length]
        sequences.append(seq)
        targets.append(tgt)
    
    return np.array(sequences), np.array(targets)


def train_lstm_model(model, train_loader, val_loader, params, device, trial=None):
    """Train LSTM model with early stopping"""
    
    # Loss function
    if params['use_focal_loss']:
        criterion = FocalLoss(alpha=params['focal_alpha'], gamma=params['focal_gamma'])
    else:
        pos_weight = torch.tensor([params['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                             weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], 
                              weight_decay=params['weight_decay'])
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'], 
                                weight_decay=params['weight_decay'])
    
    # Scheduler
    if params['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    else:  # CosineAnnealingLR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['max_epochs'])
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(params['max_epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.float())
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Scheduler step
        if params['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                break
        
        # Report to Optuna (for pruning)
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    return model, best_val_loss


def objective_function(trial: optuna.Trial, X_train, y_train, X_val, y_val, 
                      returns_val, fold_idx: int = 0) -> float:
    """
    Optuna objective function for LSTM hyperparameter optimization
    
    Returns combined score: 0.3 * ROC_AUC + 0.7 * Return_Capture_Rate
    """
    
    try:
        # Get suggested parameters
        params = suggest_lstm_params(trial)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, params['sequence_length'])
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, params['sequence_length'])
        
        # Check if we have enough data
        if len(X_train_seq) < 100 or len(X_val_seq) < 20:
            return 0.0
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.FloatTensor(y_val_seq)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = LSTMDirectionalModel(
            input_size=X_train.shape[1],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'],
            bidirectional=params['bidirectional'],
            use_attention=params['use_attention']
        ).to(device)
        
        # Train model
        model, best_val_loss = train_lstm_model(model, train_loader, val_loader, params, device, trial)
        
        # Evaluate model
        model.eval()
        val_preds = []
        val_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_probs.extend(probs)
                val_preds.extend(preds)
        
        val_probs = np.array(val_probs)
        val_preds = np.array(val_preds)
        
        # Align returns with predictions
        returns_val_seq = returns_val[params['sequence_length']:]
        y_val_seq = y_val[params['sequence_length']:]
        
        # Calculate comprehensive metrics
        metrics = financial_classification_metrics(y_val_seq, val_preds, returns_val_seq, val_probs)
        
        # Multi-objective score: balance AUC and financial performance
        auc_score = metrics.get('roc_auc', 0.0)
        financial_score = metrics.get('return_capture_rate', 0.0)
        
        # Combined objective with financial emphasis
        combined_score = 0.3 * auc_score + 0.7 * financial_score
        
        # Store detailed metrics for analysis
        trial.set_user_attr(f'fold_{fold_idx}_roc_auc', auc_score)
        trial.set_user_attr(f'fold_{fold_idx}_precision', metrics.get('precision', 0.0))
        trial.set_user_attr(f'fold_{fold_idx}_recall', metrics.get('recall', 0.0))
        trial.set_user_attr(f'fold_{fold_idx}_return_capture_rate', financial_score)
        trial.set_user_attr(f'fold_{fold_idx}_avg_return_per_tp', metrics.get('avg_return_per_tp', 0.0))
        trial.set_user_attr(f'fold_{fold_idx}_prediction_sharpe', metrics.get('prediction_sharpe', 0.0))
        trial.set_user_attr(f'fold_{fold_idx}_val_loss', best_val_loss)
        
        return combined_score
        
    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return poor score for failed trials


def tune_single_fold(X_train, y_train, X_val, y_val, returns_val, 
                     fold_idx: int, n_trials: int = 100) -> tuple:
    """Tune hyperparameters for a single fold"""
    
    print(f"\n🔍 Tuning fold {fold_idx + 1} with {n_trials} trials...")
    
    # Create study
    config = get_default_optuna_config()
    study = optuna.create_study(**config)
    
    # Add progress callback
    callback = TuningProgressCallback(f"LSTM_Fold_{fold_idx+1}", save_interval=10)
    
    # Optimize with fold-specific objective
    def fold_objective(trial):
        return objective_function(trial, X_train, y_train, X_val, y_val, 
                                returns_val, fold_idx)
    
    study.optimize(fold_objective, n_trials=n_trials, callbacks=[callback])
    
    print(f"✅ Fold {fold_idx + 1} optimization complete!")
    print(f"   Best score: {study.best_value:.4f}")
    
    return study.best_params, study


def tune_lstm_per_fold(features_dir: Path, horizons: List[int], n_trials: int = 100):
    """
    Perform per-fold hyperparameter tuning for LSTM directional model
    
    This approach tunes hyperparameters separately for each walk-forward fold,
    allowing adaptation to different market regimes.
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 LSTM HYPERPARAMETER TUNING - PER FOLD STRATEGY")
    print(f"📊 Horizons: {horizons} minutes")
    print(f"🔬 Trials per fold per horizon: {n_trials}")
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")
    
    # Load feature data
    print(f"\n📁 Loading features from {features_dir}...")
    categories = ["normal_behavior_tokens", "tokens_with_extremes", "dead_tokens"]
    
    all_data_frames = []
    for category in categories:
        cat_dir = features_dir / category
        if cat_dir.exists():
            parquet_files = list(cat_dir.glob("*.parquet"))
            print(f"  Found {len(parquet_files)} files in {category}")
            
            for file_path in tqdm(parquet_files, desc=f"Loading {category}"):
                try:
                    df = pl.read_parquet(file_path)
                    if len(df) >= 400:  # Minimum token length
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
        horizons=horizons,
        time_column='datetime'
    )
    
    if not global_splits:
        raise ValueError(f"No valid splits for any horizon!")
    
    print(f"✅ Created {len(global_splits)} folds for horizons {feasible_horizons}")
    
    # Prepare features and labels
    target_cols = [f'target_{h}m' for h in horizons]
    feature_cols = [col for col in combined_data.columns 
                   if col not in ['datetime', 'price', 'token_id'] + target_cols]
    
    print(f"🔧 Using {len(feature_cols)} features")
    
    # Store results for each horizon and fold
    horizon_results = {}
    
    for horizon in feasible_horizons:
        horizon_results[horizon] = {
            'all_best_params': [],
            'all_studies': [],
            'all_scores': []
        }
    
    # Tune each horizon and fold combination
    for horizon in feasible_horizons:
        print(f"\n{'='*60}")
        print(f"🎯 TUNING HORIZON: {horizon} MINUTES")
        print(f"{'='*60}")
        
        for fold_idx, (train_df, test_df) in enumerate(global_splits):
            print(f"\n{'='*40}")
            print(f"📊 FOLD {fold_idx + 1}/{len(global_splits)} | HORIZON {horizon}m")
            print(f"{'='*40}")
            
            # Create train/val split within fold
            fold_size = len(train_df)
            val_size = int(fold_size * 0.2)  # 20% for validation
            
            train_fold_df = train_df.head(fold_size - val_size)
            val_fold_df = train_df.tail(val_size)
            
            print(f"  Train: {len(train_fold_df):,} samples")
            print(f"  Val: {len(val_fold_df):,} samples")
            print(f"  Test: {len(test_df):,} samples")
            
            # Check if target column exists for this horizon
            target_col = f'target_{horizon}m'
            if target_col not in train_fold_df.columns:
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: target column not found")
                continue
            
            # Prepare fold data
            X_train = train_fold_df.select(feature_cols).to_numpy()
            y_train = train_fold_df[target_col].to_numpy()
            
            X_val = val_fold_df.select(feature_cols).to_numpy()
            y_val = val_fold_df[target_col].to_numpy()
            
            # Calculate returns for financial metrics
            returns_val = val_fold_df['price'].pct_change().shift(-horizon).fill_null(0).to_numpy()
            
            # Check for valid data
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: insufficient class diversity")
                continue
            
            # Tune hyperparameters for this fold and horizon
            best_params, study = tune_single_fold(
                X_train, y_train, X_val, y_val, returns_val, 
                fold_idx, n_trials
            )
            
            horizon_results[horizon]['all_best_params'].append(best_params)
            horizon_results[horizon]['all_studies'].append(study)
            horizon_results[horizon]['all_scores'].append(study.best_value)
            
            print(f"📈 Fold {fold_idx + 1}, Horizon {horizon}m best score: {study.best_value:.4f}")
    
    # Analyze results across horizons and folds
    print(f"\n{'='*60}")
    print(f"📊 CROSS-HORIZON AND CROSS-FOLD ANALYSIS")
    print(f"{'='*60}")
    
    results_dir = Path("ML/results/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    horizon_best_params = {}
    
    for horizon in feasible_horizons:
        horizon_data = horizon_results[horizon]
        all_scores = horizon_data['all_scores']
        all_best_params = horizon_data['all_best_params']
        all_studies = horizon_data['all_studies']
        
        if all_scores:
            print(f"\n📈 Horizon {horizon}m - Scores across folds:")
            print(f"  Mean: {np.mean(all_scores):.4f}")
            print(f"  Std: {np.std(all_scores):.4f}")
            print(f"  Min: {np.min(all_scores):.4f}")
            print(f"  Max: {np.max(all_scores):.4f}")
            
            # Find best performing fold for this horizon
            best_fold_idx = np.argmax(all_scores)
            best_horizon_params = all_best_params[best_fold_idx]
            best_study = all_studies[best_fold_idx]
            
            print(f"🏆 Best performing fold for {horizon}m: {best_fold_idx + 1}")
            print(f"   Score: {all_scores[best_fold_idx]:.4f}")
            
            # Save horizon-specific results
            save_best_params(
                best_horizon_params, 
                best_study,
                f"lstm_directional_{horizon}m", 
                results_dir
            )
            
            # Create visualizations for best fold of this horizon
            create_tuning_visualizations(
                best_study,
                f"LSTM_Directional_{horizon}m_BestFold",
                results_dir
            )
            
            # Print summary for this horizon
            print_tuning_summary(best_study, f"LSTM_Directional_{horizon}m")
            
            horizon_best_params[horizon] = best_horizon_params
        else:
            print(f"❌ No successful tuning runs for horizon {horizon}m!")
    
    if horizon_best_params:
        print(f"\n✅ Hyperparameter tuning complete for all horizons!")
        print(f"📁 Results saved to: {results_dir}")
        print(f"🎯 Tuned horizons: {list(horizon_best_params.keys())}")
        
        return horizon_best_params, horizon_results
    else:
        print("❌ No successful tuning runs completed for any horizon!")
        return {}, {}


def main():
    """Main function for LSTM hyperparameter tuning"""
    
    parser = argparse.ArgumentParser(description='LSTM Hyperparameter Tuning')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials per fold (default: 100)')
    parser.add_argument('--strategy', type=str, choices=['per_fold'], 
                       default='per_fold',
                       help='Tuning strategy (default: per_fold)')
    parser.add_argument('--features_dir', type=str, 
                       default='data/features_with_targets',
                       help='Directory containing pre-engineered features')
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        print("💡 Run feature_engineering/advanced_feature_engineering.py first")
        return
    
    try:
        # Define horizons to match training scripts
        horizons = [15, 30, 60, 120, 240, 360, 720]  # 15min to 12h
        
        tune_lstm_per_fold(features_dir, horizons, args.n_trials)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        raise


if __name__ == "__main__":
    main()