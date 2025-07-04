"""
Hyperparameter Tuning for LightGBM Directional Models

This script performs comprehensive hyperparameter optimization for LightGBM models
used in directional prediction (UP/DOWN movement classification) with walk-forward validation.

WORKFLOW:
1. Load pre-engineered features from feature_engineering module
2. Create walk-forward splits for realistic temporal validation  
3. For each fold, optimize hyperparameters using Optuna
4. Save best parameters for use in training scripts

OPTIMIZATION STRATEGY:
- Multi-objective: Balance ROC AUC and financial return capture
- Per-fold tuning: Adapt to market regime changes
- Pruning: Early stopping for inefficient trials
- Financial focus: Weight trading metrics heavily

USAGE:
    python tune_lightgbm_directional.py --n_trials 100 --strategy per_fold
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import polars as pl
import lightgbm as lgb
import numpy as np
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


def suggest_lightgbm_params(trial: optuna.Trial) -> dict:
    """Suggest LightGBM hyperparameters for optimization"""
    
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'force_col_wise': True,
        'deterministic': True,
        
        # Core tree parameters
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        
        # Learning parameters  
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_iterations': trial.suggest_int('num_iterations', 100, 1000),
        
        # Feature sampling
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.4, 1.0),
        
        # Bagging parameters
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
        
        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True),
        
        # Advanced parameters for memecoin data
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
        'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 100.0, log=True),
        'max_cat_threshold': trial.suggest_int('max_cat_threshold', 16, 64),
    }


def objective_function(trial: optuna.Trial, X_train, y_train, X_val, y_val, 
                      returns_val, fold_idx: int = 0) -> float:
    """
    Optuna objective function for LightGBM hyperparameter optimization
    
    Returns combined score: 0.3 * ROC_AUC + 0.7 * Return_Capture_Rate
    """
    
    try:
        # Get suggested parameters
        params = suggest_lightgbm_params(trial)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
        
        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(0)  # Silent training
            ]
        )
        
        # Predict on validation set
        val_probs = model.predict(X_val, num_iteration=model.best_iteration)
        val_preds = (val_probs > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        metrics = financial_classification_metrics(y_val, val_preds, returns_val, val_probs)
        
        # Multi-objective score: balance AUC and financial performance
        # Financial metrics are more important for trading applications
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
        
        # Store tree metrics for analysis
        trial.set_user_attr(f'fold_{fold_idx}_n_estimators', model.best_iteration)
        
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
    callback = TuningProgressCallback(f"LightGBM_Fold_{fold_idx+1}", save_interval=20)
    
    # Optimize with fold-specific objective
    def fold_objective(trial):
        return objective_function(trial, X_train, y_train, X_val, y_val, 
                                returns_val, fold_idx)
    
    study.optimize(fold_objective, n_trials=n_trials, callbacks=[callback])
    
    print(f"✅ Fold {fold_idx + 1} optimization complete!")
    print(f"   Best score: {study.best_value:.4f}")
    
    return study.best_params, study


def tune_lightgbm_per_fold(features_dir: Path, horizons: List[int], n_trials: int = 100):
    """
    Perform per-fold hyperparameter tuning for LightGBM directional model
    
    This approach tunes hyperparameters separately for each walk-forward fold,
    allowing adaptation to different market regimes.
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 LightGBM HYPERPARAMETER TUNING - PER FOLD STRATEGY")
    print(f"📊 Horizons: {horizons} minutes")
    print(f"🔬 Trials per fold per horizon: {n_trials}")
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
                f"lightgbm_directional_{horizon}m", 
                results_dir
            )
            
            # Create visualizations for best fold of this horizon
            create_tuning_visualizations(
                best_study,
                f"LightGBM_Directional_{horizon}m_BestFold",
                results_dir
            )
            
            # Print summary for this horizon
            print_tuning_summary(best_study, f"LightGBM_Directional_{horizon}m")
            
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


def tune_lightgbm_global(features_dir: Path, horizons: List[int], n_trials: int = 200):
    """
    Perform global hyperparameter tuning for LightGBM directional model
    
    This approach tunes hyperparameters once using aggregated validation data
    from all folds. Faster but less adaptive to market changes.
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 LightGBM HYPERPARAMETER TUNING - GLOBAL STRATEGY")
    print(f"📊 Horizons: {horizons} minutes")  
    print(f"🔬 Total trials: {n_trials}")
    print(f"{'='*60}")
    
    # TODO: Implement global tuning strategy
    # For now, redirect to per-fold approach
    print("🔄 Redirecting to per-fold tuning (more robust)...")
    return tune_lightgbm_per_fold(features_dir, horizons, n_trials // 4)


def main():
    """Main function for LightGBM hyperparameter tuning"""
    
    parser = argparse.ArgumentParser(description='LightGBM Hyperparameter Tuning')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials per fold (default: 100)')
    parser.add_argument('--strategy', type=str, choices=['per_fold', 'global'], 
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
        # Short-term horizons (default)
        short_term_horizons = [15, 30, 60, 120, 240, 360, 720]  # 15min to 12h
        # Medium-term horizons  
        medium_term_horizons = [240, 360, 480, 720, 960, 1380]  # 4h to 23h
        
        # Use short-term by default, can be changed manually
        # For short-term model (train_lightgbm_model.py):
        horizons = short_term_horizons
        # For medium-term model (train_lightgbm_model_medium_term.py), uncomment:
        # horizons = medium_term_horizons
        
        if args.strategy == 'per_fold':
            tune_lightgbm_per_fold(features_dir, horizons, args.n_trials)
        else:
            tune_lightgbm_global(features_dir, horizons, args.n_trials)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        raise


if __name__ == "__main__":
    main()