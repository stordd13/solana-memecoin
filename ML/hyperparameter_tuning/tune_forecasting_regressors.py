"""
Hyperparameter Tuning for Forecasting Baseline Regressors

This script performs comprehensive hyperparameter optimization for baseline regression models
used in price forecasting with walk-forward validation.

WORKFLOW:
1. Load pre-engineered features from feature_engineering module
2. Create walk-forward splits for realistic temporal validation  
3. For each fold, optimize hyperparameters using Optuna
4. Save best parameters for use in training scripts

OPTIMIZATION STRATEGY:
- Optuna for efficient hyperparameter search
- Per-fold tuning: Adapt to market regime changes
- Multi-objective: Balance MSE and financial performance
- Financial focus: Weight trading metrics heavily

USAGE:
    python tune_forecasting_regressors.py --n_trials 100 --strategy per_fold
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ML.utils.walk_forward_splitter import WalkForwardSplitter
from ML.utils.metrics_helpers import regression_metrics
from ML.hyperparameter_tuning.utils.tuning_helpers import (
    save_best_params, create_tuning_visualizations, print_tuning_summary,
    get_default_optuna_config, TuningProgressCallback
)


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Suggest XGBoost hyperparameters for optimization"""
    
    return {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        
        # Core parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        
        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True),
        
        # Sampling parameters
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
    }


def suggest_linear_params(trial: optuna.Trial) -> dict:
    """Suggest Linear Regression hyperparameters for optimization"""
    
    # For linear regression, we mainly tune the scaler and regularization
    return {
        'n_jobs': -1,
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        # We could add Ridge/Lasso here in the future
    }


def objective_function(trial: optuna.Trial, X_train, y_train, X_val, y_val, 
                      model_type: str, fold_idx: int = 0) -> float:
    """
    Optuna objective function for regression hyperparameter optimization
    
    Returns combined score: 0.4 * (1 - normalized_mse) + 0.6 * financial_score
    """
    
    try:
        if model_type == 'xgb' and HAS_XGB:
            # Get suggested parameters
            params = suggest_xgb_params(trial)
            model = XGBRegressor(**params)
            
            # XGB can handle raw features
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            
        elif model_type == 'linear':
            # Get suggested parameters
            params = suggest_linear_params(trial)
            
            # Linear regression needs scaled features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model = LinearRegression(**params)
            model.fit(X_train_scaled, y_train)
            val_preds = model.predict(X_val_scaled)
            
        else:
            return 0.0
        
        # Calculate regression metrics
        metrics = regression_metrics(y_val, val_preds)
        
        # Multi-objective score: balance MSE and financial performance
        mse_score = metrics.get('mse', float('inf'))
        r2_score = max(0.0, metrics.get('r2', 0.0))  # Clamp R² to positive
        
        # Normalize MSE (lower is better, so we use 1 - normalized_mse)
        mse_normalized = min(1.0, mse_score / (np.var(y_val) + 1e-8))
        mse_component = 1.0 - mse_normalized
        
        # Financial component (R² is a good proxy for prediction quality)
        financial_component = r2_score
        
        # Combined objective with financial emphasis
        combined_score = 0.4 * mse_component + 0.6 * financial_component
        
        # Store detailed metrics for analysis
        trial.set_user_attr(f'fold_{fold_idx}_mse', mse_score)
        trial.set_user_attr(f'fold_{fold_idx}_mae', metrics.get('mae', 0.0))
        trial.set_user_attr(f'fold_{fold_idx}_r2', r2_score)
        trial.set_user_attr(f'fold_{fold_idx}_combined_score', combined_score)
        
        return combined_score
        
    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {str(e)}")
        return 0.0  # Return poor score for failed trials


def tune_single_fold(X_train, y_train, X_val, y_val, model_type: str,
                     fold_idx: int, n_trials: int = 100) -> tuple:
    """Tune hyperparameters for a single fold"""
    
    print(f"\n🔍 Tuning {model_type.upper()} fold {fold_idx + 1} with {n_trials} trials...")
    
    # Create study
    config = get_default_optuna_config()
    study = optuna.create_study(**config)
    
    # Add progress callback
    callback = TuningProgressCallback(f"{model_type.upper()}_Fold_{fold_idx+1}", save_interval=20)
    
    # Optimize with fold-specific objective
    def fold_objective(trial):
        return objective_function(trial, X_train, y_train, X_val, y_val, 
                                model_type, fold_idx)
    
    study.optimize(fold_objective, n_trials=n_trials, callbacks=[callback])
    
    print(f"✅ Fold {fold_idx + 1} optimization complete!")
    print(f"   Best score: {study.best_value:.4f}")
    
    return study.best_params, study


def tune_forecasting_regressors_per_fold(features_dir: Path, horizons: List[int], 
                                        model_type: str, n_trials: int = 100):
    """
    Perform per-fold hyperparameter tuning for forecasting regression models
    
    This approach tunes hyperparameters separately for each walk-forward fold,
    allowing adaptation to different market regimes.
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 {model_type.upper()} FORECASTING HYPERPARAMETER TUNING - PER FOLD STRATEGY")
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
    
    # Prepare features and labels - for forecasting we need future price targets
    feature_cols = [col for col in combined_data.columns 
                   if col not in ['datetime', 'price', 'token_id'] and not col.startswith('target_')]
    
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
            
            # Create future price targets for forecasting
            train_fold_df = train_fold_df.with_columns(
                train_fold_df['price'].shift(-horizon).alias(f'future_price_{horizon}m')
            )
            val_fold_df = val_fold_df.with_columns(
                val_fold_df['price'].shift(-horizon).alias(f'future_price_{horizon}m')
            )
            
            # Remove NaN targets
            train_fold_df = train_fold_df.filter(pl.col(f'future_price_{horizon}m').is_not_null())
            val_fold_df = val_fold_df.filter(pl.col(f'future_price_{horizon}m').is_not_null())
            
            if len(train_fold_df) == 0 or len(val_fold_df) == 0:
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: no valid targets")
                continue
            
            # Prepare fold data
            X_train = train_fold_df.select(feature_cols).to_numpy()
            y_train = train_fold_df[f'future_price_{horizon}m'].to_numpy()
            
            X_val = val_fold_df.select(feature_cols).to_numpy()
            y_val = val_fold_df[f'future_price_{horizon}m'].to_numpy()
            
            # Check for valid data
            if np.isnan(y_train).any() or np.isnan(y_val).any():
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: NaN values in targets")
                continue
            
            # Tune hyperparameters for this fold and horizon
            best_params, study = tune_single_fold(
                X_train, y_train, X_val, y_val, model_type,
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
                f"{model_type}_forecasting_{horizon}m", 
                results_dir
            )
            
            # Create visualizations for best fold of this horizon
            create_tuning_visualizations(
                best_study,
                f"{model_type.upper()}_Forecasting_{horizon}m_BestFold",
                results_dir
            )
            
            # Print summary for this horizon
            print_tuning_summary(best_study, f"{model_type.upper()}_Forecasting_{horizon}m")
            
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
    """Main function for forecasting regressor hyperparameter tuning"""
    
    parser = argparse.ArgumentParser(description='Forecasting Regressor Hyperparameter Tuning')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of optimization trials per fold (default: 100)')
    parser.add_argument('--strategy', type=str, choices=['per_fold'], 
                       default='per_fold',
                       help='Tuning strategy (default: per_fold)')
    parser.add_argument('--model', type=str, choices=['linear', 'xgb', 'both'], 
                       default='both',
                       help='Model type to tune (default: both)')
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
        # Define horizons for forecasting
        horizons = [15, 30, 60, 120, 240]  # 15min to 4h forecasting
        
        models_to_tune = []
        if args.model in ['linear', 'both']:
            models_to_tune.append('linear')
        if args.model in ['xgb', 'both'] and HAS_XGB:
            models_to_tune.append('xgb')
        elif args.model in ['xgb', 'both'] and not HAS_XGB:
            print("⚠️ XGBoost not available, skipping XGB tuning")
        
        for model_type in models_to_tune:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING {model_type.upper()} HYPERPARAMETER TUNING")
            print(f"{'='*80}")
            
            tune_forecasting_regressors_per_fold(features_dir, horizons, model_type, args.n_trials)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        raise


if __name__ == "__main__":
    main()