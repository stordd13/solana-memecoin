"""
Hyperparameter Tuning for Logistic Regression Directional Models

This script performs comprehensive hyperparameter optimization for Logistic Regression models
used in directional prediction (UP/DOWN movement classification) with walk-forward validation.

WORKFLOW:
1. Load pre-engineered features from feature_engineering module
2. Create walk-forward splits for realistic temporal validation  
3. For each fold, optimize hyperparameters using Bayesian optimization (scikit-optimize)
4. Save best parameters for use in training scripts

OPTIMIZATION STRATEGY:
- Bayesian optimization for efficient search space exploration
- Per-fold tuning: Adapt to market regime changes
- Multi-objective: Balance ROC AUC and financial return capture
- Financial focus: Weight trading metrics heavily

USAGE:
    python tune_logistic_regression.py --n_calls 50 --strategy per_fold
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ML.utils.walk_forward_splitter import WalkForwardSplitter
from ML.utils.metrics_helpers import financial_classification_metrics
from ML.hyperparameter_tuning.utils.tuning_helpers import (
    save_best_params, create_tuning_visualizations, print_tuning_summary
)


# Define search space for logistic regression
search_space = [
    Real(1e-6, 1e2, prior='log-uniform', name='C'),
    Categorical(['l1', 'l2', 'elasticnet'], name='penalty'),
    Real(0.1, 1.0, name='l1_ratio'),  # Only used for elasticnet
    Integer(100, 2000, name='max_iter'),
    Categorical(['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'], name='solver'),
    Real(1e-8, 1e-4, prior='log-uniform', name='tol'),
    Categorical([True, False], name='fit_intercept'),
    Categorical(['balanced', None], name='class_weight')
]


def get_compatible_solver_penalty(penalty, solver):
    """Get compatible solver-penalty combinations"""
    compatible_combinations = {
        'liblinear': ['l1', 'l2'],
        'lbfgs': ['l2'],
        'newton-cg': ['l2'],
        'sag': ['l2'],
        'saga': ['l1', 'l2', 'elasticnet']
    }
    
    if penalty in compatible_combinations.get(solver, []):
        return solver
    
    # Find compatible solver for penalty
    for s, penalties in compatible_combinations.items():
        if penalty in penalties:
            return s
    
    return 'saga'  # Most flexible solver


def create_logistic_regression_model(C, penalty, l1_ratio, max_iter, solver, tol, fit_intercept, class_weight):
    """Create logistic regression model with given parameters"""
    
    # Ensure solver-penalty compatibility
    solver = get_compatible_solver_penalty(penalty, solver)
    
    # Build model parameters
    model_params = {
        'C': C,
        'penalty': penalty,
        'max_iter': max_iter,
        'solver': solver,
        'tol': tol,
        'fit_intercept': fit_intercept,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Add class_weight if not None
    if class_weight is not None:
        model_params['class_weight'] = class_weight
    
    # Add l1_ratio only for elasticnet
    if penalty == 'elasticnet':
        model_params['l1_ratio'] = l1_ratio
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**model_params))
    ])
    
    return pipeline


def objective_function(params_list, X_train, y_train, X_val, y_val, returns_val, fold_idx: int = 0):
    """
    Objective function for Bayesian optimization
    
    Returns combined score: 0.3 * ROC_AUC + 0.7 * Return_Capture_Rate
    """
    
    try:
        # Unpack parameters
        C, penalty, l1_ratio, max_iter, solver, tol, fit_intercept, class_weight = params_list
        
        # Create model
        model = create_logistic_regression_model(
            C, penalty, l1_ratio, max_iter, solver, tol, fit_intercept, class_weight
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = model.predict(X_val)
        
        # Calculate comprehensive metrics
        metrics = financial_classification_metrics(y_val, val_preds, returns_val, val_probs)
        
        # Multi-objective score: balance AUC and financial performance
        auc_score = metrics.get('roc_auc', 0.0)
        financial_score = metrics.get('return_capture_rate', 0.0)
        
        # Combined objective with financial emphasis
        combined_score = 0.3 * auc_score + 0.7 * financial_score
        
        return -combined_score  # Negative because gp_minimize minimizes
        
    except Exception as e:
        print(f"⚠️ Trial failed: {str(e)}")
        return 0.0  # Return poor score for failed trials


def tune_single_fold(X_train, y_train, X_val, y_val, returns_val, 
                     fold_idx: int, n_calls: int = 50):
    """Tune hyperparameters for a single fold using Bayesian optimization"""
    
    print(f"\n🔍 Tuning fold {fold_idx + 1} with {n_calls} calls...")
    
    # Create objective function with fold-specific data
    @use_named_args(search_space)
    def fold_objective(**params):
        params_list = [params[dim.name] for dim in search_space]
        return objective_function(params_list, X_train, y_train, X_val, y_val, returns_val, fold_idx)
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=fold_objective,
        dimensions=search_space,
        n_calls=n_calls,
        n_initial_points=10,
        random_state=42,
        acq_func='EI',  # Expected Improvement
        n_jobs=1  # Sequential for stability
    )
    
    # Extract best parameters
    best_params = {dim.name: result.x[i] for i, dim in enumerate(search_space)}
    
    print(f"✅ Fold {fold_idx + 1} optimization complete!")
    print(f"   Best score: {-result.fun:.4f}")
    
    return best_params, result


def tune_logistic_regression_per_fold(features_dir: Path, horizons: List[int], n_calls: int = 50, max_files: int = 10, sample_size: int = None):
    """
    Perform per-fold hyperparameter tuning for Logistic Regression directional model
    
    This approach tunes hyperparameters separately for each walk-forward fold,
    allowing adaptation to different market regimes.
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 LOGISTIC REGRESSION HYPERPARAMETER TUNING - PER FOLD STRATEGY")
    print(f"📊 Horizons: {horizons} minutes")
    print(f"🔬 Calls per fold: {n_calls}")
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
            
            # Limit number of files for tuning to prevent memory issues
            if max_files > 0:
                parquet_files = parquet_files[:max_files]
                print(f"  Using {len(parquet_files)} files for tuning (memory optimization)")
            else:
                print(f"  Using all {len(parquet_files)} files (no limit)")
            
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
    
    # Sample data if requested for faster tuning
    if sample_size and len(combined_data) > sample_size:
        print(f"🎲 Sampling {sample_size:,} rows for faster tuning...")
        combined_data = combined_data.sample(n=sample_size, seed=42)
        print(f"📊 Using {len(combined_data):,} sampled rows")
    
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
    
    # Targets should already exist from the separate target creation script
    print(f"🎯 Using pre-computed directional targets for horizons {feasible_horizons}...")
    
    # Prepare features and labels (use label_Xm format)
    target_cols = [f'label_{h}m' for h in feasible_horizons]
    return_cols = [f'return_{h}m' for h in feasible_horizons]
    feature_cols = [col for col in combined_data.columns 
                   if col not in ['datetime', 'price', 'token_id'] + target_cols + return_cols]
    
    print(f"🔧 Using {len(feature_cols)} features")
    
    # Store results for each horizon and fold
    horizon_results = {}
    
    for horizon in feasible_horizons:
        horizon_results[horizon] = {
            'all_best_params': [],
            'all_results': [],
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
            
            # Check if target column exists for this horizon (use label_Xm format)
            target_col = f'label_{horizon}m'
            if target_col not in train_fold_df.columns:
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: target column not found")
                print(f"   Available columns: {train_fold_df.columns}")
                continue
            
            # Filter out NaN values in targets first
            train_valid = train_fold_df.filter(pl.col(target_col).is_not_null())
            val_valid = val_fold_df.filter(pl.col(target_col).is_not_null())
            
            if len(train_valid) == 0 or len(val_valid) == 0:
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: no valid targets after NaN filtering")
                continue
            
            # Prepare fold data
            X_train = train_valid.select(feature_cols).to_numpy()
            y_train = train_valid[target_col].to_numpy()
            
            X_val = val_valid.select(feature_cols).to_numpy()
            y_val = val_valid[target_col].to_numpy()
            
            # Calculate returns for financial metrics
            returns_val = val_valid['price'].pct_change().shift(-horizon).fill_null(0).to_numpy()
            
            # Check for valid data
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                print(f"⚠️ Skipping fold {fold_idx + 1}, horizon {horizon}m: insufficient class diversity")
                continue
            
            # Tune hyperparameters for this fold and horizon
            best_params, result = tune_single_fold(
                X_train, y_train, X_val, y_val, returns_val, 
                fold_idx, n_calls
            )
            
            horizon_results[horizon]['all_best_params'].append(best_params)
            horizon_results[horizon]['all_results'].append(result)
            horizon_results[horizon]['all_scores'].append(-result.fun)
            
            print(f"📈 Fold {fold_idx + 1}, Horizon {horizon}m best score: {-result.fun:.4f}")
    
    # Analyze results across horizons and folds
    print(f"\n{'='*60}")
    print(f"📊 CROSS-HORIZON AND CROSS-FOLD ANALYSIS")
    print(f"{'='*60}")
    
    results_dir = Path("ML/results/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    horizon_best_params = {}
    import json
    
    for horizon in feasible_horizons:
        horizon_data = horizon_results[horizon]
        all_scores = horizon_data['all_scores']
        all_best_params = horizon_data['all_best_params']
        all_results = horizon_data['all_results']
        
        if all_scores:
            print(f"\n📈 Horizon {horizon}m - Scores across folds:")
            print(f"  Mean: {np.mean(all_scores):.4f}")
            print(f"  Std: {np.std(all_scores):.4f}")
            print(f"  Min: {np.min(all_scores):.4f}")
            print(f"  Max: {np.max(all_scores):.4f}")
            
            # Find best performing fold for this horizon
            best_fold_idx = np.argmax(all_scores)
            best_horizon_params = all_best_params[best_fold_idx]
            
            print(f"🏆 Best performing fold for {horizon}m: {best_fold_idx + 1}")
            print(f"   Score: {all_scores[best_fold_idx]:.4f}")
            
            # Save horizon-specific results (convert numpy types for JSON compatibility)
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (bool, np.bool_)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            params_file = results_dir / f"logistic_regression_directional_{horizon}m_best_params.json"
            with open(params_file, 'w') as f:
                json.dump(convert_numpy_types(best_horizon_params), f, indent=2)
            print(f"💾 Best parameters for {horizon}m saved to: {params_file}")
            
            # Save all fold results for this horizon
            all_results_file = results_dir / f"logistic_regression_directional_{horizon}m_all_folds.json"
            with open(all_results_file, 'w') as f:
                json.dump(convert_numpy_types({
                    'best_params_per_fold': all_best_params,
                    'scores_per_fold': all_scores,
                    'best_fold_idx': int(best_fold_idx),
                    'mean_score': float(np.mean(all_scores)),
                    'std_score': float(np.std(all_scores))
                }), f, indent=2)
            
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
    """Main function for Logistic Regression hyperparameter tuning"""
    
    parser = argparse.ArgumentParser(description='Logistic Regression Hyperparameter Tuning')
    parser.add_argument('--n_calls', type=int, default=50,
                       help='Number of Bayesian optimization calls per fold (default: 50)')
    parser.add_argument('--strategy', type=str, choices=['per_fold'], 
                       default='per_fold',
                       help='Tuning strategy (default: per_fold)')
    parser.add_argument('--features_dir', type=str, 
                       default='data/features_with_targets',
                       help='Directory containing pre-engineered features')
    parser.add_argument('--max_files', type=int, default=10,
                       help='Maximum files per category to prevent memory issues (default: 10, use 0 for no limit)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample only N rows from combined data for faster tuning')
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        print("💡 Run feature_engineering/advanced_feature_engineering.py first")
        return
    
    try:
        # Define horizons to match training scripts
        horizons = [15, 30, 60, 120, 240, 360, 720]  # 15min to 12h
        
        tune_logistic_regression_per_fold(features_dir, horizons, args.n_calls, args.max_files, args.sample_size)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        raise


if __name__ == "__main__":
    main()