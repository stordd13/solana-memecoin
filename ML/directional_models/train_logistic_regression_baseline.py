import sys
from pathlib import Path
import json
import joblib

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so we can import `ML.*` from
# any working directory.
# ------------------------------------------------------------------
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from ML.utils.metrics_helpers import classification_metrics, financial_classification_metrics
from ML.utils.winsorizer import Winsorizer

# --- Configuration ---
CONFIG = {
    'features_dir': Path('data/features'),
    'results_dir': Path('ML/results/logreg_short_term'),
    'categories': [
        'normal_behavior_tokens',
        'tokens_with_extremes',
        'dead_tokens'
    ],
    'horizons': [15, 30, 60],
    'random_state': 42,
    'min_rows_per_token': 60,
    'max_tokens_sample': None,  # Set to integer (e.g., 1000) to randomly sample tokens for testing
}

# --- Helper Functions ---

def add_labels_if_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure directional label columns and return columns exist; create them if they are absent."""
    if 'price' not in df.columns:
        return df
    
    missing_horizons = [h for h in CONFIG['horizons'] if f'label_{h}m' not in df.columns]
    missing_returns = [h for h in CONFIG['horizons'] if f'return_{h}m' not in df.columns]
    
    if not missing_horizons and not missing_returns:
        return df

    # Add missing labels and returns
    for h in missing_horizons:
        df = df.with_columns(
            (pl.col('price').shift(-h) > pl.col('price')).alias(f'label_{h}m')
        )
    
    for h in missing_returns:
        df = df.with_columns(
            ((pl.col('price').shift(-h) - pl.col('price')) / pl.col('price')).alias(f'return_{h}m')
        )
    
    return df


def prepare_and_split_data():
    """
    Loads all feature files, creates temporal splits, fits a winsorizer for each token
    on its training split, and applies that winsorizer to all its splits.
    """
    data_splits = {'train': [], 'val': [], 'test': []}
    token_scalers = {}
    
    paths = []
    for cat in CONFIG['categories']:
        paths += list((CONFIG['features_dir'] / cat).glob('*.parquet'))
    
    print(f"Found {len(paths)} token files to process.")
    
    # Random sampling for testing with smaller datasets
    if CONFIG['max_tokens_sample'] is not None and len(paths) > CONFIG['max_tokens_sample']:
        import random
        random.seed(CONFIG['random_state'])
        paths = random.sample(paths, CONFIG['max_tokens_sample'])
        print(f"🎲 Randomly sampled {len(paths)} tokens for testing (max_tokens_sample={CONFIG['max_tokens_sample']})")
    else:
        print(f"🔄 Processing all {len(paths)} available tokens")
    
    # 1. FIT WINSORIZERS on training split of each token
    for path in tqdm(paths, desc="1/3 Fitting Winsorizers"):
        df = pl.read_parquet(path)
        if df.height < CONFIG['min_rows_per_token']:
            continue
            
        feature_cols = [c for c in df.columns if c not in ['datetime', 'price'] and not c.startswith('label_')]
        if not feature_cols:
            continue
            
        train_split_df = df.slice(0, int(0.6 * df.height))
        X_train_token = train_split_df[feature_cols].to_numpy()
        
        # Clean non-finite values before fitting
        X_train_token[~np.isfinite(X_train_token)] = np.nan
        valid_rows = ~np.isnan(X_train_token).any(axis=1)
        
        if valid_rows.sum() < 2:  # Need at least 2 samples to fit winsorizer
            continue
            
        # Use Winsorizer instead of RobustScaler - better for crypto data
        winsorizer = Winsorizer(lower_percentile=0.005, upper_percentile=0.995)
        winsorizer.fit(X_train_token[valid_rows])
            
        token_scalers[path.stem] = winsorizer

    # 2. LOAD, SCALE, and SPLIT data for all tokens
    for path in tqdm(paths, desc="2/3 Applying Winsorization"):
        token_id = path.stem
        if token_id not in token_scalers:
            continue  # Skip tokens for which no winsorizer was fitted
            
        df = add_labels_if_missing(pl.read_parquet(path))
        feature_cols = [c for c in df.columns if c not in ['datetime', 'price'] and not c.startswith('label_') and not c.startswith('return_')]
        winsorizer = token_scalers[token_id]

        # Define splits - account for max horizon to ensure labels exist
        max_horizon = max(CONFIG['horizons'])
        usable_rows = df.height - max_horizon  # Can't use last max_horizon rows for labels
        
        if usable_rows <= 0:
            continue  # Skip tokens too short for any predictions
            
        splits_indices = {
            'train': (0, int(0.6 * usable_rows)),
            'val': (int(0.6 * usable_rows), int(0.8 * usable_rows)),
            'test': (int(0.8 * usable_rows), usable_rows)
        }
        
        for split_name, (start, end) in splits_indices.items():
            if start >= end:
                continue
                
            df_split = df.slice(start, end - start)
            X = df_split[feature_cols].to_numpy()
            y = df_split.select([f'label_{h}m' for h in CONFIG['horizons']]).to_numpy()
            returns = df_split.select([f'return_{h}m' for h in CONFIG['horizons']]).to_numpy()
            
            # Apply winsorization
            X_scaled = winsorizer.transform(X)
            
            data_splits[split_name].append({'X': X_scaled, 'y': y, 'returns': returns, 'token': token_id})

    # 3. AGGREGATE splits into final X, y, returns dictionaries
    final_data = {}
    for split_name, data_list in data_splits.items():
        if not data_list:
            empty_dict = {h: np.empty((0,0)) for h in CONFIG['horizons']}
            empty_labels = {h: np.empty((0,)) for h in CONFIG['horizons']}
            empty_returns = {h: np.empty((0,)) for h in CONFIG['horizons']}
            final_data[split_name] = (empty_dict, empty_labels, empty_returns)
            continue

        X_full = np.vstack([d['X'] for d in data_list])
        y_full = np.vstack([d['y'] for d in data_list])
        returns_full = np.vstack([d['returns'] for d in data_list])
        
        X_horizon, y_horizon, returns_horizon = {}, {}, {}
        for i, h in enumerate(CONFIG['horizons']):
            y_h = y_full[:, i].astype(float) # Use float to handle NaNs
            returns_h = returns_full[:, i].astype(float)
            
            # Filter out rows with non-finite features, labels, or returns
            finite_mask_X = np.all(np.isfinite(X_full), axis=1)
            finite_mask_y = np.isfinite(y_h)
            finite_mask_returns = np.isfinite(returns_h)
            combined_mask = finite_mask_X & finite_mask_y & finite_mask_returns
            
            X_horizon[h] = X_full[combined_mask]
            y_horizon[h] = y_h[combined_mask].astype(int) # Convert back to int for classifier
            returns_horizon[h] = returns_h[combined_mask]
            
            print(f"Horizon {h}m ({split_name}): {y_horizon[h].shape[0]:,} final samples")
            
        final_data[split_name] = (X_horizon, y_horizon, returns_horizon)
        
    return final_data['train'], final_data['val'], final_data['test'], token_scalers


def main():
    """Main training and evaluation pipeline."""
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔬 Training Logistic Regression Baseline")
    print("Using Winsorization (0.5%, 99.5%) for feature scaling")
    print("Horizons: 15min, 30min, 1h")
    if CONFIG['max_tokens_sample'] is not None:
        print(f"🎲 TESTING MODE: Using random sample of {CONFIG['max_tokens_sample']} tokens")
    else:
        print("🔄 FULL MODE: Using all available tokens")
    print("="*60)
    
    (X_train, y_train, returns_train), (X_val, y_val, returns_val), (X_test, y_test, returns_test), token_scalers = prepare_and_split_data()
    
    if not X_train:
        print("\n❌ ERROR: No training data available. Check:")
        print("1. Feature files exist in data/features/")
        print("2. Run feature engineering first if needed")
        return

    metrics_all = {}

    # Add overall progress tracking
    print(f"\n📈 Training models for {len(CONFIG['horizons'])} horizons: {CONFIG['horizons']}")
    
    for i, h in enumerate(CONFIG['horizons'], 1):
        print(f"\n--- Processing Horizon {h}m ({i}/{len(CONFIG['horizons'])}) ---")
        X_train_h, y_train_h = X_train[h], y_train[h]
        X_test_h, y_test_h = X_test[h], y_test[h]
        returns_test_h = returns_test[h]

        if X_train_h.shape[0] < 10 or X_test_h.shape[0] < 10:
            print(f"Skipping {h}m horizon – insufficient samples.")
            continue

        # Check for class balance
        unique_classes, counts = np.unique(y_train_h, return_counts=True)
        print(f"Training class distribution: {dict(zip(unique_classes, counts))}")

        if len(unique_classes) < 2:
            print(f"Skipping {h}m horizon – only one class present in training data.")
            continue

        print(f"Training logistic regression for {h}m horizon on {X_train_h.shape[0]:,} samples...")
        
        # Handle class imbalance by using class weights
        class_weight = 'balanced' if counts[0] != counts[1] else None
        if class_weight:
            print(f"Using balanced class weights due to imbalance")
        
        # SMART EARLY STOPPING: Try multiple C values and stop when no improvement
        print(f"🔄 Starting training with validation-based early stopping...")
        
        from sklearn.metrics import log_loss
        
        # Test different C values to find optimal regularization quickly
        C_values = [0.1, 1.0, 10.0]  # Higher C values for better convergence
        best_model = None
        best_val_score = float('inf')
        
        print(f"🎯 Testing {len(C_values)} different regularization values...")
        
        for i, C_val in enumerate(C_values):
            print(f"  Testing C={C_val} ({i+1}/{len(C_values)})...")
            
            model = LogisticRegression(
                max_iter=500,   # Increased for better convergence
                n_jobs=-1, 
                random_state=CONFIG['random_state'],
                class_weight=class_weight,
                solver='liblinear',  # More stable solver
                penalty='l1',
                C=C_val,
                tol=1e-4,  # Better tolerance
                verbose=0  # Disable verbose to clean up output
            )
            
            try:
                # Fit and validate
                model.fit(X_train_h, y_train_h)
                
                # Check convergence
                if not model.n_iter_[0] < 100:  # Didn't converge
                    print(f"    ⚠️  Model didn't converge with C={C_val}, skipping...")
                    continue
                
                # Validate on validation set
                X_val_h, y_val_h = X_val[h], y_val[h]
                val_pred_proba = model.predict_proba(X_val_h)[:, 1]
                val_score = log_loss(y_val_h, val_pred_proba)
                
                print(f"    ✅ C={C_val}: {model.n_iter_[0]} iterations, val_loss={val_score:.4f}")
                
                if val_score < best_val_score:
                    best_val_score = val_score
                    best_model = model
                    print(f"    🎯 New best model!")
                    
            except Exception as e:
                print(f"    ❌ Failed with C={C_val}: {e}")
                continue
        
        if best_model is None:
            print("⚠️  No model converged! Falling back to simple model...")
            model = LogisticRegression(
                max_iter=50, n_jobs=-1, random_state=CONFIG['random_state'],
                class_weight=class_weight, solver='liblinear'  # Simpler solver
            )
            model.fit(X_train_h, y_train_h)
        else:
            model = best_model
            print(f"✅ Best model: validation loss = {best_val_score:.4f}")

        # Make predictions on test set
        print(f"🔮 Making predictions on {X_test_h.shape[0]:,} test samples...")
        y_pred = model.predict(X_test_h)
        y_prob = model.predict_proba(X_test_h)[:, 1]
        print("✅ Predictions completed")
        
        print("📊 Calculating evaluation metrics (including financial metrics)...")
        metrics = financial_classification_metrics(y_test_h, y_pred, returns_test_h, y_prob)
        metrics_all[f'{h}m'] = metrics
        
        # Enhanced metrics preview
        print(f"   🎯 Accuracy: {metrics['accuracy']:.2%}")
        print(f"   📈 ROC AUC: {metrics['roc_auc']:.2%}")
        print(f"   💰 Recall by Return: {metrics['recall_by_return']:.2%}")
        print(f"   🎪 Hybrid F1 (count+return): {metrics['hybrid_f1_precision_count_recall_return']:.2%}")
        print(f"   💵 Return Capture Rate: {metrics['return_capture_rate']:.2%}")
        
        print(f"💾 Saving model...")
        joblib.dump(model, CONFIG['results_dir'] / f'logreg_{h}m.joblib')
        print(f"✅ Horizon {h}m completed!")

    # Save winsorizers and metrics
    joblib.dump(token_scalers, CONFIG['results_dir'] / 'token_winsorizers.joblib')
    with open(CONFIG['results_dir'] / 'metrics.json', 'w') as f:
        json.dump(metrics_all, f, indent=2)
    
    # Save configuration
    config_info = {
        'scaling_method': 'Winsorization',
        'lower_percentile': 0.005,
        'upper_percentile': 0.995,
        'regularization': 'L1',
        'C': 0.1,
        'solver': 'saga',
        'class_weight': 'balanced'
    }
    with open(CONFIG['results_dir'] / 'config.json', 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print("\nFinal Metrics:", json.dumps(metrics_all, indent=2))
    print(f"\n✅ Model training complete!")
    print(f"📁 Results saved to: {CONFIG['results_dir']}")

    # Create and save enhanced visualization with financial metrics
    if metrics_all:
        fig = go.Figure()
        
        # Standard metrics
        standard_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in standard_keys:
            fig.add_trace(go.Bar(
                name=key.replace('_', ' ').title(),
                x=[f'{h}m' for h, metrics in metrics_all.items()],
                y=[metrics.get(key, 0) for metrics in metrics_all.values()],
                legendgroup="standard",
                legendgrouptitle_text="Standard Metrics"
            ))
        
        # Financial metrics (new!)
        financial_keys = ['recall_by_return', 'hybrid_f1_precision_count_recall_return', 'return_capture_rate']
        for key in financial_keys:
            display_name = key.replace('_', ' ').replace('hybrid f1 precision count recall return', 'Hybrid F1').title()
            fig.add_trace(go.Bar(
                name=display_name,
                x=[f'{h}m' for h, metrics in metrics_all.items()],
                y=[metrics.get(key, 0) for metrics in metrics_all.values()],
                legendgroup="financial",
                legendgrouptitle_text="Financial Metrics"
            ))
        
        fig.update_layout(
            barmode='group',
            title='Logistic Regression: Standard + Financial Metrics',
            yaxis_range=[0,1],
            xaxis_title="Prediction Horizon",
            yaxis_title="Metric Value"
        )
        fig.write_html(CONFIG['results_dir'] / 'metrics.html')
        print(f"\n✅ Enhanced visualization with financial metrics saved to {CONFIG['results_dir'] / 'metrics.html'}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Logistic Regression Baseline with optional token sampling')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Number of tokens to randomly sample for testing (e.g., 1000). If not specified, uses all tokens.')
    parser.add_argument('--results-suffix', type=str, default='', 
                       help='Suffix to add to results directory (e.g., "_test" for test runs)')
    
    args = parser.parse_args()
    
    # Update config based on command line arguments
    if args.sample is not None:
        CONFIG['max_tokens_sample'] = args.sample
        if args.results_suffix == '':
            args.results_suffix = f'_sample_{args.sample}'
    
    if args.results_suffix:
        CONFIG['results_dir'] = Path(f'ML/results/logreg_short_term{args.results_suffix}')
    
    main() 