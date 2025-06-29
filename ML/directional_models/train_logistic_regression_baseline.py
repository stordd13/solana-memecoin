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
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from ML.utils.metrics_helpers import classification_metrics

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
}

# --- Helper Functions ---

def add_labels_if_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure directional label columns exist; create them if they are absent."""
    if 'price' not in df.columns:
        return df
    
    missing_horizons = [h for h in CONFIG['horizons'] if f'label_{h}m' not in df.columns]
    if not missing_horizons:
        return df

    for h in missing_horizons:
        df = df.with_columns(
            (pl.col('price').shift(-h) > pl.col('price')).alias(f'label_{h}m')
        )
    return df


def prepare_and_split_data():
    """
    Loads all feature files, creates temporal splits, fits a scaler for each token
    on its training split, and applies that scaler to all its splits.
    """
    data_splits = {'train': [], 'val': [], 'test': []}
    token_scalers = {}
    
    paths = []
    for cat in CONFIG['categories']:
        paths += list((CONFIG['features_dir'] / cat).glob('*.parquet'))
    
    print(f"Found {len(paths)} token files to process.")
    
    # 1. FIT SCALERS on training split of each token
    for path in tqdm(paths, desc="1/3 Fitting Scalers"):
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
        
        if valid_rows.sum() < 2:  # Need at least 2 samples to fit scaler
            continue
            
        scaler = RobustScaler()
        scaler.fit(X_train_token[valid_rows])
        
        # --- CRITICAL FIX: Patch scaler to prevent division by zero ---
        zero_iqr_mask = np.isclose(scaler.scale_, 0)
        if np.any(zero_iqr_mask):
            scaler.scale_[zero_iqr_mask] = 1.0
            
        token_scalers[path.stem] = scaler

    # 2. LOAD, SCALE, and SPLIT data for all tokens
    for path in tqdm(paths, desc="2/3 Scaling Splits"):
        token_id = path.stem
        if token_id not in token_scalers:
            continue  # Skip tokens for which no scaler was fitted
            
        df = add_labels_if_missing(pl.read_parquet(path))
        feature_cols = [c for c in df.columns if c not in ['datetime', 'price'] and not c.startswith('label_')]
        scaler = token_scalers[token_id]

        # Define splits
        n_rows = df.height
        splits_indices = {
            'train': (0, int(0.6 * n_rows)),
            'val': (int(0.6 * n_rows), int(0.8 * n_rows)),
            'test': (int(0.8 * n_rows), n_rows)
        }
        
        for split_name, (start, end) in splits_indices.items():
            if start >= end:
                continue
                
            df_split = df.slice(start, end - start)
            X = df_split[feature_cols].to_numpy()
            y = df_split.select([f'label_{h}m' for h in CONFIG['horizons']]).to_numpy()
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            data_splits[split_name].append({'X': X_scaled, 'y': y, 'token': token_id})

    # 3. AGGREGATE splits into final X, y dictionaries
    final_data = {}
    for split_name, data_list in data_splits.items():
        if not data_list:
            final_data[split_name] = ({h: np.empty((0,0)) for h in CONFIG['horizons']}, {h: np.empty((0,)) for h in CONFIG['horizons']})
            continue

        X_full = np.vstack([d['X'] for d in data_list])
        y_full = np.vstack([d['y'] for d in data_list])
        
        X_horizon, y_horizon = {}, {}
        for i, h in enumerate(CONFIG['horizons']):
            y_h = y_full[:, i].astype(float) # Use float to handle NaNs
            
            # Filter out rows with non-finite features or labels
            finite_mask_X = np.all(np.isfinite(X_full), axis=1)
            finite_mask_y = np.isfinite(y_h)
            combined_mask = finite_mask_X & finite_mask_y
            
            X_horizon[h] = X_full[combined_mask]
            y_horizon[h] = y_h[combined_mask].astype(int) # Convert back to int for classifier
            
            print(f"Horizon {h}m ({split_name}): {y_horizon[h].shape[0]:,} final samples")
            
        final_data[split_name] = (X_horizon, y_horizon)
        
    return final_data['train'], final_data['val'], final_data['test'], token_scalers


def main():
    """Main training and evaluation pipeline."""
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), token_scalers = prepare_and_split_data()

    metrics_all = {}

    for h in CONFIG['horizons']:
        print(f"\n--- Processing Horizon {h}m ---")
        X_train_h, y_train_h = X_train[h], y_train[h]
        X_test_h, y_test_h = X_test[h], y_test[h]

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
        model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=CONFIG['random_state'])
        model.fit(X_train_h, y_train_h)

        y_pred = model.predict(X_test_h)
        y_prob = model.predict_proba(X_test_h)[:, 1]
        
        metrics = classification_metrics(y_test_h, y_pred, y_prob)
        metrics_all[f'{h}m'] = metrics
        
        joblib.dump(model, CONFIG['results_dir'] / f'logreg_{h}m.joblib')

    # Save scalers and metrics
    joblib.dump(token_scalers, CONFIG['results_dir'] / 'token_scalers.joblib')
    with open(CONFIG['results_dir'] / 'metrics.json', 'w') as f:
        json.dump(metrics_all, f, indent=2)
    
    print("\nFinal Metrics:", json.dumps(metrics_all, indent=2))

    # Create and save visualization
    if metrics_all:
        fig = go.Figure()
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in metric_keys:
            fig.add_trace(go.Bar(
                name=key.replace('_', ' ').title(),
                x=[f'{h}m' for h, metrics in metrics_all.items()],
                y=[metrics.get(key, 0) for metrics in metrics_all.values()]
            ))
        fig.update_layout(barmode='group', title='Logistic Regression Baseline Metrics', yaxis_range=[0,1])
        fig.write_html(CONFIG['results_dir'] / 'metrics.html')
        print(f"\n✅ Visualization saved to {CONFIG['results_dir'] / 'metrics.html'}")

if __name__ == '__main__':
    main() 