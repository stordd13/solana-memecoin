import argparse
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH **before** importing `ML.*` so
# that running the script directly (e.g. `python ML/…/train_*.py`) works
# without an editable install.
# ------------------------------------------------------------------
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from ML.utils.metrics_helpers import regression_metrics
import joblib, json, plotly.graph_objects as go

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

CONFIG = {
    'features_dir': Path('data/features'),
    'results_dir': Path('ML/results/baseline_regressors'),
    'categories': [
        'normal_behavior_tokens',
        'tokens_with_extremes',
        'dead_tokens'
    ]
}


def build_dataset(horizon: int):
    """Return train/val/test numpy arrays and per-token scalers for linear model."""
    splits = {'train': {'X': [], 'y': []}, 'val': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}}
    token_scalers = {}

    paths = []
    for cat in CONFIG['categories']:
        paths += list((CONFIG['features_dir'] / cat).glob('*.parquet'))

    for path in tqdm(paths, desc='Building dataset'):
        df = pl.read_parquet(path)
        n_rows = df.height
        if n_rows < horizon + 60:
            continue

        # Compute regression target: future price (absolute)
        df = df.with_columns([
            (pl.col('price').shift(-horizon)).alias('target_price')
        ]).drop_nulls(subset=['target_price'])

        # Split indices
        train_idx = int(0.6 * len(df))
        val_idx = int(0.8 * len(df))

        feature_cols = [c for c in df.columns if c not in ['datetime', 'price', 'target_price'] and not c.startswith('label_')]
        X_full = df[feature_cols].to_numpy().copy()  # Make writable copy
        y_full = df['target_price'].to_numpy().copy()  # Make writable copy
        
        # Replace inf/-inf with NaN for easier handling
        X_full[~np.isfinite(X_full)] = np.nan
        y_full[~np.isfinite(y_full)] = np.nan
        
        # Create mask for rows with all finite values
        valid_mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_full)
        
        # Filter to valid samples only
        X_full = X_full[valid_mask]
        y_full = y_full[valid_mask]
        
        if len(X_full) == 0:
            print(f"Warning: No valid samples for {path.stem} after filtering non-finite values")
            continue

        # Fit per-token scaler on train split
        scaler = RobustScaler()
        scaler.fit(X_full[:train_idx])
        token_scalers[path.stem] = scaler

        X_scaled = scaler.transform(X_full)

        # Append to global splits
        for split, sl in [('train', slice(0, train_idx)),
                          ('val', slice(train_idx, val_idx)),
                          ('test', slice(val_idx, len(df)))]:
            splits[split]['X'].append(X_scaled[sl])
            splits[split]['y'].append(y_full[sl])

    # Concatenate per split
    for split in splits:
        splits[split]['X'] = np.vstack(splits[split]['X']) if splits[split]['X'] else np.empty((0,))
        splits[split]['y'] = np.concatenate(splits[split]['y']) if splits[split]['y'] else np.empty((0,))
    return splits, token_scalers


def train_and_evaluate(horizon: int, model_type: str):
    splits, token_scalers = build_dataset(horizon)
    X_train, y_train = splits['train']['X'], splits['train']['y']
    X_test, y_test = splits['test']['X'], splits['test']['y']

    if X_train.size == 0:
        print('No training samples found.')
        return None

    if model_type == 'linear':
        model = LinearRegression(n_jobs=-1)
        scaled_inputs = True  # already scaled
    else:  # xgb
        if not HAS_XGB:
            print('xgboost not installed, skipping')
            return None
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             objective='reg:squarederror', random_state=42, n_jobs=-1)
        scaled_inputs = False  # scaling not critical but we already used scaled data; still fine

    print(f"Training {model_type.upper()} model on {len(X_train):,} samples for {horizon}m forecast...")
    
    # Training with progress indication
    with tqdm(total=1, desc=f"Training {model_type.upper()} model") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    
    # Prediction with progress indication
    print(f"Making predictions on {len(X_test):,} test samples...")
    with tqdm(total=1, desc=f"Predicting with {model_type.upper()}") as pbar:
        y_pred = model.predict(X_test)
        pbar.update(1)
    
    # Evaluation with progress indication
    with tqdm(total=1, desc="Calculating metrics") as pbar:
        metrics = regression_metrics(y_test, y_pred)
        pbar.update(1)

    # Save artefacts
    out_dir = CONFIG['results_dir'] / f'{model_type}_{horizon}m'
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / 'model.joblib')
    if model_type == 'linear':
        joblib.dump(token_scalers, out_dir / 'token_scalers.joblib')
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot metrics
    fig = go.Figure([go.Bar(x=list(metrics.keys()), y=list(metrics.values()))])
    fig.update_layout(title=f'{model_type.upper()} Baseline – {horizon}m Forecast')
    fig.write_html(out_dir / 'metrics.html')
    print(f"Saved outputs to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=15, help='Forecast horizon in minutes')
    parser.add_argument('--model', type=str, choices=['linear', 'xgb', 'both'], default='both')
    args = parser.parse_args()

    if args.model in ['linear', 'both']:
        train_and_evaluate(args.horizon, 'linear')
    if args.model in ['xgb', 'both']:
        train_and_evaluate(args.horizon, 'xgb') 