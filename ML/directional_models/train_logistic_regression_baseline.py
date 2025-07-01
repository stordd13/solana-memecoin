import sys
from pathlib import Path
import json
import joblib
import polars as pl
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import plotly.graph_objects as go
from typing import List

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ML.utils.metrics_helpers import financial_classification_metrics
from ML.utils.walk_forward_splitter import WalkForwardSplitter
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
    'horizons': [15, 30, 60, 120, 240, 360],
    'random_state': 42,
    'min_rows_per_token': 100,
}

# --- Label Creation ---
def create_labels_for_horizons(df: pl.DataFrame, horizons: List[int]) -> pl.DataFrame:
    """Creates labels and returns for multiple horizons."""
    df = df.sort('datetime')
    for h in horizons:
        df = df.with_columns([
            (pl.col('price').shift(-h) > pl.col('price')).alias(f'label_{h}m'),
            ((pl.col('price').shift(-h) - pl.col('price')) / pl.col('price')).alias(f'return_{h}m')
        ])
    return df

# --- Main Training Pipeline ---
def main():
    """Main training pipeline with walk-forward validation."""
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔬 Training Logistic Regression with WALK-FORWARD VALIDATION")
    print(f"Horizons: {CONFIG['horizons']}")
    print("="*60)

    # 1. Load all data
    print("Loading all feature files...")
    all_paths = []
    for category in CONFIG['categories']:
        cat_dir = CONFIG['features_dir'] / category
        if cat_dir.exists():
            all_paths.extend(list(cat_dir.glob("*.parquet")))
    
    all_dfs = []
    for path in tqdm(all_paths, desc="Loading data"):
        df = pl.read_parquet(path)
        if df.height > CONFIG['min_rows_per_token']:
            df = df.with_columns(pl.lit(path.stem).alias("token_id"))
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data found, exiting.")
        return
        
    data = pl.concat(all_dfs)
    data = create_labels_for_horizons(data, CONFIG['horizons'])
    
    # 2. Walk-Forward Split
    print("Creating global walk-forward splits...")
    splitter = WalkForwardSplitter(config='lightgbm_short_term')
    folds = splitter.get_global_splits(data, time_column='datetime')
    
    print(f"Created {len(folds)} walk-forward folds.")
    
    # 3. Training and Evaluation Loop
    all_fold_metrics = {h: [] for h in CONFIG['horizons']}
    
    # Fit scaler on first train fold
    first_train_df, _ = folds[0]
    feature_cols = [c for c in first_train_df.columns if c not in ['datetime', 'price', 'token_id'] and not c.startswith('label_') and not c.startswith('return_')]
    
    print("Fitting Winsorizer on the first training fold...")
    winsorizer = Winsorizer(lower_percentile=0.005, upper_percentile=0.995)
    X_fit = first_train_df[feature_cols].to_numpy()
    winsorizer.fit(X_fit)

    models = {}

    for i, (train_df, test_df) in enumerate(tqdm(folds, desc="Processing Folds")):
        fold_metrics = {}
        for h in CONFIG['horizons']:
            label_col = f'label_{h}m'
            return_col = f'return_{h}m'
            
            train_fold = train_df.drop_nulls([label_col, return_col])
            test_fold = test_df.drop_nulls([label_col, return_col])
            
            X_train = winsorizer.transform(train_fold[feature_cols].to_numpy())
            y_train = train_fold[label_col].to_numpy()
            
            X_test = winsorizer.transform(test_fold[feature_cols].to_numpy())
            y_test = test_fold[label_col].to_numpy()
            returns_test = test_fold[return_col].to_numpy()

            if len(np.unique(y_train)) < 2:
                continue

            model = LogisticRegression(
                max_iter=500,
                n_jobs=-1,
                random_state=CONFIG['random_state'],
                class_weight='balanced',
                solver='liblinear',
                penalty='l1',
                C=0.1,
                tol=1e-4
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = financial_classification_metrics(y_test, y_pred, returns_test, y_prob)
            all_fold_metrics[h].append(metrics)
            models[h] = model

    # 4. Aggregate and Save Results
    print("\n--- Aggregated Walk-Forward Results ---")
    final_metrics = {}
    for h, h_metrics in all_fold_metrics.items():
        if not h_metrics: continue
        avg_metrics = {key: np.mean([m[key] for m in h_metrics]) for key in h_metrics[0]}
        std_metrics = {key: np.std([m[key] for m in h_metrics]) for key in h_metrics[0]}
        final_metrics[f'{h}m'] = {
            'mean': avg_metrics,
            'std': std_metrics
        }
        print(f"\nHorizon {h}m:")
        for key, val in avg_metrics.items():
            print(f"  {key}: {val:.4f} (+/- {std_metrics[key]:.4f})")

    # Save artifacts
    results_path = CONFIG['results_dir']
    with open(results_path / 'metrics_walkforward.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    joblib.dump(winsorizer, results_path / 'winsorizer_walkforward.joblib')
    joblib.dump(models, results_path / 'logreg_models_walkforward.joblib')

    print(f"\n✅ Walk-forward training complete!")
    print(f"📁 Results saved to: {results_path}")

if __name__ == '__main__':
    main() 