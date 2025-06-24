"""
Directional Prediction with LightGBM for Memecoin Price Movement
This model uses engineered features to predict UP/DOWN movement.
"""

import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from tqdm import tqdm
import joblib
import json
import numpy as np

# Use local smart_data_split function for consistency

# --- Configuration ---
CONFIG = {
    'base_dir': Path("data/cleaned"),
    'results_dir': Path("ML/results/lightgbm_medium_term"),
    'categories': [
        "normal_behavior_tokens",
        "tokens_with_gaps",
        "tokens_with_extremes",
    ],
    'lookback': 240,  # 4 hour lookback for medium-term patterns
    'horizons': [120, 240, 360, 720],  # 2h, 4h, 6h, 12h - medium-term trading
    'n_estimators': 500,
    'random_state': 42,
    'test_size': 0.2,
    'val_size': 0.2,
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'seed': 42,
    }
}


# --- Feature Engineering with Polars ---
def create_features(df: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """
    Engineers a rich set of features from raw price data using Polars.
    """
    if 'datetime' not in df.columns:
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        )

    df = df.sort('datetime')
    
    # Handle NaN values in price column by forward filling
    prices = df['price'].to_numpy()
    
    # Check for NaN values - if found, drop this token entirely for cleaner data
    if np.isnan(prices).any():
        nan_count = np.isnan(prices).sum()
        print(f"Dropping token with {nan_count} NaN values for cleaner baseline")
        return None
    
    # Update the dataframe with filled prices
    df = df.with_columns(pl.Series(prices).alias('price'))
    
    # --- Time-based Features ---
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.weekday().alias('weekday'),
    ])

    # --- Lag Features ---
    lags = [1, 2, 3, 5, 10, 15, 30, 60]
    df = df.with_columns(
        [pl.col('price').shift(i).alias(f'price_lag_{i}') for i in lags]
    )

    # --- Rolling Window Features ---
    windows = [5, 15, 30, 60]
    df = df.with_columns(
        [pl.col('price').rolling_mean(w).alias(f'price_rolling_mean_{w}') for w in windows] +
        [pl.col('price').rolling_std(w).alias(f'price_rolling_std_{w}') for w in windows]
    )

    # --- Momentum Indicators (RSI) ---
    delta = df.get_column('price').diff()
    gain = delta.clip(lower_bound=0).fill_null(0)
    loss = (-delta).clip(lower_bound=0).fill_null(0)
    
    avg_gain = gain.ewm_mean(span=14, adjust=False)
    avg_loss = loss.ewm_mean(span=14, adjust=False)
    
    rs = avg_gain / avg_loss.replace(0, 1e-8) # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df = df.with_columns(rsi.alias('rsi_14'))
    
    # Price compared to rolling means
    df = df.with_columns(
        [(pl.col('price') / pl.col(f'price_rolling_mean_{w}') - 1).alias(f'price_pct_from_mean_{w}') for w in windows]
    )
    
    # Create labels for each horizon BEFORE dropping nulls
    for h in CONFIG['horizons']:
        df = df.with_columns(
            (pl.col('price').shift(-h) > pl.col('price')).cast(pl.Int8).alias(f'label_{h}m')
        )
    
    return df.drop_nulls()


# --- Data Preparation ---
def prepare_data(data_paths: List[Path], lookback: int, horizons: List[int]) -> pl.DataFrame:
    """Loads all files, engineers features, and creates labels."""
    all_featured_dfs = []
    
    max_horizon = max(horizons)
    
    print(f"Processing data with lookback={lookback}, horizons={horizons}")
    
    # Calculate minimum required data points (reduced from previous)
    min_required = lookback + max_horizon + 10  # Added small buffer
    print(f"Minimum required data points per token: {min_required}")
    
    # Filter valid files first
    valid_files = []
    for file in data_paths:
        try:
            data = pl.read_parquet(file)
            if len(data) >= min_required:
                valid_files.append(file)
            else:
                print(f"Skipping {file.name}: only {len(data)} rows (need {min_required})")
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
    
    print(f"Found {len(valid_files)} valid files out of {len(data_paths)} total")
    
    if not valid_files:
        print("No valid files found for processing!")
        return None
    
    print(f"Processing {len(valid_files)} files for feature engineering...")
    
    processed_count = 0
    for path in tqdm(valid_files, desc="Processing files"):
        try:
            df = pl.read_parquet(path)
            
            # Engineer features for this file
            featured_df = create_features(df, lookback)
            
            if featured_df is None or featured_df.height == 0:
                continue
            
            all_featured_dfs.append(featured_df)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            continue
    
    print(f"Successfully processed {processed_count} files")
    
    if not all_featured_dfs:
        print("WARNING: No files could be processed!")
        return pl.DataFrame()

    # Concatenate all dataframes and drop rows with null labels
    full_df = pl.concat(all_featured_dfs)
    label_cols = [f'label_{h}m' for h in horizons]
    result_df = full_df.drop_nulls(subset=label_cols)
    
    print(f"Final dataset: {result_df.height:,} samples with {result_df.width} features")
    
    return result_df


# --- Model Training and Evaluation ---
def train_and_evaluate(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, horizon: int, params: dict):
    """Trains a LightGBM model for a specific horizon and evaluates it."""
    label_col = f'label_{horizon}m'
    feature_cols = [col for col in train_df.columns if col.startswith(('price_', 'hour', 'weekday', 'rsi'))]

    # Convert to pandas/numpy for scikit-learn API
    X_train = train_df[feature_cols].to_pandas()
    y_train = train_df[label_col].to_pandas()
    X_val = val_df[feature_cols].to_pandas()
    y_val = val_df[label_col].to_pandas()
    X_test = test_df[feature_cols].to_pandas()
    y_test = test_df[label_col].to_pandas()

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Evaluate on the unseen test set
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1_score': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs)
    }
    
    return model, metrics

def plot_metrics(metrics: Dict):
    """Plots metrics for all horizons."""
    horizons = list(metrics.keys())
    metric_names = list(metrics[horizons[0]].keys())
    
    fig = go.Figure()
    for name in metric_names:
        fig.add_trace(go.Bar(
            name=name.replace('_', ' ').title(),
            x=horizons,
            y=[metrics[h][name] for h in horizons],
            text=[f"{metrics[h][name]:.2f}" for h in horizons],
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title='Medium-Term LightGBM Directional Model Performance by Horizon',
        xaxis_title='Prediction Horizon',
        yaxis_title='Score',
        yaxis_range=[0,1],
        legend_title='Metric'
    )
    return fig


# --- Main Pipeline ---
def main():
    """Main training pipeline for the medium-term LightGBM model."""
    print("="*50)
    print("Training Medium-Term LightGBM Directional Model")
    print("Horizons: 2h, 4h, 6h, 12h")
    print("="*50)
    
    # Create results directory
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {CONFIG['results_dir']}")
    
    # Load paths using the same stratified split logic
    all_paths = []
    for category in CONFIG['categories']:
        cat_dir = CONFIG['base_dir'] / category
        if cat_dir.exists():
            all_paths.extend(list(cat_dir.glob("*.parquet")))
    
    if not all_paths:
        print(f"ERROR: No files found in {CONFIG['base_dir']}. Exiting.")
        return

    train_paths, val_paths, test_paths = smart_data_split(all_paths)

    # Prepare data for each split separately to prevent leakage
    print("\nPreparing training data...")
    train_df = prepare_data(train_paths, CONFIG['lookback'], CONFIG['horizons'])
    print(f"Training set size: {train_df.height:,} samples")

    print("\nPreparing validation data...")
    val_df = prepare_data(val_paths, CONFIG['lookback'], CONFIG['horizons'])
    print(f"Validation set size: {val_df.height:,} samples")

    print("\nPreparing test data...")
    test_df = prepare_data(test_paths, CONFIG['lookback'], CONFIG['horizons'])
    print(f"Test set size: {test_df.height:,} samples")

    if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
        print("ERROR: One or more data splits are empty. Exiting.")
        return

    # Train a model for each horizon
    all_metrics = {}
    models = {}
    
    for h in CONFIG['horizons']:
        print(f"\n--- Training for {h}m horizon ---")
        model, metrics = train_and_evaluate(train_df, val_df, test_df, h, CONFIG['lgb_params'])
        all_metrics[f'{h}m'] = metrics
        models[f'{h}m'] = model
        
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  ROC AUC: {metrics['roc_auc']:.2%}")

    # Save models
    for horizon, model in models.items():
        model_path = CONFIG['results_dir'] / f'lightgbm_model_{horizon}.joblib'
        joblib.dump(model, model_path)
    print(f"\nModels saved to: {CONFIG['results_dir']}/lightgbm_model_*.joblib")
    
    # Save metrics
    metrics_path = CONFIG['results_dir'] / 'lightgbm_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # Create and save plot
    fig = plot_metrics(all_metrics)
    plot_path = CONFIG['results_dir'] / 'lightgbm_medium_term_metrics.html'
    fig.write_html(plot_path)
    print(f"Metrics plot saved to: {plot_path}")

def smart_data_split(data_paths: List[Path], 
                    test_size: float = 0.2, 
                    val_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split data ensuring no leakage between train/val/test sets"""
    total_files = len(data_paths)
    
    np.random.seed(random_state)
    np.random.shuffle(data_paths)
    
    test_count = int(total_files * test_size)
    val_count = int(total_files * val_size)
    train_count = total_files - test_count - val_count
    
    train_paths = data_paths[:train_count]
    val_paths = data_paths[train_count:train_count + val_count]
    test_paths = data_paths[train_count + val_count:]
    
    print(f"Data split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    return train_paths, val_paths, test_paths

def forward_fill_nan(prices: np.ndarray) -> np.ndarray:
    """Forward fill NaN values in price array"""
    mask = np.isnan(prices)
    if not mask.any():
        return prices
        
    prices = prices.copy()
    indices = np.where(~mask)[0]
    
    if len(indices) == 0:
        return prices  # All NaN, can't fill
    
    for i in range(len(prices)):
        if mask[i]:
            # Find the last valid value before this point
            prev_valid = indices[indices < i]
            if len(prev_valid) > 0:
                prices[i] = prices[prev_valid[-1]]
            else:
                # If no previous valid value, use next valid value
                next_valid = indices[indices > i]
                if len(next_valid) > 0:
                    prices[i] = prices[next_valid[0]]
    
    return prices

if __name__ == "__main__":
    main() 