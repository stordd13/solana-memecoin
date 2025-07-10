"""
Medium-Term Directional Prediction with LightGBM for Memecoin Price Movement
This model loads pre-engineered features from the feature_engineering module.

IMPORTANT WORKFLOW:
1. Run feature_engineering/advanced_feature_engineering.py FIRST
2. Then run this script to train on pre-engineered features

IMPORTANT NOTES ON DATA LEAKAGE AND RESULTS:
- Fixed temporal data leakage by splitting each token temporally before feature engineering
- High accuracy (85-90%) is misleading due to severe class imbalance (10-15% UP labels)
- Model predicts DOWN most of the time, which is realistic for memecoin behavior
- ROC AUC (83-89%) is the more meaningful metric for this imbalanced problem
- Low recall (1-11%) shows model rarely predicts UP movements - conservative behavior
"""

import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from tqdm import tqdm
import joblib
import json
import sys
import numpy as np

# Add project root to path for ML imports
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ML.utils.metrics_helpers import financial_classification_metrics
# numpy import removed - using pure Polars for better performance

# --- Configuration ---
CONFIG = {
    'base_dir': Path("data/features_with_targets"),  # CHANGED: Read from features_with_targets dir
    'features_dir': Path("data/features_with_targets"),  # Pre-engineered features with targets directory
    'results_dir': Path("ML/results/lightgbm_medium_term"),
    'categories': [
        "normal_behavior_tokens",      # Highest quality for training
        "tokens_with_extremes",        # Valuable volatile patterns  
        "dead_tokens",                 # Complete token lifecycles
        # "tokens_with_gaps",          # EXCLUDED: Data quality issues
    ],
    'horizons': [240, 360, 480, 720, 960, 1380],  # 2h, 4h, 6h, 8h, 12h, 16h, 23h - medium-term trading
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
    },
    # Note: deduplicate_tokens removed - handled upstream in data_analysis
}


# --- NEW: Load Pre-Engineered Features ---
def load_features_from_file(features_path: Path) -> pl.DataFrame:
    """Load pre-engineered features from feature engineering output"""
    try:
        # Assume features are saved as parquet files by feature engineering module
        if features_path.exists():
            return pl.read_parquet(features_path)
        else:
            print(f"Features file not found: {features_path}")
            return None
    except Exception as e:
        print(f"Error loading features from {features_path}: {e}")
        return None


def validate_features_safety(features_df: pl.DataFrame, token_name: str) -> bool:
    """
    CRITICAL: Validate features for data leakage before training
    Returns True if features are safe, False otherwise
    
    IMPROVED: More intelligent detection that distinguishes between:
    - True data leakage (global features, future information)
    - Low variability (expected in dead tokens, not necessarily leakage)
    """
    if features_df is None or features_df.height == 0:
        return False
    
    # List of unsafe feature patterns that indicate data leakage
    unsafe_patterns = [
        'total_return', 'max_gain', 'max_drawdown', 'price_range',
        'global_', 'spectral_entropy', 'max_periodicity', 'dominant_period',
        'has_strong_cycles', 'cycle_interpretation', 'granularity'
    ]
    
    unsafe_features = []
    low_variability_features = []
    
    # Check each column for potential leakage
    for col in features_df.columns:
        col_lower = col.lower()
        
        # Check for unsafe patterns (TRUE data leakage)
        if any(pattern in col_lower for pattern in unsafe_patterns):
            unsafe_features.append(col)
            continue
            
        # Check for constant/low variability features
        if col not in ['datetime', 'price']:
            try:
                if features_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    unique_count = features_df[col].n_unique()
                    total_count = features_df.height
                    
                    # TRULY constant features (only 1 unique value) are suspicious
                    if unique_count == 1:
                        # But check if it's a legitimate zero/small value from dead token
                        unique_val = features_df[col].drop_nulls().unique().to_list()[0]
                        if abs(unique_val) < 1e-10:  # Very small value, likely from dead token
                            low_variability_features.append(f"{col}_near_zero")
                        else:
                            unsafe_features.append(f"{col}_constant")
                    
                    # Very low variability (< 0.1% unique values) might be suspicious
                    elif (unique_count / total_count) < 0.001:
                        # Check RELATIVE variability using pure Polars (better for crypto)
                        try:
                            stats = features_df.select([
                                pl.col(col).drop_nulls().min().alias('min_val'),
                                pl.col(col).drop_nulls().max().alias('max_val'),
                                pl.col(col).drop_nulls().mean().alias('mean_val'),
                                pl.col(col).drop_nulls().count().alias('count')
                            ])
                            
                            if stats['count'][0] > 0:
                                min_val = stats['min_val'][0]
                                max_val = stats['max_val'][0] 
                                mean_val = stats['mean_val'][0]
                                
                                # Calculate relative range (handles tiny prices correctly)
                                if abs(mean_val) > 1e-15:  # Avoid division by zero
                                    relative_range = (max_val - min_val) / abs(mean_val)
                                    if relative_range < 0.01:  # < 1% relative change is suspicious
                                        unsafe_features.append(f"{col}_no_relative_variability")
                                    else:
                                        # Low unique count but good relative range = OK for crypto
                                        low_variability_features.append(f"{col}_low_unique_count_ok")
                                else:
                                    # Values are truly zero/near-zero
                                    low_variability_features.append(f"{col}_near_zero")
                        except:
                            # Fallback: if stats calculation fails, treat as low variability
                            low_variability_features.append(f"{col}_stats_error")
            except:
                continue
    
    # Report findings
    if low_variability_features:
        print(f"\n⚠️  LOW VARIABILITY in {token_name} (expected for dead tokens):")
        for feature in low_variability_features[:3]:  # Show first 3
            print(f"   📊 {feature}")
        if len(low_variability_features) > 3:
            print(f"   ... and {len(low_variability_features) - 3} more")
    
    if unsafe_features:
        print(f"\n🚨 DATA LEAKAGE DETECTED in {token_name}:")
        for feature in unsafe_features:
            print(f"   ❌ {feature}")
        print(f"\n🛡️  Solution: Regenerate features with safe feature engineering")
        return False
    
    # Accept tokens with low variability (they're still valid for training)
    return True


# Labels are now created in pipeline - this function is no longer needed


# --- Data Preparation ---
def prepare_data_fixed(data_paths: List[Path], horizons: List[int], split_type: str = 'all') -> pl.DataFrame:
    """
    FIXED: Load pre-engineered features and apply temporal splitting
    Now reads from features/[category]/ structure instead of flat directory
    """
    all_samples = []
    processed_tokens = 0
    
    print(f"Loading pre-engineered features for {len(data_paths)} tokens ({split_type} split)...")
    
    for path in tqdm(data_paths, desc=f"Loading {split_type} features"):
        try:
            # The path is now pointing to features/[category]/[token].parquet
            # which is the actual feature file - no need to reconstruct the path
            token_name = path.stem
            
            # Since we're already reading from features dir, just load directly
            if path.exists() and path.suffix == '.parquet':
                features_df = load_features_from_file(path)
            else:
                print(f"Feature file not found: {path}")
                continue
            
            if features_df is None or len(features_df) == 0:
                continue
            
            # CRITICAL: Validate features for data leakage before using
            if not validate_features_safety(features_df, token_name):
                print(f"⚠️  SKIPPING {token_name} due to unsafe features")
                continue
            
            # Labels are already created in pipeline - no need to create them again
            
            # CRITICAL FIX: Use EXPANDING WINDOW splits to preserve historical context
            # Handle both DataFrame and LazyFrame
            if hasattr(features_df, 'height'):
                n_rows = features_df.height
            else:
                # If it's a LazyFrame, collect it first
                features_df = features_df.collect()
                n_rows = features_df.height
            
            # EXPANDING WINDOW APPROACH: Each split has access to ALL previous data
            if split_type == 'train':
                # Train on first 60% of token lifecycle
                start_idx = 0
                end_idx = int(n_rows * 0.6)
            elif split_type == 'val':
                # Validate on first 80% (includes all training data + validation period)
                start_idx = 0  # ← CRITICAL: Start from beginning!
                end_idx = int(n_rows * 0.8)
                # But only use predictions from the validation period for metrics
                prediction_start = int(n_rows * 0.6)
            elif split_type == 'test':
                # Test on entire token (includes all historical data)
                start_idx = 0  # ← CRITICAL: Start from beginning!
                end_idx = n_rows
                # But only use predictions from the test period for metrics
                prediction_start = int(n_rows * 0.8)
            else:  # 'all' for backwards compatibility
                start_idx = 0
                end_idx = n_rows
            
            # Only include if we have enough samples for this split
            # Medium-term needs more data due to longer horizons
            if end_idx - start_idx < 50:  # Need at least 50 samples for medium-term
                continue
                
            # Extract the data with full historical context
            token_split = features_df.slice(start_idx, end_idx - start_idx)
            
            # For val/test: mark which samples to use for evaluation
            if split_type in ['val', 'test']:
                # Add a column to mark evaluation samples
                eval_mask = [False] * (prediction_start - start_idx) + [True] * (end_idx - prediction_start)
                token_split = token_split.with_columns(pl.Series('eval_sample', eval_mask))
            
            # Add token identifier
            token_split = token_split.with_columns(pl.lit(path.stem).alias('token_id'))
            
            all_samples.append(token_split)
            processed_tokens += 1
                
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            continue
    
    print(f"Successfully processed {processed_tokens} tokens")
    
    if not all_samples:
        print("ERROR: No valid feature splits created!")
        return pl.DataFrame()
    
    # FIXED: Handle different feature sets by finding common columns
    if len(all_samples) > 1:
        # Find common columns across all samples
        common_cols = set(all_samples[0].columns)
        for sample_df in all_samples[1:]:
            common_cols = common_cols.intersection(set(sample_df.columns))
        
        # Convert to sorted list for consistent ordering
        common_cols = sorted(list(common_cols))
        
        print(f"Found {len(common_cols)} common columns across all tokens")
        
        # Select only common columns from each sample
        aligned_samples = []
        for sample_df in all_samples:
            aligned_samples.append(sample_df.select(common_cols))
        
        # Now concatenate the aligned samples
        result_df = pl.concat(aligned_samples)
    else:
        result_df = all_samples[0]
    
    # Drop any remaining nulls in labels
    label_cols = [f'label_{h}m' for h in horizons]
    existing_label_cols = [col for col in label_cols if col in result_df.columns]
    if existing_label_cols:
        result_df = result_df.drop_nulls(subset=existing_label_cols)
    
    print(f"{split_type.upper()} set: {result_df.height:,} samples from {len(all_samples)} tokens")
    
    return result_df


# --- Model Training and Evaluation ---
def train_and_evaluate(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, horizon: int, params: dict):
    """Trains a LightGBM model for a specific horizon and evaluates it."""
    label_col = f'label_{horizon}m'
    return_col = f'return_{horizon}m'
    
    # Select feature columns (exclude metadata, labels, and returns)
    exclude_cols = ['token_id', 'datetime', 'price'] + [f'label_{h}m' for h in CONFIG['horizons']] + [f'return_{h}m' for h in CONFIG['horizons']]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} pre-engineered features for {horizon}m prediction")

    # Convert to pandas/numpy for scikit-learn API
    print("📊 Preparing data...")
    X_train = train_df[feature_cols].to_pandas()
    y_train_raw = train_df[label_col].to_pandas()
    
    # For validation: use all data for training, but only evaluate on validation period
    X_val_full = val_df[feature_cols].to_pandas()
    y_val_full_raw = val_df[label_col].to_pandas()
    
    # For test: use all data for training, but only evaluate on test period  
    X_test_full = test_df[feature_cols].to_pandas()
    y_test_full_raw = test_df[label_col].to_pandas()
    
    # Handle NaN values BEFORE converting to int
    train_valid_mask = ~y_train_raw.isna()
    val_valid_mask = ~y_val_full_raw.isna()
    test_valid_mask = ~y_test_full_raw.isna()
    
    if train_valid_mask.sum() == 0:
        print(f"⚠️  ERROR: No valid labels for {horizon}m horizon in training data")
        dummy_metrics = {
            'validation': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5},
            'test': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5}
        }
        return None, dummy_metrics
    
    # Now safely convert to int after filtering NaN values
    y_train = y_train_raw[train_valid_mask].astype(int)
    y_val_full = y_val_full_raw[val_valid_mask].astype(int)
    y_test_full = y_test_full_raw[test_valid_mask].astype(int)
    
    # Also filter corresponding features
    X_train = X_train[train_valid_mask]
    X_val_full = X_val_full[val_valid_mask]
    X_test_full = X_test_full[test_valid_mask]
    
    # Extract returns data for financial metrics (also filter for valid labels)
    returns_val_full = val_df[return_col].to_pandas()[val_valid_mask]
    returns_test_full = test_df[return_col].to_pandas()[test_valid_mask]
    
    # Extract evaluation masks for val/test
    if 'eval_sample' in val_df.columns:
        val_eval_mask = val_df['eval_sample'].to_pandas().values[val_valid_mask]
        X_val = X_val_full[val_eval_mask]
        y_val = y_val_full[val_eval_mask]
        returns_val = returns_val_full[val_eval_mask]
    else:
        X_val = X_val_full
        y_val = y_val_full
        returns_val = returns_val_full
        
    if 'eval_sample' in test_df.columns:
        test_eval_mask = test_df['eval_sample'].to_pandas().values[test_valid_mask]
        X_test = X_test_full[test_eval_mask]
        y_test = y_test_full[test_eval_mask]
        returns_test = returns_test_full[test_eval_mask]
    else:
        X_test = X_test_full
        y_test = y_test_full
        returns_test = returns_test_full

    # Check if we have any samples after filtering
    if len(X_val) == 0:
        print(f"⚠️  WARNING: No validation samples for {horizon}m after eval_sample filtering")
        print(f"   Using full validation set instead")
        X_val = X_val_full
        y_val = y_val_full
        returns_val = returns_val_full
        
    if len(X_test) == 0:
        print(f"⚠️  WARNING: No test samples for {horizon}m after eval_sample filtering")
        print(f"   Using full test set instead")
        X_test = X_test_full
        y_test = y_test_full
        returns_test = returns_test_full

    # Show class distribution
    print(f"📈 Class distribution for {horizon}m horizon:")
    if len(y_train) > 0:
        print(f"   Train: UP={y_train.mean():.1%}, DOWN={1-y_train.mean():.1%}")
    else:
        print(f"   Train: No samples!")
        
    if len(y_val) > 0:
        print(f"   Val:   UP={y_val.mean():.1%}, DOWN={1-y_val.mean():.1%}")
    else:
        print(f"   Val:   No samples!")
        
    if len(y_test) > 0:
        print(f"   Test:  UP={y_test.mean():.1%}, DOWN={1-y_test.mean():.1%}")
    else:
        print(f"   Test:  No samples!")

    # Check if we have enough data to train after removing NaN values
    if len(X_train) < 10 or len(X_val) < 10 or len(X_test) < 10:
        print(f"⚠️  ERROR: Insufficient samples for {horizon}m horizon after NaN removal")
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"   Skipping this horizon...")
        
        # Return dummy metrics
        dummy_metrics = {
            'validation': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5},
            'test': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5}
        }
        return None, dummy_metrics
        
    # Check that we have binary labels
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        print(f"⚠️  ERROR: Labels not binary for {horizon}m horizon")
        print(f"   Train unique labels: {np.unique(y_train)}")
        print(f"   Val unique labels: {np.unique(y_val)}")
        print(f"   Test unique labels: {np.unique(y_test)}")
        print(f"   Skipping this horizon...")
        
        # Return dummy metrics
        dummy_metrics = {
            'validation': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5},
            'test': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5}
        }
        return None, dummy_metrics

    # Train model
    print(f"🌳 Training LightGBM on {len(X_train):,} samples...")
    print(f"📊 Validation set: {len(X_val):,} evaluation samples (with full historical context)")
    print(f"📊 Test set: {len(X_test):,} evaluation samples (with full historical context)")
    
    model = lgb.LGBMClassifier(**params)  # Remove duplicate verbose=-1
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],  # Use only evaluation samples for early stopping
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Evaluate on validation set with financial metrics
    print(f"🔮 Evaluating on validation set ({len(X_val):,} samples) with financial metrics...")
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    
    val_metrics = financial_classification_metrics(y_val, val_preds, returns_val, val_probs)

    # Evaluate on test set with financial metrics
    print(f"🔮 Evaluating on test set ({len(X_test):,} samples) with financial metrics...")
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    test_metrics = financial_classification_metrics(y_test, test_preds, returns_test, test_probs)
    
    # Create confusion matrices
    from sklearn.metrics import confusion_matrix
    
    val_cm = confusion_matrix(y_val, val_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    
    # Add confusion matrix info to metrics
    val_metrics['confusion_matrix'] = {
        'true_negative': int(val_cm[0, 0]),
        'false_positive': int(val_cm[0, 1]), 
        'false_negative': int(val_cm[1, 0]),
        'true_positive': int(val_cm[1, 1])
    }
    
    test_metrics['confusion_matrix'] = {
        'true_negative': int(test_cm[0, 0]),
        'false_positive': int(test_cm[0, 1]),
        'false_negative': int(test_cm[1, 0]), 
        'true_positive': int(test_cm[1, 1])
    }
    
    # Print confusion matrices for immediate feedback
    print(f"\n📊 Confusion Matrix - Validation (Horizon {horizon}m):")
    print(f"   Predicted:  DOWN  UP")
    print(f"   Actual DOWN: {val_cm[0,0]:4d} {val_cm[0,1]:3d}")
    print(f"   Actual UP:   {val_cm[1,0]:4d} {val_cm[1,1]:3d}")
    
    print(f"\n📊 Confusion Matrix - Test (Horizon {horizon}m):")
    print(f"   Predicted:  DOWN  UP") 
    print(f"   Actual DOWN: {test_cm[0,0]:4d} {test_cm[0,1]:3d}")
    print(f"   Actual UP:   {test_cm[1,0]:4d} {test_cm[1,1]:3d}")
    
    metrics = {
        'validation': val_metrics,
        'test': test_metrics
    }
    
    return model, metrics

def plot_metrics(metrics: Dict):
    """
    Plots key metrics for balanced classification (49% vs 51% is NOT imbalanced).
    FIXED: Handles missing metrics gracefully for failed horizons.
    """
    horizons = list(metrics.keys())
    
    # Filter out horizons that don't have valid metrics
    valid_horizons = []
    for h in horizons:
        if isinstance(metrics[h], dict) and 'accuracy' in metrics[h]:
            valid_horizons.append(h)
    
    if not valid_horizons:
        print("⚠️  No valid metrics to plot!")
        return go.Figure()
    
    print(f"📊 Plotting metrics for {len(valid_horizons)}/{len(horizons)} valid horizons: {valid_horizons}")
    
    # Show all meaningful metrics - accuracy IS important for balanced data
    key_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC']
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Professional color scheme
    
    for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
        # Only use horizons that have this metric
        metric_horizons = []
        metric_values = []
        metric_texts = []
        
        for h in valid_horizons:
            if metric in metrics[h]:
                metric_horizons.append(h)
                value = metrics[h][metric]
                metric_values.append(value)
                metric_texts.append(f"{value:.2f}")
        
        if metric_horizons:  # Only add trace if we have data
            fig.add_trace(go.Bar(
                name=label,
                x=metric_horizons,
                y=metric_values,
                text=metric_texts,
                textposition='auto',
                marker_color=colors[i]
            ))
    
    # Add baseline line at 50% for reference
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="50% Random Baseline"
    )
    
    fig.update_layout(
        barmode='group',
        title='Medium-Term LightGBM: Performance Metrics (Pre-Engineered Features)',
        xaxis_title='Prediction Horizon',
        yaxis_title='Score',
        yaxis_range=[0.0, 1.0],
        legend_title='Metric',
        template='plotly_white'
    )
    
    # Add annotation explaining the pre-engineered features
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="<b>Using Pre-Engineered Features:</b><br>• Enhanced technical indicators<br>• FFT analysis for medium-term cycles<br>• Advanced statistical moments<br>• Multi-granularity patterns (2m, 5m)",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        align="left"
    )
    
    return fig


# --- Main Pipeline ---
def main():
    """Main training pipeline for LightGBM medium-term directional model with walk-forward validation"""
    
    print("="*60)
    print("🔬 LightGBM Medium-Term Directional Prediction Training")
    print("Long horizons: 4h, 6h, 8h, 12h, 16h, 24h")
    print("WITH WALK-FORWARD VALIDATION")
    print("="*60)
    
    # Create results directory
    results_dir = Path("ML/results/lightgbm_medium_term")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data paths
    base_dir = Path("data/features_with_targets")
    categories = [
        "normal_behavior_tokens",
        "tokens_with_extremes", 
        "dead_tokens",
    ]
    
    all_paths = []
    for category in categories:
        cat_dir = base_dir / category
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.parquet"))
            all_paths.extend(paths)
            print(f"Found {len(paths)} files in {category}")
    
    if not all_paths:
        print("ERROR: No feature files found!")
        return  
    
    print(f"\nTotal files: {len(all_paths)}")
    
    # Import walk-forward splitter
    from ML.utils.walk_forward_splitter import WalkForwardSplitter
    
    print("\n🔄 Using Walk-Forward Validation for Medium-Term LightGBM")
    
    # Load all data first to prepare for walk-forward splitting
    print("\n📊 Loading all feature data for walk-forward validation...")
    
    all_data_frames = []
    for path in tqdm(all_paths, desc="Loading feature files"):
        try:
            df = pl.read_parquet(path)
            if len(df) < 400:  # Minimum token length
                continue
            
            # Add token identifier
            df = df.with_columns(pl.lit(path.stem).alias('token_id'))
            all_data_frames.append(df)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if not all_data_frames:
        print("ERROR: No valid feature files found!")
        return
    
    # Combine all data
    combined_data = pl.concat(all_data_frames)
    print(f"Loaded {len(combined_data):,} rows from {len(all_data_frames)} tokens")
    
    # Setup smart memecoin-aware walk-forward splitter
    splitter = WalkForwardSplitter()  # Auto-adapts to data length
    
    # For LightGBM, use smart splits that adapt to memecoin data characteristics
    print("\n🔀 Creating smart memecoin-aware global walk-forward splits...")
    global_splits, feasible_horizons = splitter.smart_split_for_memecoins(
        combined_data, 
        horizons=[240, 360, 480, 720, 960, 1380],  # 2h, 4h, 6h, 8h, 12h, 16h, 23h
        time_column='datetime'
    )
    
    print(f"Created {len(global_splits)} walk-forward folds")
    print(f"Original horizons: [240, 360, 480, 720, 960, 1380]")
    print(f"Feasible horizons: {feasible_horizons}")
    
    if len(feasible_horizons) < 7:
        skipped = [h for h in [120, 240, 360, 480, 720, 960, 1380] if h not in feasible_horizons]
        print(f"⚠️  Skipped horizons (too long for data): {skipped}")
    
    if not global_splits or not feasible_horizons:
        print("❌ No valid folds or feasible horizons!")
        return
    
    # Process each fold and collect metrics for feasible medium-term horizons
    all_fold_metrics = {h: [] for h in feasible_horizons}
    
    for fold_idx, (train_df, test_df) in enumerate(global_splits):
        print(f"\n📈 Processing fold {fold_idx + 1}/{len(global_splits)}")
        print(f"  Train: {len(train_df):,} samples, {train_df['token_id'].n_unique()} tokens")
        print(f"  Test: {len(test_df):,} samples, {test_df['token_id'].n_unique()} tokens")
        
        # Prepare data for this fold
        try:
            # Labels are already created in pipeline - no need to create them again
            
            # Split into 80% train, 20% validation from train_df
            train_tokens = train_df['token_id'].unique().to_list()
            np.random.shuffle(train_tokens)
            
            n_train_tokens = int(len(train_tokens) * 0.8)
            fold_train_tokens = train_tokens[:n_train_tokens]
            fold_val_tokens = train_tokens[n_train_tokens:]
            
            fold_train_df = train_df.filter(pl.col('token_id').is_in(fold_train_tokens))
            fold_val_df = train_df.filter(pl.col('token_id').is_in(fold_val_tokens))
            
            print(f"  Fold train: {len(fold_train_df):,} samples")
            print(f"  Fold val: {len(fold_val_df):,} samples")
            
            # Train and evaluate for each feasible horizon
            fold_metrics = {}
            for horizon in feasible_horizons:
                print(f"\n  Training horizon {horizon}min...")
                
                try:
                    # Use stronger regularization for medium-term prediction
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': 64,
                        'learning_rate': 0.03,  # Lower learning rate for stability
                        'feature_fraction': 0.7,  # More feature subsampling
                        'bagging_fraction': 0.7,  # More data subsampling
                        'bagging_freq': 5,
                        'min_child_samples': 30,  # Higher minimum samples
                        'reg_alpha': 0.1,  # L1 regularization
                        'reg_lambda': 0.1,  # L2 regularization
                        'random_state': 42,
                        'verbose': -1
                    }
                    
                    model, metrics = train_and_evaluate(fold_train_df, fold_val_df, test_df, horizon, params)
                    fold_metrics[f"{horizon}min"] = metrics
                    all_fold_metrics[horizon].append(metrics)
                    
                    if metrics and 'test' in metrics:
                        print(f"    Accuracy: {metrics['test'].get('accuracy', 0):.2%}")
                        print(f"    AUC: {metrics['test'].get('roc_auc', 0):.2%}")
                    else:
                        print(f"    No valid metrics for horizon {horizon}min")
                    
                except Exception as e:
                    print(f"    Error training horizon {horizon}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  Error processing fold {fold_idx}: {e}")
            continue
    
    # Aggregate metrics across all folds
    print("\n📊 Aggregating metrics across walk-forward folds...")
    
    final_metrics = {}
    for horizon in feasible_horizons:
        horizon_name = f"{horizon}min"
        fold_metrics_list = all_fold_metrics[horizon]
        
        if not fold_metrics_list:
            print(f"No valid metrics for horizon {horizon_name}")
            continue
            
        # Average metrics across folds (handle nested structure)
        aggregated = {}
        
        # Initialize structure for validation and test metrics
        for split_type in ['validation', 'test']:
            if split_type in fold_metrics_list[0]:
                split_metrics = {}
                first_split = fold_metrics_list[0][split_type]
                
                for metric_name in first_split.keys():
                    values = [m[split_type].get(metric_name, 0) for m in fold_metrics_list 
                             if split_type in m and metric_name in m[split_type]]
                    
                    if not values:
                        continue
                    
                    # Handle scalar metrics vs dictionary metrics
                    if metric_name == 'confusion_matrix':
                        # For confusion matrices, aggregate each component
                        if isinstance(values[0], dict):
                            avg_cm = {}
                            for key in values[0].keys():
                                cm_values = [v.get(key, 0) for v in values if isinstance(v, dict)]
                                if cm_values:
                                    avg_cm[key] = np.mean(cm_values)
                                    avg_cm[f"{key}_std"] = np.std(cm_values)
                            split_metrics[metric_name] = avg_cm
                        else:
                            split_metrics[metric_name] = values[0] if len(values) == 1 else 0
                    else:
                        # For scalar metrics, compute mean and std
                        try:
                            # Filter out non-numeric values
                            numeric_values = []
                            for v in values:
                                if isinstance(v, (int, float, np.number)) and not np.isnan(float(v)):
                                    numeric_values.append(float(v))
                            
                            if numeric_values:
                                split_metrics[metric_name] = np.mean(numeric_values)
                                split_metrics[f"{metric_name}_std"] = np.std(numeric_values)
                            else:
                                split_metrics[metric_name] = 0.0
                                split_metrics[f"{metric_name}_std"] = 0.0
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not aggregate metric {metric_name}: {e}")
                            split_metrics[metric_name] = 0.0
                            split_metrics[f"{metric_name}_std"] = 0.0
                
                aggregated[split_type] = split_metrics
        
        aggregated['num_folds'] = len(fold_metrics_list)
        final_metrics[horizon_name] = aggregated
    
    # Print final results
    print("\n" + "="*60)
    print("MEDIUM-TERM WALK-FORWARD VALIDATION RESULTS")
    print("="*60)
    
    for horizon, metrics in final_metrics.items():
        horizon_hours = int(horizon.replace('min', '')) / 60
        print(f"\nHorizon {horizon} ({horizon_hours:.1f}h) - {metrics.get('num_folds', 0)} folds:")
        
        if 'test' in metrics:
            test_metrics = metrics['test']
            print(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.2%} ± {test_metrics.get('accuracy_std', 0):.2%}")
            print(f"  Test Precision: {test_metrics.get('precision', 0):.2%} ± {test_metrics.get('precision_std', 0):.2%}")
            print(f"  Test Recall: {test_metrics.get('recall', 0):.2%} ± {test_metrics.get('recall_std', 0):.2%}")
            print(f"  Test F1 Score: {test_metrics.get('f1_score', 0):.2%} ± {test_metrics.get('f1_std', 0):.2%}")
            print(f"  Test ROC AUC: {test_metrics.get('roc_auc', 0):.2%} ± {test_metrics.get('roc_auc_std', 0):.2%}")
        
        if 'validation' in metrics:
            val_metrics = metrics['validation']
            print(f"  Val Accuracy: {val_metrics.get('accuracy', 0):.2%} ± {val_metrics.get('accuracy_std', 0):.2%}")
            print(f"  Val ROC AUC: {val_metrics.get('roc_auc', 0):.2%} ± {val_metrics.get('roc_auc_std', 0):.2%}")
    
    # Save results
    print(f"\n💾 Saving results to {results_dir}")
    
    # Save metrics JSON
    metrics_path = results_dir / 'metrics_walkforward.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Create and save plots
    if final_metrics:
        # Performance plot
        metrics_fig = plot_metrics(final_metrics)
        metrics_fig.update_layout(title="LightGBM Medium-Term Performance (Walk-Forward Validation)")
        metrics_plot_path = results_dir / 'performance_metrics_walkforward.html'
        metrics_fig.write_html(metrics_plot_path)
        
        print(f"  📊 Performance plot: {metrics_plot_path}")
        print(f"  📋 Metrics JSON: {metrics_path}")
    
    print("\n✅ LightGBM medium-term walk-forward validation training complete!")
    print(f"   More realistic metrics due to temporal validation")
    print(f"   Results averaged across {len(global_splits)} walk-forward folds")
    print(f"   Focused on longer horizons (2-24 hours) for extended predictions")
    
    return final_metrics


if __name__ == "__main__":
    main() 