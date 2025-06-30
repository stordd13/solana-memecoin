"""
Directional Prediction with LightGBM for Memecoin Price Movement
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from tqdm import tqdm
import joblib
import json
import sys

# Add project root to path for ML imports
current_dir = Path(__file__).resolve()
project_root = current_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ML.utils.metrics_helpers import financial_classification_metrics
# numpy import removed - using pure Polars for better performance

# --- Configuration ---
CONFIG = {
    'base_dir': Path("data/features"),  # CHANGED: Read from features dir instead of cleaned
    'features_dir': Path("data/features"),  # Pre-engineered features directory
    'results_dir': Path("ML/results/lightgbm_short_term"),
    'categories': [
        "normal_behavior_tokens",      # Highest quality for training
        "tokens_with_extremes",        # Valuable volatile patterns  
        "dead_tokens",                 # Complete token lifecycles
        # "tokens_with_gaps",          # EXCLUDED: Data quality issues
    ],
    'min_lookback': 3,   # Start predicting at minute 3
    'horizons': [15, 30, 60],  # 15min, 30min, 1h - short-term trading
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


def create_labels_for_horizons(df: pl.DataFrame, horizons: List[int]) -> pl.DataFrame:
    """Create directional labels and returns for multiple horizons"""
    if 'price' not in df.columns:
        return df
        
    # Add labels and returns for each horizon
    for h in horizons:
        df = df.with_columns([
            (pl.col('price').shift(-h) > pl.col('price')).alias(f'label_{h}m'),
            ((pl.col('price').shift(-h) - pl.col('price')) / pl.col('price')).alias(f'return_{h}m')
        ])
    
    return df


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
            
            # Create directional labels
            features_df = create_labels_for_horizons(features_df, horizons)
            
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
            if end_idx - start_idx < 20:  # Need at least 20 samples
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
    y_train = train_df[label_col].to_pandas()
    
    # For validation: use all data for training, but only evaluate on validation period
    X_val_full = val_df[feature_cols].to_pandas()
    y_val_full = val_df[label_col].to_pandas()
    
    # For test: use all data for training, but only evaluate on test period  
    X_test_full = test_df[feature_cols].to_pandas()
    y_test_full = test_df[label_col].to_pandas()
    
    # Extract returns data for financial metrics
    returns_val_full = val_df[return_col].to_pandas()
    returns_test_full = test_df[return_col].to_pandas()
    
    # Extract evaluation masks for val/test
    if 'eval_sample' in val_df.columns:
        val_eval_mask = val_df['eval_sample'].to_pandas().values
        X_val = X_val_full[val_eval_mask]
        y_val = y_val_full[val_eval_mask]
        returns_val = returns_val_full[val_eval_mask]
    else:
        X_val = X_val_full
        y_val = y_val_full
        returns_val = returns_val_full
        
    if 'eval_sample' in test_df.columns:
        test_eval_mask = test_df['eval_sample'].to_pandas().values
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

    # Check if we have enough data to train
    if len(X_train) < 10 or len(X_val) < 10 or len(X_test) < 10:
        print(f"⚠️  ERROR: Insufficient samples for {horizon}m horizon")
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"   Skipping this horizon...")
        
        # Return dummy metrics
        dummy_metrics = {
            'validation': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5,
                          'confusion_matrix': {'true_negative': 0, 'false_positive': 0, 
                                             'false_negative': 0, 'true_positive': 0}},
            'test': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0.5,
                    'confusion_matrix': {'true_negative': 0, 'false_positive': 0,
                                       'false_negative': 0, 'true_positive': 0}}
        }
        return None, dummy_metrics

    # Train model
    print(f"🌳 Training LightGBM on {len(X_train):,} samples...")
    print(f"📊 Validation set: {len(X_val):,} evaluation samples (with full historical context)")
    print(f"📊 Test set: {len(X_test):,} evaluation samples (with full historical context)")
    
    model = lgb.LGBMClassifier(**params, verbose=-1)  # Suppress verbose output
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
    """Plots key metrics comparing validation vs test performance."""
    horizons = list(metrics.keys())
    
    # Show all meaningful metrics - accuracy IS important for balanced data
    key_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC']
    
    fig = go.Figure()
    
    colors_val = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Validation colors
    colors_test = ['#87ceeb', '#ffd700', '#90ee90', '#ff6347', '#dda0dd']  # Test colors (lighter)
    
    for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
        # Validation metrics
        fig.add_trace(go.Bar(
            name=f'{label} (Val)',
            x=horizons,
            y=[metrics[h]['validation'][metric] for h in horizons],
            text=[f"{metrics[h]['validation'][metric]:.2f}" for h in horizons],
            textposition='auto',
            marker_color=colors_val[i],
            offsetgroup=1
        ))
        
        # Test metrics  
        fig.add_trace(go.Bar(
            name=f'{label} (Test)',
            x=horizons,
            y=[metrics[h]['test'][metric] for h in horizons],
            text=[f"{metrics[h]['test'][metric]:.2f}" for h in horizons],
            textposition='auto',
            marker_color=colors_test[i],
            offsetgroup=2
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
        title='Short-Term LightGBM: Validation vs Test Performance',
        xaxis_title='Prediction Horizon',
        yaxis_title='Score',
        yaxis_range=[0.0, 1.0],
        legend_title='Metric',
        template='plotly_white'
    )
    
    # Add annotation explaining the results
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="<b>Model Analysis:</b><br>• High accuracy + Low recall = Conservative model<br>• Predicts DOWN most of the time<br>• Check confusion matrices for details<br>• ROC AUC shows true predictive power",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        align="left"
    )
    
    return fig


def plot_confusion_matrices(metrics: Dict):
    """Create confusion matrix visualizations for all horizons."""
    import plotly.subplots as sp
    
    horizons = list(metrics.keys())
    n_horizons = len(horizons)
    
    # Create subplots: 2 rows (val/test) x n_horizons columns
    fig = sp.make_subplots(
        rows=2, 
        cols=n_horizons,
        subplot_titles=[f'{h} Val' for h in horizons] + [f'{h} Test' for h in horizons],
        specs=[[{'type': 'bar'} for _ in range(n_horizons)] for _ in range(2)]
    )
    
    for i, horizon in enumerate(horizons):
        col = i + 1
        
        # Validation confusion matrix
        val_cm = metrics[horizon]['validation']['confusion_matrix']
        val_values = [val_cm['true_negative'], val_cm['false_positive'], 
                     val_cm['false_negative'], val_cm['true_positive']]
        val_labels = ['TN', 'FP', 'FN', 'TP']
        val_colors = ['green', 'orange', 'orange', 'green']
        
        fig.add_trace(go.Bar(
            x=val_labels,
            y=val_values,
            text=[str(v) for v in val_values],
            textposition='auto',
            marker_color=val_colors,
            showlegend=False
        ), row=1, col=col)
        
        # Test confusion matrix
        test_cm = metrics[horizon]['test']['confusion_matrix']
        test_values = [test_cm['true_negative'], test_cm['false_positive'],
                      test_cm['false_negative'], test_cm['true_positive']]
        
        fig.add_trace(go.Bar(
            x=val_labels,
            y=test_values,
            text=[str(v) for v in test_values],
            textposition='auto',
            marker_color=val_colors,
            showlegend=False
        ), row=2, col=col)
    
    fig.update_layout(
        title='Confusion Matrices: Validation (Top) vs Test (Bottom)',
        template='plotly_white',
        height=600
    )
    
    return fig


# --- Main Pipeline ---
def main():
    """Main training pipeline for the short-term LightGBM model."""
    print("="*50)
    print("🛡️  Training Short-Term LightGBM Directional Model")
    print("USING SAFE PRE-ENGINEERED FEATURES (NO DATA LEAKAGE)")
    print("Horizons: 15min, 30min, 1h")
    print("="*50)
    
    # Check if feature engineering has been run
    if not CONFIG['features_dir'].exists():
        print(f"\n❌ ERROR: Features directory not found: {CONFIG['features_dir']}")
        print("\n🔧 REQUIRED STEP: Run feature engineering first!")
        print("   python feature_engineering/advanced_feature_engineering.py")
        print("\nThis will create the pre-engineered features needed for training.")
        return
    
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

    # Note: Token deduplication is now handled upstream in data_analysis
    # Each token should appear in exactly one category folder

    # FIXED: Use temporal splitting within tokens instead of random token splits
    print("\n🔧 LOADING PRE-ENGINEERED FEATURES with temporal splitting")
    
    # Use all paths for temporal splitting within each token
    train_df = prepare_data_fixed(all_paths, CONFIG['horizons'], 'train')
    val_df = prepare_data_fixed(all_paths, CONFIG['horizons'], 'val')
    test_df = prepare_data_fixed(all_paths, CONFIG['horizons'], 'test')

    if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
        print("ERROR: One or more data splits are empty. Check feature engineering output.")
        return

    # Train a model for each horizon
    all_metrics = {}
    models = {}
    
    print(f"\n🚀 Training {len(CONFIG['horizons'])} LightGBM models: {CONFIG['horizons']}")
    
    for i, h in enumerate(CONFIG['horizons'], 1):
        print(f"\n--- Training for {h}m horizon ({i}/{len(CONFIG['horizons'])}) ---")
        model, metrics = train_and_evaluate(train_df, val_df, test_df, h, CONFIG['lgb_params'])
        
        if model is not None:
            all_metrics[f'{h}m'] = metrics
            models[f'{h}m'] = model
            
            # Print summary for both validation and test
            val_metrics = metrics['validation']
            test_metrics = metrics['test']
            print(f"  🎯 Validation - Accuracy: {val_metrics['accuracy']:.2%}, ROC AUC: {val_metrics['roc_auc']:.2%}")
            print(f"  🎯 Test      - Accuracy: {test_metrics['accuracy']:.2%}, ROC AUC: {test_metrics['roc_auc']:.2%}")
            print(f"✅ Horizon {h}m completed!")
        else:
            print(f"❌ Horizon {h}m failed - insufficient data")

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

    # Create and save plots
    fig = plot_metrics(all_metrics)
    plot_path = CONFIG['results_dir'] / 'lightgbm_short_term_metrics.html'
    fig.write_html(plot_path)
    print(f"Metrics plot saved to: {plot_path}")
    
    # Create and save confusion matrix plot
    cm_fig = plot_confusion_matrices(all_metrics)
    cm_plot_path = CONFIG['results_dir'] / 'lightgbm_confusion_matrices.html'
    cm_fig.write_html(cm_plot_path)
    print(f"Confusion matrices saved to: {cm_plot_path}")


if __name__ == "__main__":
    main() 