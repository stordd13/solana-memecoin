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
from typing import List, Dict

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
    'features_dir': Path('data/features_with_targets'),
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
# Labels are now created in pipeline - this function is no longer needed

# --- Plotting Functions ---
def plot_metrics(metrics: Dict):
    """Plots key metrics comparing validation vs test performance."""
    horizons = list(metrics.keys())
    
    # Show all meaningful metrics - accuracy IS important for balanced data
    key_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC']
    
    fig = go.Figure()
    
    colors_mean = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Mean colors
    colors_std = ['#87ceeb', '#ffd700', '#90ee90', '#ff6347', '#dda0dd']   # Std colors (lighter)
    
    for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
        # Mean metrics
        mean_values = [metrics[h]['mean'][metric] for h in horizons]
        std_values = [metrics[h]['std'][metric] for h in horizons]
        
        fig.add_trace(go.Bar(
            name=f'{label} (Mean)',
            x=horizons,
            y=mean_values,
            text=[f"{val:.2f}" for val in mean_values],
            textposition='auto',
            marker_color=colors_mean[i],
            error_y=dict(type='data', array=std_values, visible=True),
            offsetgroup=1
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
        title='Logistic Regression: Walk-Forward Cross-Validation Performance',
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
        text="<b>Walk-Forward Validation:</b><br>• Error bars show std across folds<br>• High accuracy + Low recall = Conservative model<br>• Check confusion matrices for details<br>• ROC AUC shows true predictive power",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        align="left"
    )
    
    return fig


def plot_confusion_matrices(metrics: Dict):
    """Create confusion matrix heatmaps for all horizons."""
    import plotly.subplots as sp
    
    horizons = list(metrics.keys())
    n_horizons = len(horizons)
    
    # Create subplots: 1 row x n_horizons columns for mean confusion matrices
    fig = sp.make_subplots(
        rows=1, 
        cols=n_horizons,
        subplot_titles=[f'{h} (Mean)' for h in horizons],
        specs=[[{'type': 'heatmap'} for _ in range(n_horizons)]]
    )
    
    for i, horizon in enumerate(horizons):
        col = i + 1
        
        # Get mean confusion matrix values (these should be aggregated means)
        horizon_metrics = metrics[horizon]['mean']
        
        # Extract confusion matrix components if they exist
        if 'confusion_matrix' in horizon_metrics:
            cm = horizon_metrics['confusion_matrix']
            # Build 2x2 confusion matrix
            confusion_matrix = [
                [cm.get('true_negative', 0), cm.get('false_positive', 0)],
                [cm.get('false_negative', 0), cm.get('true_positive', 0)]
            ]
        else:
            # If no confusion matrix data, create dummy
            confusion_matrix = [[0, 0], [0, 0]]
        
        # Create heatmap
        fig.add_trace(go.Heatmap(
            z=confusion_matrix,
            x=['Predicted DOWN', 'Predicted UP'],
            y=['Actual DOWN', 'Actual UP'],
            text=[[f'{val:.0f}' for val in row] for row in confusion_matrix],
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='Blues',
            showscale=(i == 0),  # Only show colorbar for first subplot
            name=f'CM {horizon}'
        ), row=1, col=col)
    
    fig.update_layout(
        title='Confusion Matrices: Walk-Forward Validation (Averaged)',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_financial_metrics(metrics: Dict):
    """Plot financial-specific metrics."""
    horizons = list(metrics.keys())
    
    # Financial metrics to plot
    financial_metrics = ['avg_return_per_tp', 'avg_return_per_fp', 'prediction_sharpe', 'return_capture_rate']
    metric_labels = ['Avg Return per TP', 'Avg Return per FP', 'Prediction Sharpe', 'Return Capture Rate']
    
    fig = go.Figure()
    
    colors = ['#2E8B57', '#DC143C', '#4169E1', '#FF8C00']  # Different colors for financial metrics
    
    for i, (metric, label) in enumerate(zip(financial_metrics, metric_labels)):
        mean_values = []
        std_values = []
        
        for h in horizons:
            if metric in metrics[h]['mean']:
                mean_val = metrics[h]['mean'][metric]
                std_val = metrics[h]['std'][metric]
                
                # Convert to percentage for some metrics
                if metric in ['return_capture_rate']:
                    mean_val *= 100
                    std_val *= 100
                    
                mean_values.append(mean_val)
                std_values.append(std_val)
            else:
                mean_values.append(0)
                std_values.append(0)
        
        fig.add_trace(go.Bar(
            name=label,
            x=horizons,
            y=mean_values,
            text=[f"{val:.3f}" if abs(val) < 1 else f"{val:.1f}" for val in mean_values],
            textposition='auto',
            marker_color=colors[i],
            error_y=dict(type='data', array=std_values, visible=True),
            offsetgroup=i
        ))
    
    fig.update_layout(
        barmode='group',
        title='Financial Metrics: Walk-Forward Cross-Validation',
        xaxis_title='Prediction Horizon',
        yaxis_title='Value',
        legend_title='Financial Metric',
        template='plotly_white'
    )
    
    return fig


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
    
    # 2. Smart Walk-Forward Split (memecoin-aware) - BEFORE creating labels
    print("Creating smart memecoin-aware walk-forward splits...")
    splitter = WalkForwardSplitter()  # Auto-adapts to data length
    
    folds, feasible_horizons = splitter.smart_split_for_memecoins(
        data, 
        horizons=CONFIG['horizons'],
        time_column='datetime'
    )
    
    print(f"Created {len(folds)} walk-forward folds.")
    print(f"Original horizons: {CONFIG['horizons']}")
    print(f"Feasible horizons: {feasible_horizons}")
    
    if len(feasible_horizons) < len(CONFIG['horizons']):
        skipped = [h for h in CONFIG['horizons'] if h not in feasible_horizons]
        print(f"⚠️  Skipped horizons (too long for data): {skipped}")
    
    if not folds:
        print("❌ No valid folds created - all tokens too short!")
        return
        
    if not feasible_horizons:
        print("❌ No feasible horizons - tokens too short for any prediction!")
        return
    
    # 3. Training and Evaluation Loop (only for feasible horizons)
    all_fold_metrics = {h: [] for h in feasible_horizons}
    
    # Prepare for fitting scaler on first train fold
    first_train_df, _ = folds[0]
    feature_cols = [c for c in first_train_df.columns if c not in ['datetime', 'price', 'token_id'] and not c.startswith('label_') and not c.startswith('return_')]
    
    # We'll fit the winsorizer on the first fold's training data after creating labels
    winsorizer = None
    models = {}

    for i, (train_df, test_df) in enumerate(tqdm(folds, desc="Processing Folds")):
        # Labels are already created in pipeline - no need to create them again
        
        # Fit winsorizer on first fold's training data
        if winsorizer is None:
            print("Fitting Winsorizer on the first training fold...")
            winsorizer = Winsorizer(lower_percentile=0.005, upper_percentile=0.995)
            X_fit = train_df[feature_cols].to_numpy()
            winsorizer.fit(X_fit)
        
        fold_metrics = {}
        for h in feasible_horizons:
            label_col = f'label_{h}m'
            return_col = f'return_{h}m'
            
            train_fold = train_df.drop_nulls([label_col, return_col])
            test_fold = test_df.drop_nulls([label_col, return_col])
            
            # Check if we have enough samples after creating labels and dropping nulls
            if len(train_fold) == 0:
                # This can happen for long horizons - expected behavior
                continue
                
            if len(test_fold) == 0:
                # This is expected for longer horizons where we can't look far enough ahead
                continue
            
            X_train = winsorizer.transform(train_fold[feature_cols].to_numpy())
            y_train = train_fold[label_col].to_numpy().astype(int)  # Ensure binary labels are integers
            
            X_test = winsorizer.transform(test_fold[feature_cols].to_numpy())
            y_test = test_fold[label_col].to_numpy().astype(int)  # Ensure binary labels are integers
            returns_test = test_fold[return_col].to_numpy()
            
            # Remove NaN values from labels and corresponding features
            train_valid_mask = ~np.isnan(y_train)
            X_train = X_train[train_valid_mask]
            y_train = y_train[train_valid_mask]
            
            test_valid_mask = ~np.isnan(y_test)
            X_test = X_test[test_valid_mask]
            y_test = y_test[test_valid_mask]
            returns_test = returns_test[test_valid_mask]

            # Check that we have enough samples after NaN removal
            if len(X_train) < 10 or len(X_test) < 10:
                print(f"⚠️  Warning: Insufficient samples for {h}m after NaN removal")
                print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
                continue
                
            # Check that we have binary labels  
            if len(np.unique(y_train)) < 2:
                print(f"⚠️  Warning: Labels not binary for {h}m horizon")
                print(f"   Train unique labels: {np.unique(y_train)}")
                continue

            model = LogisticRegression(
                max_iter=300,
                n_jobs=-1,
                random_state=CONFIG['random_state'],
                class_weight='balanced',
                solver='liblinear',
                penalty='l2',
                C=0.1,
                tol=1e-4
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = financial_classification_metrics(y_test, y_pred, returns_test, y_prob)
            
            # Add confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
            
            all_fold_metrics[h].append(metrics)
            models[h] = model

    # 4. Aggregate and Save Results
    print("\n--- Aggregated Walk-Forward Results ---")
    final_metrics = {}
    for h, h_metrics in all_fold_metrics.items():
        if not h_metrics: continue
        
        # Separate confusion matrix from other metrics
        avg_metrics = {}
        std_metrics = {}
        
        for key in h_metrics[0]:
            if key == 'confusion_matrix':
                # Average confusion matrix components
                cm_avg = {}
                cm_std = {}
                for cm_key in h_metrics[0]['confusion_matrix']:
                    values = [m['confusion_matrix'][cm_key] for m in h_metrics]
                    cm_avg[cm_key] = np.mean(values)
                    cm_std[cm_key] = np.std(values)
                avg_metrics['confusion_matrix'] = cm_avg
                std_metrics['confusion_matrix'] = cm_std
            else:
                # Regular metrics
                values = [m[key] for m in h_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
                    std_metrics[key] = np.std(values)
        
        final_metrics[f'{h}m'] = {
            'mean': avg_metrics,
            'std': std_metrics
        }
        print(f"\nHorizon {h}m:")
        for key, val in avg_metrics.items():
            if key == 'confusion_matrix':
                print(f"  {key}:")
                for cm_key, cm_val in val.items():
                    print(f"    {cm_key}: {cm_val:.4f} (+/- {std_metrics[key][cm_key]:.4f})")
            else:
                print(f"  {key}: {val:.4f} (+/- {std_metrics[key]:.4f})")

    # Save artifacts
    results_path = CONFIG['results_dir']
    with open(results_path / 'metrics_walkforward.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    joblib.dump(winsorizer, results_path / 'winsorizer_walkforward.joblib')
    joblib.dump(models, results_path / 'logreg_models_walkforward.joblib')

    # Generate and save HTML plots
    print("\n📊 Generating visualization plots...")
    
    # Performance metrics plot
    metrics_fig = plot_metrics(final_metrics)
    metrics_path = results_path / 'performance_metrics_walkforward.html'
    metrics_fig.write_html(metrics_path)
    print(f"  📈 Performance metrics: {metrics_path}")
    
    # Confusion matrices plot
    confusion_fig = plot_confusion_matrices(final_metrics)
    confusion_path = results_path / 'confusion_matrices_walkforward.html'
    confusion_fig.write_html(confusion_path)
    print(f"  🔥 Confusion matrices: {confusion_path}")
    
    # Financial metrics plot
    financial_fig = plot_financial_metrics(final_metrics)
    financial_path = results_path / 'financial_metrics_walkforward.html'
    financial_fig.write_html(financial_path)
    print(f"  💰 Financial metrics: {financial_path}")

    print(f"\n✅ Walk-forward training complete!")
    print(f"📁 Results saved to: {results_path}")

if __name__ == '__main__':
    main() 