#!/usr/bin/env python3
"""
Baseline model training with LightGBM fallback for macOS OpenMP issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime
import json
import argparse

from scripts.utils import setup_logger
import config

logger = setup_logger(__name__)

def train_baseline_model(df: pl.DataFrame, interval: str):
    """Train baseline model with LightGBM (better macOS compatibility)"""
    
    try:
        from lightgbm import LGBMClassifier
        model_type = "LightGBM"
        logger.info("Using LightGBM (better macOS compatibility)")
    except ImportError:
        try:
            from xgboost import XGBClassifier as LGBMClassifier
            model_type = "XGBoost"  
            logger.info("Using XGBoost (may have OpenMP issues on macOS)")
        except ImportError:
            logger.error("Neither LightGBM nor XGBoost available")
            return None
    
    logger.info(f"Training baseline {model_type} model for {interval} interval")
    logger.info(f"Dataset shape: {df.shape}")
    
    # Split data
    train = df.filter(pl.col("split") == "train")
    test = df.filter(pl.col("split") == "test")
    
    logger.info(f"Train samples: {train.height}, Test samples: {test.height}")
    
    # Feature columns (same as original)
    feature_cols = ["scaled_returns", "ma_5", "rsi_14", "imbalance_ratio", 
                    "initial_dump_flag", "acf_lag_1", "intra_interval_max_return", 
                    "max_returns", "vol_return_ratio", "vol_std_5", "momentum_lag1",
                    "vol_std_10", "momentum_lag5", "ma_10"]
    
    # Add new unified features if available
    unified_features = ["token_lifetime_minutes", "avg_volatility", "max_total_return"]
    for feat in unified_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    logger.info(f"Using {len(available_features)} features: {available_features}")
    
    # Prepare data
    X_train = train.select(available_features).fill_null(0).to_numpy()
    y_train = train["pump_label"].to_numpy()
    X_test = test.select(available_features).fill_null(0).to_numpy()
    y_test = test["pump_label"].to_numpy()
    
    # Calculate class weights
    pos_samples = sum(y_train)
    neg_samples = len(y_train) - pos_samples
    pos_weight = neg_samples / (pos_samples + 1e-6)
    
    logger.info(f"Class balance: {pos_samples} positive, {neg_samples} negative")
    logger.info(f"Pos weight: {pos_weight:.2f}")
    
    pump_rate = pos_samples / len(y_train)
    logger.info(f"Pump rate: {pump_rate:.3%}")
    
    # Train model
    if model_type == "LightGBM":
        model = LGBMClassifier(
            scale_pos_weight=pos_weight,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1  # Suppress LightGBM output
        )
    else:  # XGBoost
        model = LGBMClassifier(
            scale_pos_weight=pos_weight,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = list(zip(available_features, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:5]
    else:
        top_features = []
    
    # Results
    results = {
        "model_type": model_type,
        "interval": interval,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "pump_rate": pump_rate,
        "pos_weight": pos_weight,
        "top_features": [(feat, float(imp)) for feat, imp in top_features],
        "model": model,
        "features": available_features
    }
    
    logger.info(f"\n{model_type} Results for {interval}:")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"ROC AUC: {roc_auc:.3f}")
    logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logger.info(f"Top features: {[f'{feat}: {imp:.3f}' for feat, imp in top_features]}")
    
    # Save model if performance is reasonable
    save_model = (f1 > 0.10) or (roc_auc > 0.65)
    
    if save_model:
        # Use unified naming convention
        unified_model_name = f"baseline_{interval}_unified"
        model_path = os.path.join(config.BOT_NEW_ROOT, f"models/{unified_model_name}.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': model,
            'features': available_features,
            'metrics': {k: v for k, v in results.items() if k not in ['model', 'features']},
            'interval': interval,
            'model_type': f'{model_type.lower()}_baseline',
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, model_path)
        logger.info(f"✅ Model saved to {model_path} (F1={f1:.3f}, AUC={roc_auc:.3f})")
        
        # Also save results to analysis folder
        analysis_dir = os.path.join(config.BOT_NEW_ROOT, 'analysis', interval)
        os.makedirs(analysis_dir, exist_ok=True)
        
        results_file = os.path.join(analysis_dir, f'baseline_results_{model_type.lower()}.json')
        with open(results_file, 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'model'}, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    else:
        logger.warning(f"❌ Model performance too low: F1={f1:.3f}, AUC={roc_auc:.3f}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train baseline model with LightGBM fallback')
    parser.add_argument('--interval', default='5m', choices=['1m', '5m'], help='Data interval')
    parser.add_argument('--min-f1', type=float, default=0.10, help='Minimum F1 to save model')
    args = parser.parse_args()
    
    # Load unified data
    data_file = f"processed_features_{args.interval}_unified.parquet"
    data_path = os.path.join(config.BOT_NEW_ROOT, data_file)
    
    if not os.path.exists(data_path):
        logger.error(f"Unified data file not found: {data_path}")
        logger.info("Run 'python scripts/run_pipeline3.py' first")
        return
    
    logger.info(f"Loading unified data from: {data_path}")
    df = pl.read_parquet(data_path)
    
    # Train model
    results = train_baseline_model(df, args.interval)
    
    if results and results.get('f1_score', 0) > args.min_f1:
        logger.info("✅ Baseline model training completed successfully!")
    else:
        logger.warning("⚠️  Model performance below threshold")

if __name__ == "__main__":
    main()