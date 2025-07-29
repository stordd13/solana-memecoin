#!/usr/bin/env python3
"""
Run baseline XGBoost model training - Fast path to trading strategy
Focuses on getting F1 > 15% for immediate signal generation (realistic for memecoin data)

Usage:
    python ml/run_baseline.py [--interval 5m] [--no-save-models] [--use-smote]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime
import json
import argparse

from scripts.utils import setup_logger
import config

logger = setup_logger(__name__)

def load_and_validate_data(file_path: str) -> pl.DataFrame:
    """Load data and validate required columns exist."""
    logger.info(f"Loading data from {file_path}")
    df = pl.read_parquet(file_path)
    
    # Core required columns for unified data (no archetypes)
    required_cols = ["token_id", "split", "pump_label"]
    
    # Optional feature columns - add defaults if missing
    optional_cols = ["ma_5", "rsi_14", "imbalance_ratio", "initial_dump_flag", "acf_lag_1",
                     "intra_interval_max_return", "max_returns", "vol_return_ratio",
                     "vol_std_5", "momentum_lag1", "vol_std_10", "momentum_lag5", "ma_10"]
    
    # Check required columns
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # Add missing optional columns with default values
    missing_optional = [col for col in optional_cols if col not in df.columns]
    if missing_optional:
        logger.info(f"Adding default values for missing features: {missing_optional}")
        for col in missing_optional:
            df = df.with_columns(pl.lit(0.0).alias(col))
    
    # Create scaled_returns from returns if available
    if "returns" in df.columns and "scaled_returns" not in df.columns:
        logger.info("Creating scaled_returns from returns column")
        df = df.with_columns(pl.col("returns").alias("scaled_returns"))
    elif "scaled_returns" not in df.columns:
        logger.info("No returns data available, using ma_5 as proxy")
        df = df.with_columns(pl.col("ma_5").alias("scaled_returns"))
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Available columns: {df.columns[:10]}... (showing first 10)")
    
    # Pump statistics for unified data
    pump_rate = df['pump_label'].sum() / df.height
    logger.info(f"Unified pump rate: {pump_rate:.3%} ({df['pump_label'].sum()} pumps out of {df.height} samples)")
    logger.info(f"Train/test split: {df['split'].value_counts()}")
    
    return df

def train_baseline_model(df: pl.DataFrame, archetype=None, min_samples: int = 100, interval: str = "5m") -> dict:
    """Train XGBoost on unified data (no archetype separation)."""
    
    # Use all data for unified training
    df_train = df
    model_name = f"baseline_{interval}_unified"
    logger.info(f"Training unified baseline model ({df_train.height} samples)")
    
    # Check minimum samples
    if df_train.height < min_samples:
        logger.warning(f"Too few samples ({df_train.height}) for {model_name}")
        return None
    
    # Split data
    train = df_train.filter(pl.col("split") == "train")
    test = df_train.filter(pl.col("split") == "test")
    
    logger.info(f"\nTraining {model_name}:")
    logger.info(f"Train samples: {train.height}, Test samples: {test.height}")
    logger.info(f"Train pump rate: {(train['pump_label'].sum() / train.height):.2%}")
    
    # Feature columns (LEGITIMATE ONLY - NO DATA LEAKAGE)
    feature_cols = ["scaled_returns", "ma_5", "rsi_14", "imbalance_ratio", 
                    "initial_dump_flag", "acf_lag_1", "intra_interval_max_return", 
                    "max_returns", "vol_return_ratio", "vol_std_5", "momentum_lag1",
                    "vol_std_10", "momentum_lag5", "ma_10", "initial_price",
                    "minutes_since_start", "current_total_return", "recent_avg_volatility"]
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    logger.info(f"Using features: {available_features}")
    
    # Prepare data
    X_train = train.select(available_features).fill_null(0).to_numpy()
    y_train = train["pump_label"].to_numpy()
    X_test = test.select(available_features).fill_null(0).to_numpy()
    y_test = test["pump_label"].to_numpy()
    
    # Calculate class weight for imbalanced data (no SMOTE - memory efficient)
    pos_samples = sum(y_train)
    neg_samples = len(y_train) - pos_samples
    pos_weight = neg_samples / (pos_samples + 1e-6)
    
    logger.info(f"Class balance: {pos_samples} positive, {neg_samples} negative")
    logger.info(f"XGBoost scale_pos_weight: {pos_weight:.2f}")
    
    # Additional class balance info
    pump_rate = pos_samples / len(y_train)
    logger.info(f"Pump rate: {pump_rate:.3%} - {'Good balance' if pump_rate > 0.01 else 'Severe imbalance'}")
    
    # Train model with tuned parameters
    model = XGBClassifier(
        scale_pos_weight=pos_weight,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Fit without early stopping rounds (compatibility fix)
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC (useful even with imbalanced classes)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Additional metrics for trading context
    total_positive_preds = fp + tp
    precision_if_all_positive = sum(y_test) / len(y_test)  # Baseline if we predict all as pump
    
    # Feature importance
    feature_importance = dict(zip(available_features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    results = {
        "model_name": model_name,
        "interval": interval,
        "train_samples": train.height,
        "test_samples": test.height,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_positive_predictions": int(total_positive_preds),
        "baseline_precision": precision_if_all_positive,
        "top_features": top_features,
        "model": model,
        "features": available_features
    }
    
    logger.info(f"\nResults for {model_name}:")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info(f"Precision: {precision:.3f} (baseline: {precision_if_all_positive:.3f})")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"ROC AUC: {roc_auc:.3f}")
    logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logger.info(f"Positive predictions: {total_positive_preds} ({total_positive_preds/len(y_test):.1%})")
    logger.info(f"Top features: {[f'{feat}: {imp:.3f}' for feat, imp in top_features]}")
    
    # Save model with lower threshold for extreme imbalance OR good ROC AUC
    save_model = (f1 > 0.15) or (roc_auc > 0.7 and precision > 0.1) or (precision > 0.2)
    
    if save_model:
        # Use unified naming convention
        unified_model_name = f"baseline_{interval}_unified"
        model_path = os.path.join(config.BOT_NEW_ROOT, f"models/{unified_model_name}.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': model,
            'features': available_features,
            'metrics': {k: v for k, v in results.items() if k != 'model'},
            'interval': interval,
            'model_type': 'xgboost_baseline',
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, model_path)
        logger.info(f"✅ Model saved to {model_path} (F1={f1:.3f}, AUC={roc_auc:.3f})")
    else:
        logger.warning(f"❌ Model performance too low: F1={f1:.3f}, AUC={roc_auc:.3f}, Precision={precision:.3f}")
    
    return results

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train baseline XGBoost models for memecoin pump prediction')
    parser.add_argument('--interval', default='5m', choices=['1m', '5m'], 
                       help='Data interval to use (default: 5m)')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Skip saving trained models')
    parser.add_argument('--min-f1', type=float, default=0.15,
                       help='Minimum F1 score to save models (default: 0.15)')
    args = parser.parse_args()
    
    # Load unified data
    data_filename = f"processed_features_{args.interval}_unified.parquet" 
    data_path = os.path.join(config.BOT_NEW_ROOT, data_filename)
    
    if not os.path.exists(data_path):
        logger.error(f"Unified data file not found: {data_path}")
        logger.error("Please run the unified pipeline first:")
        logger.error(f"python scripts/run_pipeline3.py")
        return
    
    logger.info(f"Using {args.interval} interval data")
    df = load_and_validate_data(data_path)
    
    # Train unified model (no archetypes)
    logger.info(f"\n{'='*50}")
    logger.info(f"Training unified baseline model ({args.interval} data)")
    logger.info(f"{'='*50}")
    
    results = train_baseline_model(df, archetype=None, interval=args.interval)
    
    if results:
        logger.info(f"\n🎯 Unified model training completed")
        logger.info(f"F1={results['f1_score']:.3f}, AUC={results['roc_auc']:.3f}")
        
        if results["f1_score"] > args.min_f1:
            logger.info("✅ Model performance above threshold - saved successfully")
        else:
            logger.warning(f"⚠️  Model F1 ({results['f1_score']:.3f}) below threshold ({args.min_f1})")
    else:
        logger.error("❌ Model training failed")
    
    # Save results summary if model was trained successfully
    if results:
        interval_dir = os.path.join(config.BOT_NEW_ROOT, "analysis", args.interval)
        os.makedirs(interval_dir, exist_ok=True)
        summary_path = os.path.join(interval_dir, "baseline_results.json")
        
        # Save simplified results
        simplified_results = {k: v for k, v in results.items() if k != "model"}
        
        with open(summary_path, 'w') as f:
            json.dump(simplified_results, f, indent=2, default=str)
        
        logger.info(f"📊 Results summary saved to {summary_path}")
        
        # Final summary
        logger.info("\n" + "="*50)
        logger.info("UNIFIED BASELINE MODEL SUMMARY")
        logger.info("="*50)
        logger.info(f"F1 Score: {results['f1_score']:.3f}")
        logger.info(f"Precision: {results['precision']:.3f}") 
        logger.info(f"Recall: {results['recall']:.3f}")
        logger.info(f"ROC AUC: {results['roc_auc']:.3f}")
        
        if results['f1_score'] > 0.15:
            logger.info("✅ Model ready for transformer and RL training!")
        else:
            logger.info("⚠️  Consider feature engineering or threshold adjustment")
    

if __name__ == "__main__":
    main()
