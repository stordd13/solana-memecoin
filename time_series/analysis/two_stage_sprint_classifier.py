#!/usr/bin/env python3
"""
Two-Stage Sprint Classification Pipeline for Memecoin Analysis

Stage 1: Binary Sprint Detector (Sprint vs Non-Sprint)
Stage 2: Marathon vs Standard (Non-Sprint only)

Features: 33 total (5min window) or 38 total (10min window)
Custom Scorer: 0.7*recall + 0.3*precision
Full Dataset: 30k tokens with 80/20 stratified split
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import joblib

# Import from same directory
from archetype_classifier import ArchetypeClassifier

class TwoStageSprintClassifier:
    """Two-stage classification pipeline for sprint detection."""
    
    def __init__(self, results_dir: Path = None, window: int = 5):
        self.results_dir = results_dir or Path("../results")
        self.archetype_classifier = ArchetypeClassifier()
        self.window = window  # Analysis window in minutes
        
        # Models
        self.stage1_model = None  # Sprint vs Non-Sprint
        self.stage2_model = None  # Marathon vs Standard
        self.scaler = StandardScaler()
        
        # Custom scorer
        self.custom_scorer = make_scorer(self._weighted_recall_precision_scorer)
        
        # Results storage
        self.stage1_results = {}
        self.stage2_results = {}
        self.volatility_results = {}
        
    def _weighted_recall_precision_scorer(self, y_true, y_pred):
        """Custom scorer: 0.7*recall + 0.3*precision"""
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        return 0.7 * recall + 0.3 * precision
    
    def load_full_dataset(self, archetype_results_path: Path, data_dir: Path) -> pd.DataFrame:
        """Load full 30k dataset with no sampling."""
        print("📊 Loading full dataset (no sampling)...")
        
        # Load archetype results and token data
        self.archetype_classifier.load_archetype_results(archetype_results_path)
        token_data = self.archetype_classifier.load_token_data(data_dir)
        
        # Get category distribution
        category_counts = {}
        for token_name, labels in self.archetype_classifier.token_labels.items():
            category = labels['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"📈 Total tokens available: {len(self.archetype_classifier.token_labels)}")
        print(f"📊 Category distribution: {category_counts}")
        
        # Use all available tokens from each category
        all_tokens = {}
        for token_name, labels in self.archetype_classifier.token_labels.items():
            if token_name in token_data:
                all_tokens[token_name] = token_data[token_name]
        
        print(f"✅ Full dataset loaded: {len(all_tokens)} tokens")
        
        # Extract features with dynamic window sizing
        features_df = self.archetype_classifier.extract_features(all_tokens, minutes=self.window)
        
        # Verify final distribution
        actual_counts = features_df['category'].value_counts()
        print(f"📊 Final dataset distribution: {actual_counts.to_dict()}")
        print(f"📊 Total features: {len([col for col in features_df.columns if col not in ['token_name', 'category', 'cluster', 'archetype']])}")
        
        return features_df
    
    def prepare_stage_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for both stages."""
        print("🔧 Preparing stage data...")
        
        # Feature columns (exclude labels and token_name)
        feature_cols = [col for col in features_df.columns if col not in ['token_name', 'category', 'cluster', 'archetype']]
        X = features_df[feature_cols].values
        X = np.nan_to_num(X)  # Handle any NaN values
        
        # Stage 1: Sprint vs Non-Sprint
        y_stage1 = (features_df['category'] == 'sprint').astype(int).values
        
        # Stage 2: Marathon vs Standard (Non-Sprint only)
        non_sprint_mask = features_df['category'] != 'sprint'
        X_stage2 = X[non_sprint_mask]
        y_stage2 = (features_df[non_sprint_mask]['category'] == 'marathon').astype(int).values
        
        print(f"📊 Stage 1 - Sprint vs Non-Sprint:")
        print(f"  Total samples: {len(X)}")
        print(f"  Sprint: {np.sum(y_stage1)}")
        print(f"  Non-Sprint: {np.sum(1 - y_stage1)}")
        
        print(f"📊 Stage 2 - Marathon vs Standard:")
        print(f"  Total samples: {len(X_stage2)}")
        print(f"  Marathon: {np.sum(y_stage2)}")
        print(f"  Standard: {np.sum(1 - y_stage2)}")
        
        return X, y_stage1, X_stage2, y_stage2
    
    def train_stage1_sprint_detector(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Stage 1: Binary Sprint Detector."""
        print("🚀 Training Stage 1: Binary Sprint Detector...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for balanced training
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Optuna optimization
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0,
                'use_label_encoder': False,
                'scale_pos_weight': class_weights[1] / class_weights[0]  # Positive class weight
            }
            
            model = xgb.XGBClassifier(**params)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                      scoring=self.custom_scorer, n_jobs=-1)
            return cv_scores.mean()
        
        # Optimize (enhanced trials)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=15, show_progress_bar=True)
        
        # Train final model
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'use_label_encoder': False
        })
        
        self.stage1_model = xgb.XGBClassifier(**best_params)
        
        # Train with balanced approach (no additional sample weights needed)
        self.stage1_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.stage1_model.predict(X_test_scaled)
        
        results = {
            'best_params': best_params,
            'optimization_score': study.best_value,
            'test_accuracy': np.mean(y_pred == y_test),
            'test_precision': precision_score(y_test, y_pred, average='binary'),
            'test_recall': recall_score(y_test, y_pred, average='binary'),
            'test_f1': f1_score(y_test, y_pred, average='binary'),
            'custom_score': self._weighted_recall_precision_scorer(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=['Non-Sprint', 'Sprint']),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.stage1_results = results
        
        print(f"✅ Stage 1 Results:")
        print(f"  Optimization Score: {results['optimization_score']:.4f}")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Sprint Precision: {results['test_precision']:.4f}")
        print(f"  Sprint Recall: {results['test_recall']:.4f}")
        print(f"  Sprint F1: {results['test_f1']:.4f}")
        print(f"  Custom Score: {results['custom_score']:.4f}")
        
        return results
    
    def train_stage2_marathon_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Stage 2: Marathon vs Standard Classifier."""
        print("🚀 Training Stage 2: Marathon vs Standard Classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (use existing scaler)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for balanced training
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Optuna optimization
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0,
                'use_label_encoder': False,
                'scale_pos_weight': class_weights[1] / class_weights[0]  # Positive class weight
            }
            
            model = xgb.XGBClassifier(**params)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                      scoring=self.custom_scorer, n_jobs=-1)
            return cv_scores.mean()
        
        # Optimize (enhanced trials)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=15, show_progress_bar=True)
        
        # Train final model
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'use_label_encoder': False
        })
        
        self.stage2_model = xgb.XGBClassifier(**best_params)
        
        # Train with balanced approach (no additional sample weights needed)
        self.stage2_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.stage2_model.predict(X_test_scaled)
        
        results = {
            'best_params': best_params,
            'optimization_score': study.best_value,
            'test_accuracy': np.mean(y_pred == y_test),
            'test_precision': precision_score(y_test, y_pred, average='binary'),
            'test_recall': recall_score(y_test, y_pred, average='binary'),
            'test_f1': f1_score(y_test, y_pred, average='binary'),
            'custom_score': self._weighted_recall_precision_scorer(y_test, y_pred),
            'false_negative_rate': 1 - recall_score(y_test, y_pred, average='binary'),
            'classification_report': classification_report(y_test, y_pred, target_names=['Standard', 'Marathon']),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.stage2_results = results
        
        print(f"✅ Stage 2 Results:")
        print(f"  Optimization Score: {results['optimization_score']:.4f}")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Marathon Precision: {results['test_precision']:.4f}")
        print(f"  Marathon Recall: {results['test_recall']:.4f}")
        print(f"  Marathon F1: {results['test_f1']:.4f}")
        print(f"  Custom Score: {results['custom_score']:.4f}")
        print(f"  False Negative Rate: {results['false_negative_rate']:.4f}")
        
        return results
    
    def run_volatility_analysis(self, data_dir: Path) -> Dict:
        """Run volatility analysis for all categories."""
        print("📊 Running volatility analysis for all categories...")
        
        # Initialize results for all categories
        categories = ['sprint', 'standard', 'marathon']
        all_results = {}
        
        for category in categories:
            print(f"  📈 Analyzing {category} tokens...")
            
            # Load tokens for this category
            category_tokens = []
            for token_name, labels in self.archetype_classifier.token_labels.items():
                if labels['category'] == category:
                    token_file = data_dir / f"{token_name}.parquet"
                    if token_file.exists():
                        try:
                            df = pl.read_parquet(token_file)
                            prices = df['price'].to_numpy()
                            if len(prices) >= 6:  # Minimum length requirement
                                category_tokens.append({'name': token_name, 'prices': prices})
                        except:
                            continue
            
            print(f"    📊 Loaded {len(category_tokens)} {category} tokens")
            
            # Run volatility analysis for this category
            results = {'high_vol_count': 0, 'high_vol_pumps': 0, 'times_to_1_5x': []}
            
            for token in category_tokens:
                prices = token['prices']
                if len(prices) < 6:
                    continue
                    
                # Calculate early returns and volatility (consistent 5min for all)
                returns_early = np.diff(prices[:5]) / prices[:4]
                vol_5min = np.std(returns_early) / np.mean(prices[:5]) if np.mean(prices[:5]) > 0 else 0
                
                if vol_5min > 0.8:
                    results['high_vol_count'] += 1
                    # Check for pump after analysis window (use prices[9] for 10min, prices[4] for 5min)
                    analysis_end = min(self.window, len(prices) - 1)
                    base_price_idx = 9 if self.window == 10 and len(prices) > 10 else 4
                    base_price_idx = min(base_price_idx, len(prices) - 1)
                    
                    post_prices = prices[analysis_end:]
                    
                    if len(post_prices) > 0 and np.max(post_prices) / prices[base_price_idx] > 1.5:
                        results['high_vol_pumps'] += 1
                        pump_indices = np.where(post_prices > prices[base_price_idx] * 1.5)[0]
                        if len(pump_indices) > 0:
                            time_to_1_5x = pump_indices[0] + 1  # Minutes after analysis window
                            results['times_to_1_5x'].append(time_to_1_5x)
            
            # Calculate final metrics for this category
            pump_rate = results['high_vol_pumps'] / results['high_vol_count'] * 100 if results['high_vol_count'] > 0 else 0
            avg_time = np.mean(results['times_to_1_5x']) if results['times_to_1_5x'] else 0
            overall_1_5x_rate = len(results['times_to_1_5x']) / results['high_vol_count'] * 100 if results['high_vol_count'] > 0 else 0
            
            category_results = {
                'high_vol_count': results['high_vol_count'],
                'high_vol_pumps': results['high_vol_pumps'],
                'pump_rate_percent': pump_rate,
                'avg_time_to_1_5x': avg_time,
                'overall_1_5x_rate': overall_1_5x_rate,
                'times_to_1_5x': results['times_to_1_5x']
            }
            
            all_results[category] = category_results
            
            print(f"    ✅ {category}: {results['high_vol_count']} high-vol, {pump_rate:.1f}% pump rate")
        
        # Add overall summary
        final_results = {
            'categories': all_results,
            'marathon_false_negative_rate': self.stage2_results.get('false_negative_rate', 0),
            'api_status': 'Soon',
            'analysis_window': self.window
        }
        
        self.volatility_results = final_results
        
        print(f"✅ Volatility Analysis Summary:")
        for category, results in all_results.items():
            print(f"  {category}: {results['high_vol_count']} high-vol, {results['pump_rate_percent']:.1f}% pump rate, {results['avg_time_to_1_5x']:.1f}min avg time")
        
        return final_results
    
    def save_models(self) -> None:
        """Save trained models with window suffix."""
        models_dir = self.results_dir / "two_stage_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models with window suffix
        window_suffix = f"_{self.window}min"
        joblib.dump(self.stage1_model, models_dir / f'stage1_sprint_detector{window_suffix}.pkl')
        joblib.dump(self.stage2_model, models_dir / f'stage2_marathon_classifier{window_suffix}.pkl')
        joblib.dump(self.scaler, models_dir / f'scaler{window_suffix}.pkl')
        
        # Save results with window suffix
        all_results = {
            'window_minutes': self.window,
            'stage1_results': self.stage1_results,
            'stage2_results': self.stage2_results,
            'volatility_results': self.volatility_results
        }
        
        with open(models_dir / f'results{window_suffix}.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"💾 Models and results saved to: {models_dir} (window: {self.window}min)")
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("# Two-Stage Sprint Classification Results")
        report.append("=" * 50)
        
        # Stage 1 Results
        report.append("\n## Stage 1: Binary Sprint Detector")
        report.append(f"- **Optimization Score**: {self.stage1_results['optimization_score']:.4f}")
        report.append(f"- **Test Accuracy**: {self.stage1_results['test_accuracy']:.4f}")
        report.append(f"- **Sprint Precision**: {self.stage1_results['test_precision']:.4f}")
        report.append(f"- **Sprint Recall**: {self.stage1_results['test_recall']:.4f} (Target: >0.75)")
        report.append(f"- **Sprint F1**: {self.stage1_results['test_f1']:.4f}")
        report.append(f"- **Custom Score**: {self.stage1_results['custom_score']:.4f}")
        
        # Stage 2 Results
        report.append("\n## Stage 2: Marathon vs Standard Classifier")
        report.append(f"- **Optimization Score**: {self.stage2_results['optimization_score']:.4f}")
        report.append(f"- **Test Accuracy**: {self.stage2_results['test_accuracy']:.4f}")
        report.append(f"- **Marathon Precision**: {self.stage2_results['test_precision']:.4f}")
        report.append(f"- **Marathon Recall**: {self.stage2_results['test_recall']:.4f}")
        report.append(f"- **Marathon F1**: {self.stage2_results['test_f1']:.4f}")
        report.append(f"- **False Negative Rate**: {self.stage2_results['false_negative_rate']:.4f} (Target: <0.2)")
        
        # Volatility Analysis
        report.append("\n## Volatility Analysis (All Categories)")
        report.append(f"- **Analysis Window**: {self.volatility_results['analysis_window']} minutes")
        for category, results in self.volatility_results['categories'].items():
            report.append(f"- **{category.capitalize()}**: {results['high_vol_count']} high-vol, {results['pump_rate_percent']:.1f}% pump rate, {results['avg_time_to_1_5x']:.1f}min avg time")
        report.append(f"- **API Status**: {self.volatility_results['api_status']}")
        
        # Success Criteria (Updated Targets)
        report.append("\n## Success Criteria Check")
        stage1_f1 = self.stage1_results['test_f1']
        sprint_recall = self.stage1_results['test_recall']
        marathon_false_neg = self.stage2_results['false_negative_rate']
        
        report.append(f"- **Stage 1 F1 > 0.65**: {stage1_f1:.4f} {'✅' if stage1_f1 > 0.65 else '❌'}")
        report.append(f"- **Sprint Recall > 0.7**: {sprint_recall:.4f} {'✅' if sprint_recall > 0.7 else '❌'}")
        report.append(f"- **Marathon False Neg < 0.25**: {marathon_false_neg:.4f} {'✅' if marathon_false_neg < 0.25 else '❌'}")
        
        return "\n".join(report)
    
    def run_full_pipeline(self, archetype_results_path: Path, data_dir: Path) -> Dict:
        """Run complete two-stage classification pipeline."""
        print("🚀 Starting Two-Stage Sprint Classification Pipeline...")
        print(f"📊 Using {self.window}-minute analysis window")
        
        # Load full dataset
        features_df = self.load_full_dataset(archetype_results_path, data_dir)
        
        # Prepare stage data
        X, y_stage1, X_stage2, y_stage2 = self.prepare_stage_data(features_df)
        
        # Train Stage 1
        stage1_results = self.train_stage1_sprint_detector(X, y_stage1)
        
        # Train Stage 2
        stage2_results = self.train_stage2_marathon_classifier(X_stage2, y_stage2)
        
        # Run volatility analysis
        volatility_results = self.run_volatility_analysis(data_dir)
        
        # Save models
        self.save_models()
        
        # Generate report
        report = self.generate_report()
        print(f"\n{report}")
        
        return {
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'volatility_results': volatility_results,
            'report': report
        }

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Two-Stage Sprint Classification Pipeline")
    parser.add_argument('--window', type=int, default=10, choices=[5, 10], 
                       help='Analysis window in minutes (default: 10)')
    args = parser.parse_args()
    
    print("🚀 Two-Stage Sprint Classification Pipeline")
    print("=" * 50)
    print(f"📊 Analysis window: {args.window} minutes")
    
    # Paths
    results_dir = Path("../results/phase1_day9_10_archetypes")
    data_dir = Path("../../data/with_archetypes_fixed")
    
    # Find latest archetype results
    json_files = list(results_dir.glob("archetype_characterization_*.json"))
    if not json_files:
        print("❌ No archetype results found. Run Phase 1 first.")
        return
    
    latest_results = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using archetype results: {latest_results}")
    
    # Initialize and run pipeline
    classifier = TwoStageSprintClassifier(window=args.window)
    results = classifier.run_full_pipeline(latest_results, data_dir)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📊 Feature count: {args.window*6 + 8} total features")  # Approximate based on window
    print(f"🏆 Performance summary:")
    print(f"  Stage 1 F1: {results['stage1_results']['test_f1']:.4f} (Target: >0.65)")
    print(f"  Sprint Recall: {results['stage1_results']['test_recall']:.4f} (Target: >0.7)")
    print(f"  Marathon False Neg: {results['stage2_results']['false_negative_rate']:.4f} (Target: <0.25)")

if __name__ == "__main__":
    main()