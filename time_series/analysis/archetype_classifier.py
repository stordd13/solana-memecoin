#!/usr/bin/env python3
"""
HIERARCHICAL Archetype Classification Model

Creates a HIERARCHICAL classification model that predicts:
1. FIRST: Category (sprint/standard/marathon) 
2. THEN: Cluster (0-5) within that predicted category

IMPORTANT: This is NOT independent classification!
- marathon_cluster_1 ≠ standard_cluster_1 (completely different behaviors)
- Clusters only have meaning within their category context
- The model first determines category, then determines cluster within that category

Usage:
    python archetype_classifier.py --archetype-results PATH_TO_RESULTS --data-dir PATH_TO_DATA
    
Key Features:
- Uses first 5 minutes of price data for prediction
- Hierarchical approach: Category → Cluster (not independent)
- Handles 3-category scenario (sprint/standard/marathon)
- Works with both old and new directory structures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class ArchetypeClassifier:
    """Dual XGBoost classification model with Bayesian optimization."""
    
    def __init__(self, results_dir: Path = None):
        # Fix path resolution - ensure we save to time_series/results/
        if results_dir is None:
            # From analysis/ folder, go up one level to time_series/, then to results/
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = results_dir
        self.archetype_data = {}
        self.token_labels = {}
        self.category_model = None
        self.archetype_model = None
        self.cluster_models = {}  # Keep for backward compatibility
        self.scaler = StandardScaler()
        
        # Model optimization tracking
        self.category_best_params = None
        self.archetype_best_params = None
        self.optimization_results = {}
        
    def load_archetype_results(self, archetype_results_path: Path) -> None:
        """Load archetype characterization results."""
        print(f"📊 Loading archetype results from: {archetype_results_path}")
        
        with open(archetype_results_path, 'r') as f:
            results = json.load(f)
        
        self.archetype_data = results.get('archetype_data', {})
        
        # Create token-to-label mapping
        self.token_labels = {}
        for category, category_archetypes in self.archetype_data.items():
            for archetype_name, archetype_info in category_archetypes.items():
                cluster_id = archetype_info.get('cluster_id', 0)
                tokens = archetype_info.get('tokens', [])
                
                for token in tokens:
                    self.token_labels[token] = {
                        'category': category,
                        'cluster': cluster_id,
                        'archetype': archetype_name,
                        'archetype_full': archetype_name  # Full archetype name for 17-class model
                    }
        
        print(f"📈 Loaded labels for {len(self.token_labels)} tokens")
        print(f"📊 Categories: {list(self.archetype_data.keys())}")
        
    def load_token_data(self, data_dir: Path) -> Dict[str, pl.DataFrame]:
        """Load raw token data."""
        print(f"📁 Loading token data from: {data_dir}")
        
        all_tokens = {}
        
        # Try both directory structures: 
        # 1. New simplified structure (all files in one directory)
        # 2. Old structure (files in subdirectories)
        
        # Check if files are directly in data_dir (new structure)
        direct_files = list(data_dir.glob("*.parquet"))
        
        if direct_files:
            print(f"  📁 Using simplified directory structure")
            print(f"  📁 Loading tokens directly from: {data_dir}")
            
            for token_file in direct_files:
                token_name = token_file.stem
                if token_name in self.token_labels:  # Only load tokens we have labels for
                    try:
                        df = pl.read_parquet(token_file)
                        all_tokens[token_name] = df
                    except Exception as e:
                        print(f"    Warning: Failed to load {token_name}: {e}")
        else:
            # Fall back to old directory structure
            print(f"  📁 Using old directory structure with subdirectories")
            
            for category_name in ['dead_tokens', 'normal_behavior_tokens', 'tokens_with_extremes', 'tokens_with_gaps']:
                category_path = data_dir / category_name
                if category_path.exists():
                    print(f"  Loading {category_name} tokens...")
                    
                    token_files = list(category_path.glob("*.parquet"))
                    for token_file in token_files:
                        token_name = token_file.stem
                        if token_name in self.token_labels:  # Only load tokens we have labels for
                            try:
                                df = pl.read_parquet(token_file)
                                all_tokens[token_name] = df
                            except Exception as e:
                                print(f"    Warning: Failed to load {token_name}: {e}")
        
        print(f"📊 Loaded {len(all_tokens)} tokens with labels")
        return all_tokens
    
    def extract_features(self, token_data: Dict[str, pl.DataFrame], minutes: int = 5) -> pd.DataFrame:
        """Extract features from first N minutes of token data."""
        print(f"⚡ Extracting features from first {minutes} minutes...")
        
        features_list = []
        
        for token_name, df in token_data.items():
            if token_name not in self.token_labels:
                continue
                
            try:
                # Convert to pandas for easier manipulation
                pdf = df.to_pandas()
                
                # Take first N minutes
                if len(pdf) > minutes:
                    pdf = pdf.iloc[:minutes]
                elif len(pdf) < minutes:
                    # Skip tokens with insufficient data
                    continue
                
                # Calculate features
                prices = pdf['price'].values
                returns = pdf['price'].pct_change().dropna().values
                
                if len(returns) < 2:
                    continue
                
                # REAL-TIME SAFE FEATURES - NO DATA LEAKAGE
                # All features use only progressive/rolling calculations
                
                features = {
                    'token_name': token_name,
                    
                    # Progressive price features (safe - only current state)
                    'initial_price': prices[0] if len(prices) > 0 else 0,
                    'current_price': prices[-1] if len(prices) > 0 else 0,  # Last available price
                    
                    # Progressive return features (cumulative up to current minute)
                    'cumulative_return_1m': (prices[0] - prices[0]) / prices[0] if len(prices) >= 1 and prices[0] != 0 else 0,
                    'cumulative_return_2m': (prices[1] - prices[0]) / prices[0] if len(prices) >= 2 and prices[0] != 0 else 0,
                    'cumulative_return_3m': (prices[2] - prices[0]) / prices[0] if len(prices) >= 3 and prices[0] != 0 else 0,
                    'cumulative_return_4m': (prices[3] - prices[0]) / prices[0] if len(prices) >= 4 and prices[0] != 0 else 0,
                    'cumulative_return_5m': (prices[4] - prices[0]) / prices[0] if len(prices) >= 5 and prices[0] != 0 else 0,
                    
                    # Rolling volatility (expanding window - safe)
                    'rolling_volatility_1m': 0,  # Cannot calculate with 1 point
                    'rolling_volatility_2m': np.std(prices[:2]) / np.mean(prices[:2]) if len(prices) >= 2 and np.mean(prices[:2]) != 0 else 0,
                    'rolling_volatility_3m': np.std(prices[:3]) / np.mean(prices[:3]) if len(prices) >= 3 and np.mean(prices[:3]) != 0 else 0,
                    'rolling_volatility_4m': np.std(prices[:4]) / np.mean(prices[:4]) if len(prices) >= 4 and np.mean(prices[:4]) != 0 else 0,
                    'rolling_volatility_5m': np.std(prices[:5]) / np.mean(prices[:5]) if len(prices) >= 5 and np.mean(prices[:5]) != 0 else 0,
                    
                    # Momentum features (safe - only recent data)
                    'momentum_1min': returns[-1] if len(returns) >= 1 else 0,
                    'momentum_2min': np.mean(returns[-2:]) if len(returns) >= 2 else 0,
                    'momentum_3min': np.mean(returns[-3:]) if len(returns) >= 3 else 0,
                    
                    # Progressive trend features (expanding window - safe)
                    'trend_slope_2m': np.polyfit(range(2), prices[:2], 1)[0] if len(prices) >= 2 else 0,
                    'trend_slope_3m': np.polyfit(range(3), prices[:3], 1)[0] if len(prices) >= 3 else 0,
                    'trend_slope_4m': np.polyfit(range(4), prices[:4], 1)[0] if len(prices) >= 4 else 0,
                    'trend_slope_5m': np.polyfit(range(5), prices[:5], 1)[0] if len(prices) >= 5 else 0,
                    
                    # Progressive statistical features (expanding window - safe)
                    'returns_mean_2m': np.mean(returns[:1]) if len(returns) >= 1 else 0,  # 2min = 1 return
                    'returns_mean_3m': np.mean(returns[:2]) if len(returns) >= 2 else 0,  # 3min = 2 returns
                    'returns_mean_4m': np.mean(returns[:3]) if len(returns) >= 3 else 0,
                    'returns_mean_5m': np.mean(returns[:4]) if len(returns) >= 4 else 0,
                    
                    'returns_std_2m': 0,  # Cannot calculate std with 1 point
                    'returns_std_3m': np.std(returns[:2]) if len(returns) >= 2 else 0,
                    'returns_std_4m': np.std(returns[:3]) if len(returns) >= 3 else 0,
                    'returns_std_5m': np.std(returns[:4]) if len(returns) >= 4 else 0,
                    
                    # Labels
                    'category': self.token_labels[token_name]['category'],
                    'cluster': self.token_labels[token_name]['cluster'],
                    'archetype': self.token_labels[token_name]['archetype']
                }
                
                features_list.append(features)
                
            except Exception as e:
                print(f"    Warning: Failed to extract features for {token_name}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        print(f"✅ Extracted features for {len(features_df)} tokens")
        
        return features_df
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0
        
        # Calculate cumulative maximum
        cummax = np.maximum.accumulate(prices)
        # Calculate drawdown
        drawdown = (prices - cummax) / cummax
        # Return maximum drawdown (most negative)
        return np.min(drawdown)
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for models."""
        print(f"🔧 Preparing training data...")
        
        # Feature columns (exclude labels and token_name)
        feature_cols = [col for col in features_df.columns if col not in ['token_name', 'category', 'cluster', 'archetype']]
        
        X = features_df[feature_cols].values
        y_category = features_df['category'].values
        y_cluster = features_df['cluster'].values
        
        # Handle any NaN values
        X = np.nan_to_num(X)
        
        # Store original category labels for later use
        self.category_labels = sorted(np.unique(y_category))
        self.category_label_map = {label: i for i, label in enumerate(self.category_labels)}
        self.category_inverse_map = {i: label for i, label in enumerate(self.category_labels)}
        
        # Convert string categories to numeric for XGBoost
        y_category_numeric = np.array([self.category_label_map[cat] for cat in y_category])
        
        # Create archetype labels for 17-class model
        y_archetype = features_df['archetype'].values
        self.archetype_labels = sorted(np.unique(y_archetype))
        self.archetype_label_map = {label: i for i, label in enumerate(self.archetype_labels)}
        self.archetype_inverse_map = {i: label for i, label in enumerate(self.archetype_labels)}
        
        # Convert string archetypes to numeric for XGBoost
        y_archetype_numeric = np.array([self.archetype_label_map[arch] for arch in y_archetype])
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📈 Category distribution: {pd.Series(y_category).value_counts().to_dict()}")
        print(f"📈 Category mapping: {self.category_label_map}")
        print(f"📈 Archetype distribution: {pd.Series(y_archetype).value_counts().to_dict()}")
        print(f"📈 Total archetype classes: {len(self.archetype_labels)}")
        print(f"📈 Cluster distribution: {pd.Series(y_cluster).value_counts().to_dict()}")
        
        return X, y_category_numeric, y_cluster, y_archetype_numeric
    
    def train_models(self, X: np.ndarray, y_category: np.ndarray, y_cluster: np.ndarray, y_archetype: np.ndarray) -> Dict:
        """Train the dual XGBoost classification models with Bayesian optimization.
        
        IMPORTANT: This uses DUAL classification:
        1. Model 1: Category prediction (sprint/standard/marathon) - 3 classes
        2. Model 2: Archetype prediction (all 17 archetypes) - 17 classes
        
        Both models are optimized separately using Bayesian optimization.
        """
        print(f"🚀 Training DUAL XGBoost classification models with Bayesian optimization...")
        print(f"📊 Model 1: Category prediction (3 classes)")
        print(f"📊 Model 2: Archetype prediction ({len(self.archetype_labels)} classes)")
        
        # Split data - stratify by category to ensure balanced splits
        X_train, X_test, y_cat_train, y_cat_test, y_clust_train, y_clust_test, y_arch_train, y_arch_test = train_test_split(
            X, y_category, y_cluster, y_archetype, test_size=0.2, random_state=42, stratify=y_category
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train both models with Bayesian optimization
        print("\n🔍 Starting Bayesian optimization for both models...")
        category_results = self._train_category_model_with_optimization(X_train_scaled, X_test_scaled, y_cat_train, y_cat_test)
        archetype_results = self._train_archetype_model_with_optimization(X_train_scaled, X_test_scaled, y_arch_train, y_arch_test)
        
        results.update(category_results)
        results.update(archetype_results)
        
        return results
        
        # LEVEL 1: Train Category Model (Hierarchical Level 1)
        print("\n  🎯 LEVEL 1: Training Category model (sprint vs standard vs marathon)...")
        self.category_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0  # Suppress XGBoost warnings
        )
        
        # Handle class imbalance with sample weights
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_cat_train)
        self.category_model.fit(X_train_scaled, y_cat_train, sample_weight=sample_weights)
        
        # Evaluate category model
        cat_pred_numeric = self.category_model.predict(X_test_scaled)
        cat_accuracy = np.mean(cat_pred_numeric == y_cat_test)
        cat_cv_scores = cross_val_score(self.category_model, X_train_scaled, y_cat_train, cv=5, n_jobs=-1)
        
        # Convert numeric predictions back to string labels for reporting
        cat_pred_labels = [self.category_inverse_map[pred] for pred in cat_pred_numeric]
        y_cat_test_labels = [self.category_inverse_map[true] for true in y_cat_test]
        
        results['category'] = {
            'accuracy': cat_accuracy,
            'cv_scores': cat_cv_scores,
            'cv_mean': cat_cv_scores.mean(),
            'cv_std': cat_cv_scores.std(),
            'classification_report': classification_report(y_cat_test_labels, cat_pred_labels),
            'confusion_matrix': confusion_matrix(y_cat_test_labels, cat_pred_labels, labels=self.category_labels)
        }
        
        print(f"    ✅ Category accuracy: {cat_accuracy:.3f}")
        print(f"    ✅ Category CV: {cat_cv_scores.mean():.3f} (+/- {cat_cv_scores.std()*2:.3f})")
        
        # Display classification report with precision/recall
        print(f"\n    📊 Category Classification Report:")
        print(classification_report(y_cat_test_labels, cat_pred_labels))
        
        # Display confusion matrix
        print(f"    📊 Category Confusion Matrix:")
        cm = confusion_matrix(y_cat_test_labels, cat_pred_labels, labels=self.category_labels)
        print(f"    Predicted: {' '.join([f'{cat:>10}' for cat in self.category_labels])}")
        for i, cat in enumerate(self.category_labels):
            print(f"    {cat:>10}: {' '.join([f'{cm[i,j]:>10}' for j in range(len(self.category_labels))])}")
        
        # LEVEL 2: Train Cluster Models (Hierarchical Level 2) - one for each category
        print("\n  🎯 LEVEL 2: Training Cluster models (within each category)...")
        self.cluster_models = {}
        
        # Predict categories for the test set to evaluate hierarchical performance
        test_cat_predictions_numeric = self.category_model.predict(X_test_scaled)
        hierarchical_correct = 0
        hierarchical_total = 0
        
        # Convert numeric predictions back to string labels for cluster training
        test_cat_predictions = [self.category_inverse_map[pred] for pred in test_cat_predictions_numeric]
        
        for category in self.category_labels:
            print(f"\n    🔍 Training cluster model for '{category}' category...")
            
            # Filter TRAINING data for this category
            category_numeric = self.category_label_map[category]
            category_train_mask = y_cat_train == category_numeric
            if np.sum(category_train_mask) < 10:  # Skip if too few samples
                print(f"      ⚠️ Skipping {category} - too few training samples ({np.sum(category_train_mask)})")
                continue
            
            X_cat_train = X_train_scaled[category_train_mask]
            y_cat_clust_train = y_clust_train[category_train_mask]
            
            print(f"      📊 Training on {len(X_cat_train)} {category} samples")
            print(f"      📊 Clusters in {category}: {sorted(np.unique(y_cat_clust_train))}")
            
            # Train cluster model for this category
            cluster_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0  # Suppress XGBoost warnings
            )
            
            # Handle class imbalance with sample weights for cluster model
            cluster_sample_weights = compute_sample_weight('balanced', y_cat_clust_train)
            cluster_model.fit(X_cat_train, y_cat_clust_train, sample_weight=cluster_sample_weights)
            
            # Evaluate cluster model on TEST data from this category
            category_test_mask = y_cat_test == category_numeric
            if np.sum(category_test_mask) > 0:
                X_cat_test = X_test_scaled[category_test_mask]
                y_cat_clust_test = y_clust_test[category_test_mask]
                
                clust_pred = cluster_model.predict(X_cat_test)
                clust_accuracy = np.mean(clust_pred == y_cat_clust_test)
                
                # Cross-validation for cluster model
                cv_folds = min(5, len(np.unique(y_cat_clust_train)))
                clust_cv_scores = cross_val_score(cluster_model, X_cat_train, y_cat_clust_train, cv=cv_folds, n_jobs=-1)
                
                results[f'cluster_{category}'] = {
                    'accuracy': clust_accuracy,
                    'cv_scores': clust_cv_scores,
                    'cv_mean': clust_cv_scores.mean(),
                    'cv_std': clust_cv_scores.std(),
                    'classification_report': classification_report(y_cat_clust_test, clust_pred),
                    'confusion_matrix': confusion_matrix(y_cat_clust_test, clust_pred),
                    'train_samples': len(X_cat_train),
                    'test_samples': len(X_cat_test),
                    'clusters': sorted(np.unique(y_cat_clust_train))
                }
                
                print(f"      ✅ {category} cluster accuracy: {clust_accuracy:.3f}")
                print(f"      ✅ {category} cluster CV: {clust_cv_scores.mean():.3f} (+/- {clust_cv_scores.std()*2:.3f})")
                
                # Display cluster classification report
                print(f"\n      📊 {category.capitalize()} Cluster Classification Report:")
                print(classification_report(y_cat_clust_test, clust_pred))
                
                # Calculate hierarchical accuracy (correct category AND correct cluster)
                category_pred_mask = test_cat_predictions == category
                correct_category_predictions = category_test_mask & category_pred_mask
                if np.sum(correct_category_predictions) > 0:
                    hierarchical_correct += np.sum(clust_pred == y_cat_clust_test)
                    hierarchical_total += len(y_cat_clust_test)
            
            self.cluster_models[category] = cluster_model
        
        # Calculate overall hierarchical accuracy
        if hierarchical_total > 0:
            hierarchical_accuracy = hierarchical_correct / hierarchical_total
            results['hierarchical'] = {
                'accuracy': hierarchical_accuracy,
                'description': 'Accuracy of predicting BOTH category AND cluster correctly'
            }
            print(f"\n  🎯 HIERARCHICAL ACCURACY: {hierarchical_accuracy:.3f}")
            print(f"    (Both category AND cluster predictions correct)")
        
        return results
    
    def _train_category_model_with_optimization(self, X_train: np.ndarray, X_test: np.ndarray, 
                                               y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train category model with Bayesian optimization."""
        print(f"\n  🎯 MODEL 1: Training Category model with Bayesian optimization...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'eval_metric': 'logloss',
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            
            # Use cross-validation to evaluate F1 score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
            return cv_scores.mean()
        
        # Create study for category model
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        # Get best parameters
        self.category_best_params = study.best_params
        print(f"    ✅ Best category F1 score: {study.best_value:.4f}")
        print(f"    ✅ Best category parameters: {self.category_best_params}")
        
        # Train final model with best parameters
        best_params = self.category_best_params.copy()
        best_params.update({'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1, 'verbosity': 0})
        
        self.category_model = xgb.XGBClassifier(**best_params)
        
        # Handle class imbalance with sample weights
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)
        self.category_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate final model
        y_pred = self.category_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Convert numeric predictions back to string labels for reporting
        y_pred_labels = [self.category_inverse_map[pred] for pred in y_pred]
        y_test_labels = [self.category_inverse_map[true] for true in y_test]
        
        print(f"    ✅ Final category accuracy: {accuracy:.4f}")
        print(f"    ✅ Final category F1 score: {f1_macro:.4f}")
        
        return {
            'category': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'best_params': self.category_best_params,
                'optimization_score': study.best_value,
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'confusion_matrix': confusion_matrix(y_test_labels, y_pred_labels, labels=self.category_labels)
            }
        }
    
    def _train_archetype_model_with_optimization(self, X_train: np.ndarray, X_test: np.ndarray,
                                                y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train archetype model with Bayesian optimization."""
        print(f"\n  🎯 MODEL 2: Training Archetype model with Bayesian optimization...")
        print(f"    📊 Training on {len(self.archetype_labels)} archetype classes")
        
        def objective(trial):
            # Define hyperparameter search space (potentially different from category model)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.5),
                'random_state': 42,
                'eval_metric': 'logloss',
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            
            # Use cross-validation to evaluate F1 score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
            return cv_scores.mean()
        
        # Create study for archetype model
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=100, show_progress_bar=True)  # More trials for complex 17-class problem
        
        # Get best parameters
        self.archetype_best_params = study.best_params
        print(f"    ✅ Best archetype F1 score: {study.best_value:.4f}")
        print(f"    ✅ Best archetype parameters: {self.archetype_best_params}")
        
        # Train final model with best parameters
        best_params = self.archetype_best_params.copy()
        best_params.update({'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1, 'verbosity': 0})
        
        self.archetype_model = xgb.XGBClassifier(**best_params)
        
        # Handle class imbalance with sample weights
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)
        self.archetype_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate final model
        y_pred = self.archetype_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Convert numeric predictions back to string labels for reporting
        y_pred_labels = [self.archetype_inverse_map[pred] for pred in y_pred]
        y_test_labels = [self.archetype_inverse_map[true] for true in y_test]
        
        print(f"    ✅ Final archetype accuracy: {accuracy:.4f}")
        print(f"    ✅ Final archetype F1 score: {f1_macro:.4f}")
        
        return {
            'archetype': {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'best_params': self.archetype_best_params,
                'optimization_score': study.best_value,
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'confusion_matrix': confusion_matrix(y_test_labels, y_pred_labels, labels=self.archetype_labels)
            }
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make dual predictions using both category and archetype models.
        
        This implements DUAL prediction:
        1. Category model: Predicts category (sprint/standard/marathon)
        2. Archetype model: Predicts archetype (17 classes)
        
        Returns:
            category_pred: Array of category predictions (string labels)
            archetype_pred: Array of archetype predictions (string labels)
            cluster_pred: Array of cluster predictions (for backward compatibility)
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # MODEL 1: Predict category (3 classes)
        category_pred_numeric = self.category_model.predict(X_scaled)
        
        # Convert numeric predictions to string labels
        category_pred = np.array([self.category_inverse_map[pred] for pred in category_pred_numeric])
        
        # MODEL 2: Predict archetype (17 classes)
        archetype_pred_numeric = self.archetype_model.predict(X_scaled)
        archetype_pred = np.array([self.archetype_inverse_map[pred] for pred in archetype_pred_numeric])
        
        # Extract cluster numbers from archetype predictions for backward compatibility
        cluster_pred = np.zeros(len(X_scaled), dtype=int)
        for i, archetype in enumerate(archetype_pred):
            # Extract cluster ID from archetype name (e.g., "sprint_cluster_1" -> 1)
            if '_cluster_' in archetype:
                cluster_id = int(archetype.split('_cluster_')[1])
                cluster_pred[i] = cluster_id
        
        return category_pred, archetype_pred, cluster_pred
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make hierarchical predictions with confidence scores.
        
        Returns:
            category_pred: Array of category predictions
            cluster_pred: Array of cluster predictions  
            category_confidence: Array of category prediction confidence scores
            cluster_confidence: Array of cluster prediction confidence scores
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # LEVEL 1: Predict category with confidence
        category_pred_numeric = self.category_model.predict(X_scaled)
        category_pred = np.array([self.category_inverse_map[pred] for pred in category_pred_numeric])
        category_proba = self.category_model.predict_proba(X_scaled)
        category_confidence = np.max(category_proba, axis=1)
        
        # LEVEL 2: Predict cluster with confidence
        cluster_pred = np.zeros(len(X_scaled), dtype=int)
        cluster_confidence = np.zeros(len(X_scaled), dtype=float)
        
        # Group predictions by category
        unique_categories = np.unique(category_pred)
        
        for category in unique_categories:
            if category in self.cluster_models:
                category_mask = category_pred == category
                category_indices = np.where(category_mask)[0]
                
                if len(category_indices) > 0:
                    X_category = X_scaled[category_mask]
                    
                    # Predict clusters
                    cluster_predictions = self.cluster_models[category].predict(X_category)
                    cluster_proba = self.cluster_models[category].predict_proba(X_category)
                    cluster_conf = np.max(cluster_proba, axis=1)
                    
                    # Assign predictions
                    cluster_pred[category_indices] = cluster_predictions
                    cluster_confidence[category_indices] = cluster_conf
            else:
                # Default for unknown categories
                category_mask = category_pred == category
                cluster_pred[category_mask] = 0
                cluster_confidence[category_mask] = 0.0
        
        return category_pred, cluster_pred, category_confidence, cluster_confidence
    
    def save_models(self, output_dir: Path) -> None:
        """Save trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        
        # Save category model
        joblib.dump(self.category_model, output_dir / 'category_model.pkl')
        
        # Save archetype model
        joblib.dump(self.archetype_model, output_dir / 'archetype_model.pkl')
        
        # Save cluster models (for backward compatibility)
        for category, model in self.cluster_models.items():
            joblib.dump(model, output_dir / f'cluster_model_{category}.pkl')
        
        # Save optimization results and parameters
        optimization_results = {
            'category_best_params': self.category_best_params,
            'archetype_best_params': self.archetype_best_params,
            'category_labels': self.category_labels,
            'archetype_labels': self.archetype_labels,
            'category_label_map': self.category_label_map,
            'archetype_label_map': self.archetype_label_map
        }
        
        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        print(f"💾 Models saved to: {output_dir}")
        print(f"📊 Category model classes: {len(self.category_labels)}")
        print(f"📊 Archetype model classes: {len(self.archetype_labels)}")
    
    def load_models(self, output_dir: Path = None) -> None:
        """Load trained models."""
        if output_dir is None:
            output_dir = self.results_dir / "classification_models"
        
        # Check if models exist
        if not output_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {output_dir}")
        
        required_files = ['scaler.pkl', 'category_model.pkl', 'archetype_model.pkl', 'optimization_results.json']
        missing_files = [f for f in required_files if not (output_dir / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Load scaler
        self.scaler = joblib.load(output_dir / 'scaler.pkl')
        
        # Load category model
        self.category_model = joblib.load(output_dir / 'category_model.pkl')
        
        # Load archetype model
        self.archetype_model = joblib.load(output_dir / 'archetype_model.pkl')
        
        # Load optimization results and label mappings
        with open(output_dir / 'optimization_results.json', 'r') as f:
            optimization_results = json.load(f)
        
        self.category_best_params = optimization_results.get('category_best_params')
        self.archetype_best_params = optimization_results.get('archetype_best_params')
        self.category_labels = optimization_results.get('category_labels', [])
        self.archetype_labels = optimization_results.get('archetype_labels', [])
        self.category_label_map = optimization_results.get('category_label_map', {})
        self.archetype_label_map = optimization_results.get('archetype_label_map', {})
        
        # Create inverse mappings
        self.category_inverse_map = {v: k for k, v in self.category_label_map.items()}
        self.archetype_inverse_map = {v: k for k, v in self.archetype_label_map.items()}
        
        # Load cluster models (for backward compatibility)
        self.cluster_models = {}
        for model_file in output_dir.glob('cluster_model_*.pkl'):
            category = model_file.stem.replace('cluster_model_', '')
            self.cluster_models[category] = joblib.load(model_file)
        
        print(f"📦 Models loaded from: {output_dir}")
        print(f"📊 Category model classes: {len(self.category_labels)}")
        print(f"📊 Archetype model classes: {len(self.archetype_labels)}")
        print(f"🎯 Category best params: {self.category_best_params}")
        print(f"🎭 Archetype best params: {self.archetype_best_params}")
    
    def create_visualizations(self, results: Dict, features_df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create visualizations for model performance."""
        print(f"📊 Creating visualizations...")
        
        figures = {}
        
        # 1. Confusion Matrix for Category Model
        if 'category' in results and 'confusion_matrix' in results['category']:
            cm = results['category']['confusion_matrix']
            categories = self.category_labels  # Use actual categories from the model
            
            fig_confusion = go.Figure(data=go.Heatmap(
                z=cm,
                x=categories,
                y=categories,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
            ))
            
            fig_confusion.update_layout(
                title='Category Classification Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                yaxis={'autorange': 'reversed'}  # Flip y-axis to match sklearn convention
            )
            
            figures['confusion_matrix'] = fig_confusion
        
        # 2. Model Performance Comparison
        categories = []
        accuracies = []
        cv_means = []
        cv_stds = []
        
        for model_name, model_results in results.items():
            if 'accuracy' in model_results:  # Skip hierarchical result which has different structure
                categories.append(model_name)
                accuracies.append(model_results['accuracy'])
                cv_means.append(model_results.get('cv_mean', 0))
                cv_stds.append(model_results.get('cv_std', 0))
        
        fig_performance = go.Figure()
        
        fig_performance.add_trace(go.Bar(
            x=categories,
            y=accuracies,
            name='Test Accuracy',
            marker_color='blue'
        ))
        
        fig_performance.add_trace(go.Bar(
            x=categories,
            y=cv_means,
            name='CV Mean',
            marker_color='green',
            error_y=dict(type='data', array=cv_stds)
        ))
        
        fig_performance.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            barmode='group'
        )
        
        figures['performance'] = fig_performance
        
        # 3. Precision/Recall Chart for Category Model
        if 'category' in results and 'classification_report' in results['category']:
            # Parse classification report to extract precision/recall
            report_lines = results['category']['classification_report'].split('\n')
            precision_recall_data = []
            
            for line in report_lines:
                if 'marathon' in line or 'standard' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        label = parts[0]
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1 = float(parts[3])
                        precision_recall_data.append({
                            'Category': label,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1
                        })
            
            if precision_recall_data:
                import pandas as pd
                pr_df = pd.DataFrame(precision_recall_data)
                
                fig_pr = go.Figure()
                
                fig_pr.add_trace(go.Bar(
                    x=pr_df['Category'],
                    y=pr_df['Precision'],
                    name='Precision',
                    marker_color='lightblue'
                ))
                
                fig_pr.add_trace(go.Bar(
                    x=pr_df['Category'],
                    y=pr_df['Recall'],
                    name='Recall',
                    marker_color='lightcoral'
                ))
                
                fig_pr.add_trace(go.Bar(
                    x=pr_df['Category'],
                    y=pr_df['F1-Score'],
                    name='F1-Score',
                    marker_color='lightgreen'
                ))
                
                fig_pr.update_layout(
                    title='Precision, Recall, and F1-Score by Category',
                    xaxis_title='Category',
                    yaxis_title='Score',
                    barmode='group',
                    yaxis={'range': [0, 1]}
                )
                
                figures['precision_recall'] = fig_pr
        
        # 4. Feature Importance (for category model - XGBoost)
        if self.category_model:
            feature_cols = [col for col in features_df.columns if col not in ['token_name', 'category', 'cluster', 'archetype']]
            importances = self.category_model.feature_importances_
            
            fig_importance = go.Figure()
            
            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1]
            
            fig_importance.add_trace(go.Bar(
                x=[feature_cols[i] for i in sorted_idx[:15]],  # Top 15 features
                y=importances[sorted_idx[:15]],
                marker_color='orange',
                hovertemplate='<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>'
            ))
            
            fig_importance.update_layout(
                title='XGBoost Feature Importance (Category Model)',
                xaxis_title='Features',
                yaxis_title='XGBoost Importance Score',
                xaxis={'tickangle': 45},
                annotations=[
                    dict(
                        text="Based on XGBoost gain metric",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98, xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="gray")
                    )
                ]
            )
            
            figures['feature_importance'] = fig_importance
        
        # 5. Category Distribution
        fig_category = go.Figure()
        
        category_counts = features_df['category'].value_counts()
        
        fig_category.add_trace(go.Bar(
            x=category_counts.index,
            y=category_counts.values,
            marker_color='purple'
        ))
        
        fig_category.update_layout(
            title='Category Distribution',
            xaxis_title='Category',
            yaxis_title='Count'
        )
        
        figures['category_distribution'] = fig_category
        
        # 6. Cluster Distribution by Category
        fig_cluster = make_subplots(
            rows=1, cols=len(features_df['category'].unique()),
            subplot_titles=list(features_df['category'].unique()),
            specs=[[{"secondary_y": False}] * len(features_df['category'].unique())]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, category in enumerate(features_df['category'].unique()):
            category_data = features_df[features_df['category'] == category]
            cluster_counts = category_data['cluster'].value_counts().sort_index()
            
            fig_cluster.add_trace(
                go.Bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    name=f'{category}',
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig_cluster.update_layout(title='Cluster Distribution by Category')
        
        figures['cluster_distribution'] = fig_cluster
        
        return figures
    
    def run_training(self, archetype_results_path: Path, data_dir: Path, minutes: int = 5) -> Dict:
        """Run complete training pipeline."""
        print(f"🚀 Starting Dual XGBoost Classification Training")
        
        # Load data
        self.load_archetype_results(archetype_results_path)
        token_data = self.load_token_data(data_dir)
        
        # Extract features
        features_df = self.extract_features(token_data, minutes)
        
        # Prepare training data
        X, y_category, y_cluster, y_archetype = self.prepare_training_data(features_df)
        
        # Train models
        results = self.train_models(X, y_category, y_cluster, y_archetype)
        
        # Save models
        models_dir = self.results_dir / "classification_models"
        self.save_models(models_dir)
        
        # Create visualizations
        figures = self.create_visualizations(results, features_df)
        
        # Save visualizations
        viz_dir = self.results_dir / "classification_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for fig_name, fig in figures.items():
            fig_path = viz_dir / f"{fig_name}.html"
            fig.write_html(fig_path)
        
        print(f"📊 Training complete!")
        print(f"📈 Models saved to: {models_dir}")
        print(f"📋 Visualizations saved to: {viz_dir}")
        
        return {
            'results': results,
            'features_df': features_df,
            'models_dir': models_dir,
            'viz_dir': viz_dir
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Level Archetype Classification")
    parser.add_argument("--archetype-results", type=Path,
                       help="Path to archetype characterization results JSON")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("../../data/processed"),
                       help="Path to processed token data directory")
    parser.add_argument("--minutes", type=int, default=5,
                       help="Number of minutes to use for feature extraction")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Find latest results if not specified
    if not args.archetype_results:
        results_dir = Path("../results/phase1_day9_10_archetypes")
        if results_dir.exists():
            json_files = list(results_dir.glob("archetype_characterization_*.json"))
            if json_files:
                args.archetype_results = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest results: {args.archetype_results}")
            else:
                print("❌ No archetype results found. Run the Phase 1 pipeline first.")
                return
        else:
            print("❌ Results directory not found. Run the Phase 1 pipeline first.")
            return
    
    # Initialize classifier
    classifier = ArchetypeClassifier(args.output_dir)
    
    try:
        # Run training
        training_results = classifier.run_training(
            args.archetype_results,
            args.data_dir,
            args.minutes
        )
        
        # Print summary
        print(f"\n📊 TRAINING SUMMARY")
        print(f"=" * 60)
        
        for model_name, model_results in training_results['results'].items():
            print(f"{model_name}:")
            print(f"  Accuracy: {model_results['accuracy']:.3f}")
            if 'cv_mean' in model_results:
                print(f"  CV Score: {model_results['cv_mean']:.3f} (+/- {model_results['cv_std']*2:.3f})")
            else:
                print(f"  Description: {model_results.get('description', 'N/A')}")
        
        print(f"\n🎉 Training complete!")
        print(f"📈 Feature extraction window: {args.minutes} minutes")
        print(f"📊 Total tokens processed: {len(training_results['features_df'])}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()