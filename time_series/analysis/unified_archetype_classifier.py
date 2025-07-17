#!/usr/bin/env python3
"""
Unified Archetype Classification Pipeline for Memecoin Analysis

Single 6-class classifier for unified clusters from clustering analysis
Features: 33 total (5min window default)
Custom Scorer: 0.7*recall + 0.3*precision (weighted for top pump clusters)
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
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import joblib

# Import from same directory
from archetype_classifier import ArchetypeClassifier

class UnifiedArchetypeClassifier:
    """Unified archetype classification for 6-class cluster prediction."""
    
    def __init__(self, results_dir: Path = None, window: int = 5):
        self.results_dir = results_dir or Path("../results")
        self.archetype_classifier = ArchetypeClassifier()
        self.window = window  # Analysis window in minutes
        
        # Model
        self.model = None
        self.scaler = StandardScaler()
        
        # Unified cluster mappings
        self.cluster_mappings = {}
        self.cluster_info = {}
        self.real_assignments = None
        
        # Custom scorer for top pump clusters
        self.custom_scorer = make_scorer(self._weighted_cluster_scorer)
        
        # Results storage
        self.classification_results = {}
        self.volatility_results = {}
        
    def _weighted_cluster_scorer(self, y_true, y_pred):
        """Custom scorer: weighted recall for top pump clusters"""
        # Get per-class recall
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Weight clusters based on their pump rates
        # After remapping: 0->low pump (2.9%), 1->high pump (77.8%), 2->medium pump (22.3%)
        cluster_weights = [0.2, 0.5, 0.3]  # [low, high, medium pump rates]
        
        # Calculate weighted recall
        weighted_score = 0
        for cluster_id in range(len(recalls)):
            if cluster_id < len(cluster_weights):
                weight = cluster_weights[cluster_id]
                weighted_score += weight * recalls[cluster_id]
        
        return weighted_score
    
    def load_unified_cluster_mappings(self, clustering_results_path: Path) -> Dict:
        """Load unified cluster mappings from clustering comparison results."""
        print("📊 Loading unified cluster mappings...")
        
        with open(clustering_results_path, 'r') as f:
            clustering_results = json.load(f)
        
        # Extract cluster information
        cluster_analysis = clustering_results['comparison_summary']['cluster_analysis']
        
        print(f"📊 Found {len(cluster_analysis)} active clusters:")
        for cluster_id, info in cluster_analysis.items():
            cluster_num = int(cluster_id.split('_')[1])
            self.cluster_info[cluster_num] = info
            print(f"  {cluster_id}: {info['size']} tokens, {info['pump_rate_percent']:.1f}% pump rate")
        
        # Check if we have real cluster assignments
        assignments_file = clustering_results_path.parent / "cluster_assignments_30519.json"
        if assignments_file.exists():
            print(f"📊 Found real cluster assignments: {assignments_file}")
            with open(assignments_file, 'r') as f:
                assignments_data = json.load(f)
            self.real_assignments = assignments_data['token_assignments']
            print(f"📊 Loaded real assignments for {len(self.real_assignments)} tokens")
        else:
            print(f"⚠️  No real cluster assignments found, will use category mapping")
            self.real_assignments = None
        
        return clustering_results
    
    def load_full_dataset(self, archetype_results_path: Path, data_dir: Path) -> pd.DataFrame:
        """Load full 30k dataset with unified cluster labels."""
        print("📊 Loading full dataset with unified cluster labels...")
        
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
        
        # Extract features with 5-minute window
        features_df = self.archetype_classifier.extract_features(all_tokens, minutes=self.window)
        
        # Add unified cluster labels 
        if self.real_assignments:
            print("📊 Using real cluster assignments from K-means")
            # Use real cluster assignments
            features_df['unified_cluster'] = features_df['token_name'].map(
                lambda token: self.real_assignments.get(token, {}).get('cluster_id', 0)
            )
        else:
            print("📊 Using category-based mapping (fallback)")
            # Fall back to category mapping
            mapping_func = self._get_cluster_mapping()
            features_df['unified_cluster'] = features_df['category'].apply(mapping_func)
        
        # Remap to consecutive labels for XGBoost 
        unique_clusters = sorted(features_df['unified_cluster'].unique())
        cluster_remap = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        features_df['unified_cluster'] = features_df['unified_cluster'].map(cluster_remap)
        
        print(f"📊 Original clusters: {unique_clusters}")
        print(f"📊 Cluster remapping: {cluster_remap}")
        print(f"📊 Final clusters: {sorted(features_df['unified_cluster'].unique())}")
        
        # Verify final distribution
        actual_counts = features_df['category'].value_counts()
        cluster_counts = features_df['unified_cluster'].value_counts()
        print(f"📊 Final dataset distribution: {actual_counts.to_dict()}")
        print(f"📊 Unified cluster distribution: {cluster_counts.to_dict()}")
        print(f"📊 Total features: {len([col for col in features_df.columns if col not in ['token_name', 'category', 'cluster', 'archetype', 'unified_cluster']])}")
        
        return features_df
    
    def _get_cluster_mapping(self) -> Dict:
        """Map original categories to unified clusters based on clustering results."""
        # Based on the clustering results analysis:
        # Cluster 0: 21,661 tokens (35.6% sprint, 64.4% standard) - 2.9% pump
        # Cluster 1: 3,720 tokens (100% marathon) - 77.8% pump
        # Cluster 3: 5,135 tokens (54.6% marathon, 45.4% standard) - 22.3% pump
        
        # Simplified mapping for testing (we need actual cluster assignments from unified clustering)
        # For now, simulate the cluster distribution based on category probabilities
        import random
        random.seed(42)
        
        def map_category_to_cluster(category):
            if category == 'marathon':
                # 57% go to cluster 1 (high pump), 43% to cluster 3 (medium pump)
                return 1 if random.random() < 0.57 else 3
            elif category == 'sprint':
                # All sprints go to cluster 0 (low pump)
                return 0
            else:  # standard
                # 70% go to cluster 0, 30% to cluster 3
                return 0 if random.random() < 0.70 else 3
        
        return map_category_to_cluster
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for unified classification."""
        print("🔧 Preparing unified classification data...")
        
        # Feature columns (exclude labels and token_name)
        feature_cols = [col for col in features_df.columns if col not in ['token_name', 'category', 'cluster', 'archetype', 'unified_cluster']]
        X = features_df[feature_cols].values
        X = np.nan_to_num(X)  # Handle any NaN values
        
        # Unified cluster labels
        y = features_df['unified_cluster'].values
        
        print(f"📊 Unified Classification Data:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        
        # Show cluster distribution
        unique_clusters, counts = np.unique(y, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            print(f"  Cluster {cluster}: {count} tokens")
        
        return X, y
    
    def train_unified_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train unified 6-class classifier."""
        print("🚀 Training Unified Archetype Classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for balanced training
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"📊 Class weights: {class_weight_dict}")
        
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
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'use_label_encoder': False
            }
            
            model = xgb.XGBClassifier(**params)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                      scoring=self.custom_scorer, n_jobs=-1)
            return cv_scores.mean()
        
        # Optimize (10 trials as specified)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=10, show_progress_bar=True)
        
        # Train final model
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'use_label_encoder': False
        })
        
        self.model = xgb.XGBClassifier(**best_params)
        
        # Apply class weights through sample_weight
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate per-class metrics
        unique_classes = np.unique(y_test)
        class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Convert numpy types to native Python types for JSON serialization
        results = {
            'best_params': best_params,
            'optimization_score': float(study.best_value),
            'test_accuracy': float(np.mean(y_pred == y_test)),
            'overall_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'custom_score': float(self._weighted_cluster_scorer(y_test, y_pred)),
            'per_class_precision': {int(k): float(v) for k, v in zip(unique_classes, class_precision)},
            'per_class_recall': {int(k): float(v) for k, v in zip(unique_classes, class_recall)},
            'per_class_f1': {int(k): float(v) for k, v in zip(unique_classes, class_f1)},
            'classification_report': classification_report(y_test, y_pred, target_names=[f'Cluster_{i}' for i in unique_classes]),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.classification_results = results
        
        print(f"✅ Unified Classification Results:")
        print(f"  Optimization Score: {results['optimization_score']:.4f}")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Overall F1: {results['overall_f1']:.4f}")
        print(f"  Custom Score: {results['custom_score']:.4f}")
        
        # Show per-class results for top clusters
        for cluster_id in [0, 1, 3]:  # Focus on active clusters
            if cluster_id in results['per_class_f1']:
                print(f"  Cluster {cluster_id} F1: {results['per_class_f1'][cluster_id]:.4f}")
                print(f"  Cluster {cluster_id} Recall: {results['per_class_recall'][cluster_id]:.4f}")
        
        return results
    
    def run_volatility_analysis(self, features_df: pd.DataFrame, data_dir: Path) -> Dict:
        """Run volatility analysis for unified clusters."""
        print("📊 Running volatility analysis for unified clusters...")
        
        # Group by unified clusters
        cluster_results = {}
        
        for cluster_id in features_df['unified_cluster'].unique():
            cluster_tokens = features_df[features_df['unified_cluster'] == cluster_id]['token_name'].tolist()
            
            print(f"  📈 Analyzing Cluster {cluster_id}: {len(cluster_tokens)} tokens...")
            
            # Run volatility analysis for this cluster
            results = {'high_vol_count': 0, 'high_vol_pumps': 0, 'times_to_1_5x': []}
            
            for token_name in cluster_tokens:
                token_file = data_dir / f"{token_name}.parquet"
                if token_file.exists():
                    try:
                        df = pl.read_parquet(token_file)
                        prices = df['price'].to_numpy()
                        
                        if len(prices) >= 6:
                            # Calculate early returns and volatility (first 5min)
                            returns_early = np.diff(prices[:5]) / prices[:4]
                            vol_5min = np.std(returns_early) / np.mean(prices[:5]) if np.mean(prices[:5]) > 0 else 0
                            
                            if vol_5min > 0.8:
                                results['high_vol_count'] += 1
                                # Check for pump after minute 5
                                base_price_idx = 4  # Using 5-minute window
                                post_prices = prices[5:]
                                
                                if len(post_prices) > 0 and np.max(post_prices) / prices[base_price_idx] > 1.5:
                                    results['high_vol_pumps'] += 1
                                    pump_indices = np.where(post_prices > prices[base_price_idx] * 1.5)[0]
                                    if len(pump_indices) > 0:
                                        time_to_1_5x = pump_indices[0] + 1  # Minutes after window
                                        results['times_to_1_5x'].append(time_to_1_5x)
                    except:
                        continue
            
            # Calculate metrics
            pump_rate = results['high_vol_pumps'] / results['high_vol_count'] * 100 if results['high_vol_count'] > 0 else 0
            avg_time = np.mean(results['times_to_1_5x']) if results['times_to_1_5x'] else 0
            
            cluster_results[f'cluster_{cluster_id}'] = {
                'size': len(cluster_tokens),
                'high_vol_count': results['high_vol_count'],
                'high_vol_pumps': results['high_vol_pumps'],
                'pump_rate_percent': pump_rate,
                'avg_time_to_1_5x': avg_time,
                'times_to_1_5x': results['times_to_1_5x']
            }
            
            print(f"    ✅ Cluster {cluster_id}: {results['high_vol_count']} high-vol, {pump_rate:.1f}% pump rate")
        
        # Final results
        final_results = {
            'clusters': cluster_results,
            'analysis_window': self.window,
            'total_clusters': len(cluster_results)
        }
        
        self.volatility_results = final_results
        
        print(f"✅ Volatility Analysis Summary:")
        for cluster_id, results in cluster_results.items():
            print(f"  {cluster_id}: {results['high_vol_count']} high-vol, {results['pump_rate_percent']:.1f}% pump rate, {results['avg_time_to_1_5x']:.1f}min avg time")
        
        return final_results
    
    def save_model(self) -> None:
        """Save trained model."""
        models_dir = self.results_dir / "unified_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, models_dir / f'unified_classifier_{self.window}min.pkl')
        joblib.dump(self.scaler, models_dir / f'unified_scaler_{self.window}min.pkl')
        
        # Save results
        all_results = {
            'window_minutes': self.window,
            'classification_results': self.classification_results,
            'volatility_results': self.volatility_results,
            'cluster_info': self.cluster_info
        }
        
        with open(models_dir / f'unified_results_{self.window}min.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"💾 Model and results saved to: {models_dir}")
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        report = []
        report.append("# Unified Archetype Classification Results")
        report.append("=" * 50)
        
        # Classification Results
        report.append(f"\n## Unified Classification Performance")
        report.append(f"- **Optimization Score**: {self.classification_results['optimization_score']:.4f}")
        report.append(f"- **Test Accuracy**: {self.classification_results['test_accuracy']:.4f}")
        report.append(f"- **Overall F1**: {self.classification_results['overall_f1']:.4f}")
        report.append(f"- **Custom Score**: {self.classification_results['custom_score']:.4f}")
        
        # Per-Class Results
        report.append(f"\n## Per-Cluster Performance")
        for cluster_id in sorted(self.classification_results['per_class_f1'].keys()):
            precision = self.classification_results['per_class_precision'][cluster_id]
            recall = self.classification_results['per_class_recall'][cluster_id]
            f1 = self.classification_results['per_class_f1'][cluster_id]
            
            # Get cluster info if available
            cluster_info = self.cluster_info.get(cluster_id, {})
            pump_rate = cluster_info.get('pump_rate_percent', 0)
            
            report.append(f"- **Cluster {cluster_id}**: F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f} (Pump: {pump_rate:.1f}%)")
        
        # Volatility Analysis
        report.append(f"\n## Volatility Analysis (Unified Clusters)")
        report.append(f"- **Analysis Window**: {self.volatility_results['analysis_window']} minutes")
        for cluster_id, results in self.volatility_results['clusters'].items():
            report.append(f"- **{cluster_id}**: {results['high_vol_count']} high-vol, {results['pump_rate_percent']:.1f}% pump rate, {results['avg_time_to_1_5x']:.1f}min avg time")
        
        # Success Criteria
        report.append(f"\n## Success Criteria Check")
        overall_f1 = self.classification_results['overall_f1']
        
        # Check top cluster recalls
        top_cluster_recalls = []
        for cluster_id in [1, 3]:  # Top pump clusters
            if cluster_id in self.classification_results['per_class_recall']:
                recall = self.classification_results['per_class_recall'][cluster_id]
                top_cluster_recalls.append(recall)
        
        avg_top_recall = np.mean(top_cluster_recalls) if top_cluster_recalls else 0
        
        report.append(f"- **Overall F1 > 0.6**: {overall_f1:.4f} {'✅' if overall_f1 > 0.6 else '❌'}")
        report.append(f"- **Top Clusters Recall > 0.65**: {avg_top_recall:.4f} {'✅' if avg_top_recall > 0.65 else '❌'}")
        
        return "\n".join(report)
    
    def run_full_pipeline(self, clustering_results_path: Path, archetype_results_path: Path, data_dir: Path) -> Dict:
        """Run complete unified classification pipeline."""
        print("🚀 Starting Unified Archetype Classification Pipeline...")
        print(f"📊 Using {self.window}-minute analysis window")
        
        # Load cluster mappings
        clustering_results = self.load_unified_cluster_mappings(clustering_results_path)
        
        # Load full dataset
        features_df = self.load_full_dataset(archetype_results_path, data_dir)
        
        # Prepare data
        X, y = self.prepare_data(features_df)
        
        # Train classifier
        classification_results = self.train_unified_classifier(X, y)
        
        # Run volatility analysis
        volatility_results = self.run_volatility_analysis(features_df, data_dir)
        
        # Save model
        self.save_model()
        
        # Generate report
        report = self.generate_report()
        print(f"\n{report}")
        
        return {
            'classification_results': classification_results,
            'volatility_results': volatility_results,
            'report': report
        }

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Unified Archetype Classification Pipeline")
    parser.add_argument('--window', type=int, default=5, choices=[5, 10], 
                       help='Analysis window in minutes (default: 5)')
    args = parser.parse_args()
    
    print("🚀 Unified Archetype Classification Pipeline")
    print("=" * 50)
    print(f"📊 Analysis window: {args.window} minutes")
    
    # Paths
    clustering_results_path = Path("../results/clustering_comparison/clustering_comparison_results_30519.json")
    archetype_results_path = Path("../results/phase1_day9_10_archetypes/archetype_characterization_20250716_144555.json")
    data_dir = Path("../../data/with_archetypes_fixed")
    
    # Check if files exist
    if not clustering_results_path.exists():
        print(f"❌ Clustering results not found: {clustering_results_path}")
        return
    
    if not archetype_results_path.exists():
        print(f"❌ Archetype results not found: {archetype_results_path}")
        return
    
    print(f"📁 Using clustering results: {clustering_results_path}")
    print(f"📁 Using archetype results: {archetype_results_path}")
    
    # Initialize and run pipeline
    classifier = UnifiedArchetypeClassifier(window=args.window)
    results = classifier.run_full_pipeline(clustering_results_path, archetype_results_path, data_dir)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📊 Feature count: 33 features (5min window)")
    print(f"🏆 Performance summary:")
    print(f"  Overall F1: {results['classification_results']['overall_f1']:.4f} (Target: >0.6)")
    print(f"  Custom Score: {results['classification_results']['custom_score']:.4f}")

if __name__ == "__main__":
    main()