#!/usr/bin/env python3
"""
XGBoost Misclassification Analysis

Analyzes how well the XGBoost classifier predicts marathon vs standard categories
compared to the true lifespan-based categories and archetype assignments.

Usage:
    python xgboost_misclassification_analyzer.py [--data-dir PATH] [--archetype-results PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our XGBoost classifier
from analysis.archetype_classifier import ArchetypeClassifier


class XGBoostMisclassificationAnalyzer:
    """Analyzes XGBoost classifier misclassification patterns."""
    
    def __init__(self, results_dir: Path = None):
        # Fix path resolution - ensure we save to time_series/results/
        if results_dir is None:
            # From analysis/ folder, go up one level to time_series/, then to results/
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = results_dir
        self.classifier = ArchetypeClassifier(results_dir=self.results_dir)
        self.token_data = {}
        self.predictions = {}
        self.misclassification_analysis = {}
        
    def load_and_predict(self, archetype_results_path: Path, data_dir: Path) -> None:
        """Load pre-trained models and run XGBoost predictions."""
        print(f"🔥 Loading pre-trained XGBoost models...")
        
        # Try to load pre-trained models first
        try:
            self.classifier.load_models()
            print(f"✅ Pre-trained XGBoost models loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠️ Pre-trained models not found: {e}")
            print(f"🔄 Training models from scratch...")
            
            # Fallback to training if models don't exist
            training_results = self.classifier.run_training(
                archetype_results_path,
                data_dir,
                minutes=5
            )
            print(f"✅ XGBoost classifier trained successfully")
        
        # Load archetype results for token mapping (must be done before loading tokens)
        if not hasattr(self.classifier, 'archetype_data') or not self.classifier.archetype_data:
            self.classifier.load_archetype_results(archetype_results_path)
        
        # Load all token data to make predictions
        self._load_all_tokens(data_dir)
        
        # Make predictions on all tokens
        self._predict_all_tokens()
        
    def _load_all_tokens(self, data_dir: Path) -> None:
        """Load all token data for prediction."""
        print(f"📁 Loading all token data from: {data_dir}")
        
        # Check if files are directly in data_dir (simplified structure)
        direct_files = list(data_dir.glob("*.parquet"))
        
        if direct_files:
            print(f"  📁 Loading from simplified directory structure")
            
            loaded_count = 0
            for i, token_file in enumerate(direct_files):
                if i >= 5000:  # Limit to 5000 tokens for performance
                    break
                    
                token_name = token_file.stem
                try:
                    df = pl.read_parquet(token_file)
                    self.token_data[token_name] = df
                    loaded_count += 1
                    
                    if loaded_count % 1000 == 0:
                        print(f"    Loaded {loaded_count} tokens...")
                        
                except Exception as e:
                    continue  # Skip failed loads
                    
            print(f"📊 Loaded {loaded_count} token files for prediction")
        else:
            print(f"❌ No token files found in {data_dir}")
    
    def _predict_all_tokens(self) -> None:
        """Make predictions on all loaded tokens."""
        print(f"🔮 Making XGBoost predictions on {len(self.token_data)} tokens...")
        
        prediction_results = []
        
        for i, (token_name, df) in enumerate(self.token_data.items()):
            try:
                # Convert to pandas and extract features (same as classifier)
                pdf = df.to_pandas()
                
                if len(pdf) < 5:  # Need at least 5 minutes
                    continue
                
                # Calculate true category based on lifespan (3-category system)
                lifespan_minutes = len(pdf)
                if lifespan_minutes >= 1200:
                    true_category = 'marathon'
                elif lifespan_minutes >= 400:
                    true_category = 'standard'
                else:  # 0-399 minutes
                    true_category = 'sprint'
                
                # Get archetype assignment if available
                archetype_category = 'unknown'
                archetype_cluster = -1
                archetype_name = 'unknown'
                
                # Check if token has archetype assignment
                for category, category_archetypes in self.classifier.archetype_data.items():
                    for arch_name, arch_info in category_archetypes.items():
                        if token_name in arch_info.get('tokens', []):
                            archetype_category = category
                            archetype_cluster = arch_info.get('cluster_id', -1)
                            archetype_name = arch_name
                            break
                    if archetype_category != 'unknown':
                        break
                
                # Extract features (first 5 minutes)
                features_dict = self._extract_token_features(pdf)
                
                if features_dict is None:
                    continue
                
                # Convert to feature array for prediction
                feature_cols = [col for col in features_dict.keys() if col not in ['token_name']]
                X = np.array([[features_dict[col] for col in feature_cols]])
                
                # Make prediction with dual model
                try:
                    predicted_category, predicted_archetype, predicted_cluster = self.classifier.predict(X)
                    predicted_category = predicted_category[0]
                    predicted_archetype = predicted_archetype[0]
                    predicted_cluster = predicted_cluster[0]
                    
                    # Calculate prediction confidence (for backward compatibility)
                    try:
                        _, _, category_confidence, cluster_confidence = self.classifier.predict_with_confidence(X)
                        category_conf = category_confidence[0]
                        cluster_conf = cluster_confidence[0]
                    except:
                        # If confidence prediction fails, use placeholder values
                        category_conf = 0.0
                        cluster_conf = 0.0
                    
                except Exception as pred_error:
                    # Skip tokens that cause prediction errors
                    continue
                
                # Calculate basic metrics
                prices = pdf['price'].values
                returns = pdf['price'].pct_change().dropna().values
                volatility_cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                total_return = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
                
                prediction_results.append({
                    'token_name': token_name,
                    'lifespan_minutes': lifespan_minutes,
                    'true_category': true_category,
                    'predicted_category': predicted_category,
                    'predicted_archetype': predicted_archetype,
                    'predicted_cluster': predicted_cluster,
                    'category_confidence': category_conf,
                    'cluster_confidence': cluster_conf,
                    'archetype_category': archetype_category,
                    'archetype_cluster': archetype_cluster,
                    'archetype_name': archetype_name,
                    'volatility_cv': volatility_cv,
                    'total_return': total_return,
                    'misclassified': true_category != predicted_category,
                    'has_archetype': archetype_category != 'unknown'
                })
                
                if len(prediction_results) % 500 == 0:
                    print(f"    Processed {len(prediction_results)} predictions...")
                
            except Exception as e:
                continue  # Skip tokens that cause errors
        
        self.predictions = pd.DataFrame(prediction_results)
        print(f"✅ Generated {len(self.predictions)} predictions")
    
    def _extract_token_features(self, pdf: pd.DataFrame) -> Optional[Dict]:
        """Extract features for a single token (matches classifier logic)."""
        if len(pdf) < 5:
            return None
        
        try:
            # Take first 5 minutes
            pdf_5min = pdf.iloc[:5]
            
            # Calculate features
            prices = pdf_5min['price'].values
            returns = pdf_5min['price'].pct_change().dropna().values
            
            if len(returns) < 2:
                return None
            
            # REAL-TIME SAFE FEATURES - NO DATA LEAKAGE (same as classifier)
            features = {
                'initial_price': prices[0] if len(prices) > 0 else 0,
                'current_price': prices[-1] if len(prices) > 0 else 0,
                
                # Progressive return features
                'cumulative_return_1m': (prices[0] - prices[0]) / prices[0] if len(prices) >= 1 and prices[0] != 0 else 0,
                'cumulative_return_2m': (prices[1] - prices[0]) / prices[0] if len(prices) >= 2 and prices[0] != 0 else 0,
                'cumulative_return_3m': (prices[2] - prices[0]) / prices[0] if len(prices) >= 3 and prices[0] != 0 else 0,
                'cumulative_return_4m': (prices[3] - prices[0]) / prices[0] if len(prices) >= 4 and prices[0] != 0 else 0,
                'cumulative_return_5m': (prices[4] - prices[0]) / prices[0] if len(prices) >= 5 and prices[0] != 0 else 0,
                
                # Rolling volatility
                'rolling_volatility_1m': 0,
                'rolling_volatility_2m': np.std(prices[:2]) / np.mean(prices[:2]) if len(prices) >= 2 and np.mean(prices[:2]) != 0 else 0,
                'rolling_volatility_3m': np.std(prices[:3]) / np.mean(prices[:3]) if len(prices) >= 3 and np.mean(prices[:3]) != 0 else 0,
                'rolling_volatility_4m': np.std(prices[:4]) / np.mean(prices[:4]) if len(prices) >= 4 and np.mean(prices[:4]) != 0 else 0,
                'rolling_volatility_5m': np.std(prices[:5]) / np.mean(prices[:5]) if len(prices) >= 5 and np.mean(prices[:5]) != 0 else 0,
                
                # Momentum features
                'momentum_1min': returns[-1] if len(returns) >= 1 else 0,
                'momentum_2min': np.mean(returns[-2:]) if len(returns) >= 2 else 0,
                'momentum_3min': np.mean(returns[-3:]) if len(returns) >= 3 else 0,
                
                # Progressive trend features
                'trend_slope_2m': np.polyfit(range(2), prices[:2], 1)[0] if len(prices) >= 2 else 0,
                'trend_slope_3m': np.polyfit(range(3), prices[:3], 1)[0] if len(prices) >= 3 else 0,
                'trend_slope_4m': np.polyfit(range(4), prices[:4], 1)[0] if len(prices) >= 4 else 0,
                'trend_slope_5m': np.polyfit(range(5), prices[:5], 1)[0] if len(prices) >= 5 else 0,
                
                # Progressive statistical features
                'returns_mean_2m': np.mean(returns[:1]) if len(returns) >= 1 else 0,
                'returns_mean_3m': np.mean(returns[:2]) if len(returns) >= 2 else 0,
                'returns_mean_4m': np.mean(returns[:3]) if len(returns) >= 3 else 0,
                'returns_mean_5m': np.mean(returns[:4]) if len(returns) >= 4 else 0,
                
                'returns_std_2m': 0,
                'returns_std_3m': np.std(returns[:2]) if len(returns) >= 2 else 0,
                'returns_std_4m': np.std(returns[:3]) if len(returns) >= 3 else 0,
                'returns_std_5m': np.std(returns[:4]) if len(returns) >= 4 else 0,
            }
            
            # Handle any NaN values
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            return None
    
    def analyze_misclassification(self) -> Dict:
        """Analyze misclassification patterns."""
        print(f"🔍 Analyzing XGBoost misclassification patterns...")
        
        if self.predictions.empty:
            return {}
        
        df = self.predictions
        
        # Overall misclassification analysis
        total_tokens = len(df)
        total_misclassified = df['misclassified'].sum()
        overall_misclassification_rate = (total_misclassified / total_tokens * 100) if total_tokens > 0 else 0
        
        # Marathon-specific misclassification
        true_marathons = df[df['true_category'] == 'marathon']
        marathon_misclassified = true_marathons[true_marathons['predicted_category'] != 'marathon']
        marathon_misclassification_rate = (len(marathon_misclassified) / len(true_marathons) * 100) if len(true_marathons) > 0 else 0
        
        # Standard-specific misclassification
        true_standards = df[df['true_category'] == 'standard']
        standard_misclassified = true_standards[true_standards['predicted_category'] != 'standard']
        standard_misclassification_rate = (len(standard_misclassified) / len(true_standards) * 100) if len(true_standards) > 0 else 0
        
        # Sprint-specific misclassification
        true_sprints = df[df['true_category'] == 'sprint']
        sprint_misclassified = true_sprints[true_sprints['predicted_category'] != 'sprint']
        sprint_misclassification_rate = (len(sprint_misclassified) / len(true_sprints) * 100) if len(true_sprints) > 0 else 0
        
        # Analyze borderline cases (1150-1250 minutes)
        borderline_tokens = df[(df['lifespan_minutes'] >= 1150) & (df['lifespan_minutes'] <= 1250)]
        borderline_misclassified = borderline_tokens[borderline_tokens['misclassified']]
        
        # Confidence analysis
        low_confidence_threshold = 0.6
        low_confidence_misclassified = df[(df['category_confidence'] < low_confidence_threshold) & (df['misclassified'])]
        
        # Archetype vs prediction comparison
        archetype_tokens = df[df['has_archetype']]
        archetype_vs_prediction_mismatch = archetype_tokens[archetype_tokens['archetype_category'] != archetype_tokens['predicted_category']]
        
        # Archetype-level analysis (17-class predictions)
        archetype_perfect_match = archetype_tokens[archetype_tokens['archetype_name'] == archetype_tokens['predicted_archetype']]
        archetype_accuracy = len(archetype_perfect_match) / len(archetype_tokens) * 100 if len(archetype_tokens) > 0 else 0
        
        # Marathon→Standard misclassification specifically (Q3 focus)
        marathon_to_standard = true_marathons[true_marathons['predicted_category'] == 'standard']
        marathon_to_standard_rate = (len(marathon_to_standard) / len(true_marathons) * 100) if len(true_marathons) > 0 else 0
        
        self.misclassification_analysis = {
            'total_tokens': total_tokens,
            'total_misclassified': total_misclassified,
            'overall_misclassification_rate': overall_misclassification_rate,
            
            'true_marathons': len(true_marathons),
            'marathon_misclassified': len(marathon_misclassified),
            'marathon_misclassification_rate': marathon_misclassification_rate,
            
            'true_standards': len(true_standards),
            'standard_misclassified': len(standard_misclassified),
            'standard_misclassification_rate': standard_misclassification_rate,
            
            'true_sprints': len(true_sprints),
            'sprint_misclassified': len(sprint_misclassified),
            'sprint_misclassification_rate': sprint_misclassification_rate,
            
            'borderline_tokens': len(borderline_tokens),
            'borderline_misclassified': len(borderline_misclassified),
            'borderline_misclassification_rate': (len(borderline_misclassified) / len(borderline_tokens) * 100) if len(borderline_tokens) > 0 else 0,
            
            'low_confidence_misclassified': len(low_confidence_misclassified),
            'archetype_tokens': len(archetype_tokens),
            'archetype_prediction_mismatch': len(archetype_vs_prediction_mismatch),
            
            # Archetype-level metrics (17-class model)
            'archetype_accuracy': archetype_accuracy,
            'archetype_perfect_matches': len(archetype_perfect_match),
            
            # Q3-specific metric: Marathon→Standard misclassification
            'marathon_to_standard_misclassified': len(marathon_to_standard),
            'marathon_to_standard_rate': marathon_to_standard_rate,
            
            'accuracy': 100 - overall_misclassification_rate
        }
        
        print(f"📊 XGBoost Classification Results:")
        print(f"   Overall accuracy: {self.misclassification_analysis['accuracy']:.1f}%")
        print(f"   Marathon misclassification rate: {marathon_misclassification_rate:.1f}%")
        print(f"   Standard misclassification rate: {standard_misclassification_rate:.1f}%")
        print(f"   Borderline case misclassification: {self.misclassification_analysis['borderline_misclassification_rate']:.1f}%")
        
        return self.misclassification_analysis
    
    def print_key_answer(self) -> None:
        """Print clear answer to Question 3."""
        if not self.misclassification_analysis:
            print("❌ No misclassification analysis available")
            return
        
        analysis = self.misclassification_analysis
        
        print("\n" + "="*80)
        print("🎯 KEY TRADING STRATEGY ANSWER")
        print("="*80)
        
        print(f"\n3️⃣  QUESTION 3: What % of true marathons get misclassified?")
        print(f"   📊 ANSWER: {analysis['marathon_misclassification_rate']:.1f}%")
        print(f"   🎯 Specifically Marathon→Standard: {analysis['marathon_to_standard_rate']:.1f}%")
        print(f"   📈 Coverage Impact: {analysis['true_marathons'] - analysis['marathon_misclassified']}/{analysis['true_marathons']} marathons correctly identified")
        
        print(f"\n📋 DETAILED BREAKDOWN (3-Category System):")
        print(f"   • Total true marathons: {analysis['true_marathons']:,}")
        print(f"   • Marathon misclassified: {analysis['marathon_misclassified']:,}")
        print(f"   • Marathon→Standard: {analysis['marathon_to_standard_misclassified']:,}")
        print(f"   • Correctly identified: {analysis['true_marathons'] - analysis['marathon_misclassified']:,}")
        
        print(f"\n📊 ALL CATEGORIES PERFORMANCE:")
        print(f"   • Sprint misclassification: {analysis['sprint_misclassification_rate']:.1f}% ({analysis['sprint_misclassified']:,}/{analysis['true_sprints']:,})")
        print(f"   • Standard misclassification: {analysis['standard_misclassification_rate']:.1f}% ({analysis['standard_misclassified']:,}/{analysis['true_standards']:,})")
        print(f"   • Marathon misclassification: {analysis['marathon_misclassification_rate']:.1f}% ({analysis['marathon_misclassified']:,}/{analysis['true_marathons']:,})")
        
        print(f"\n🎭 ARCHETYPE MODEL PERFORMANCE (17-Class):")
        print(f"   • Archetype accuracy: {analysis['archetype_accuracy']:.1f}%")
        print(f"   • Perfect archetype matches: {analysis['archetype_perfect_matches']:,}/{analysis['archetype_tokens']:,}")
        
        if analysis['borderline_tokens'] > 0:
            print(f"   • Borderline cases (1150-1250 min): {analysis['borderline_tokens']:,}")
        
        if analysis.get('archetype_prediction_mismatch', 0) > 0:
            print(f"   • Archetype vs prediction mismatch: {analysis['archetype_prediction_mismatch']:,}")
        
        print("="*80)
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Create misclassification analysis visualizations."""
        print(f"📊 Creating XGBoost misclassification visualizations...")
        
        if self.predictions.empty:
            return {}
        
        figures = {}
        df = self.predictions
        
        # 1. Confusion Matrix
        figures['confusion_matrix'] = self._create_confusion_matrix(df)
        
        # 2. Misclassification by Lifespan
        figures['misclass_by_lifespan'] = self._create_lifespan_analysis(df)
        
        # 3. Confidence Analysis
        figures['confidence_analysis'] = self._create_confidence_analysis(df)
        
        # 4. Comprehensive Dashboard
        figures['xgboost_misclass_dashboard'] = self._create_misclass_dashboard(df)
        
        return figures
    
    def _create_confusion_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create confusion matrix visualization."""
        # Create confusion matrix
        cm = confusion_matrix(df['true_category'], df['predicted_category'], labels=['marathon', 'standard'])
        
        # Calculate percentages
        cm_percent = cm / cm.sum() * 100
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Confusion Matrix (Counts)', 'Confusion Matrix (Percentages)']
        )
        
        # Count matrix
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Marathon', 'Standard'],
                y=['Marathon', 'Standard'],
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues',
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
                name='Counts'
            ),
            row=1, col=1
        )
        
        # Percentage matrix
        fig.add_trace(
            go.Heatmap(
                z=cm_percent,
                x=['Marathon', 'Standard'],
                y=['Standard', 'Marathon'],
                text=[[f"{val:.1f}%" for val in row] for row in cm_percent],
                texttemplate="%{text}",
                colorscale='Reds',
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
                name='Percentages'
            ),
            row=1, col=2
        )
        
        # Calculate key metrics
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum() * 100
        marathon_precision = cm[0,0] / (cm[0,0] + cm[1,0]) * 100 if (cm[0,0] + cm[1,0]) > 0 else 0
        marathon_recall = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
        
        fig.update_layout(
            title=f'XGBoost Classification Confusion Matrix<br><sub>Accuracy: {accuracy:.1f}% | Marathon Precision: {marathon_precision:.1f}% | Marathon Recall: {marathon_recall:.1f}%</sub>',
            height=500
        )
        
        return fig
    
    def _create_lifespan_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create misclassification by lifespan analysis."""
        # Create lifespan bins
        df_copy = df.copy()
        df_copy['lifespan_bin'] = pd.cut(df_copy['lifespan_minutes'], bins=30)
        
        # Calculate misclassification rate by bin
        misclass_by_bin = df_copy.groupby('lifespan_bin').agg({
            'misclassified': ['sum', 'count', 'mean'],
            'category_confidence': 'mean'
        }).round(3)
        
        misclass_by_bin.columns = ['misclassified_count', 'total_count', 'misclass_rate', 'avg_confidence']
        misclass_by_bin = misclass_by_bin.reset_index()
        misclass_by_bin['lifespan_midpoint'] = misclass_by_bin['lifespan_bin'].apply(lambda x: x.mid if hasattr(x, 'mid') else 0)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                'Misclassification Rate by Token Lifespan',
                'Prediction Confidence by Token Lifespan'
            ],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Misclassification rate
        fig.add_trace(
            go.Scatter(
                x=misclass_by_bin['lifespan_midpoint'],
                y=misclass_by_bin['misclass_rate'] * 100,
                mode='lines+markers',
                name='Misclassification Rate',
                line=dict(color='red', width=2),
                hovertemplate='Lifespan: %{x:.0f} min<br>Misclass Rate: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Token count (secondary y-axis)
        fig.add_trace(
            go.Bar(
                x=misclass_by_bin['lifespan_midpoint'],
                y=misclass_by_bin['total_count'],
                name='Token Count',
                marker_color='lightblue',
                opacity=0.6,
                hovertemplate='Lifespan: %{x:.0f} min<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add vertical line at 1200 minutes threshold
        fig.add_vline(x=1200, line_dash="dash", line_color="green", annotation_text="Marathon Threshold (1200 min)", row=1, col=1)
        
        # Confidence analysis
        fig.add_trace(
            go.Scatter(
                x=misclass_by_bin['lifespan_midpoint'],
                y=misclass_by_bin['avg_confidence'],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='blue', width=2),
                hovertemplate='Lifespan: %{x:.0f} min<br>Avg Confidence: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='XGBoost Misclassification Analysis by Token Lifespan',
            height=800
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Misclassification Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Token Count", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Prediction Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Token Lifespan (minutes)", row=2, col=1)
        
        return fig
    
    def _create_confidence_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create confidence vs accuracy analysis."""
        # Create confidence bins
        df_copy = df.copy()
        df_copy['confidence_bin'] = pd.cut(df_copy['category_confidence'], bins=10)
        
        confidence_analysis = df_copy.groupby('confidence_bin').agg({
            'misclassified': ['count', 'sum', 'mean'],
            'category_confidence': 'mean'
        }).round(3)
        
        confidence_analysis.columns = ['total_count', 'misclassified_count', 'misclass_rate', 'avg_confidence']
        confidence_analysis = confidence_analysis.reset_index()
        confidence_analysis['confidence_midpoint'] = confidence_analysis['confidence_bin'].apply(lambda x: x.mid if hasattr(x, 'mid') else 0)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Accuracy vs Confidence',
                'Confidence Distribution',
                'Misclassified Token Characteristics',
                'High vs Low Confidence Comparison'
            ]
        )
        
        # 1. Accuracy vs Confidence
        fig.add_trace(
            go.Scatter(
                x=confidence_analysis['avg_confidence'],
                y=(1 - confidence_analysis['misclass_rate']) * 100,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='green', width=3),
                marker=dict(size=confidence_analysis['total_count']/10, sizemin=5),
                hovertemplate='Confidence: %{x:.3f}<br>Accuracy: %{y:.1f}%<br>Count: %{marker.size:.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Confidence distribution
        fig.add_trace(
            go.Histogram(
                x=df['category_confidence'],
                nbinsx=20,
                name='Confidence Distribution',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Misclassified token characteristics
        misclassified_tokens = df[df['misclassified']]
        if not misclassified_tokens.empty:
            fig.add_trace(
                go.Scatter(
                    x=misclassified_tokens['volatility_cv'],
                    y=misclassified_tokens['category_confidence'],
                    mode='markers',
                    marker=dict(
                        color=misclassified_tokens['lifespan_minutes'],
                        colorscale='Viridis',
                        size=8,
                        showscale=True,
                        colorbar=dict(title="Lifespan (min)")
                    ),
                    name='Misclassified Tokens',
                    hovertemplate='Volatility CV: %{x:.3f}<br>Confidence: %{y:.3f}<br>Lifespan: %{marker.color} min<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. High vs Low confidence comparison
        high_conf_threshold = 0.8
        low_conf_threshold = 0.6
        
        high_conf_tokens = df[df['category_confidence'] >= high_conf_threshold]
        low_conf_tokens = df[df['category_confidence'] <= low_conf_threshold]
        
        comparison_data = {
            'High Confidence (>0.8)': [
                len(high_conf_tokens),
                high_conf_tokens['misclassified'].mean() * 100 if len(high_conf_tokens) > 0 else 0
            ],
            'Low Confidence (<0.6)': [
                len(low_conf_tokens),
                low_conf_tokens['misclassified'].mean() * 100 if len(low_conf_tokens) > 0 else 0
            ]
        }
        
        fig.add_trace(
            go.Bar(
                x=['High Confidence', 'Low Confidence'],
                y=[comparison_data['High Confidence (>0.8)'][1], comparison_data['Low Confidence (<0.6)'][1]],
                text=[f"{rate:.1f}%<br>(n={count})" for count, rate in [comparison_data['High Confidence (>0.8)'], comparison_data['Low Confidence (<0.6)']]],
                textposition='auto',
                name='Misclassification Rate',
                marker_color=['green', 'red']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='XGBoost Prediction Confidence Analysis',
            height=800
        )
        
        return fig
    
    def _create_misclass_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive misclassification dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{"type": "pie"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "pie"}], 
                   [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=[
                'Overall Classification Performance',
                'Misclassification by Category',
                'Borderline Cases (1150-1250 min)',
                'Archetype vs Prediction Agreement',
                'Error Distribution by Volatility',
                'Summary Statistics'
            ]
        )
        
        # 1. Overall performance metrics
        accuracy = (1 - df['misclassified'].mean()) * 100
        
        categories = ['Correct', 'Misclassified']
        counts = [len(df) - df['misclassified'].sum(), df['misclassified'].sum()]
        
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=counts,
                name="Overall Performance",
                hovertemplate='%{label}: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Misclassification by true category
        misclass_by_category = df.groupby('true_category')['misclassified'].agg(['count', 'sum', 'mean']).reset_index()
        misclass_by_category['accuracy'] = (1 - misclass_by_category['mean']) * 100
        
        fig.add_trace(
            go.Bar(
                x=misclass_by_category['true_category'],
                y=misclass_by_category['accuracy'],
                text=[f"{acc:.1f}%<br>({correct}/{total})" for acc, correct, total in 
                      zip(misclass_by_category['accuracy'], 
                          misclass_by_category['count'] - misclass_by_category['sum'],
                          misclass_by_category['count'])],
                textposition='auto',
                name='Accuracy by Category',
                marker_color=['lightblue', 'lightcoral']
            ),
            row=1, col=2
        )
        
        # 3. Borderline cases
        borderline = df[(df['lifespan_minutes'] >= 1150) & (df['lifespan_minutes'] <= 1250)]
        if not borderline.empty:
            borderline_breakdown = borderline.groupby(['true_category', 'predicted_category']).size().reset_index(name='count')
            
            fig.add_trace(
                go.Bar(
                    x=[f"{row['true_category']} → {row['predicted_category']}" for _, row in borderline_breakdown.iterrows()],
                    y=borderline_breakdown['count'],
                    text=borderline_breakdown['count'],
                    textposition='auto',
                    name='Borderline Classifications',
                    marker_color='yellow'
                ),
                row=2, col=1
            )
        
        # 4. Archetype vs Prediction agreement
        archetype_tokens = df[df['has_archetype']]
        if not archetype_tokens.empty:
            agreement = archetype_tokens['archetype_category'] == archetype_tokens['predicted_category']
            agreement_rate = agreement.mean() * 100
            
            agreement_counts = [agreement.sum(), len(archetype_tokens) - agreement.sum()]
            fig.add_trace(
                go.Pie(
                    labels=['Agreement', 'Disagreement'],
                    values=agreement_counts,
                    name="Archetype Agreement",
                    hovertemplate='%{label}: %{value}<br>Percentage: %{percent}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Error distribution by volatility
        misclassified_tokens = df[df['misclassified']]
        if not misclassified_tokens.empty:
            fig.add_trace(
                go.Histogram(
                    x=misclassified_tokens['volatility_cv'],
                    nbinsx=20,
                    name='Misclassified Volatility',
                    marker_color='red',
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # 6. Summary statistics
        summary_text = f"""
        📊 XGBoost Classification Summary
        
        🎯 Overall Performance:
        • Accuracy: {accuracy:.1f}%
        • Total Tokens: {len(df):,}
        • Misclassified: {df['misclassified'].sum():,}
        
        🏃 Marathon Performance:
        • Total Marathons: {len(df[df['true_category'] == 'marathon']):,}
        • Marathon Accuracy: {(1-df[df['true_category'] == 'marathon']['misclassified'].mean())*100:.1f}%
        • Misclassified as Standard: {len(df[(df['true_category'] == 'marathon') & (df['predicted_category'] == 'standard')]):,}
        
        🚶 Standard Performance:
        • Total Standards: {len(df[df['true_category'] == 'standard']):,}
        • Standard Accuracy: {(1-df[df['true_category'] == 'standard']['misclassified'].mean())*100:.1f}%
        • Misclassified as Marathon: {len(df[(df['true_category'] == 'standard') & (df['predicted_category'] == 'marathon')]):,}
        
        🎲 Confidence Analysis:
        • Avg Confidence: {df['category_confidence'].mean():.3f}
        • High Confidence (>0.8): {len(df[df['category_confidence'] > 0.8]):,}
        • Low Confidence (<0.6): {len(df[df['category_confidence'] < 0.6]):,}
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=9, family="monospace"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            row=3, col=2
        )
        
        fig.update_layout(
            title='XGBoost Misclassification Analysis Dashboard<br><sub>Comprehensive analysis of prediction accuracy and error patterns</sub>',
            height=1200,
            showlegend=False
        )
        
        return fig
    
    def save_results(self, output_dir: Path = None) -> str:
        """Save misclassification analysis results."""
        if output_dir is None:
            output_dir = self.results_dir / "xgboost_misclassification"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prediction results
        if not self.predictions.empty:
            predictions_path = output_dir / f"xgboost_predictions_{timestamp}.csv"
            self.predictions.to_csv(predictions_path, index=False)
            print(f"📊 Predictions saved: {predictions_path}")
        
        # Save misclassification analysis
        analysis_path = output_dir / f"misclassification_analysis_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(self.misclassification_analysis, f, indent=2, default=str)
        print(f"📋 Analysis saved: {analysis_path}")
        
        # Create and save visualizations
        figures = self.create_visualizations()
        for fig_name, fig in figures.items():
            fig_path = output_dir / f"{fig_name}_{timestamp}.html"
            fig.write_html(fig_path)
            print(f"📈 Visualization saved: {fig_path}")
        
        print(f"✅ All results saved to: {output_dir}")
        return timestamp
    
    def run_complete_analysis(self, archetype_results_path: Path, data_dir: Path) -> Dict:
        """Run complete misclassification analysis."""
        print(f"🚀 Starting XGBoost Misclassification Analysis")
        
        # Load data and train classifier
        self.load_and_predict(archetype_results_path, data_dir)
        
        if self.predictions.empty:
            print(f"❌ No predictions generated")
            return {}
        
        # Analyze misclassification patterns
        misclass_metrics = self.analyze_misclassification()
        
        # Save results
        timestamp = self.save_results()
        
        return {
            'misclassification_analysis': misclass_metrics,
            'timestamp': timestamp,
            'total_predictions': len(self.predictions)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="XGBoost Misclassification Analysis")
    parser.add_argument("--archetype-results", type=Path,
                       help="Path to archetype characterization results JSON")
    parser.add_argument("--data-dir", type=Path,
                       default=Path("../../data/with_archetypes_fixed"),
                       help="Path to token data directory")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Find latest archetype results if not specified
    if not args.archetype_results:
        results_dir = Path("../results/phase1_day9_10_archetypes")
        if results_dir.exists():
            json_files = list(results_dir.glob("archetype_characterization_*.json"))
            if json_files:
                args.archetype_results = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest archetype results: {args.archetype_results}")
            else:
                print("❌ No archetype results found. Run the Phase 1 pipeline first.")
                return
        else:
            print("❌ Results directory not found. Run the Phase 1 pipeline first.")
            return
    
    # Initialize analyzer
    analyzer = XGBoostMisclassificationAnalyzer(args.output_dir)
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis(args.archetype_results, args.data_dir)
        
        # Print key answer clearly
        analyzer.print_key_answer()
        
        print(f"\n🎉 XGBoost misclassification analysis complete!")
        print(f"📈 Results saved with timestamp: {results.get('timestamp', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()