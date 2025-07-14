#!/usr/bin/env python3
# phase1_day3_4_feature_standardization.py
"""
Phase 1 Day 3-4: Feature Standardization

CEO Roadmap Implementation:
- Adopt winning returns type from Day 1-2 baseline assessment
- Implement robust 15-feature pipeline with edge case handling
- Test on full dataset categories (normal/extreme/dead)
- Validate feature consistency and stability

Usage:
    python phase1_day3_4_feature_standardization.py [--baseline-results PATH] [--data-dir PATH] [--output-dir PATH]
    
Interactive Mode:
    python phase1_day3_4_feature_standardization.py --interactive
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_categorized_tokens, extract_features_from_returns, ClusteringEngine,
    ResultsManager, GradioVisualizer, prepare_token_for_analysis
)


class FeatureStandardizationAnalyzer:
    """
    Implements CEO roadmap Phase 1 Day 3-4: Feature Standardization.
    Adopts winner from baseline assessment and creates robust feature pipeline.
    """
    
    def __init__(self, results_dir: Path = None):
        self.results_manager = ResultsManager(results_dir or Path("../results"))
        self.clustering_engine = ClusteringEngine(random_state=42)
        
    def load_baseline_results(self, baseline_results_path: Path) -> Dict:
        """Load baseline assessment results to determine winner."""
        if not baseline_results_path.exists():
            raise ValueError(f"Baseline results not found: {baseline_results_path}")
        
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        
        return baseline_results
    
    def determine_winning_method(self, baseline_results: Dict) -> Tuple[str, Dict]:
        """Extract winning method and metrics from baseline results."""
        winner_selection = baseline_results['winner_selection']
        winner = winner_selection['winner']
        winner_metrics = winner_selection['winner_metrics']
        
        print(f"📊 Baseline Assessment Winner: {winner.upper()} returns")
        print(f"   ARI: {winner_metrics['ari']:.4f}")
        print(f"   Silhouette: {winner_metrics['silhouette']:.4f}")
        print(f"   Primary Score: {winner_metrics['primary_score']:.4f}")
        
        return winner, winner_metrics
    
    def run_feature_standardization(self, processed_dir: Path, 
                                  baseline_results_path: Path = None,
                                  max_tokens_per_category: int = None) -> Dict:
        """
        Run feature standardization analysis using winner from baseline assessment.
        
        Args:
            processed_dir: Path to processed data directory
            baseline_results_path: Path to baseline assessment results
            max_tokens_per_category: Maximum tokens per category for testing
            
        Returns:
            Complete feature standardization results
        """
        print(f"🚀 Starting Phase 1 Day 3-4 Feature Standardization")
        
        # Step 1: Load baseline results and determine winner
        if baseline_results_path:
            print(f"\n📁 Loading baseline results from {baseline_results_path}")
            baseline_results = self.load_baseline_results(baseline_results_path)
            winner, winner_metrics = self.determine_winning_method(baseline_results)
        else:
            print("\n⚠️  No baseline results provided, defaulting to raw returns")
            winner = 'raw'
            winner_metrics = {'ari': 0.0, 'silhouette': 0.0, 'primary_score': 0.0}
        
        use_log = (winner == 'log')
        
        # Step 2: Load categorized tokens
        print(f"\n📁 Loading categorized tokens...")
        categorized_data = load_categorized_tokens(
            processed_dir, 
            max_tokens_per_category=max_tokens_per_category
        )
        
        total_tokens = sum(len(tokens) for tokens in categorized_data.values())
        print(f"✅ Loaded {total_tokens} tokens across {len(categorized_data)} categories")
        
        # Step 3: Extract features and categorize by lifespan
        print(f"\n🔧 Extracting features and categorizing by lifespan using {winner} returns...")
        lifespan_categories = {
            'sprint': {},      # 0-400 active minutes
            'standard': {},    # 400-1200 active minutes
            'marathon': {}     # 1200+ active minutes
        }
        
        source_category_stats = {}
        lifespan_stats = {
            'sprint': {'count': 0, 'min_lifespan': float('inf'), 'max_lifespan': 0},
            'standard': {'count': 0, 'min_lifespan': float('inf'), 'max_lifespan': 0},
            'marathon': {'count': 0, 'min_lifespan': float('inf'), 'max_lifespan': 0}
        }
        
        for source_category, token_data in categorized_data.items():
            print(f"  Processing {source_category}: {len(token_data)} tokens")
            
            source_features_dict = self._extract_category_features(token_data, use_log)
            source_category_stats[source_category] = self._calculate_feature_statistics(source_features_dict)
            
            # Categorize by lifespan
            for token_name, features in source_features_dict.items():
                lifespan = features['lifespan_minutes']
                
                # Determine lifespan category
                if lifespan < 400:
                    target_category = 'sprint'
                elif lifespan < 1200:
                    target_category = 'standard'
                else:
                    target_category = 'marathon'
                
                # Add to lifespan category
                lifespan_categories[target_category][token_name] = features
                
                # Update lifespan statistics
                lifespan_stats[target_category]['count'] += 1
                lifespan_stats[target_category]['min_lifespan'] = min(
                    lifespan_stats[target_category]['min_lifespan'], lifespan
                )
                lifespan_stats[target_category]['max_lifespan'] = max(
                    lifespan_stats[target_category]['max_lifespan'], lifespan
                )
            
            print(f"    ✅ Extracted features for {len(source_features_dict)} tokens")
        
        # Clean up infinite values in lifespan stats
        for category in lifespan_stats:
            if lifespan_stats[category]['count'] == 0:
                lifespan_stats[category]['min_lifespan'] = 0
        
        print(f"\n✅ Lifespan categorization complete:")
        for category, stats in lifespan_stats.items():
            print(f"  {category}: {stats['count']} tokens "
                  f"({stats['min_lifespan']}-{stats['max_lifespan']} min)")
        
        # Calculate feature statistics for lifespan categories
        lifespan_feature_stats = {}
        for category, features_dict in lifespan_categories.items():
            lifespan_feature_stats[category] = self._calculate_feature_statistics(features_dict)
        
        # Step 4: Validate feature consistency across lifespan categories
        print(f"\n🔍 Validating feature consistency...")
        consistency_analysis = self._validate_feature_consistency(lifespan_categories)
        
        # Step 5: Test feature stability on subsamples
        print(f"\n📊 Testing feature stability...")
        stability_analysis = self._test_feature_stability(lifespan_categories, use_log)
        
        # Step 6: Generate edge case handling report
        print(f"\n⚠️  Analyzing edge cases...")
        edge_case_analysis = self._analyze_edge_cases(lifespan_categories)
        
        # Step 7: Create standardized feature pipeline
        print(f"\n🏭 Creating standardized feature pipeline...")
        pipeline_config = self._create_feature_pipeline_config(
            winner, winner_metrics, consistency_analysis, stability_analysis
        )
        
        # Compile complete results
        complete_results = {
            'analysis_type': 'feature_standardization',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'winning_method': winner,
                'use_log_returns': use_log,
                'max_tokens_per_category': max_tokens_per_category,
                'total_tokens_processed': total_tokens
            },
            'baseline_winner': {
                'method': winner,
                'metrics': winner_metrics
            },
            'source_category_features': {
                category: {
                    'n_tokens': len(features),
                    'feature_names': list(next(iter(features.values())).keys()) if features else []
                }
                for category, features in source_category_stats.items()
            },
            'lifespan_categories': lifespan_categories,
            'lifespan_statistics': lifespan_stats,
            'source_category_statistics': source_category_stats,
            'lifespan_feature_statistics': lifespan_feature_stats,
            'consistency_analysis': consistency_analysis,
            'stability_analysis': stability_analysis,
            'edge_case_analysis': edge_case_analysis,
            'standardized_pipeline': pipeline_config,
            'recommendations': self._generate_recommendations(
                consistency_analysis, stability_analysis, edge_case_analysis
            )
        }
        
        print(f"\n🎉 Feature standardization complete!")
        print(f"📊 Processed {total_tokens} tokens using {winner} returns")
        print(f"✅ Feature consistency: {consistency_analysis['overall_consistency']:.3f}")
        print(f"🏃 Sprint tokens: {lifespan_stats['sprint']['count']}")
        print(f"🚶 Standard tokens: {lifespan_stats['standard']['count']}")
        print(f"🏃‍♂️ Marathon tokens: {lifespan_stats['marathon']['count']}")
        
        return complete_results
    
    def _extract_category_features(self, token_data: Dict, use_log: bool) -> Dict[str, Dict[str, float]]:
        """Extract features for all tokens in a category."""
        features_dict = {}
        
        for token_name, token_df in token_data.items():
            try:
                prices, returns = prepare_token_for_analysis(token_df)
                features = extract_features_from_returns(returns, prices, use_log=use_log)
                features_dict[token_name] = features
            except Exception as e:
                print(f"    Warning: Failed to extract features for {token_name}: {e}")
                continue
        
        return features_dict
    
    def _calculate_feature_statistics(self, features_dict: Dict[str, Dict[str, float]]) -> Dict:
        """Calculate comprehensive statistics for features in a category."""
        if not features_dict:
            return {}
        
        # Convert to matrix for easier calculation
        feature_names = list(next(iter(features_dict.values())).keys())
        feature_matrix = np.array([[features[fname] for fname in feature_names] 
                                  for features in features_dict.values()])
        
        stats = {}
        for i, feature_name in enumerate(feature_names):
            feature_values = feature_matrix[:, i]
            
            # Filter out non-finite values
            finite_values = feature_values[np.isfinite(feature_values)]
            
            if len(finite_values) > 0:
                stats[feature_name] = {
                    'mean': float(np.mean(finite_values)),
                    'std': float(np.std(finite_values)),
                    'median': float(np.median(finite_values)),
                    'min': float(np.min(finite_values)),
                    'max': float(np.max(finite_values)),
                    'n_finite': len(finite_values),
                    'n_total': len(feature_values),
                    'pct_finite': len(finite_values) / len(feature_values) * 100
                }
            else:
                stats[feature_name] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0,
                    'min': 0.0, 'max': 0.0,
                    'n_finite': 0, 'n_total': len(feature_values),
                    'pct_finite': 0.0
                }
        
        return stats
    
    def _validate_feature_consistency(self, category_features: Dict[str, Dict]) -> Dict:
        """Validate that features are consistent across categories."""
        # Check feature names consistency
        all_feature_names = []
        for category, features_dict in category_features.items():
            if features_dict:
                feature_names = set(next(iter(features_dict.values())).keys())
                all_feature_names.append((category, feature_names))
        
        if not all_feature_names:
            return {'overall_consistency': 0.0, 'issues': ['No features extracted']}
        
        # Check if all categories have the same feature names
        reference_features = all_feature_names[0][1]
        consistency_issues = []
        
        for category, feature_names in all_feature_names[1:]:
            if feature_names != reference_features:
                missing = reference_features - feature_names
                extra = feature_names - reference_features
                if missing:
                    consistency_issues.append(f"{category} missing: {missing}")
                if extra:
                    consistency_issues.append(f"{category} extra: {extra}")
        
        # Calculate overall consistency score
        if consistency_issues:
            consistency_score = 0.5  # Partial consistency
        else:
            consistency_score = 1.0  # Full consistency
        
        return {
            'overall_consistency': consistency_score,
            'reference_features': list(reference_features),
            'n_features': len(reference_features),
            'issues': consistency_issues,
            'categories_processed': [cat for cat, _ in all_feature_names]
        }
    
    def _test_feature_stability(self, category_features: Dict[str, Dict], use_log: bool) -> Dict:
        """Test feature stability across different subsamples."""
        stability_results = {}
        
        for category, features_dict in category_features.items():
            if len(features_dict) < 10:  # Need at least 10 tokens for stability testing
                continue
            
            # Create 3 different subsamples
            token_names = list(features_dict.keys())
            np.random.seed(42)  # For reproducibility
            
            subsample_size = min(len(token_names) // 2, 50)  # Use half or max 50 tokens
            
            stability_scores = []
            feature_correlations = []
            
            for i in range(3):  # 3 stability runs
                # Create random subsample
                np.random.seed(42 + i)
                subsample_tokens = np.random.choice(token_names, subsample_size, replace=False)
                subsample_features = {t: features_dict[t] for t in subsample_tokens}
                
                # Extract feature matrix
                if subsample_features:
                    feature_names = list(next(iter(subsample_features.values())).keys())
                    feature_matrix = np.array([[features[fname] for fname in feature_names] 
                                              for features in subsample_features.values()])
                    
                    # Filter out non-finite values and calculate correlation matrix
                    finite_mask = np.all(np.isfinite(feature_matrix), axis=1)
                    if np.sum(finite_mask) > 1:
                        clean_matrix = feature_matrix[finite_mask]
                        correlation_matrix = np.corrcoef(clean_matrix.T)
                        
                        # Calculate stability score as mean absolute correlation
                        if correlation_matrix.size > 1:
                            stability_score = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
                            stability_scores.append(stability_score)
                            feature_correlations.append(correlation_matrix.tolist())
            
            if stability_scores:
                stability_results[category] = {
                    'mean_stability': float(np.mean(stability_scores)),
                    'std_stability': float(np.std(stability_scores)),
                    'stability_scores': stability_scores,
                    'n_subsamples': len(stability_scores),
                    'subsample_size': subsample_size
                }
            else:
                stability_results[category] = {
                    'mean_stability': 0.0,
                    'std_stability': 0.0,
                    'stability_scores': [],
                    'n_subsamples': 0,
                    'subsample_size': 0
                }
        
        return stability_results
    
    def _analyze_edge_cases(self, category_features: Dict[str, Dict]) -> Dict:
        """Analyze edge cases and potential issues with feature extraction."""
        edge_cases = {
            'infinite_values': {},
            'nan_values': {},
            'zero_variance': {},
            'extreme_outliers': {}
        }
        
        for category, features_dict in category_features.items():
            if not features_dict:
                continue
            
            # Convert to matrix
            feature_names = list(next(iter(features_dict.values())).keys())
            feature_matrix = np.array([[features[fname] for fname in feature_names] 
                                      for features in features_dict.values()])
            
            category_edge_cases = {
                'infinite_values': [],
                'nan_values': [],
                'zero_variance': [],
                'extreme_outliers': []
            }
            
            for i, feature_name in enumerate(feature_names):
                feature_values = feature_matrix[:, i]
                
                # Check for infinite values
                n_inf = np.sum(np.isinf(feature_values))
                if n_inf > 0:
                    category_edge_cases['infinite_values'].append({
                        'feature': feature_name,
                        'count': int(n_inf),
                        'percentage': float(n_inf / len(feature_values) * 100)
                    })
                
                # Check for NaN values
                n_nan = np.sum(np.isnan(feature_values))
                if n_nan > 0:
                    category_edge_cases['nan_values'].append({
                        'feature': feature_name,
                        'count': int(n_nan),
                        'percentage': float(n_nan / len(feature_values) * 100)
                    })
                
                # Check for zero variance (constant features)
                finite_values = feature_values[np.isfinite(feature_values)]
                if len(finite_values) > 1 and np.var(finite_values) < 1e-12:
                    category_edge_cases['zero_variance'].append({
                        'feature': feature_name,
                        'value': float(finite_values[0]) if len(finite_values) > 0 else 0.0
                    })
                
                # Check for extreme outliers (values beyond 5 standard deviations)
                if len(finite_values) > 2:
                    mean_val = np.mean(finite_values)
                    std_val = np.std(finite_values)
                    if std_val > 0:
                        outliers = np.abs(finite_values - mean_val) > 5 * std_val
                        n_outliers = np.sum(outliers)
                        if n_outliers > 0:
                            category_edge_cases['extreme_outliers'].append({
                                'feature': feature_name,
                                'count': int(n_outliers),
                                'percentage': float(n_outliers / len(finite_values) * 100)
                            })
            
            edge_cases['infinite_values'][category] = category_edge_cases['infinite_values']
            edge_cases['nan_values'][category] = category_edge_cases['nan_values']
            edge_cases['zero_variance'][category] = category_edge_cases['zero_variance']
            edge_cases['extreme_outliers'][category] = category_edge_cases['extreme_outliers']
        
        return edge_cases
    
    def _create_feature_pipeline_config(self, winner: str, winner_metrics: Dict,
                                      consistency_analysis: Dict, stability_analysis: Dict) -> Dict:
        """Create standardized feature pipeline configuration."""
        return {
            'method': winner,
            'use_log_returns': winner == 'log',
            'feature_names': consistency_analysis.get('reference_features', []),
            'n_features': consistency_analysis.get('n_features', 0),
            'baseline_performance': {
                'ari': winner_metrics.get('ari', 0.0),
                'silhouette': winner_metrics.get('silhouette', 0.0),
                'primary_score': winner_metrics.get('primary_score', 0.0)
            },
            'consistency_score': consistency_analysis.get('overall_consistency', 0.0),
            'stability_scores': {
                category: stats.get('mean_stability', 0.0)
                for category, stats in stability_analysis.items()
            },
            'preprocessing_steps': [
                f"Use {winner} returns for feature extraction",
                "Handle infinite values with winsorization",
                "Replace NaN values with feature-specific defaults",
                "Apply feature scaling per category if needed"
            ],
            'validation_requirements': {
                'min_consistency_score': 0.8,
                'min_stability_score': 0.3,
                'max_nan_percentage': 10.0,
                'max_infinite_percentage': 5.0
            }
        }
    
    def _generate_recommendations(self, consistency_analysis: Dict, 
                                stability_analysis: Dict, edge_case_analysis: Dict) -> List[str]:
        """Generate actionable recommendations for feature pipeline improvements."""
        recommendations = []
        
        # Consistency recommendations
        if consistency_analysis['overall_consistency'] < 0.8:
            recommendations.append("⚠️ Feature consistency is below threshold (0.8). Review feature extraction logic.")
            if consistency_analysis['issues']:
                recommendations.append(f"Address consistency issues: {'; '.join(consistency_analysis['issues'])}")
        
        # Stability recommendations
        mean_stability = np.mean([stats.get('mean_stability', 0) for stats in stability_analysis.values()])
        if mean_stability < 0.3:
            recommendations.append("⚠️ Feature stability is low. Consider feature engineering improvements.")
        
        # Edge case recommendations
        total_nan_features = sum(len(issues) for issues in edge_case_analysis['nan_values'].values())
        total_inf_features = sum(len(issues) for issues in edge_case_analysis['infinite_values'].values())
        
        if total_nan_features > 0:
            recommendations.append(f"Handle {total_nan_features} features with NaN values using imputation.")
        
        if total_inf_features > 0:
            recommendations.append(f"Handle {total_inf_features} features with infinite values using winsorization.")
        
        # General recommendations
        recommendations.extend([
            f"✅ Pipeline ready for Phase 5-6: Use {consistency_analysis.get('n_features', 0)} standardized features",
            "✅ Proceed with category-specific K-selection using validated features",
            "✅ Monitor feature stability in production pipeline"
        ])
        
        return recommendations
    
    def save_results(self, results: Dict, output_dir: Path = None) -> str:
        """Save complete analysis results."""
        if output_dir:
            self.results_manager = ResultsManager(output_dir)
        
        timestamp = self.results_manager.save_analysis_results(
            results, 
            analysis_name="feature_standardization", 
            phase_dir="phase1_day3_4_features",
            include_plots=False  # No plots for this phase
        )
        
        return timestamp


def create_gradio_interface():
    """Create interactive Gradio interface for feature standardization."""
    visualizer = GradioVisualizer("Phase 1 Day 3-4: Feature Standardization")
    analyzer = FeatureStandardizationAnalyzer()
    
    def run_interactive_analysis(data_dir_str: str, baseline_results_str: str, 
                                max_tokens: int):
        """Run analysis with Gradio inputs."""
        try:
            data_dir = Path(data_dir_str)
            if not data_dir.exists():
                return "❌ Data directory not found"
            
            baseline_results_path = None
            if baseline_results_str.strip():
                baseline_results_path = Path(baseline_results_str)
                if not baseline_results_path.exists():
                    return f"❌ Baseline results file not found: {baseline_results_str}"
            
            # Run analysis
            results = analyzer.run_feature_standardization(
                data_dir, baseline_results_path, max_tokens
            )
            
            # Save results
            try:
                timestamp = analyzer.save_results(results)
                print(f"✅ Results successfully saved with timestamp: {timestamp}")
            except Exception as save_error:
                print(f"❌ Error saving results: {save_error}")
                timestamp = "save_failed"
            
            # Generate summary
            winner = results['parameters']['winning_method']
            consistency = results['consistency_analysis']['overall_consistency']
            total_tokens = results['parameters']['total_tokens_processed']
            
            summary = f"""
            ## 🎉 Feature Standardization Complete!
            
            **Winning Method**: {winner.upper()} returns
            **Total Tokens Processed**: {total_tokens:,}
            **Feature Consistency Score**: {consistency:.3f}
            
            **Lifespan Categorization**:
            {chr(10).join([f"- {cat.title()}: {stats['count']} tokens ({stats['min_lifespan']}-{stats['max_lifespan']} min)" 
                          for cat, stats in results['lifespan_statistics'].items()])}
            
            **Source Categories**:
            {chr(10).join([f"- {cat}: {info['n_tokens']} tokens" 
                          for cat, info in results['source_category_features'].items()])}
            
            **Key Recommendations**:
            {chr(10).join(['- ' + rec for rec in results['recommendations'][:5]])}
            
            **Results saved with timestamp**: {timestamp}
            """
            
            return summary
            
        except Exception as e:
            error_msg = f"## ❌ Error During Analysis\n\n```\n{str(e)}\n```"
            return error_msg
    
    import gradio as gr
    
    interface = gr.Interface(
        fn=run_interactive_analysis,
        inputs=[
            gr.Textbox(
                value="/Users/brunostordeur/Docs/GitHub/Solana/memecoin2/data/processed",
                label="Processed Data Directory",
                placeholder="Path to processed data directory"
            ),
            gr.Textbox(
                value="",
                label="Baseline Results File (JSON)",
                placeholder="Path to Day 1-2 baseline results JSON file (optional)"
            ),
            gr.Number(
                value=100,
                label="Max Tokens per Category (optional)",
                precision=0
            )
        ],
        outputs=[
            gr.Markdown(label="Feature Standardization Results")
        ],
        title="🚀 Phase 1 Day 3-4: Feature Standardization",
        description="Adopt winning method from baseline assessment and create robust feature pipeline"
    )
    
    return interface


def main():
    """Main entry point with CLI and interactive modes."""
    parser = argparse.ArgumentParser(description="Phase 1 Day 3-4 Feature Standardization")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path("../../data/processed"),
                       help="Path to processed data directory")
    parser.add_argument("--baseline-results", type=Path,
                       help="Path to Day 1-2 baseline results JSON file")
    parser.add_argument("--max-tokens", type=int,
                       help="Maximum tokens per category")
    parser.add_argument("--output-dir", type=Path, default=Path("../results"),
                       help="Output directory for results")
    parser.add_argument("--interactive", action="store_true",
                       help="Launch interactive Gradio interface")
    parser.add_argument("--share", action="store_true",
                       help="Create public Gradio link")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("🚀 Launching interactive Gradio interface...")
        interface = create_gradio_interface()
        interface.launch(share=args.share)
    else:
        # CLI mode
        analyzer = FeatureStandardizationAnalyzer(args.output_dir)
        results = analyzer.run_feature_standardization(
            args.data_dir, args.baseline_results, args.max_tokens
        )
        try:
            timestamp = analyzer.save_results(results)
            print(f"✅ Results successfully saved with timestamp: {timestamp}")
        except Exception as save_error:
            print(f"❌ Error saving results: {save_error}")
            timestamp = "save_failed"
        
        print(f"\n📁 Results saved to: {args.output_dir}/phase1_day3_4_features/")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"✅ Method: {results['parameters']['winning_method']} returns")


if __name__ == "__main__":
    main()