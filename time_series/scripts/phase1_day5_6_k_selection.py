#!/usr/bin/env python3
# phase1_day5_6_k_selection.py
"""
Phase 1 Day 5-6: Optimal K Selection per Category

CEO Roadmap Implementation:
- Load lifespan-categorized features from Day 3-4 standardization results
- Analyze sprint (0-400 min), standard (400-1200 min), marathon (1200+ min) categories separately
- Apply elbow method, silhouette analysis, and gap statistic
- Generate category-specific clustering recommendations
- Prepare optimal parameters for Phase 7-8 stability testing

Usage:
    python phase1_day5_6_k_selection.py --standardization-results PATH [--output-dir PATH]
    
Interactive Mode:
    python phase1_day5_6_k_selection.py --interactive
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_categorized_tokens, extract_features_from_returns, ClusteringEngine,
    ResultsManager, GradioVisualizer, prepare_token_for_analysis
)


class KSelectionAnalyzer:
    """
    Implements CEO roadmap Phase 1 Day 5-6: Optimal K Selection per Category.
    Uses standardized features to find optimal K for each lifespan category.
    """
    
    def __init__(self, results_dir: Path = None):
        self.results_manager = ResultsManager(results_dir or Path("../results"))
        self.clustering_engine = ClusteringEngine(random_state=42)
        
    def load_standardization_results(self, standardization_results_path: Path) -> Dict:
        """Load feature standardization results to get winning method."""
        if not standardization_results_path.exists():
            raise ValueError(f"Standardization results not found: {standardization_results_path}")
        
        with open(standardization_results_path, 'r') as f:
            standardization_results = json.load(f)
        
        return standardization_results
    
    def load_lifespan_categories_from_day34(self, standardization_results: Dict) -> Tuple[Dict[str, Dict], Dict]:
        """
        Load lifespan-categorized features from Day 3-4 standardization results.
        
        Args:
            standardization_results: Results from Day 3-4 feature standardization
            
        Returns:
            Tuple of (lifespan_categories, category_stats)
        """
        if 'lifespan_categories' not in standardization_results:
            raise ValueError("Day 3-4 results missing lifespan categorization. Please re-run Day 3-4 with updated script.")
        
        lifespan_categories = standardization_results['lifespan_categories']
        category_stats = standardization_results['lifespan_statistics']
        
        print(f"✅ Loaded lifespan categories from Day 3-4:")
        for category, stats in category_stats.items():
            print(f"  {category}: {stats['count']} tokens "
                  f"({stats['min_lifespan']}-{stats['max_lifespan']} min)")
        
        return lifespan_categories, category_stats
    
    def calculate_gap_statistic(self, features: np.ndarray, k_range: range, n_refs: int = 10) -> Dict:
        """
        Calculate gap statistic for optimal K selection.
        
        Args:
            features: Feature matrix
            k_range: Range of K values to test
            n_refs: Number of reference datasets for gap calculation
            
        Returns:
            Dictionary with gap statistics
        """
        n_samples, n_features = features.shape
        
        gaps = []
        log_wks = []
        sk_values = []
        
        for k in k_range:
            # Calculate log(Wk) for actual data
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(features)
            
            # Calculate within-cluster sum of squares
            wk = 0
            for i in range(k):
                cluster_points = features[labels == i]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    wk += np.sum((cluster_points - centroid) ** 2)
            
            log_wk = np.log(wk) if wk > 0 else 0
            log_wks.append(log_wk)
            
            # Calculate expected log(Wk) under null hypothesis
            ref_log_wks = []
            for _ in range(n_refs):
                # Generate random reference data with same bounds
                ref_data = np.random.uniform(
                    features.min(axis=0), features.max(axis=0), (n_samples, n_features)
                )
                
                kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels_ref = kmeans_ref.fit_predict(ref_data)
                
                wk_ref = 0
                for i in range(k):
                    cluster_points_ref = ref_data[labels_ref == i]
                    if len(cluster_points_ref) > 0:
                        centroid_ref = np.mean(cluster_points_ref, axis=0)
                        wk_ref += np.sum((cluster_points_ref - centroid_ref) ** 2)
                
                log_wk_ref = np.log(wk_ref) if wk_ref > 0 else 0
                ref_log_wks.append(log_wk_ref)
            
            # Calculate gap and standard error
            expected_log_wk = np.mean(ref_log_wks)
            gap = expected_log_wk - log_wk
            gaps.append(gap)
            
            # Calculate standard error
            sk = np.std(ref_log_wks) * np.sqrt(1 + 1/n_refs)
            sk_values.append(sk)
        
        # Find optimal K using gap statistic rule
        optimal_k_gap = list(k_range)[0]
        for i, k in enumerate(k_range):
            if i < len(gaps) - 1:
                if gaps[i] >= gaps[i + 1] - sk_values[i + 1]:
                    optimal_k_gap = k
                    break
        
        return {
            'k_range': list(k_range),
            'gaps': gaps,
            'log_wks': log_wks,
            'sk_values': sk_values,
            'optimal_k_gap': optimal_k_gap,
            'max_gap': max(gaps) if gaps else 0,
            'max_gap_k': list(k_range)[np.argmax(gaps)] if gaps else list(k_range)[0]
        }
    
    def analyze_category_k_selection(self, category_name: str, features_dict: Dict, 
                                   k_max: int = 12) -> Dict:
        """
        Perform comprehensive K-selection analysis for a single category.
        
        Args:
            category_name: Name of the lifespan category
            features_dict: Features for tokens in this category
            k_max: Maximum K to test
            
        Returns:
            Complete K-selection analysis results
        """
        if len(features_dict) < 6:  # Need minimum tokens for meaningful clustering
            return {
                'category': category_name,
                'n_tokens': len(features_dict),
                'analysis_possible': False,
                'reason': f'Insufficient tokens for analysis (need ≥6, got {len(features_dict)})'
            }
        
        print(f"    Analyzing {category_name}: {len(features_dict)} tokens")
        
        # Prepare features for clustering
        features, token_names = self.clustering_engine.prepare_features_for_clustering(features_dict)
        
        # Determine appropriate K range
        max_k = min(k_max, len(features_dict) // 2, 12)
        k_range = range(2, max_k + 1)
        
        # Standard elbow and silhouette analysis
        k_analysis = self.clustering_engine.find_optimal_k(features, k_range)
        
        # Gap statistic analysis
        gap_analysis = self.calculate_gap_statistic(features, k_range)
        
        # Additional clustering metrics for each K
        detailed_metrics = {}
        for k in k_range:
            result = self.clustering_engine.cluster_and_evaluate(features, k)
            
            # Calculate additional metrics
            detailed_metrics[k] = {
                'silhouette_score': result['silhouette_score'],
                'inertia': result['inertia'],
                'calinski_harabasz': self._calculate_calinski_harabasz(features, result['labels']),
                'davies_bouldin': self._calculate_davies_bouldin(features, result['labels'])
            }
        
        # CEO-style optimal K selection: prioritize elbow, then gap, then silhouette
        optimal_k_elbow = k_analysis['optimal_k_elbow']
        optimal_k_gap = gap_analysis['optimal_k_gap']
        optimal_k_silhouette = k_analysis['optimal_k_silhouette']
        
        # Final recommendation logic
        final_k = self._select_final_k(optimal_k_elbow, optimal_k_gap, optimal_k_silhouette,
                                     detailed_metrics, category_name)
        
        # Perform final clustering with recommended K
        final_clustering = self.clustering_engine.cluster_and_evaluate(features, final_k)
        
        return {
            'category': category_name,
            'n_tokens': len(features_dict),
            'analysis_possible': True,
            'k_range_tested': list(k_range),
            'elbow_analysis': k_analysis,
            'gap_analysis': gap_analysis,
            'detailed_metrics': detailed_metrics,
            'recommendations': {
                'elbow_method': optimal_k_elbow,
                'gap_statistic': optimal_k_gap,
                'silhouette_method': optimal_k_silhouette,
                'final_recommendation': final_k,
                'selection_rationale': self._get_selection_rationale(
                    optimal_k_elbow, optimal_k_gap, optimal_k_silhouette, final_k
                )
            },
            'final_clustering': final_clustering,
            'features': features,
            'token_names': token_names
        }
    
    def _calculate_calinski_harabasz(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz index (variance ratio criterion)."""
        from sklearn.metrics import calinski_harabasz_score
        try:
            return calinski_harabasz_score(features, labels)
        except:
            return 0.0
    
    def _calculate_davies_bouldin(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin index (lower is better)."""
        from sklearn.metrics import davies_bouldin_score
        try:
            return davies_bouldin_score(features, labels)
        except:
            return float('inf')
    
    def _select_final_k(self, k_elbow: int, k_gap: int, k_silhouette: int, 
                       detailed_metrics: Dict, category_name: str) -> int:
        """
        Select final K using CEO priorities: elbow first, gap second, silhouette third.
        Updated to properly respect CEO methodology priorities.
        """
        # Set category-specific silhouette thresholds
        if category_name == 'marathon':
            min_threshold = 0.2  # Lower threshold for complex marathon tokens
        else:
            min_threshold = 0.25  # Standard threshold for sprint/standard
        
        # CEO Priority #1: Elbow method - ALWAYS trust structural breaks
        if k_elbow in detailed_metrics:
            elbow_silhouette = detailed_metrics[k_elbow]['silhouette_score']
            # For elbow method, accept even lower thresholds (structural importance)
            if elbow_silhouette >= (min_threshold - 0.05):
                return k_elbow
        
        # CEO Priority #2: Gap statistic - Good statistical foundation
        if k_gap in detailed_metrics:
            gap_silhouette = detailed_metrics[k_gap]['silhouette_score']
            if gap_silhouette >= min_threshold:
                return k_gap
        
        # CEO Priority #3: Silhouette method - Pure optimization
        if k_silhouette in detailed_metrics:
            return k_silhouette
        
        # Ultimate fallback: choose K with best silhouette score
        best_k = max(detailed_metrics.keys(), 
                    key=lambda k: detailed_metrics[k]['silhouette_score'])
        return best_k
    
    def _get_selection_rationale(self, k_elbow: int, k_gap: int, k_silhouette: int, 
                               final_k: int) -> str:
        """Generate explanation for K selection."""
        if final_k == k_elbow:
            return f"Selected elbow method K={k_elbow} (CEO priority #1) - trusting structural break detection"
        elif final_k == k_gap:
            return f"Selected gap statistic K={k_gap} (CEO priority #2) after elbow K={k_elbow} failed minimum silhouette threshold"
        elif final_k == k_silhouette:
            return f"Selected silhouette method K={k_silhouette} (CEO priority #3) as fallback after elbow/gap failed thresholds"
        else:
            return f"Selected K={final_k} based on best overall silhouette score (ultimate fallback)"
    
    def run_k_selection_analysis(self, processed_dir: Path, 
                               standardization_results_path: Path = None,
                               max_tokens_per_category: int = None) -> Dict:
        """
        Run complete K-selection analysis across all lifespan categories.
        
        Args:
            processed_dir: Path to processed data directory (not used - loads from Day 3-4)
            standardization_results_path: Path to Day 3-4 standardization results
            max_tokens_per_category: Maximum tokens per source category (not used - loads from Day 3-4)
            
        Returns:
            Complete K-selection analysis results
        """
        print(f"🚀 Starting Phase 1 Day 5-6 K-Selection Analysis")
        
        # Step 1: Load standardization results - REQUIRED
        if not standardization_results_path:
            raise ValueError("Day 3-4 standardization results are required for K-selection analysis")
        
        print(f"\n📁 Loading standardization results from {standardization_results_path}")
        standardization_results = self.load_standardization_results(standardization_results_path)
        winner = standardization_results['parameters']['winning_method']
        use_log = standardization_results['parameters']['use_log_returns']
        total_tokens = standardization_results['parameters']['total_tokens_processed']
        print(f"✅ Using {winner} returns (log={use_log}) from Day 3-4 winner")
        
        # Step 2: Load lifespan-categorized features from Day 3-4
        print(f"\n📁 Loading lifespan-categorized features from Day 3-4...")
        lifespan_categories, category_stats = self.load_lifespan_categories_from_day34(standardization_results)
        
        # Step 3: Perform K-selection analysis for each category
        print(f"\n🔍 Performing K-selection analysis...")
        category_analyses = {}
        
        for category_name, features_dict in lifespan_categories.items():
            if category_stats[category_name]['count'] > 0:
                category_analyses[category_name] = self.analyze_category_k_selection(
                    category_name, features_dict
                )
            else:
                category_analyses[category_name] = {
                    'category': category_name,
                    'n_tokens': 0,
                    'analysis_possible': False,
                    'reason': 'No tokens in this category'
                }
        
        # Step 4: Generate cross-category insights
        print(f"\n💡 Generating cross-category insights...")
        cross_category_insights = self._generate_cross_category_insights(category_analyses)
        
        # Step 5: Create Phase 7-8 preparation recommendations
        phase78_recommendations = self._create_phase78_recommendations(category_analyses)
        
        # Compile complete results
        complete_results = {
            'analysis_type': 'k_selection_per_category',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'winning_method': winner,
                'use_log_returns': use_log,
                'total_tokens_processed': total_tokens,
                'day34_standardization_file': standardization_results_path.name if standardization_results_path else None
            },
            'standardization_reference': standardization_results_path.name if standardization_results_path else None,
            'lifespan_categorization': category_stats,
            'category_analyses': category_analyses,
            'cross_category_insights': cross_category_insights,
            'phase78_recommendations': phase78_recommendations,
            'summary_recommendations': self._generate_summary_recommendations(category_analyses)
        }
        
        print(f"\n🎉 K-selection analysis complete!")
        
        # Print summary
        for category, analysis in category_analyses.items():
            if analysis['analysis_possible']:
                final_k = analysis['recommendations']['final_recommendation']
                silhouette = analysis['final_clustering']['silhouette_score']
                print(f"  {category}: K={final_k} (silhouette={silhouette:.3f})")
            else:
                print(f"  {category}: {analysis['reason']}")
        
        return complete_results
    
    def _generate_cross_category_insights(self, category_analyses: Dict) -> Dict:
        """Generate insights comparing K-selection across categories."""
        insights = {}
        
        # Collect successful analyses
        successful_analyses = {k: v for k, v in category_analyses.items() 
                             if v['analysis_possible']}
        
        if not successful_analyses:
            return {'no_successful_analyses': True}
        
        # Compare optimal K values
        k_values = {}
        silhouette_scores = {}
        
        for category, analysis in successful_analyses.items():
            k_values[category] = analysis['recommendations']['final_recommendation']
            silhouette_scores[category] = analysis['final_clustering']['silhouette_score']
        
        insights['optimal_k_comparison'] = k_values
        insights['silhouette_comparison'] = silhouette_scores
        
        # Analyze patterns
        if len(k_values) > 1:
            k_variance = np.var(list(k_values.values()))
            avg_k = np.mean(list(k_values.values()))
            
            insights['k_consistency'] = {
                'average_k': avg_k,
                'k_variance': k_variance,
                'consistent': k_variance < 1.0,
                'pattern': self._describe_k_pattern(k_values)
            }
        
        # Quality assessment
        avg_silhouette = np.mean(list(silhouette_scores.values()))
        min_silhouette = min(silhouette_scores.values())
        
        insights['clustering_quality'] = {
            'average_silhouette': avg_silhouette,
            'minimum_silhouette': min_silhouette,
            'quality_assessment': self._assess_clustering_quality(avg_silhouette, min_silhouette)
        }
        
        return insights
    
    def _describe_k_pattern(self, k_values: Dict[str, int]) -> str:
        """Describe the pattern of K values across categories."""
        categories = ['sprint', 'standard', 'marathon']
        available_k = [k_values.get(cat) for cat in categories if cat in k_values]
        
        if len(available_k) < 2:
            return "insufficient_data"
        elif len(set(available_k)) == 1:
            return "uniform_k"
        elif len(available_k) == 3:
            sprint_k, standard_k, marathon_k = [k_values.get(cat, 0) for cat in categories]
            if sprint_k < standard_k < marathon_k:
                return "increasing_with_lifespan"
            elif sprint_k > standard_k > marathon_k:
                return "decreasing_with_lifespan"
            else:
                return "mixed_pattern"
        else:
            return "partial_pattern"
    
    def _assess_clustering_quality(self, avg_silhouette: float, min_silhouette: float) -> str:
        """Assess overall clustering quality."""
        if avg_silhouette >= 0.5 and min_silhouette >= 0.3:
            return "excellent"
        elif avg_silhouette >= 0.3 and min_silhouette >= 0.2:
            return "good"
        elif avg_silhouette >= 0.2:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def _create_phase78_recommendations(self, category_analyses: Dict) -> Dict:
        """Create recommendations for Phase 7-8 stability testing."""
        recommendations = {}
        
        for category, analysis in category_analyses.items():
            if analysis['analysis_possible']:
                optimal_k = analysis['recommendations']['final_recommendation']
                silhouette = analysis['final_clustering']['silhouette_score']
                
                recommendations[category] = {
                    'recommended_k': optimal_k,
                    'expected_silhouette': silhouette,
                    'stability_test_parameters': {
                        'n_stability_runs': 10 if silhouette >= 0.4 else 20,
                        'bootstrap_samples': 5,
                        'target_ari': 0.75,
                        'min_silhouette': 0.3
                    },
                    'clustering_method': 'kmeans',
                    'preprocessing': f"standardization_winner_method",
                    'ready_for_stability_testing': silhouette >= 0.2
                }
            else:
                recommendations[category] = {
                    'ready_for_stability_testing': False,
                    'reason': analysis['reason'],
                    'suggested_action': 'increase_token_count' if 'insufficient' in analysis.get('reason', '').lower() else 'review_data_quality'
                }
        
        return recommendations
    
    def _generate_summary_recommendations(self, category_analyses: Dict) -> List[str]:
        """Generate actionable summary recommendations."""
        recommendations = []
        
        # Count successful analyses
        successful = sum(1 for analysis in category_analyses.values() 
                        if analysis['analysis_possible'])
        total = len(category_analyses)
        
        if successful == 0:
            recommendations.extend([
                "⚠️ No categories had sufficient tokens for K-selection analysis",
                "💡 Increase token sample sizes or review data quality",
                "🔍 Consider relaxing minimum token requirements for analysis"
            ])
        elif successful < total:
            recommendations.append(f"✅ {successful}/{total} categories ready for stability testing")
            recommendations.append("⚠️ Some categories need more tokens for robust analysis")
        else:
            recommendations.append(f"🎉 All {total} categories ready for Phase 7-8 stability testing")
        
        # Quality-based recommendations
        if successful > 0:
            avg_silhouette = np.mean([
                analysis['final_clustering']['silhouette_score'] 
                for analysis in category_analyses.values() 
                if analysis['analysis_possible']
            ])
            
            if avg_silhouette >= 0.4:
                recommendations.append("✅ Excellent clustering quality across categories")
            elif avg_silhouette >= 0.3:
                recommendations.append("✅ Good clustering quality - proceed with stability testing")
            else:
                recommendations.append("⚠️ Clustering quality below optimal - consider feature engineering improvements")
        
        # Next steps
        recommendations.extend([
            "🚀 Ready for Phase 7-8: Use recommended K values for stability testing",
            "📊 Monitor ARI scores ≥0.75 and silhouette scores ≥0.3 in stability tests",
            "🔄 Consider bootstrapping for categories with marginal clustering quality"
        ])
        
        return recommendations
    
    def save_results(self, results: Dict, output_dir: Path = None) -> str:
        """Save complete analysis results."""
        if output_dir:
            self.results_manager = ResultsManager(output_dir)
        
        timestamp = self.results_manager.save_analysis_results(
            results, 
            analysis_name="k_selection", 
            phase_dir="phase1_day5_6_k_selection",
            include_plots=True
        )
        
        return timestamp


def create_gradio_interface():
    """Create interactive Gradio interface for K-selection analysis."""
    visualizer = GradioVisualizer("Phase 1 Day 5-6: K-Selection per Category")
    analyzer = KSelectionAnalyzer()
    
    def run_interactive_analysis(data_dir_str: str, standardization_results_str: str, 
                                max_tokens: int):
        """Run analysis with Gradio inputs."""
        try:
            data_dir = Path(data_dir_str)  # Not used anymore, kept for interface compatibility
            
            if not standardization_results_str.strip():
                return "❌ Day 3-4 standardization results file is required", None, None, None
            
            standardization_results_path = Path(standardization_results_str)
            if not standardization_results_path.exists():
                return f"❌ Standardization results file not found: {standardization_results_str}", None, None, None
            
            # Run analysis
            results = analyzer.run_k_selection_analysis(
                data_dir, standardization_results_path, max_tokens
            )
            
            # Save results
            timestamp = analyzer.save_results(results)
            
            # Generate summary
            successful_categories = sum(1 for analysis in results['category_analyses'].values() 
                                      if analysis['analysis_possible'])
            total_categories = len(results['category_analyses'])
            
            summary = f"""
            ## 🎉 K-Selection Analysis Complete!
            
            **Method Used**: {results['parameters']['winning_method'].upper()} returns
            **Total Tokens Processed**: {results['parameters']['total_tokens_processed']:,}
            **Categories Analyzed**: {successful_categories}/{total_categories}
            
            **Optimal K Values**:
            """
            
            # Add category-specific results
            for category, analysis in results['category_analyses'].items():
                if analysis['analysis_possible']:
                    k = analysis['recommendations']['final_recommendation']
                    sil = analysis['final_clustering']['silhouette_score']
                    summary += f"\n- **{category.title()}**: K={k} (silhouette={sil:.3f})"
                else:
                    summary += f"\n- **{category.title()}**: ❌ {analysis['reason']}"
            
            summary += f"\n\n**Quality Assessment**: {results['cross_category_insights'].get('clustering_quality', {}).get('quality_assessment', 'N/A').title()}"
            
            summary += f"\n\n**Key Recommendations**:\n"
            summary += "\n".join([f"- {rec}" for rec in results['summary_recommendations'][:5]])
            
            summary += f"\n\n**Results saved with timestamp**: {timestamp}"
            
            # Generate plots (simplified for K-selection)
            plots = [None, None, None]  # Placeholder for consistency
            
            # Create K-selection comparison plot
            if successful_categories > 0:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Optimal K by Category', 'Silhouette Scores'),
                    specs=[[{"type": "bar"}, {"type": "bar"}]]
                )
                
                categories = []
                k_values = []
                silhouette_values = []
                
                for category, analysis in results['category_analyses'].items():
                    if analysis['analysis_possible']:
                        categories.append(category.title())
                        k_values.append(analysis['recommendations']['final_recommendation'])
                        silhouette_values.append(analysis['final_clustering']['silhouette_score'])
                
                fig.add_trace(
                    go.Bar(x=categories, y=k_values, name='Optimal K', 
                          marker_color='steelblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=categories, y=silhouette_values, name='Silhouette Score',
                          marker_color='lightcoral'),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title='K-Selection Results by Lifespan Category',
                    height=400,
                    showlegend=False
                )
                
                plots[0] = fig
            
            return summary, *plots
            
        except Exception as e:
            error_msg = f"## ❌ Error During Analysis\n\n```\n{str(e)}\n```"
            return error_msg, None, None, None
    
    import gradio as gr
    
    interface = gr.Interface(
        fn=run_interactive_analysis,
        inputs=[
            gr.Textbox(
                value="/Users/brunostordeur/Docs/GitHub/Solana/memecoin2/data/processed",
                label="Processed Data Directory (not used - kept for compatibility)",
                placeholder="Path to processed data directory"
            ),
            gr.Textbox(
                value="",
                label="Day 3-4 Standardization Results File (JSON) - REQUIRED",
                placeholder="Path to Day 3-4 standardization results JSON file (REQUIRED)"
            ),
            gr.Number(
                value=200,
                label="Max Tokens per Source Category (not used - kept for compatibility)",
                precision=0
            )
        ],
        outputs=[
            gr.Markdown(label="K-Selection Analysis Results"),
            gr.Plot(label="K-Selection Comparison"),
            gr.Plot(label="Analysis Metrics"),
            gr.Plot(label="Category Insights")
        ],
        title="🚀 Phase 1 Day 5-6: K-Selection per Category",
        description="Find optimal number of clusters for each lifespan category using standardized features"
    )
    
    return interface


def main():
    """Main entry point with CLI and interactive modes."""
    parser = argparse.ArgumentParser(description="Phase 1 Day 5-6 K-Selection Analysis")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path("../../data/processed"),
                       help="Path to processed data directory (not used - kept for compatibility)")
    parser.add_argument("--standardization-results", type=Path,
                       help="Path to Day 3-4 standardization results JSON file (REQUIRED for CLI mode)")
    parser.add_argument("--max-tokens", type=int,
                       help="Maximum tokens per source category (not used - kept for compatibility)")
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
        # CLI mode - require standardization results
        if not args.standardization_results:
            print("❌ Error: --standardization-results is required for CLI mode")
            print("Use --interactive for Gradio interface or provide --standardization-results PATH")
            return
        
        analyzer = KSelectionAnalyzer(args.output_dir)
        results = analyzer.run_k_selection_analysis(
            args.data_dir, args.standardization_results, args.max_tokens
        )
        timestamp = analyzer.save_results(results)
        
        print(f"\n📁 Results saved to: {args.output_dir}/phase1_day5_6_k_selection/")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"✅ Analysis complete for {len([a for a in results['category_analyses'].values() if a['analysis_possible']])} categories")


if __name__ == "__main__":
    main()