#!/usr/bin/env python3
# phase1_day1_2_baseline_assessment.py
"""
Phase 1 Day 1-2: Baseline Assessment & Returns Testing

CEO Roadmap Implementation:
- Extract 15 exact features from raw data
- A/B test raw vs safe-log returns on 1k subsample  
- Evaluate ARI, silhouette, behavioral separation, scale invariance
- Pick winner by highest separation + cross-ARI (60%), then ARI/sil
- Output winner selection + initial cleaning refinement ideas

Usage:
    python phase1_day1_2_baseline_assessment.py [--data-dir PATH] [--n-tokens 1000] [--output-dir PATH]
    
Interactive Mode:
    python phase1_day1_2_baseline_assessment.py --interactive
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
    load_subsample_tokens, extract_features_from_returns, ClusteringEngine,
    ResultsManager, GradioVisualizer, create_comparison_interface,
    get_base_price_groups, prepare_token_for_analysis
)


class BaselineAssessmentAnalyzer:
    """
    Implements CEO roadmap Phase 1 Day 1-2: Baseline Assessment & Returns Testing.
    """
    
    def __init__(self, results_dir: Path = None):
        self.results_manager = ResultsManager(results_dir or Path("../results"))
        self.clustering_engine = ClusteringEngine(random_state=42)
        
    def run_ab_test_analysis(self, processed_dir: Path, n_tokens: int = 1000, 
                           categories: List[str] = None) -> Dict:
        """
        Run A/B test comparing raw vs log returns following CEO specifications.
        
        Args:
            processed_dir: Path to processed data directory
            n_tokens: Number of tokens to sample for testing
            categories: Categories to sample from (default: all available)
            
        Returns:
            Complete A/B test results with winner selection
        """
        print(f"🚀 Starting Phase 1 Day 1-2 Baseline Assessment")
        print(f"📊 Testing raw vs log returns on {n_tokens} token subsample")
        
        # Step 1: Load subsample tokens
        print("\n📁 Loading token subsample...")
        token_data = load_subsample_tokens(
            processed_dir, n_tokens=n_tokens, categories=categories, seed=42
        )
        
        if not token_data:
            raise ValueError("No token data loaded")
        
        print(f"✅ Loaded {len(token_data)} tokens")
        
        # Step 2: Extract features using both methods
        print("\n🔧 Extracting features...")
        raw_features_dict = self._extract_features_dict(token_data, use_log=False)
        log_features_dict = self._extract_features_dict(token_data, use_log=True)
        
        print(f"✅ Extracted 15 features for {len(raw_features_dict)} tokens")
        
        # Step 3: Perform clustering analysis for both methods
        print("\n🔍 Performing clustering analysis...")
        raw_results = self._perform_clustering_analysis(raw_features_dict, "raw_returns")
        log_results = self._perform_clustering_analysis(log_features_dict, "log_returns")
        
        # Step 4: Scale invariance testing
        print("\n⚖️ Testing scale invariance...")
        scale_results = self._test_scale_invariance(token_data, raw_features_dict, log_features_dict)
        
        # Step 5: Behavioral separation testing (if labels available)
        print("\n🎯 Testing behavioral separation...")
        separation_results = self._test_behavioral_separation(raw_results, log_results)
        
        # Step 6: Winner selection based on CEO criteria
        print("\n🏆 Selecting winner based on CEO criteria...")
        winner_analysis = self._select_winner(raw_results, log_results, scale_results, separation_results)
        
        # Step 7: Generate cleaning refinement ideas
        print("\n💡 Generating cleaning refinement ideas...")
        cleaning_ideas = self._generate_cleaning_ideas(winner_analysis, raw_results, log_results)
        
        # Compile complete results
        complete_results = {
            'analysis_type': 'baseline_assessment_ab_test',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'n_tokens': len(token_data),
                'categories': categories,
                'random_seed': 42
            },
            'raw_returns_analysis': raw_results,
            'log_returns_analysis': log_results,
            'scale_invariance': scale_results,
            'behavioral_separation': separation_results,
            'winner_selection': winner_analysis,
            'cleaning_refinement_ideas': cleaning_ideas,
            'token_list': list(token_data.keys())
        }
        
        print(f"\n🎉 Analysis complete! Winner: {winner_analysis['winner']}")
        print(f"📊 Results: ARI={winner_analysis['winner_metrics']['ari']:.3f}, "
              f"Silhouette={winner_analysis['winner_metrics']['silhouette']:.3f}")
        
        return complete_results
    
    def _extract_features_dict(self, token_data: Dict, use_log: bool) -> Dict[str, Dict[str, float]]:
        """Extract features for all tokens using specified method."""
        features_dict = {}
        
        for token_name, token_df in token_data.items():
            try:
                prices, returns = prepare_token_for_analysis(token_df)
                features = extract_features_from_returns(returns, prices, use_log=use_log)
                features_dict[token_name] = features
            except Exception as e:
                print(f"Warning: Failed to extract features for {token_name}: {e}")
                continue
        
        return features_dict
    
    def _perform_clustering_analysis(self, features_dict: Dict, method_name: str) -> Dict:
        """Perform comprehensive clustering analysis."""
        print(f"  🔍 Analyzing {method_name}...")
        
        # Adjust K-range based on number of samples (need at least 2 samples per cluster)
        n_samples = len(features_dict)
        max_k = min(10, max(3, n_samples // 2))
        k_range = range(2, max_k + 1)
        
        # Use clustering engine for comprehensive analysis
        analysis = self.clustering_engine.comprehensive_analysis(
            features_dict, k_range=k_range, stability_runs=5
        )
        
        # Add method identifier
        analysis['method'] = method_name
        
        return analysis
    
    def _test_scale_invariance(self, token_data: Dict, raw_features: Dict, log_features: Dict) -> Dict:
        """Test scale invariance across different base price groups."""
        # Split tokens by base price
        low_tokens, high_tokens = get_base_price_groups(token_data)
        
        if len(low_tokens) < 10 or len(high_tokens) < 10:
            return {
                'scale_invariance_testable': False,
                'reason': 'Insufficient tokens in price groups'
            }
        
        # Test raw returns scale invariance
        raw_low_features = {t: raw_features[t] for t in low_tokens if t in raw_features}
        raw_high_features = {t: raw_features[t] for t in high_tokens if t in raw_features}
        
        raw_scale_test = self.clustering_engine.test_scale_invariance(
            self.clustering_engine.prepare_features_for_clustering(raw_low_features)[0],
            self.clustering_engine.prepare_features_for_clustering(raw_high_features)[0],
            k=5
        )
        
        # Test log returns scale invariance  
        log_low_features = {t: log_features[t] for t in low_tokens if t in log_features}
        log_high_features = {t: log_features[t] for t in high_tokens if t in log_features}
        
        log_scale_test = self.clustering_engine.test_scale_invariance(
            self.clustering_engine.prepare_features_for_clustering(log_low_features)[0],
            self.clustering_engine.prepare_features_for_clustering(log_high_features)[0],
            k=5
        )
        
        return {
            'scale_invariance_testable': True,
            'n_low_price_tokens': len(low_tokens),
            'n_high_price_tokens': len(high_tokens),
            'raw_returns_scale_test': raw_scale_test,
            'log_returns_scale_test': log_scale_test,
            'raw_scale_score': self._calculate_scale_score(raw_scale_test),
            'log_scale_score': self._calculate_scale_score(log_scale_test)
        }
    
    def _calculate_scale_score(self, scale_test: Dict) -> float:
        """Calculate scale invariance score (higher = more scale invariant)."""
        # Score based on similarity of silhouette scores across price groups
        sil_diff = scale_test['silhouette_difference']
        return max(0, 1.0 - sil_diff)  # Higher score = more similar
    
    def _test_behavioral_separation(self, raw_results: Dict, log_results: Dict) -> Dict:
        """Test behavioral separation quality."""
        # For now, use silhouette score as proxy for behavioral separation
        # In a real implementation, this would use manually labeled behavioral patterns
        
        raw_separation = raw_results['final_clustering']['silhouette_score']
        log_separation = log_results['final_clustering']['silhouette_score']
        
        return {
            'separation_method': 'silhouette_proxy',
            'raw_separation_score': raw_separation,
            'log_separation_score': log_separation,
            'separation_winner': 'raw' if raw_separation > log_separation else 'log'
        }
    
    def _select_winner(self, raw_results: Dict, log_results: Dict, 
                      scale_results: Dict, separation_results: Dict) -> Dict:
        """Select winner based on CEO criteria: separation + cross-ARI (60%), then ARI/sil."""
        
        # Extract key metrics
        raw_ari = raw_results['stability']['mean_ari']
        log_ari = log_results['stability']['mean_ari']
        
        raw_sil = raw_results['final_clustering']['silhouette_score']
        log_sil = log_results['final_clustering']['silhouette_score']
        
        raw_separation = separation_results['raw_separation_score']
        log_separation = separation_results['log_separation_score']
        
        # Scale invariance scores
        if scale_results['scale_invariance_testable']:
            raw_scale = scale_results['raw_scale_score']
            log_scale = scale_results['log_scale_score']
        else:
            raw_scale = 0.5  # Neutral score if not testable
            log_scale = 0.5
        
        # CEO scoring: separation + cross-ARI weighted 60%, then ARI/sil
        raw_primary_score = 0.6 * (raw_separation + raw_scale) / 2 + 0.4 * (raw_ari + raw_sil) / 2
        log_primary_score = 0.6 * (log_separation + log_scale) / 2 + 0.4 * (log_ari + log_sil) / 2
        
        # Determine winner
        if raw_primary_score > log_primary_score:
            winner = 'raw'
            winner_metrics = {
                'ari': raw_ari,
                'silhouette': raw_sil,
                'separation': raw_separation,
                'scale_invariance': raw_scale,
                'primary_score': raw_primary_score
            }
        else:
            winner = 'log'
            winner_metrics = {
                'ari': log_ari,
                'silhouette': log_sil,
                'separation': log_separation,
                'scale_invariance': log_scale,
                'primary_score': log_primary_score
            }
        
        # Check CEO thresholds
        meets_ceo_requirements = {
            'ari_threshold': winner_metrics['ari'] >= 0.75,
            'silhouette_threshold': winner_metrics['silhouette'] >= 0.5,
            'overall_pass': winner_metrics['ari'] >= 0.75 and winner_metrics['silhouette'] >= 0.5
        }
        
        return {
            'winner': winner,
            'winner_metrics': winner_metrics,
            'loser_metrics': {
                'ari': log_ari if winner == 'raw' else raw_ari,
                'silhouette': log_sil if winner == 'raw' else raw_sil,
                'separation': log_separation if winner == 'raw' else raw_separation,
                'scale_invariance': log_scale if winner == 'raw' else raw_scale,
                'primary_score': log_primary_score if winner == 'raw' else raw_primary_score
            },
            'score_difference': abs(raw_primary_score - log_primary_score),
            'meets_ceo_requirements': meets_ceo_requirements,
            'ceo_scoring_breakdown': {
                'separation_scale_weight': 0.6,
                'ari_silhouette_weight': 0.4,
                'raw_primary_score': raw_primary_score,
                'log_primary_score': log_primary_score
            }
        }
    
    def _generate_cleaning_ideas(self, winner_analysis: Dict, raw_results: Dict, log_results: Dict) -> Dict:
        """Generate archetype-based cleaning refinement ideas."""
        winner = winner_analysis['winner']
        winner_results = raw_results if winner == 'raw' else log_results
        
        # Analyze cluster characteristics for cleaning insights
        cluster_labels = winner_results['final_clustering']['labels']
        features_matrix = winner_results['features']
        token_names = winner_results['token_names']
        
        # Simple clustering-based insights
        n_clusters = winner_results['optimal_k']
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        
        cleaning_ideas = {
            'recommended_method': f"{winner}_returns",
            'method_justification': f"Won with primary score {winner_analysis['winner_metrics']['primary_score']:.3f}",
            'cluster_analysis': {
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes,
                'largest_cluster_pct': max(cluster_sizes) / len(cluster_labels) * 100,
                'smallest_cluster_pct': min(cluster_sizes) / len(cluster_labels) * 100
            },
            'cleaning_recommendations': [
                f"Use {winner} returns for feature extraction in subsequent phases",
                f"Consider cluster-specific filtering if largest cluster >70% (current: {max(cluster_sizes) / len(cluster_labels) * 100:.1f}%)",
                "Investigate smallest clusters for potential noise/outliers",
                f"Target ARI >0.75 (current: {winner_analysis['winner_metrics']['ari']:.3f}) in stability improvements",
                f"Target silhouette >0.5 (current: {winner_analysis['winner_metrics']['silhouette']:.3f}) in future iterations"
            ],
            'next_phase_parameters': {
                'use_log_returns': winner == 'log',
                'optimal_k_estimate': n_clusters,
                'stability_baseline': winner_analysis['winner_metrics']['ari']
            }
        }
        
        return cleaning_ideas
    
    def save_results(self, results: Dict, output_dir: Path = None) -> str:
        """Save complete analysis results."""
        if output_dir:
            self.results_manager = ResultsManager(output_dir)
        
        timestamp = self.results_manager.save_analysis_results(
            results, 
            analysis_name="baseline_assessment", 
            phase_dir="phase1_day1_2_baseline",
            include_plots=True
        )
        
        return timestamp


def create_gradio_interface():
    """Create interactive Gradio interface for baseline assessment."""
    visualizer = GradioVisualizer("Phase 1 Day 1-2: Baseline Assessment")
    analyzer = BaselineAssessmentAnalyzer()
    
    def run_interactive_analysis(data_dir_str: str, n_tokens: int, categories_str: str):
        """Run analysis with Gradio inputs."""
        try:
            data_dir = Path(data_dir_str)
            if not data_dir.exists():
                return "❌ Data directory not found", None, None, None
            
            # Parse categories
            if categories_str.strip():
                categories = [c.strip() for c in categories_str.split(',')]
            else:
                categories = None
            
            # Run analysis
            results = analyzer.run_ab_test_analysis(data_dir, n_tokens, categories)
            
            # Save results
            timestamp = analyzer.save_results(results)
            
            # Generate summary
            winner = results['winner_selection']['winner']
            metrics = results['winner_selection']['winner_metrics']
            
            summary = f"""
            ## 🎉 Analysis Complete!
            
            **Winner**: {winner.upper()} returns
            
            **Key Metrics**:
            - ARI: {metrics['ari']:.4f} {'✅' if metrics['ari'] >= 0.75 else '❌'} (target: ≥0.75)
            - Silhouette: {metrics['silhouette']:.4f} {'✅' if metrics['silhouette'] >= 0.5 else '❌'} (target: ≥0.5)
            - Primary Score: {metrics['primary_score']:.4f}
            
            **Recommendations for Phase 2**:
            {chr(10).join(['- ' + rec for rec in results['cleaning_refinement_ideas']['cleaning_recommendations']])}
            
            **Results saved with timestamp**: {timestamp}
            """
            
            # Generate plots
            winner_results = (results['raw_returns_analysis'] if winner == 'raw' 
                            else results['log_returns_analysis'])
            
            k_plot = visualizer.create_k_selection_plot(winner_results['k_analysis'])
            stability_plot = visualizer.create_stability_plot(winner_results['stability'])
            tsne_plot_2d = visualizer.create_tsne_plot(
                winner_results['tsne_2d'],
                winner_results['final_clustering']['labels'],
                winner_results['token_names']
            )
            tsne_plot_3d = visualizer.create_tsne_plot_3d(
                winner_results['tsne_3d'],
                winner_results['final_clustering']['labels'],
                winner_results['token_names']
            )
            
            return summary, k_plot, stability_plot, tsne_plot_2d, tsne_plot_3d
            
        except Exception as e:
            error_msg = f"## ❌ Error During Analysis\n\n```\n{str(e)}\n```"
            return error_msg, None, None, None, None
    
    import gradio as gr
    
    interface = gr.Interface(
        fn=run_interactive_analysis,
        inputs=[
            gr.Textbox(
                value="/Users/brunostordeur/Docs/GitHub/Solana/memecoin2/data/processed",
                label="Processed Data Directory",
                placeholder="Path to processed data directory"
            ),
            gr.Number(value=1000, label="Number of Tokens to Sample"),
            gr.Textbox(
                value="normal_tokens,extreme_tokens,dead_tokens",
                label="Categories (comma-separated, leave blank for all)",
                placeholder="normal_tokens,extreme_tokens"
            )
        ],
        outputs=[
            gr.Markdown(label="Analysis Results"),
            gr.Plot(label="K-Selection Analysis"),
            gr.Plot(label="Stability Analysis"),
            gr.Plot(label="2D t-SNE Visualization"),
            gr.Plot(label="3D t-SNE Visualization")
        ],
        title="🚀 Phase 1 Day 1-2: Baseline Assessment & A/B Testing",
        description="Compare raw vs log returns for memecoin behavioral archetype analysis"
    )
    
    return interface


def main():
    """Main entry point with CLI and interactive modes."""
    parser = argparse.ArgumentParser(description="Phase 1 Day 1-2 Baseline Assessment")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path("../data/processed"),
                       help="Path to processed data directory")
    parser.add_argument("--n-tokens", type=int, default=1000,
                       help="Number of tokens to sample")
    parser.add_argument("--categories", nargs="+", 
                       default=["normal_tokens", "extreme_tokens", "dead_tokens"],
                       help="Categories to include")
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
        analyzer = BaselineAssessmentAnalyzer(args.output_dir)
        results = analyzer.run_ab_test_analysis(args.data_dir, args.n_tokens, args.categories)
        timestamp = analyzer.save_results(results)
        
        print(f"\n📁 Results saved to: {args.output_dir}/phase1_day1_2_baseline/")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"🏆 Winner: {results['winner_selection']['winner']} returns")


if __name__ == "__main__":
    main()