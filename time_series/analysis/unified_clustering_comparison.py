#!/usr/bin/env python3
"""
Unified vs Per-Category Clustering Comparison

Tests whether unified clustering (all categories together) outperforms 
per-category clustering using the ESSENTIAL_FEATURES set.

Strategy:
1. Test on 5k subsample using ESSENTIAL_FEATURES
2. If silhouette > 0.5, test on full 30k dataset
3. Compare performance between unified vs per-category approaches
"""

import json
import argparse
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import clustering engine
import sys
sys.path.append('../utils')
from clustering_engine import ClusteringEngine
from archetype_classifier import ArchetypeClassifier

# Import feature extraction
sys.path.append('../utils')
import feature_extraction_15

class UnifiedClusteringComparison:
    """Compare unified vs per-category clustering performance."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("../results")
        self.archetype_classifier = ArchetypeClassifier()
        self.clustering_engine = ClusteringEngine(random_state=42)
        
        # Use the actual ESSENTIAL_FEATURES from feature_extraction_15
        self.ESSENTIAL_FEATURES = list(feature_extraction_15.ESSENTIAL_FEATURES.keys())
    
    def load_and_prepare_data(self, archetype_results_path: Path, data_dir: Path, 
                             use_full_dataset: bool = True) -> Dict[str, Dict[str, float]]:
        """Load and prepare data for clustering comparison."""
        dataset_desc = "full 30k dataset" if use_full_dataset else "5k sample"
        print(f"📊 Loading data for clustering comparison ({dataset_desc})...")
        
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
        
        # Use either full dataset or sample
        all_tokens = [
            token_name for token_name, labels in self.archetype_classifier.token_labels.items()
            if token_name in token_data
        ]
        
        if use_full_dataset:
            selected_tokens = all_tokens
            print(f"✅ Using full dataset: {len(selected_tokens)} tokens")
        else:
            sample_size = 5000
            if len(all_tokens) > sample_size:
                np.random.seed(42)  # For reproducibility
                selected_tokens = np.random.choice(all_tokens, size=sample_size, replace=False).tolist()
            else:
                selected_tokens = all_tokens
            print(f"✅ Sample dataset created: {len(selected_tokens)} tokens")
        
        # Create token dataset
        balanced_tokens = {
            token_name: token_data[token_name] for token_name in selected_tokens
        }
        
        # Extract features using the proper feature extraction function
        features_dict = {}
        
        print(f"📊 Extracting {len(self.ESSENTIAL_FEATURES)} essential features...")
        
        for token_name, token_df in balanced_tokens.items():
            # Get token category
            category = self.archetype_classifier.token_labels[token_name]['category']
            
            # Extract prices and calculate returns
            prices = token_df['price'].to_numpy()
            if len(prices) < 2:
                continue
                
            returns = np.diff(prices) / prices[:-1]
            
            # Extract essential features
            token_features = feature_extraction_15.extract_features_from_returns(returns, prices, use_log=False)
            
            # Add category for analysis
            token_features['category'] = category
            
            features_dict[token_name] = token_features
        
        print(f"📊 Essential features extracted: {len(self.ESSENTIAL_FEATURES)}")
        print(f"📊 Tokens with features: {len(features_dict)}")
        
        return features_dict
    
    def run_unified_clustering(self, features_dict: Dict[str, Dict[str, float]]) -> Dict:
        """Run unified clustering on all tokens together."""
        print("🔄 Running unified clustering (all categories together)...")
        
        # Prepare features for clustering (remove category)
        clustering_features = {}
        for token_name, features in features_dict.items():
            clustering_features[token_name] = {
                k: v for k, v in features.items() if k != 'category'
            }
        
        # Run comprehensive clustering analysis with extended K range
        print("🔬 Testing extended K range (3-30) for better cluster discovery...")
        unified_results = self.clustering_engine.comprehensive_analysis(
            clustering_features, 
            k_range=range(3, 31),  # Extended range to find more nuanced patterns
            stability_runs=5,
            category='unified'
        )
        
        # Add category information back for evaluation
        unified_results['token_categories'] = {
            token_name: features['category'] for token_name, features in features_dict.items()
        }
        
        print(f"✅ Unified clustering completed:")
        print(f"  Optimal K: {unified_results['optimal_k']}")
        print(f"  Silhouette Score: {unified_results['final_clustering']['silhouette_score']:.4f}")
        print(f"  Stability (ARI): {unified_results['stability']['mean_ari']:.4f}")
        print(f"  CEO Requirements: {unified_results['meets_ceo_requirements']['stability_achieved']}")
        
        return unified_results
    
    def run_per_category_clustering(self, features_dict: Dict[str, Dict[str, float]]) -> Dict:
        """Run per-category clustering (current approach)."""
        print("🔄 Running per-category clustering (current approach)...")
        
        # Separate tokens by category
        category_tokens = {}
        for token_name, features in features_dict.items():
            category = features['category']
            if category not in category_tokens:
                category_tokens[category] = {}
            category_tokens[category][token_name] = {
                k: v for k, v in features.items() if k != 'category'
            }
        
        # Run clustering for each category
        per_category_results = {}
        overall_tokens = 0
        overall_silhouette_sum = 0
        overall_ari_sum = 0
        
        for category, tokens in category_tokens.items():
            if len(tokens) < 10:  # Skip categories with too few tokens
                print(f"⚠️  Skipping {category}: only {len(tokens)} tokens")
                continue
            
            print(f"📊 Clustering {category}: {len(tokens)} tokens")
            
            category_result = self.clustering_engine.comprehensive_analysis(
                tokens,
                k_range=range(3, min(11, len(tokens))),
                stability_runs=5,
                category=category
            )
            
            per_category_results[category] = category_result
            overall_tokens += len(tokens)
            overall_silhouette_sum += category_result['final_clustering']['silhouette_score'] * len(tokens)
            overall_ari_sum += category_result['stability']['mean_ari'] * len(tokens)
            
            print(f"  ✅ {category} - K: {category_result['optimal_k']}, "
                  f"Silhouette: {category_result['final_clustering']['silhouette_score']:.4f}, "
                  f"ARI: {category_result['stability']['mean_ari']:.4f}")
        
        # Calculate weighted averages
        overall_silhouette = overall_silhouette_sum / overall_tokens if overall_tokens > 0 else 0
        overall_ari = overall_ari_sum / overall_tokens if overall_tokens > 0 else 0
        
        print(f"✅ Per-category clustering completed:")
        print(f"  Weighted Avg Silhouette: {overall_silhouette:.4f}")
        print(f"  Weighted Avg ARI: {overall_ari:.4f}")
        
        return {
            'category_results': per_category_results,
            'overall_silhouette': overall_silhouette,
            'overall_ari': overall_ari,
            'total_tokens': overall_tokens
        }
    
    def compare_approaches(self, unified_results: Dict, per_category_results: Dict, 
                          cluster_analysis: Dict = None) -> Dict:
        """Compare unified vs per-category clustering approaches."""
        print("🔍 Comparing clustering approaches...")
        
        # Extract key metrics
        unified_silhouette = unified_results['final_clustering']['silhouette_score']
        unified_ari = unified_results['stability']['mean_ari']
        unified_meets_ceo = unified_results['meets_ceo_requirements']['stability_achieved']
        
        per_category_silhouette = per_category_results['overall_silhouette']
        per_category_ari = per_category_results['overall_ari']
        
        # Calculate performance differences
        silhouette_diff = unified_silhouette - per_category_silhouette
        ari_diff = unified_ari - per_category_ari
        
        # Determine winner using enhanced criteria
        unified_better = (silhouette_diff > 0.05) and (ari_diff > 0.05)
        per_category_better = (silhouette_diff < -0.05) and (ari_diff < -0.05)
        
        # Enhanced decision criteria with cluster analysis
        should_switch = False
        if cluster_analysis:
            # Calculate top cluster pump rates
            top_clusters = sorted(cluster_analysis.items(), 
                                key=lambda x: x[1]['pump_rate_percent'], reverse=True)[:2]
            
            max_pump_rate = max([cluster[1]['pump_rate_percent'] for cluster in top_clusters]) if top_clusters else 0
            
            # Enhanced criteria: F1 >0.65 equivalent (silhouette + ARI > 1.3) AND pumps >60%
            unified_strong = (unified_silhouette + unified_ari > 1.3) and (max_pump_rate > 60)
            
            if unified_strong:
                should_switch = True
                winner = "unified"
                recommendation = f"Switch to unified clustering (F1 criteria met + {max_pump_rate:.1f}% top pump rate)"
            elif unified_better:
                winner = "unified"
                recommendation = "Switch to unified clustering (metrics improved)"
            elif per_category_better:
                winner = "per_category"
                recommendation = "Continue with per-category clustering"
            else:
                winner = "tie"
                recommendation = "No clear winner - continue with current approach"
        else:
            if unified_better:
                winner = "unified"
                recommendation = "Switch to unified clustering approach"
            elif per_category_better:
                winner = "per_category"
                recommendation = "Continue with per-category clustering"
            else:
                winner = "tie"
                recommendation = "No clear winner - continue with current approach"
        
        comparison_results = {
            'unified_metrics': {
                'silhouette_score': unified_silhouette,
                'mean_ari': unified_ari,
                'meets_ceo_requirements': unified_meets_ceo,
                'optimal_k': unified_results['optimal_k']
            },
            'per_category_metrics': {
                'weighted_silhouette': per_category_silhouette,
                'weighted_ari': per_category_ari,
                'total_tokens': per_category_results['total_tokens'],
                'num_categories': len(per_category_results['category_results'])
            },
            'performance_differences': {
                'silhouette_diff': silhouette_diff,
                'ari_diff': ari_diff,
                'unified_better': unified_better,
                'per_category_better': per_category_better
            },
            'conclusion': {
                'winner': winner,
                'recommendation': recommendation,
                'should_switch': should_switch
            }
        }
        
        print(f"📊 Comparison Results:")
        print(f"  Unified Silhouette: {unified_silhouette:.4f}")
        print(f"  Per-Category Silhouette: {per_category_silhouette:.4f}")
        print(f"  Silhouette Difference: {silhouette_diff:+.4f}")
        print(f"  Unified ARI: {unified_ari:.4f}")
        print(f"  Per-Category ARI: {per_category_ari:.4f}")
        print(f"  ARI Difference: {ari_diff:+.4f}")
        if cluster_analysis:
            top_cluster_rates = [f"{cluster[1]['pump_rate_percent']:.1f}%" for cluster in sorted(cluster_analysis.items(), key=lambda x: x[1]['pump_rate_percent'], reverse=True)[:2]]
            print(f"  Top Cluster Pump Rates: {top_cluster_rates}")
        print(f"  🏆 Winner: {winner}")
        print(f"  📋 Recommendation: {recommendation}")
        
        return comparison_results
    
    def save_results(self, comparison_results: Dict, unified_results: Dict, 
                    per_category_results: Dict, sample_size: int) -> None:
        """Save comparison results to JSON file."""
        output_dir = self.results_dir / "clustering_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare complete results
        complete_results = {
            'comparison_summary': comparison_results,
            'unified_results': {
                'optimal_k': unified_results['optimal_k'],
                'silhouette_score': unified_results['final_clustering']['silhouette_score'],
                'mean_ari': unified_results['stability']['mean_ari'],
                'meets_ceo_requirements': unified_results['meets_ceo_requirements']
            },
            'per_category_results': {
                'overall_silhouette': per_category_results['overall_silhouette'],
                'overall_ari': per_category_results['overall_ari'],
                'total_tokens': per_category_results['total_tokens'],
                'category_summaries': {
                    category: {
                        'optimal_k': results['optimal_k'],
                        'silhouette_score': results['final_clustering']['silhouette_score'],
                        'mean_ari': results['stability']['mean_ari'],
                        'meets_ceo_requirements': results['meets_ceo_requirements']
                    }
                    for category, results in per_category_results['category_results'].items()
                }
            },
            'experimental_setup': {
                'sample_size': sample_size,
                'features_used': self.ESSENTIAL_FEATURES,
                'random_seed': 42
            }
        }
        
        # Save results
        output_file = output_dir / f"clustering_comparison_results_{sample_size}.json"
        with open(output_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")
    
    def save_cluster_assignments(self, unified_results: Dict, features_dict: Dict, dataset_size: int) -> None:
        """Save actual token-to-cluster assignments for unified classifier."""
        output_dir = self.results_dir / "clustering_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get cluster labels and token names
        cluster_labels = unified_results['final_clustering']['labels']
        token_names = unified_results['token_names']
        
        # Create token-to-cluster mapping
        token_cluster_mapping = {}
        for i, token_name in enumerate(token_names):
            token_cluster_mapping[token_name] = {
                'cluster_id': int(cluster_labels[i]),
                'category': features_dict[token_name]['category']
            }
        
        # Create cluster summary
        cluster_summary = {}
        for cluster_id in range(unified_results['optimal_k']):
            cluster_tokens = [name for i, name in enumerate(token_names) if cluster_labels[i] == cluster_id]
            cluster_categories = [features_dict[token]['category'] for token in cluster_tokens]
            
            # Category distribution
            from collections import Counter
            category_counts = Counter(cluster_categories)
            category_dist = {cat: count/len(cluster_tokens) for cat, count in category_counts.items()}
            
            cluster_summary[cluster_id] = {
                'size': len(cluster_tokens),
                'category_distribution': category_dist,
                'dominant_category': max(category_dist, key=category_dist.get) if category_dist else None,
                'purity': max(category_dist.values()) if category_dist else 0
            }
        
        # Save assignments
        assignments_data = {
            'optimal_k': unified_results['optimal_k'],
            'total_tokens': len(token_names),
            'cluster_summary': cluster_summary,
            'token_assignments': token_cluster_mapping,
            'silhouette_score': unified_results['final_clustering']['silhouette_score'],
            'mean_ari': unified_results['stability']['mean_ari']
        }
        
        assignments_file = output_dir / f"cluster_assignments_{dataset_size}.json"
        with open(assignments_file, 'w') as f:
            json.dump(assignments_data, f, indent=2)
        
        print(f"💾 Cluster assignments saved to: {assignments_file}")
    
    def analyze_cluster_characteristics(self, features_dict: Dict[str, Dict[str, float]], 
                                      unified_results: Dict, data_dir: Path) -> Dict:
        """Analyze cluster characteristics for recommendation."""
        print("🔍 Analyzing unified cluster characteristics...")
        
        # Get cluster labels
        cluster_labels = unified_results['final_clustering']['labels']
        token_names = unified_results['token_names']
        
        # DEBUG: Show all cluster sizes before filtering
        print("🔍 DEBUG: All cluster sizes before filtering:")
        all_cluster_sizes = {}
        for i in range(unified_results['optimal_k']):
            cluster_tokens = [token_names[j] for j in range(len(token_names)) if cluster_labels[j] == i]
            all_cluster_sizes[i] = len(cluster_tokens)
            print(f"  Cluster {i}: {len(cluster_tokens)} tokens")
        
        # Create cluster analysis
        cluster_stats = {}
        for i in range(unified_results['optimal_k']):
            cluster_tokens = [token_names[j] for j in range(len(token_names)) if cluster_labels[j] == i]
            
            # REMOVED FILTERING - Analyze ALL clusters regardless of size
            print(f"🔍 Analyzing Cluster {i}: {len(cluster_tokens)} tokens")
            
            if len(cluster_tokens) > 0:  # Only need at least 1 token
                # Calculate cluster characteristics
                categories = [features_dict[token]['category'] for token in cluster_tokens]
                category_dist = {cat: categories.count(cat) / len(categories) for cat in set(categories)}
                
                # Calculate returns, volatility, and pump rates for cluster
                cluster_returns = []
                cluster_volatilities = []
                high_vol_pumps = 0
                high_vol_count = 0
                
                for token in cluster_tokens:
                    token_file = data_dir / f"{token}.parquet"
                    if token_file.exists():
                        try:
                            df = pl.read_parquet(token_file)
                            prices = df['price'].to_numpy()
                            
                            if len(prices) >= 11:  # Need at least 11 minutes for post-min 10 analysis
                                # Calculate returns and volatility
                                returns = np.diff(prices) / prices[:-1]
                                cluster_returns.extend(returns)
                                
                                # Calculate volatility from first 5 minutes
                                if len(prices) >= 6:
                                    returns_early = np.diff(prices[:5]) / prices[:4]
                                    volatility = np.std(returns_early) / np.mean(prices[:5]) if np.mean(prices[:5]) > 0 else 0
                                    cluster_volatilities.append(volatility)
                                    
                                    # Check for high volatility (>0.8)
                                    if volatility > 0.8:
                                        high_vol_count += 1
                                        
                                        # Check for pump >50% after minute 10
                                        post_min_10_prices = prices[10:]
                                        if len(post_min_10_prices) > 0:
                                            max_return = np.max(post_min_10_prices) / prices[9]
                                            if max_return > 1.5:  # 50% pump
                                                high_vol_pumps += 1
                        except:
                            continue
                
                # Calculate aggregate metrics
                avg_return = np.mean(cluster_returns) if cluster_returns else 0
                avg_volatility = np.mean(cluster_volatilities) if cluster_volatilities else 0
                pump_rate = (high_vol_pumps / high_vol_count * 100) if high_vol_count > 0 else 0
                
                cluster_stats[f'cluster_{i}'] = {
                    'size': len(cluster_tokens),
                    'category_distribution': category_dist,
                    'dominant_category': max(category_dist, key=category_dist.get),
                    'purity': max(category_dist.values()),
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'high_vol_count': high_vol_count,
                    'high_vol_pumps': high_vol_pumps,
                    'pump_rate_percent': pump_rate
                }
        
        return cluster_stats
    
    def run_full_comparison(self, archetype_results_path: Path, data_dir: Path) -> Dict:
        """Run complete unified vs per-category clustering comparison."""
        print("🚀 Starting Unified vs Per-Category Clustering Comparison...")
        print("=" * 70)
        
        # Use full 30k dataset directly as requested
        features_dict = self.load_and_prepare_data(archetype_results_path, data_dir, use_full_dataset=True)
        dataset_size = len(features_dict)
        
        # Run unified clustering on full dataset
        unified_results = self.run_unified_clustering(features_dict)
        
        # Check unified clustering performance
        unified_silhouette = unified_results['final_clustering']['silhouette_score']
        unified_ari = unified_results['stability']['mean_ari']
        
        print(f"📊 Unified clustering results: Silhouette={unified_silhouette:.4f}, ARI={unified_ari:.4f}")
        
        if unified_silhouette > 0.5 and unified_ari > 0.5:
            print("🎯 Unified clustering meets performance thresholds (>0.5)")
        else:
            print("⚠️  Unified clustering below thresholds, but continuing with analysis")
        
        # Run per-category clustering
        per_category_results = self.run_per_category_clustering(features_dict)
        
        # Add cluster analysis for recommendation
        cluster_analysis = self.analyze_cluster_characteristics(features_dict, unified_results, data_dir)
        
        # Compare approaches with cluster analysis
        comparison_results = self.compare_approaches(unified_results, per_category_results, cluster_analysis)
        comparison_results['cluster_analysis'] = cluster_analysis
        
        # Save results
        self.save_results(comparison_results, unified_results, per_category_results, dataset_size)
        
        # Save actual cluster assignments for unified classifier
        self.save_cluster_assignments(unified_results, features_dict, dataset_size)
        
        return comparison_results

def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Unified vs Per-Category Clustering Comparison")
    parser.add_argument('--window', type=int, default=5, choices=[5, 10], 
                       help='Analysis window in minutes (default: 5)')
    args = parser.parse_args()
    
    print("🚀 Unified vs Per-Category Clustering Comparison")
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
    
    # Initialize and run comparison
    comparison = UnifiedClusteringComparison()
    results = comparison.run_full_comparison(latest_results, data_dir)
    
    print("\n🎉 Clustering comparison completed successfully!")
    print(f"🏆 Winner: {results['conclusion']['winner']}")
    print(f"📋 Recommendation: {results['conclusion']['recommendation']}")

if __name__ == "__main__":
    main()