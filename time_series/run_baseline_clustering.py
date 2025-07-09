#!/usr/bin/env python3
"""
Baseline Clustering Analysis Script
Runs clustering analysis for Sprint/Standard/Marathon categories and saves results.
Follows CEO's Day 1-2 requirements.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from time_series.behavioral_archetype_analysis import BehavioralArchetypeAnalyzer
from time_series.archetype_utils import categorize_by_lifespan

# Constants
PROCESSED_DIR = project_root / 'data' / 'processed'
RESULTS_DIR = project_root / 'time_series' / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_baseline_clustering():
    """
    Run baseline clustering analysis for all lifespan categories.
    CEO's Day 1-2 requirements: 15 features, elbow method, no PCA.
    """
    print("="*80)
    print("BASELINE CLUSTERING ANALYSIS")
    print("CEO's Day 1-2 Requirements: 15 features, elbow method, no PCA")
    print("="*80)
    
    # Initialize analyzer
    analyzer = BehavioralArchetypeAnalyzer()
    
    # Load tokens from processed categories (limit for testing)
    print("\n1. Loading tokens from processed categories...")
    token_data = analyzer.load_categorized_tokens(PROCESSED_DIR, limit=1000)  # Limit to 1000 tokens per category for testing
    
    if not token_data:
        print("❌ No tokens found in processed directory!")
        return
    
    print(f"✅ Loaded {len(token_data)} tokens")
    
    # Extract 15 essential features
    print("\n2. Extracting 15 essential features...")
    features_df = analyzer.extract_all_features(token_data)
    
    # Categorize tokens by lifespan
    print("\n3. Categorizing tokens by lifespan...")
    features_df = categorize_by_lifespan(features_df)
    
    # Count tokens per category
    category_counts = features_df['lifespan_category'].value_counts()
    print("\nLifespan Category Distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count:,} tokens")
    
    # Process each lifespan category
    lifespan_categories = ['Sprint', 'Standard', 'Marathon']
    baseline_results = {}
    
    for category in lifespan_categories:
        print(f"\n{'='*60}")
        print(f"PROCESSING {category.upper()} CATEGORY")
        print(f"{'='*60}")
        
        # Filter tokens for this category
        category_tokens = features_df[features_df['lifespan_category'] == category]
        
        if len(category_tokens) < 10:
            print(f"❌ Too few tokens in {category} category ({len(category_tokens)}). Skipping...")
            continue
        
        print(f"Processing {len(category_tokens)} {category} tokens...")
        
        # Perform clustering
        print(f"\n4. Running clustering for {category} category...")
        clustering_results = analyzer.perform_clustering(category_tokens)
        
        # Get results
        best_k = clustering_results['best_k']
        elbow_results = clustering_results['elbow_results']
        cluster_labels = clustering_results['kmeans'][best_k]['labels']
        silhouette_score = clustering_results['kmeans'][best_k]['silhouette_score']
        
        # Add cluster labels to dataframe
        category_tokens_with_clusters = category_tokens.copy()
        category_tokens_with_clusters['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in range(best_k):
            cluster_data = category_tokens_with_clusters[category_tokens_with_clusters['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'n_tokens': len(cluster_data),
                'pct_of_category': len(cluster_data) / len(category_tokens) * 100,
                'pct_dead': cluster_data['is_dead'].mean() * 100,
                'avg_lifespan': cluster_data['lifespan_minutes'].mean(),
                'avg_return_magnitude': cluster_data['return_magnitude_5min'].mean(),
                'avg_volatility_5min': cluster_data['volatility_5min'].mean(),
                'avg_mean_return': cluster_data['mean_return'].mean(),
                'avg_std_return': cluster_data['std_return'].mean(),
                'avg_max_drawdown': cluster_data['max_drawdown'].mean(),
                'example_tokens': cluster_data['token'].head(5).tolist()
            }
            cluster_stats.append(stats)
        
        # Save results
        results_file = RESULTS_DIR / f"baseline_{category.lower()}_k{best_k}.csv"
        category_tokens_with_clusters.to_csv(results_file, index=False)
        
        # Store summary
        baseline_results[category] = {
            'n_tokens': len(category_tokens),
            'optimal_k': best_k,
            'elbow_k': elbow_results['optimal_k_elbow'],
            'silhouette_k': elbow_results['optimal_k_silhouette'],
            'silhouette_score': silhouette_score,
            'cluster_stats': cluster_stats,
            'results_file': str(results_file)
        }
        
        # Print results
        print(f"\n✅ {category} Results:")
        print(f"  - Tokens: {len(category_tokens):,}")
        print(f"  - Optimal K (elbow): {best_k}")
        print(f"  - Silhouette score: {silhouette_score:.3f}")
        print(f"  - Results saved to: {results_file}")
        
        print(f"\n  Cluster Summary:")
        for stats in cluster_stats:
            print(f"    Cluster {stats['cluster_id']}: {stats['n_tokens']} tokens "
                  f"({stats['pct_of_category']:.1f}%), "
                  f"{stats['pct_dead']:.1f}% dead, "
                  f"avg lifespan: {stats['avg_lifespan']:.0f} min")
    
    # Save overall summary
    summary_file = RESULTS_DIR / "baseline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("BASELINE CLUSTERING ANALYSIS COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"{'='*80}")
    
    return baseline_results


def print_elbow_plots_info(baseline_results):
    """Print information about elbow plots for each category."""
    print(f"\n{'='*60}")
    print("ELBOW PLOTS INFORMATION")
    print(f"{'='*60}")
    
    for category, results in baseline_results.items():
        print(f"\n{category} Category:")
        print(f"  - Elbow method suggests K = {results['elbow_k']}")
        print(f"  - Silhouette method suggests K = {results['silhouette_k']}")
        print(f"  - Selected K = {results['optimal_k']} (using elbow method)")
        print(f"  - Final silhouette score = {results['silhouette_score']:.3f}")


if __name__ == "__main__":
    # Run baseline clustering analysis
    baseline_results = run_baseline_clustering()
    
    # Print elbow plots information
    if baseline_results:
        print_elbow_plots_info(baseline_results)
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Run stability test: python run_stability_test.py")
        print(f"2. Check results files in: {RESULTS_DIR}")
        print(f"3. Review cluster assignments and example tokens")
        print(f"4. Validate that clustering is stable and interpretable")