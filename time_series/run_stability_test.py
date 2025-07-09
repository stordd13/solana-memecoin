#!/usr/bin/env python3
"""
Stability Test Script
Runs clustering 5 times with different seeds and calculates ARI scores.
CEO's Day 1-2 requirements for stability validation.
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
from sklearn.metrics import adjusted_rand_score

# Constants
PROCESSED_DIR = project_root / 'data' / 'processed'
RESULTS_DIR = project_root / 'time_series' / 'results'
STABILITY_DIR = RESULTS_DIR / 'stability'
STABILITY_DIR.mkdir(exist_ok=True)

def run_stability_test():
    """
    Run stability test (5 runs) and calculate ARI scores.
    CEO's Day 1-2 requirements: ARI > 0.7 for stable clustering.
    """
    print("="*80)
    print("STABILITY TEST ANALYSIS")
    print("CEO's Day 1-2 Requirements: ARI > 0.7 for stable clustering")
    print("="*80)
    
    # Initialize analyzer
    analyzer = BehavioralArchetypeAnalyzer()
    
    # Load tokens (same as baseline)
    print("\n1. Loading tokens from processed categories...")
    token_data = analyzer.load_categorized_tokens(PROCESSED_DIR, limit=1000)
    
    if not token_data:
        print("❌ No tokens found in processed directory!")
        return
    
    print(f"✅ Loaded {len(token_data)} tokens")
    
    # Extract features
    print("\n2. Extracting 14 essential features...")
    features_df = analyzer.extract_all_features(token_data)
    
    # Categorize by lifespan
    print("\n3. Categorizing tokens by lifespan...")
    features_df = categorize_by_lifespan(features_df)
    
    # Process each category
    lifespan_categories = ['Sprint', 'Standard', 'Marathon']
    stability_results = {}
    
    for category in lifespan_categories:
        print(f"\n{'='*60}")
        print(f"STABILITY TEST: {category.upper()} CATEGORY")
        print(f"{'='*60}")
        
        # Filter tokens for this category
        category_tokens = features_df[features_df['lifespan_category'] == category]
        
        if len(category_tokens) < 10:
            print(f"❌ Too few tokens in {category} category ({len(category_tokens)}). Skipping...")
            continue
        
        print(f"Testing stability with {len(category_tokens)} {category} tokens...")
        
        # Run clustering 5 times with different seeds
        print(f"\n4. Running clustering 5 times with different seeds...")
        
        cluster_results = []
        all_labels = []
        
        for run in range(5):
            print(f"\n  Run {run + 1}/5 (seed={run})...")
            
            # Create new analyzer instance for each run
            run_analyzer = BehavioralArchetypeAnalyzer()
            
            # Perform clustering with different random seed
            np.random.seed(run)
            clustering_results = run_analyzer.perform_clustering(category_tokens)
            
            # Get results
            best_k = clustering_results['best_k']
            cluster_labels = clustering_results['kmeans'][best_k]['labels']
            silhouette_score = clustering_results['kmeans'][best_k]['silhouette_score']
            
            # Store results
            cluster_results.append({
                'run': run,
                'seed': run,
                'best_k': best_k,
                'silhouette_score': silhouette_score,
                'labels': cluster_labels
            })
            
            all_labels.append(cluster_labels)
            
            print(f"    K={best_k}, Silhouette={silhouette_score:.3f}")
        
        # Calculate ARI between all pairs of runs
        print(f"\n5. Calculating ARI between all pairs of runs...")
        
        n_runs = len(all_labels)
        ari_matrix = np.zeros((n_runs, n_runs))
        
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                ari_score = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_matrix[i, j] = ari_score
                ari_matrix[j, i] = ari_score
                print(f"    ARI(Run {i+1}, Run {j+1}) = {ari_score:.3f}")
        
        # Set diagonal to 1.0 (perfect agreement with itself)
        np.fill_diagonal(ari_matrix, 1.0)
        
        # Calculate summary statistics
        upper_triangle = ari_matrix[np.triu_indices_from(ari_matrix, k=1)]
        mean_ari = np.mean(upper_triangle)
        std_ari = np.std(upper_triangle)
        min_ari = np.min(upper_triangle)
        max_ari = np.max(upper_triangle)
        
        # Check stability criteria (CEO requirement: ARI > 0.7)
        stability_status = "STABLE" if mean_ari > 0.7 else "UNSTABLE"
        
        # Store results
        stability_results[category] = {
            'n_tokens': len(category_tokens),
            'n_runs': n_runs,
            'cluster_results': cluster_results,
            'ari_matrix': ari_matrix.tolist(),
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'min_ari': min_ari,
            'max_ari': max_ari,
            'stability_status': stability_status
        }
        
        # Save detailed results
        stability_file = STABILITY_DIR / f"stability_{category.lower()}.json"
        with open(stability_file, 'w') as f:
            json.dump(stability_results[category], f, indent=2)
        
        # Print summary
        print(f"\n✅ {category} Stability Results:")
        print(f"  - Tokens: {len(category_tokens):,}")
        print(f"  - Runs: {n_runs}")
        print(f"  - Mean ARI: {mean_ari:.3f}")
        print(f"  - Std ARI: {std_ari:.3f}")
        print(f"  - Min ARI: {min_ari:.3f}")
        print(f"  - Max ARI: {max_ari:.3f}")
        print(f"  - Status: {stability_status}")
        print(f"  - Details saved to: {stability_file}")
        
        # Print k-value consistency
        k_values = [r['best_k'] for r in cluster_results]
        k_consistency = len(set(k_values)) == 1
        print(f"  - K-value consistency: {'✅ CONSISTENT' if k_consistency else '❌ INCONSISTENT'}")
        print(f"  - K-values across runs: {k_values}")
    
    # Save overall summary
    summary_file = STABILITY_DIR / "stability_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(stability_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("STABILITY TEST COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print(f"Details directory: {STABILITY_DIR}")
    print(f"{'='*80}")
    
    return stability_results


def print_stability_recommendations(stability_results):
    """Print recommendations based on stability results."""
    print(f"\n{'='*60}")
    print("STABILITY RECOMMENDATIONS")
    print(f"{'='*60}")
    
    for category, results in stability_results.items():
        print(f"\n{category} Category:")
        mean_ari = results['mean_ari']
        status = results['stability_status']
        
        if status == "STABLE":
            print(f"  ✅ STABLE (ARI={mean_ari:.3f} > 0.7)")
            print(f"  - Clustering is reliable and reproducible")
            print(f"  - Can proceed with archetype identification")
        else:
            print(f"  ❌ UNSTABLE (ARI={mean_ari:.3f} < 0.7)")
            print(f"  - Clustering results vary significantly between runs")
            print(f"  - Recommendations:")
            
            if mean_ari > 0.5:
                print(f"    • Semi-stable. Try reducing features further")
                print(f"    • Consider increasing K or using different clustering method")
            else:
                print(f"    • Highly unstable. Major changes needed:")
                print(f"    • Reduce features to top 10 by variance")
                print(f"    • Try RobustScaler instead of StandardScaler")
                print(f"    • Consider DBSCAN or different algorithm")
                print(f"    • Remove outliers (top/bottom 1%)")


if __name__ == "__main__":
    # Run stability test
    stability_results = run_stability_test()
    
    # Print recommendations
    if stability_results:
        print_stability_recommendations(stability_results)
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Review stability results in: {STABILITY_DIR}")
        print(f"2. If stable (ARI > 0.7): Create baseline_results.md report")
        print(f"3. If unstable (ARI < 0.7): Implement stability improvements")
        print(f"4. Validate clustering interpretability with example tokens")