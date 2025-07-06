#!/usr/bin/env python3
"""
Run autocorrelation and clustering analysis on memecoin time series data
"""

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Run autocorrelation and clustering analysis')
    parser.add_argument('--data_dir', type=str, default='data/raw/dataset',
                       help='Directory containing token parquet files')
    parser.add_argument('--output_dir', type=str, default='time_series/results',
                       help='Directory to save results')
    parser.add_argument('--max_tokens', type=int, default=None,
                       help='Maximum number of tokens to analyze (None = no limit)')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (None = find optimal)')
    parser.add_argument('--find_optimal_k', action='store_true', default=True,
                       help='Find optimal number of clusters using elbow method')
    parser.add_argument('--use_log_price', action='store_true', default=True,
                       help='Use log prices instead of raw prices')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("🔄 Initializing Autocorrelation & Clustering Analysis")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Max tokens: {args.max_tokens}")
    print(f"🎯 Number of clusters: {args.n_clusters}")
    print(f"📈 Using {'log' if args.use_log_price else 'raw'} prices")
    
    # Initialize analyzer
    analyzer = AutocorrelationClusteringAnalyzer()
    
    try:
        # Run complete analysis
        print("\n🚀 Running analysis...")
        results = analyzer.run_complete_analysis(
            Path(args.data_dir),
            use_log_price=args.use_log_price,
            n_clusters=args.n_clusters,
            find_optimal_k=args.find_optimal_k,
            max_tokens=args.max_tokens
        )
        
        print(f"\n✅ Analysis complete! Analyzed {len(results['token_names'])} tokens")
        
        # Print summary statistics
        print("\n📊 CLUSTER SUMMARY:")
        print("-" * 50)
        for cluster_id, stats in results['cluster_stats'].items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Tokens: {stats['n_tokens']}")
            print(f"  Avg Length: {stats['avg_length']:.0f} minutes")
            print(f"  Avg Volatility: {stats['price_characteristics']['avg_volatility']:.4f}")
            print(f"  Avg Return: {stats['price_characteristics']['avg_return']:.4f}")
            print(f"  Sample tokens: {', '.join(stats['tokens'][:5])}")
        
        # Save results
        print(f"\n💾 Saving results to {output_dir}")
        
        # Save cluster assignments
        cluster_assignments = {
            token: int(label) 
            for token, label in zip(results['token_names'], results['cluster_labels'])
        }
        
        with open(output_dir / 'cluster_assignments.json', 'w') as f:
            json.dump(cluster_assignments, f, indent=2)
        
        # Save cluster statistics
        cluster_stats_serializable = {}
        for cluster_id, stats in results['cluster_stats'].items():
            cluster_stats_serializable[str(cluster_id)] = {
                'n_tokens': stats['n_tokens'],
                'avg_length': float(stats['avg_length']),
                'tokens': stats['tokens'],
                'price_characteristics': {
                    k: float(v) for k, v in stats['price_characteristics'].items()
                }
            }
        
        with open(output_dir / 'cluster_statistics.json', 'w') as f:
            json.dump(cluster_stats_serializable, f, indent=2)
        
        # Save ACF summary
        acf_summary = {}
        for token, acf_result in results['acf_results'].items():
            acf_summary[token] = {
                'significant_lags': acf_result['significant_lags'],
                'decay_rate': float(acf_result['decay_rate']) if not np.isnan(acf_result['decay_rate']) else None,
                'first_zero_crossing': int(acf_result['first_zero_crossing'])
            }
        
        with open(output_dir / 'acf_summary.json', 'w') as f:
            json.dump(acf_summary, f, indent=2)
        
        # Save plots if requested
        if args.save_plots:
            print("\n🎨 Creating visualizations...")
            
            # Create visualization plots
            figures = analyzer.create_visualization_plots(results)
            
            # Save each figure
            for name, fig in figures.items():
                fig.write_html(output_dir / f'{name}.html')
                print(f"  Saved {name}.html")
            
            # Also create a matplotlib version of key plots
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. t-SNE 2D plot
            plt.figure(figsize=(10, 8))
            embedding = results['t_sne_2d']
            labels = results['cluster_labels']
            
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                          label=f'Cluster {cluster_id}', s=50, alpha=0.7)
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('t-SNE Visualization of Token Clusters')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'tsne_2d.png', dpi=300)
            plt.close()
            
            # 2. Average ACF by cluster
            fig, axes = plt.subplots(1, args.n_clusters, figsize=(4*args.n_clusters, 4))
            if args.n_clusters == 1:
                axes = [axes]
            
            for cluster_id, ax in enumerate(axes):
                if cluster_id in results['acf_by_cluster']:
                    acf_data = results['acf_by_cluster'][cluster_id]
                    avg_acf = np.mean(acf_data, axis=0)
                    
                    ax.bar(range(len(avg_acf)), avg_acf, alpha=0.7)
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
                    ax.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
                    ax.set_title(f'Cluster {cluster_id}')
                    ax.set_xlabel('Lag')
                    ax.set_ylabel('ACF')
                    ax.set_ylim(-0.2, 1.0)
            
            plt.suptitle('Average Autocorrelation Function by Cluster')
            plt.tight_layout()
            plt.savefig(output_dir / 'acf_by_cluster.png', dpi=300)
            plt.close()
            
            print("  Saved static plots as PNG files")
        
        print(f"\n✨ Analysis complete! Results saved to {output_dir}")
        print("\n📝 Generated files:")
        print("  - cluster_assignments.json: Token to cluster mapping")
        print("  - cluster_statistics.json: Detailed cluster statistics")
        print("  - acf_summary.json: Autocorrelation summary for each token")
        if args.save_plots:
            print("  - Interactive HTML plots and static PNG images")
        
        print("\n🎯 Next steps:")
        print("  1. Run the Streamlit app for interactive exploration:")
        print("     streamlit run time_series/autocorrelation_app.py")
        print("  2. Analyze cluster assignments to find similar tokens")
        print("  3. Use ACF patterns to understand token dynamics")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import numpy as np  # Import needed for the script
    exit(main()) 