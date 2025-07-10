"""
Streamlit App for Autocorrelation and Time Series Clustering Analysis
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from streamlit_utils.formatting import format_large_number, format_percentage, format_data_points

# Import our analyzer
from time_series.autocorrelation_clustering import AutocorrelationClusteringAnalyzer
from time_series.behavioral_archetype_analysis import BehavioralArchetypeAnalyzer


def run_baseline_clustering_analysis(processed_data_path, token_limits, k_range, n_stability_runs, export_results):
    """
    Run baseline clustering analysis with CEO requirements (14 features, elbow method, stability testing).
    """
    from time_series.archetype_utils import categorize_by_lifespan
    from sklearn.metrics import adjusted_rand_score
    import json
    from datetime import datetime
    
    # Initialize analyzer
    analyzer = BehavioralArchetypeAnalyzer()
    
    # Load tokens from processed categories
    st.info("Loading categorized token data...")
    token_data = analyzer.load_categorized_tokens(processed_data_path, limit=None)
    
    if not token_data:
        st.error("No token data found!")
        return {}
    
    # Categorize by lifespan
    st.info("Categorizing tokens by lifespan...")
    categorized_tokens = categorize_by_lifespan(token_data, token_limits)
    
    # Extract 14 features for each category
    st.info("Extracting 14 essential features...")
    results = {
        'analysis_type': 'Baseline Clustering (CEO)',
        'categories': {},
        'total_tokens_analyzed': 0,
        'stability_summary': {}
    }
    
    for category_name, category_tokens in categorized_tokens.items():
        if not category_tokens:
            continue
            
        st.info(f"Processing {category_name}: {len(category_tokens)} tokens")
        
        # Extract 14 features
        features_df = analyzer.extract_14_features(category_tokens)
        
        if features_df is None or len(features_df) == 0:
            st.warning(f"No features extracted for {category_name} - skipping")
            continue
        
        st.info(f"Extracted {len(features_df)} features for {category_name}")
        
        # Check if we have enough samples for clustering
        if len(features_df) < 3:
            st.warning(f"Category {category_name} has only {len(features_df)} samples - skipping clustering (need at least 3)")
            continue
        
        # Find optimal K using elbow method
        optimal_k = analyzer.find_optimal_k_elbow(features_df, k_range)
        
        # Ensure K is reasonable for the number of samples
        optimal_k = min(optimal_k, len(features_df) - 1)
        optimal_k = max(optimal_k, 2)  # At least 2 clusters
        
        st.info(f"Using K={optimal_k} for {category_name}")
        
        # Run clustering
        clustering_results = analyzer.run_kmeans_clustering(features_df, optimal_k)
        
        # Test stability
        stability_results = analyzer.test_clustering_stability(features_df, optimal_k, n_stability_runs)
        
        # Store results
        results['categories'][category_name] = {
            'n_tokens': len(category_tokens),
            'features': features_df,
            'optimal_k': optimal_k,
            'clustering_results': clustering_results,
            'stability_results': stability_results,
            'token_data': category_tokens
        }
        
        results['total_tokens_analyzed'] += len(category_tokens)
    
    # Calculate overall stability summary
    all_ari_scores = []
    for category_results in results['categories'].values():
        stability = category_results['stability_results']
        if 'ari_scores' in stability:
            all_ari_scores.extend(stability['ari_scores'])
    
    if all_ari_scores:
        results['stability_summary'] = {
            'average_ari': np.mean(all_ari_scores),
            'min_ari': np.min(all_ari_scores),
            'max_ari': np.max(all_ari_scores),
            'n_runs': len(all_ari_scores)
        }
    
    # Export results if requested
    if export_results:
        st.info("Exporting results...")
        export_baseline_results(results, processed_data_path.parent / "time_series" / "results")
    
    return results


def export_baseline_results(results, output_dir):
    """Export baseline clustering results to files."""
    from pathlib import Path
    import json
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export summary
    summary_file = output_dir / f"baseline_clustering_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'analysis_type': results['analysis_type'],
            'total_tokens_analyzed': results['total_tokens_analyzed'],
            'stability_summary': results['stability_summary'],
            'categories': {name: {
                'n_tokens': cat['n_tokens'],
                'optimal_k': cat['optimal_k'],
                'stability_results': cat['stability_results']
            } for name, cat in results['categories'].items()}
        }, f, indent=2)
    
    # Export detailed results per category
    for category_name, category_results in results['categories'].items():
        category_file = output_dir / f"baseline_{category_name.lower()}_{timestamp}.csv"
        features_df = category_results['features']
        clustering_results = category_results['clustering_results']
        
        # Add cluster assignments to features
        features_with_clusters = features_df.copy()
        features_with_clusters['cluster'] = clustering_results['labels']
        
        # Save to CSV
        features_with_clusters.to_csv(category_file, index=False)
    
    st.success(f"Results exported to: {output_dir}")


def main():
    st.set_page_config(page_title="Time Series Autocorrelation & Clustering", layout="wide")
    
    st.title("🔄 Time Series Autocorrelation & Clustering Analysis")
    st.markdown("""
    This app analyzes raw price time series data using:
    - **Autocorrelation (ACF)** and **Partial Autocorrelation (PACF)** - computed for all analysis types
    - **Time Series Clustering** to find similar patterns
    - **t-SNE Visualization** for dimensionality reduction
    
    **Analysis Types:**
    - **Feature-based**: Uses 16 engineered features (ACF values + statistical measures) for clustering
    - **Price-only**: Clusters directly on price series (returns, log returns, raw prices, log prices, DTW features) + computes ACF
    """)
    
    # Initialize analyzer
    analyzer = AutocorrelationClusteringAnalyzer()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data directory selection
    # Try to find the correct data directory automatically
    possible_paths = [
        "data/raw/dataset",
        "../data/raw/dataset", 
        "../../data/raw/dataset",
        Path(__file__).parent.parent / "data/raw/dataset"
    ]
    
    default_path = "data/raw/dataset"
    for path in possible_paths:
        if Path(path).exists():
            default_path = str(path)
            break
    
    data_dir = st.sidebar.text_input(
        "Data Directory", 
        value=default_path,
        help="Path to directory containing token parquet files"
    )
    
    # Analysis parameters will be moved to analysis-type-specific sections
    
    # Analysis type selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Type")
    
    analysis_type = st.sidebar.radio(
        "Choose analysis approach:",
        ["Feature-based (ACF + Statistics)", "Price-only", "Multi-Resolution ACF + Baseline Clustering"],
        help="Feature-based uses 16 engineered features. Price-only clusters on price series. Multi-Resolution performs comprehensive analysis across Sprint/Standard/Marathon categories with 14-feature baseline clustering and stability testing."
    )
    
    # Analysis-specific configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    # Initialize default variables (will be overridden by analysis-specific sections)
    use_log_price = True
    max_tokens = None
    find_optimal_k = True
    n_clusters = None
    clustering_method = 'kmeans'
    
    # Feature-based specific options
    if analysis_type == "Feature-based (ACF + Statistics)":
        with st.sidebar.expander("📊 Feature-based Settings", expanded=True):
            use_log_price = st.checkbox("Use log prices", value=True,
                                      help="Use log-transformed prices for feature engineering")
            
            max_tokens = st.number_input("Max tokens to analyze:", 
                                       min_value=10, value=1000,
                                       help="Limit for faster analysis (use higher values for comprehensive analysis)")
            
            find_optimal_k = st.checkbox("Find optimal K", value=True,
                                       help="Automatically find optimal clusters using elbow method")
            
            n_clusters = None
            if not find_optimal_k:
                n_clusters = st.slider("Number of clusters:", 
                                      min_value=2, max_value=20, value=5)
            
            clustering_method = st.selectbox("Clustering method:",
                                           ["kmeans", "hierarchical", "dbscan"],
                                           help="Algorithm for clustering engineered features")
    
    # Price-only specific options
    elif analysis_type == "Price-only":
        with st.sidebar.expander("💰 Price-only Settings", expanded=True):
            price_method = st.selectbox(
                "Price transformation:",
                ["returns", "log_returns", "prices", "log_prices", "dtw_features"],
                help="How to convert price series for clustering"
            )
            
            # Add guidance for extreme volatility
            if price_method == "returns":
                st.info("💡 **Returns** work well for normal volatility. For extreme moves (>1000%), consider **log_returns** or **log_prices**.")
            elif price_method == "log_returns":
                st.info("💡 **Log Returns** handle extreme volatility better than regular returns. Good for pumps/dumps >100%.")
            elif price_method == "log_prices":
                st.info("💡 **Log Prices** are best for extreme volatility (10M%+ pumps, 99.9% dumps). Most stable for memecoin analysis.")
            elif price_method == "dtw_features":
                st.info("💡 **DTW Features** find similar temporal patterns regardless of scale. Good for discovering behavioral patterns.")
            else:
                st.info("💡 **Raw Prices** can be unstable with extreme volatility. Consider log_prices for better results.")
            
            max_tokens = st.number_input("Max tokens to analyze:", 
                                       min_value=10, value=1000,
                                       help="Limit for faster analysis (use higher values for comprehensive analysis)")
            
            max_sequence_length = st.number_input(
                "Max sequence length:",
                min_value=50, value=500,
                help="Maximum length of price sequences"
            )
            
            use_max_length = st.checkbox("Apply sequence length limit", value=True)
            
            find_optimal_k = st.checkbox("Find optimal K", value=True,
                                       help="Automatically find optimal clusters")
            
            n_clusters = None
            if not find_optimal_k:
                n_clusters = st.slider("Number of clusters:", 
                                      min_value=2, max_value=20, value=5)
            
            clustering_method = st.selectbox("Clustering method:",
                                           ["kmeans", "hierarchical", "dbscan"],
                                           help="Algorithm for clustering price data")
            
            st.markdown("""
            **Price Methods:**
            - **returns**: Raw returns (more stationary)
            - **log_returns**: Log returns (better statistical properties)
            - **prices**: Raw prices (as-is)
            - **log_prices**: Log-transformed prices
            - **dtw_features**: Statistical features from price series
            """)
    
    # Multi-Resolution ACF + Baseline Clustering specific options
    elif analysis_type == "Multi-Resolution ACF + Baseline Clustering":
        with st.sidebar.expander("🚀 Multi-Resolution Settings", expanded=True):
            st.markdown("**⭐ PHASE 1A: Multi-Resolution Analysis**")
            
            multi_method = st.selectbox(
                "Price transformation:",
                ["returns", "log_returns", "prices", "dtw_features"],
                help="Method for analyzing across lifespan categories"
            )
            
            # Token limit with support for unlimited analysis
            token_limit_input = st.text_input(
                "Max tokens per category (or 'none' for unlimited):",
                value="1000",
                help="Limit per category for balanced analysis across Sprint/Standard/Marathon. Use 'none' for unlimited analysis."
            )
            
            # Parse token limit
            try:
                if token_limit_input.lower() == 'none':
                    max_tokens_per_category = None
                else:
                    max_tokens_per_category = int(token_limit_input)
                    if max_tokens_per_category < 10:
                        st.error("Token limit must be at least 10 or 'none'")
                        max_tokens_per_category = 1000
            except ValueError:
                st.error("Invalid input. Please enter a number or 'none'")
                max_tokens_per_category = 1000
            
            enable_dtw_clustering = st.checkbox(
                "Enable DTW clustering",
                value=False,
                help="Variable-length sequence clustering (slower but more accurate)"
            )
            
            compare_across_categories = st.checkbox(
                "Cross-category ACF comparison",
                value=True,
                help="Compare ACF patterns across lifespan categories"
            )
            
            # Baseline clustering options
            st.markdown("---")
            st.markdown("**⭐ Baseline Clustering (CEO Requirements):**")
            
            enable_baseline_clustering = st.checkbox(
                "Enable 14-feature baseline clustering",
                value=True,
                help="Run baseline clustering with 14 features, elbow method, and stability testing"
            )
            
            if enable_baseline_clustering:
                # Clustering parameters
                k_range_min = st.number_input("Min clusters (K):", min_value=3, max_value=10, value=3)
                k_range_max = st.number_input("Max clusters (K):", min_value=3, max_value=15, value=10)
                
                # Stability testing
                n_stability_runs = st.number_input(
                    "Number of stability runs:", 
                    min_value=3, max_value=10, value=5,
                    help="Number of runs to test clustering stability (ARI calculation)"
                )
                
                # Export options
                export_results = st.checkbox("Export results to files", value=True)
            
            st.markdown("""
            **Lifespan Categories (Death-Aware):**
            - **Sprint**: 0-400 active min (includes ALL dead tokens)
            - **Standard**: 400-1200 active min (typical lifecycle)
            - **Marathon**: 1200+ active min (extended development)
            
            **Analysis Features:**
            - Multi-resolution ACF analysis across categories
            - 14-feature baseline clustering (CEO requirements)
            - Stability testing with ARI > 0.7 threshold
            - Elbow method for optimal K selection
            
            **Goal**: Discover behavioral archetypes including death patterns
            """)
        
        # Multi-resolution uses fixed optimal settings
        use_log_price = True
        find_optimal_k = True
        n_clusters = None
        clustering_method = 'kmeans'
        max_tokens = None
        
        # Initialize multi-resolution specific variables for other analysis types
        # multi_method is set by UI above, don't override
        if not enable_baseline_clustering:
            enable_dtw_clustering = False
            compare_across_categories = True
            k_range_min = 3
            k_range_max = 10
            n_stability_runs = 5
            export_results = True
    
    # Initialize variables for other analysis types that don't have multi-resolution settings
    if analysis_type not in ["Multi-Resolution ACF + Baseline Clustering"]:
        # Default values for multi-resolution variables not used in other types
        multi_method = "returns"  # Default for non-multi-resolution analysis
        max_tokens_per_category = None  # Use None as default for unlimited
        enable_dtw_clustering = False
        compare_across_categories = True
        enable_baseline_clustering = False
        k_range_min = 3
        k_range_max = 10
        n_stability_runs = 5
        export_results = True
    
    # Initialize price-only specific variables for other analysis types
    if analysis_type != "Price-only":
        price_method = "returns"
        max_sequence_length = 500
        use_max_length = True
    
    
    # Quick rerun option if results exist (only for non-multi-resolution analysis)
    if 'results' in st.session_state and analysis_type not in ["Multi-Resolution ACF + Baseline Clustering"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Quick Rerun Options:**")
        
        if 'n_clusters' in st.session_state['results']:
            current_k = st.session_state['results']['n_clusters']
            st.sidebar.text(f"Current K: {current_k}")
            
            quick_k = st.sidebar.number_input("Quick rerun with K:", 
                                             min_value=2, max_value=20, value=current_k,
                                             help="Quickly rerun analysis with different number of clusters")
            
            if st.sidebar.button("🔄 Quick Rerun", help="Rerun analysis with the K specified above"):
                st.session_state['rerun_with_k'] = quick_k
                st.session_state['rerun_requested'] = True
                st.rerun()
    
    # Show debug info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Debug Info:**")
    st.sidebar.text(f"Current working directory: {Path.cwd()}")
    st.sidebar.text(f"Data directory exists: {Path(data_dir).exists()}")
    if Path(data_dir).exists():
        parquet_count = len(list(Path(data_dir).rglob("*.parquet")))
        st.sidebar.text(f"Parquet files found: {parquet_count}")
    
    # Check if rerun was requested from elbow analysis
    run_analysis = st.sidebar.button("Run Analysis", type="primary")
    rerun_requested = st.session_state.get('rerun_requested', False)
    
    if run_analysis or rerun_requested:
        # Clear rerun flag
        if rerun_requested:
            st.session_state['rerun_requested'] = False
            # Use the selected K from elbow analysis
            if 'rerun_with_k' in st.session_state:
                n_clusters = st.session_state['rerun_with_k']
                find_optimal_k = False  # Don't find optimal K again, use selected K
                st.info(f"🔄 Rerunning analysis with K={n_clusters}")
        
        with st.spinner("Running analysis... This may take a few minutes."):
            try:
                # Validate data directory first
                data_path = Path(data_dir)
                if not data_path.exists():
                    st.error(f"Data directory does not exist: {data_path.absolute()}")
                    return
                
                parquet_files = list(data_path.rglob("*.parquet"))
                if len(parquet_files) == 0:
                    st.error(f"No parquet files found in: {data_path.absolute()}")
                    return
                
                st.info(f"Found {len(parquet_files)} parquet files in {data_path.absolute()}")
                
                # Run analysis based on selected type
                if analysis_type == "Price-only":
                    # Run price-only analysis
                    results = analyzer.run_price_only_analysis(
                        data_path,
                        method=price_method,
                        use_log_price=use_log_price,
                        n_clusters=n_clusters,
                        find_optimal_k=find_optimal_k,
                        clustering_method=clustering_method,
                        max_tokens=max_tokens,
                        max_length=max_sequence_length if use_max_length else None
                    )
                elif analysis_type == "Multi-Resolution ACF + Baseline Clustering":
                    # Run multi-resolution lifespan category analysis
                    st.info("🚀 Running Phase 1A: Multi-Resolution ACF Analysis...")
                    
                    # Use processed data directory for multi-resolution analysis
                    processed_data_path = data_path.parent / "processed"
                    if not processed_data_path.exists():
                        st.error(f"Processed data directory not found: {processed_data_path.absolute()}")
                        st.error("Please run data analysis first to generate processed categories.")
                        return
                    
                    st.info(f"Using processed data from: {processed_data_path.absolute()}")
                    
                    results = analyzer.analyze_by_lifespan_category(
                        processed_data_path,
                        method=multi_method,
                        use_log_price=use_log_price,
                        max_tokens_per_category=max_tokens_per_category
                    )
                    
                    # If DTW clustering is enabled, run it on each category
                    if enable_dtw_clustering:
                        st.info("Running DTW clustering for variable-length sequences...")
                        for category_name, category_results in results['categories'].items():
                            try:
                                dtw_results = analyzer.dtw_clustering_variable_length(
                                    category_results['token_data'],
                                    use_log_price=use_log_price,
                                    n_clusters=5,
                                    max_tokens=50  # Limit for DTW performance
                                )
                                category_results['dtw_clustering'] = dtw_results
                            except Exception as e:
                                st.warning(f"DTW clustering failed for {category_name}: {e}")
                    
                    # Compare ACF across categories if enabled
                    if compare_across_categories:
                        st.info("Comparing ACF patterns across lifespan categories...")
                        try:
                            acf_comparison = analyzer.compare_acf_across_lifespans(results)
                            results['acf_comparison'] = acf_comparison
                        except Exception as e:
                            st.warning(f"ACF comparison failed: {e}")
                    
                    # Run baseline clustering if enabled
                    if enable_baseline_clustering:
                        st.info("🎯 Running Baseline Clustering Analysis (CEO Requirements)...")
                        
                        baseline_results = run_baseline_clustering_analysis(
                            processed_data_path,
                            token_limits={
                                'sprint': None,  # No limits - use all tokens
                                'standard': None, 
                                'marathon': None
                            },
                            k_range=(k_range_min, k_range_max),
                            n_stability_runs=n_stability_runs,
                            export_results=export_results
                        )
                        
                        # Merge baseline results into main results
                        results['baseline_clustering'] = baseline_results
                
                
                else:
                    # Run complete feature-based analysis
                    results = analyzer.run_complete_analysis(
                        data_path,
                        use_log_price=use_log_price,
                        n_clusters=n_clusters,
                        find_optimal_k=find_optimal_k,
                        clustering_method=clustering_method,
                        max_tokens=max_tokens
                    )
                
                # Store results in session state
                st.session_state['results'] = results
                
                # Success message depends on analysis type
                if analysis_type == "Multi-Resolution ACF + Baseline Clustering":
                    total_tokens = results.get('total_tokens_analyzed', 0)
                    n_categories = len(results.get('categories', {}))
                    success_msg = f"✅ Multi-Resolution Analysis complete! Analyzed {total_tokens} tokens across {n_categories} lifespan categories."
                    
                    # Add baseline clustering info if enabled
                    if enable_baseline_clustering and 'baseline_clustering' in results:
                        baseline_results = results['baseline_clustering']
                        stability_summary = baseline_results.get('stability_summary', {})
                        avg_ari = stability_summary.get('average_ari', 0)
                        success_msg += f" Baseline clustering: Average ARI {avg_ari:.3f}"
                    
                    st.success(success_msg)
                else:
                    n_tokens = len(results.get('token_names', []))
                    st.success(f"✅ Analysis complete! Analyzed {n_tokens} tokens.")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.error(f"Full traceback: {traceback.format_exc()}")
                return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Create tabs based on analysis type  
        if 'categories' in results:  # Multi-Resolution analysis (includes baseline clustering)
            # Check if baseline clustering was run
            has_baseline = 'baseline_clustering' in results
            
            if has_baseline:
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "📊 Multi-Resolution Overview", 
                    "🏃 Category Comparison",
                    "📈 Cross-Category ACF",
                    "🎯 Baseline Clustering",
                    "📈 Stability Analysis",
                    "🗺️ Combined t-SNE",
                    "🎭 Behavioral Archetypes"
                ])
            else:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Multi-Resolution Overview", 
                    "🏃 Category Comparison",
                    "📈 Cross-Category ACF",
                    "🎯 Category Clustering",
                    "🗺️ Combined t-SNE",
                    "🎭 Behavioral Archetypes"
                ])
        else:  # Standard analysis
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Overview", 
                "📈 Autocorrelation Analysis",
                "🎯 Clustering Results",
                "📈 Elbow Analysis",
                "🗺️ t-SNE Visualization",
                "🔍 Token Explorer",
                "🎭 Behavioral Archetypes"
            ])
        
        # Handle tab content based on analysis type
        if 'categories' in results:  # Multi-Resolution analysis (includes baseline clustering)
            with tab1:
                display_multi_resolution_overview(results)
                
            with tab2:
                display_category_comparison(results)
                
            with tab3:
                display_cross_category_acf(results)
                
            if has_baseline:
                with tab4:
                    # Display baseline clustering results
                    if 'baseline_clustering' in results:
                        baseline_results = results['baseline_clustering']
                        display_baseline_cluster_overview(baseline_results)
                    else:
                        st.info("Baseline clustering not available")
                        
                with tab5:
                    # Display stability analysis
                    if 'baseline_clustering' in results:
                        baseline_results = results['baseline_clustering']
                        display_baseline_stability_analysis(baseline_results)
                    else:
                        st.info("Stability analysis not available")
                        
                with tab6:
                    display_combined_tsne(results)
                    
                with tab7:
                    display_behavioral_archetypes(results)
            else:
                with tab4:
                    display_category_clustering(results)
                    
                with tab5:
                    display_combined_tsne(results)
                    
                with tab6:
                    display_behavioral_archetypes(results)
        else:  # Standard analysis
            with tab1:
                display_overview(results)
                
            with tab2:
                display_autocorrelation_analysis(results, analyzer)
                
            with tab3:
                display_clustering_results(results)
                
            with tab4:
                display_elbow_analysis(results)
                
            with tab5:
                display_tsne_visualization(results, analyzer)
                
            with tab6:
                display_token_explorer(results)
                
            with tab7:
                display_behavioral_archetypes(results)
            
    else:
        st.info("👈 Configure parameters and click 'Run Analysis' to start")


def display_overview(results: Dict):
    """Display analysis overview"""
    st.header("📊 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tokens", len(results['token_names']))
        
    with col2:
        st.metric("Number of Clusters", results['n_clusters'])
        
    with col3:
        analysis_method = results.get('analysis_method', 'feature_based')
        if analysis_method.startswith('price_only'):
            method_display = f"Price-only ({analysis_method.split('_')[-1]})"
        else:
            method_display = "Feature-based"
        st.metric("Analysis Type", method_display)
        
    with col4:
        avg_length = np.mean([len(df) for df in results['token_data'].values()])
        st.metric("Avg Token Length", f"{avg_length:.0f} minutes")
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    
    cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(x=[f"Cluster {i}" for i in cluster_counts.index],
               y=cluster_counts.values,
               text=cluster_counts.values,
               textposition='auto')
    ])
    fig.update_layout(
        title="Number of Tokens per Cluster",
        xaxis_title="Cluster",
        yaxis_title="Number of Tokens",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics summary
    st.subheader("Cluster Characteristics")
    
    cluster_summary = []
    for cluster_id, stats in results['cluster_stats'].items():
        cluster_summary.append({
            'Cluster': cluster_id,
            'Tokens': stats['n_tokens'],
            'Avg Length': f"{stats['avg_length']:.0f}",
            'Avg Volatility': f"{stats['price_characteristics']['avg_volatility']:.4f}",
            'Avg Return': f"{stats['price_characteristics']['avg_return']:.4f}",
            'Sample Tokens': ', '.join(stats['tokens'][:3]) + '...'
        })
    
    st.dataframe(pd.DataFrame(cluster_summary), use_container_width=True)


def display_autocorrelation_analysis(results: Dict, analyzer: AutocorrelationClusteringAnalyzer):
    """Display autocorrelation analysis"""
    st.header("📈 Autocorrelation Analysis")
    
    # Check if ACF results are available
    if 'acf_results' not in results:
        st.warning("⚠️ Autocorrelation data is not available for this analysis.")
        st.info("💡 ACF computation may have failed or been skipped.")
        return
    
    # Add info about analysis type
    analysis_method = results.get('analysis_method', 'feature_based')
    if analysis_method.startswith('price_only'):
        st.info(f"🎯 **Price-only Analysis** ({analysis_method.split('_')[-1]}) with ACF computation")
    else:
        st.info("🎯 **Feature-based Analysis** with engineered features and ACF")
    
    # Select cluster to analyze
    cluster_id = st.selectbox("Select Cluster", 
                             sorted(np.unique(results['cluster_labels'])),
                             key="acf_cluster_selector")
    
    # Get tokens in selected cluster
    mask = results['cluster_labels'] == cluster_id
    cluster_tokens = [results['token_names'][i] for i in np.where(mask)[0]]
    
    st.info(f"Cluster {cluster_id} contains {len(cluster_tokens)} tokens")
    
    # Average ACF for cluster
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average ACF for Cluster")
        
        # Get ACF data for cluster
        cluster_acfs = []
        for token in cluster_tokens:
            if token in results.get('acf_results', {}):
                acf = results['acf_results'][token]['acf']
                if len(acf) > 0:
                    cluster_acfs.append(acf)
        
        if cluster_acfs:
            # Pad to same length
            max_len = max(len(acf) for acf in cluster_acfs)
            padded_acfs = []
            for acf in cluster_acfs:
                padded = np.pad(acf, (0, max_len - len(acf)), mode='constant', constant_values=0)
                padded_acfs.append(padded)
            
            avg_acf = np.mean(padded_acfs, axis=0)
            std_acf = np.std(padded_acfs, axis=0)
            
            # Plot average ACF with confidence band
            fig = go.Figure()
            
            x = list(range(len(avg_acf)))
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=list(avg_acf + 1.96 * std_acf) + list((avg_acf - 1.96 * std_acf)[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=False
            ))
            
            # Add average ACF
            fig.add_trace(go.Scatter(
                x=x,
                y=avg_acf,
                mode='lines+markers',
                name='Average ACF',
                line=dict(color='blue', width=2)
            ))
            
            # Add significance threshold
            n_avg = np.mean([len(df) for token in cluster_tokens if token in results['token_data'] for df in [results['token_data'][token]]])
            threshold = 1.96 / np.sqrt(n_avg)
            
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                         annotation_text="95% significance")
            fig.add_hline(y=-threshold, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            
            fig.update_layout(
                title=f"Average Autocorrelation Function - Cluster {cluster_id}",
                xaxis_title="Lag",
                yaxis_title="ACF",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ACF Characteristics Distribution")
        
        # Extract ACF characteristics
        decay_rates = []
        first_zeros = []
        significant_lags = []
        
        for token in cluster_tokens:
            if token in results.get('acf_results', {}):
                acf_res = results['acf_results'][token]
                if not np.isnan(acf_res['decay_rate']):
                    decay_rates.append(acf_res['decay_rate'])
                first_zeros.append(acf_res['first_zero_crossing'])
                significant_lags.append(len(acf_res['significant_lags']))
        
        # Create distribution plots
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Decay Rates', 'First Zero Crossings', 'Number of Significant Lags'))
        
        if decay_rates:
            fig.add_trace(go.Histogram(x=decay_rates, nbinsx=20, name='Decay Rate'),
                         row=1, col=1)
        
        fig.add_trace(go.Histogram(x=first_zeros, nbinsx=20, name='First Zero'),
                     row=2, col=1)
        
        fig.add_trace(go.Histogram(x=significant_lags, nbinsx=20, name='Significant Lags'),
                     row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual token ACF viewer
    st.subheader("Individual Token ACF - Multiple Tokens")
    
    # Allow selection of multiple tokens
    num_tokens_to_show = st.slider("Number of tokens to display", min_value=1, max_value=len(cluster_tokens), value=min(5, len(cluster_tokens)), key="acf_tokens_slider")
    
    # Let user select specific tokens or use first N
    use_custom_selection = st.checkbox("Select specific tokens", value=False, key="acf_custom_selection")
    
    if use_custom_selection:
        selected_tokens = st.multiselect(
            "Select tokens from cluster", 
            cluster_tokens, 
            default=cluster_tokens[:num_tokens_to_show],
            key="acf_token_multiselect"
        )
    else:
        selected_tokens = cluster_tokens[:num_tokens_to_show]
    
    if selected_tokens:
        # Create explanation box for ACF metrics
        with st.expander("📚 Understanding ACF Metrics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **Decay Rate**
                - How quickly autocorrelation diminishes
                - High rate = random behavior 
                - Low rate = trending/momentum
                """)
            with col2:
                st.markdown("""
                **First Zero Crossing**
                - First lag where ACF crosses zero
                - Early = short-term dependencies
                - Late = long-term memory
                """)
            with col3:
                st.markdown("""
                **Significant Lags**
                - Lags exceeding 95% confidence
                - More lags = stronger patterns
                - Indicates predictability
                """)
        
        # Create subplot for multiple tokens
        n_tokens = len(selected_tokens)
        n_cols = min(3, n_tokens)
        n_rows = (n_tokens + n_cols - 1) // n_cols
        
        # ACF plots
        st.write("**Autocorrelation Functions (ACF)**")
        fig_acf = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"ACF - {token[:15]}..." if len(token) > 15 else f"ACF - {token}" for token in selected_tokens],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # PACF plots  
        st.write("**Partial Autocorrelation Functions (PACF)**")
        fig_pacf = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"PACF - {token[:15]}..." if len(token) > 15 else f"PACF - {token}" for token in selected_tokens],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for idx, token in enumerate(selected_tokens):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            if token in results.get('acf_results', {}) and token in results['token_data']:
                acf_data = results['acf_results'][token]
                df = results['token_data'][token]
                n = len(df)
                threshold = 1.96 / np.sqrt(n)
                
                # ACF plot
                fig_acf.add_trace(go.Bar(
                    x=list(range(len(acf_data['acf']))),
                    y=acf_data['acf'],
                    name=f'ACF-{token}',
                    showlegend=False
                ), row=row, col=col)
                
                # Add significance lines for ACF
                fig_acf.add_hline(y=threshold, line_dash="dash", line_color="red", row=row, col=col)
                fig_acf.add_hline(y=-threshold, line_dash="dash", line_color="red", row=row, col=col)
                fig_acf.add_hline(y=0, line_color="black", line_width=0.5, row=row, col=col)
                
                # PACF plot
                if len(acf_data['pacf']) > 0:
                    fig_pacf.add_trace(go.Bar(
                        x=list(range(len(acf_data['pacf']))),
                        y=acf_data['pacf'],
                        name=f'PACF-{token}',
                        showlegend=False
                    ), row=row, col=col)
                    
                    # Add significance lines for PACF
                    fig_pacf.add_hline(y=threshold, line_dash="dash", line_color="red", row=row, col=col)
                    fig_pacf.add_hline(y=-threshold, line_dash="dash", line_color="red", row=row, col=col)
                    fig_pacf.add_hline(y=0, line_color="black", line_width=0.5, row=row, col=col)
        
        # Update layouts
        fig_acf.update_layout(height=300*n_rows, showlegend=False)
        fig_pacf.update_layout(height=300*n_rows, showlegend=False)
        
        # Update axes labels
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                fig_acf.update_xaxes(title_text="Lag", row=row, col=col)
                fig_acf.update_yaxes(title_text="ACF", row=row, col=col)
                fig_pacf.update_xaxes(title_text="Lag", row=row, col=col)
                fig_pacf.update_yaxes(title_text="PACF", row=row, col=col)
        
        st.plotly_chart(fig_acf, use_container_width=True)
        st.plotly_chart(fig_pacf, use_container_width=True)
        
        # Summary table for selected tokens
        st.subheader("ACF Summary for Selected Tokens")
        summary_data = []
        for token in selected_tokens:
            if token in results.get('acf_results', {}):
                acf_res = results['acf_results'][token]
                summary_data.append({
                    'Token': token[:20] + '...' if len(token) > 20 else token,
                    'Significant Lags': len(acf_res['significant_lags']),
                    'Decay Rate': f"{acf_res['decay_rate']:.4f}" if not np.isnan(acf_res['decay_rate']) else "N/A",
                    'First Zero Crossing': acf_res['first_zero_crossing'],
                    'ACF at Lag 1': f"{acf_res['acf'][1]:.4f}" if len(acf_res['acf']) > 1 else "N/A"
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


def display_clustering_results(results: Dict):
    """Display clustering results"""
    st.header("🎯 Clustering Results")
    
    # Feature importance (which features contribute most to clustering)
    st.subheader("Feature Contributions")
    
    # Determine feature names based on analysis type
    analysis_method = results.get('analysis_method', 'feature_based')
    feature_matrix = results['feature_matrix']
    
    if analysis_method.startswith('price_only'):
        # For price-only analysis, features are time series values
        price_method = analysis_method.split('_')[-1]
        sequence_length = results.get('sequence_length', feature_matrix.shape[1])
        
        if price_method == 'dtw':
            # DTW features are statistical measures
            feature_names = [
                'Q10', 'Q25', 'Q50', 'Q75', 'Q90',  # Quantiles
                'Mean Return', 'Std Return', 'Skew Return',  # Returns stats
                'Trend Slope',  # Trend
                'Vol Window 5', 'Vol Window 10', 'Vol Window 20'  # Rolling volatility
            ]
        else:
            # Time series features (returns, log_returns, prices, log_prices)
            if sequence_length == 'variable':
                feature_names = [f'{price_method.title()} Feature {i+1}' for i in range(feature_matrix.shape[1])]
            else:
                feature_names = [f'{price_method.title()} T-{i+1}' for i in range(min(feature_matrix.shape[1], 20))]
                if feature_matrix.shape[1] > 20:
                    feature_names.extend([f'{price_method.title()} T-{i+1}' for i in range(20, feature_matrix.shape[1])])
    else:
        # Feature-based analysis - use the original 16 engineered features
        feature_names = [
            'Mean Price', 'Std Price', 'Median Price', 'Q1 Price', 'Q3 Price',
            'Mean Return', 'Std Return', 'Min Return', 'Max Return',
            'ACF Lag 1', 'ACF Lag 5', 'ACF Lag 10',
            'Significant Lags', 'Decay Rate', 'First Zero',
            'Trend Slope'
        ]
    
    # Calculate feature variance across clusters
    feature_matrix = results['feature_matrix']
    labels = results['cluster_labels']
    
    feature_importance = []
    for i, feature_name in enumerate(feature_names[:feature_matrix.shape[1]]):
        # Calculate between-cluster variance
        cluster_means = []
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_means.append(np.mean(feature_matrix[mask, i]))
        
        variance = np.var(cluster_means)
        feature_importance.append({
            'Feature': feature_name,
            'Variance': variance
        })
    
    importance_df = pd.DataFrame(feature_importance).sort_values('Variance', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(x=importance_df['Variance'],
               y=importance_df['Feature'],
               orientation='h')
    ])
    fig.update_layout(
        title="Feature Importance for Clustering (Between-Cluster Variance)",
        xaxis_title="Variance",
        yaxis_title="Feature",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster comparison heatmap
    st.subheader("Cluster Feature Comparison")
    
    # Calculate mean features per cluster
    cluster_features = []
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        mean_features = np.mean(feature_matrix[mask], axis=0)
        cluster_features.append(mean_features)
    
    cluster_features = np.array(cluster_features)
    
    # Normalize features for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cluster_features_norm = scaler.fit_transform(cluster_features.T).T
    
    fig = go.Figure(data=go.Heatmap(
        z=cluster_features_norm,
        x=feature_names[:cluster_features_norm.shape[1]],
        y=[f"Cluster {i}" for i in range(len(cluster_features_norm))],
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(
        title="Normalized Feature Values by Cluster",
        xaxis_title="Features",
        yaxis_title="Clusters",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def display_elbow_analysis(results: Dict):
    """Display elbow method analysis for optimal K selection"""
    st.header("📈 Elbow Analysis - Optimal Number of Clusters")
    
    if 'clustering_results' in results and results['clustering_results']['elbow_analysis']:
        elbow_data = results['clustering_results']['elbow_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Optimal K (Elbow)", elbow_data['optimal_k_elbow'])
            
        with col2:
            st.metric("Optimal K (Silhouette)", elbow_data['optimal_k_silhouette'])
            
        with col3:
            actual_k = results['clustering_results']['n_clusters']
            st.metric("Used K", actual_k)
        
        # Create elbow curve plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method (Inertia)', 'Silhouette Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Elbow curve
        fig.add_trace(
            go.Scatter(
                x=elbow_data['k_range'],
                y=elbow_data['inertias'],
                mode='lines+markers',
                name='Inertia',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Mark optimal K from elbow
        optimal_k_elbow = elbow_data['optimal_k_elbow']
        if optimal_k_elbow in elbow_data['k_range']:
            idx = elbow_data['k_range'].index(optimal_k_elbow)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k_elbow],
                    y=[elbow_data['inertias'][idx]],
                    mode='markers',
                    name=f'Elbow K={optimal_k_elbow}',
                    marker=dict(color='red', size=15, symbol='star'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=optimal_k_elbow,
                y=elbow_data['inertias'][idx],
                text=f"Elbow<br>K={optimal_k_elbow}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                arrowwidth=2,
                row=1, col=1
            )
        
        # Silhouette scores
        fig.add_trace(
            go.Scatter(
                x=elbow_data['k_range'],
                y=elbow_data['silhouette_scores'],
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Mark optimal K from silhouette
        optimal_k_sil = elbow_data['optimal_k_silhouette']
        if optimal_k_sil in elbow_data['k_range']:
            idx = elbow_data['k_range'].index(optimal_k_sil)
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k_sil],
                    y=[elbow_data['silhouette_scores'][idx]],
                    mode='markers',
                    name=f'Best Silhouette K={optimal_k_sil}',
                    marker=dict(color='orange', size=15, symbol='star'),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Add annotation
            fig.add_annotation(
                x=optimal_k_sil,
                y=elbow_data['silhouette_scores'][idx],
                text=f"Best Silhouette<br>K={optimal_k_sil}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="orange",
                arrowwidth=2,
                row=1, col=2
            )
        
        fig.update_layout(
            title='Optimal Number of Clusters Analysis',
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig.update_yaxes(title_text="Inertia (Within-cluster sum of squares)", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.subheader("📚 How to Interpret")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Elbow Method:**
            - Look for the "elbow" where inertia stops decreasing rapidly
            - Point of diminishing returns in cluster quality
            - Lower inertia = tighter clusters
            """)
            
        with col2:
            st.markdown("""
            **Silhouette Score:**
            - Measures how similar tokens are to their own cluster vs other clusters
            - Range: -1 to 1 (higher is better)
            - > 0.5 = good clustering, > 0.7 = strong clustering
            """)
        
        # Show detailed scores table
        st.subheader("📊 Detailed Scores")
        
        scores_df = pd.DataFrame({
            'K': elbow_data['k_range'],
            'Inertia': elbow_data['inertias'],
            'Silhouette Score': elbow_data['silhouette_scores']
        })
        
        # Highlight optimal values
        def highlight_optimal(row):
            styles = [''] * len(row)
            if row['K'] == optimal_k_elbow:
                styles[0] = 'background-color: #ffcccc'  # Light red for elbow
                styles[1] = 'background-color: #ffcccc'
            if row['K'] == optimal_k_sil:
                styles[0] = 'background-color: #ffffcc'  # Light yellow for silhouette
                styles[2] = 'background-color: #ffffcc'
            return styles
        
        st.dataframe(scores_df.style.apply(highlight_optimal, axis=1), use_container_width=True)
        
        # Add option to rerun with selected K
        st.subheader("🔄 Rerun Analysis with Selected K")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_k = st.selectbox(
                "Select number of clusters to use:",
                options=elbow_data['k_range'],
                index=elbow_data['k_range'].index(elbow_data['optimal_k_silhouette']) if elbow_data['optimal_k_silhouette'] in elbow_data['k_range'] else 0,
                help="Choose the number of clusters based on the elbow analysis above",
                key="elbow_k_selector"
            )
        
        with col2:
            if st.button("🚀 Rerun with K=" + str(selected_k), type="primary"):
                # Store the selected K in session state and trigger rerun
                st.session_state['rerun_with_k'] = selected_k
                st.session_state['rerun_requested'] = True
                st.rerun()
        
        if selected_k != results['n_clusters']:
            st.info(f"💡 Current analysis uses K={results['n_clusters']}. Click 'Rerun' to analyze with K={selected_k}")
        
    else:
        st.info("Elbow analysis not available. Run analysis with 'Find Optimal K' enabled.")


def display_tsne_visualization(results: Dict, analyzer: AutocorrelationClusteringAnalyzer):
    """Display t-SNE visualization"""
    st.header("🗺️ t-SNE Visualization")
    
    # 2D vs 3D selection
    viz_type = st.radio("Visualization Type", ["2D", "3D"])
    
    if viz_type == "2D":
        embedding = results['t_sne_2d']
        
        # Create interactive scatter plot
        fig = go.Figure()
        
        # Add traces for each cluster
        for cluster_id in sorted(np.unique(results['cluster_labels'])):
            mask = results['cluster_labels'] == cluster_id
            cluster_tokens = [results['token_names'][i] for i in np.where(mask)[0]]
            
            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=cluster_tokens,
                marker=dict(size=8),
                hovertemplate='Token: %{text}<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="2D t-SNE Visualization of Token Clusters",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # 3D
        embedding = results['t_sne_3d']
        
        fig = go.Figure()
        
        for cluster_id in sorted(np.unique(results['cluster_labels'])):
            mask = results['cluster_labels'] == cluster_id
            cluster_tokens = [results['token_names'][i] for i in np.where(mask)[0]]
            
            fig.add_trace(go.Scatter3d(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                z=embedding[mask, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=cluster_tokens,
                marker=dict(size=5),
                hovertemplate='Token: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="3D t-SNE Visualization of Token Clusters",
            scene=dict(
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                zaxis_title="t-SNE 3"
            ),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Distance matrix visualization
    st.subheader("Token Similarity Matrix")
    
    selected_cluster = st.selectbox("Select Cluster for Distance Matrix", 
                                   sorted(np.unique(results['cluster_labels'])),
                                   key="tsne_cluster_selector")
    
    # Get tokens in cluster
    mask = results['cluster_labels'] == selected_cluster
    cluster_indices = np.where(mask)[0][:20]  # Limit to 20 tokens for visibility
    
    if len(cluster_indices) > 1:
        # Calculate pairwise distances in t-SNE space
        cluster_embedding = embedding[cluster_indices]
        
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(cluster_embedding))
        
        cluster_token_names = [results['token_names'][i] for i in cluster_indices]
        
        fig = go.Figure(data=go.Heatmap(
            z=distances,
            x=cluster_token_names,
            y=cluster_token_names,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f"Token Distance Matrix - Cluster {selected_cluster} (t-SNE space)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_token_explorer(results: Dict):
    """Display individual token explorer"""
    st.header("🔍 Token Explorer")
    
    # Mode selection
    exploration_mode = st.radio("Exploration Mode", ["Individual Token", "Cluster Analysis"])
    
    if exploration_mode == "Individual Token":
        # Original individual token explorer
        selected_token = st.selectbox("Select Token", sorted(results['token_names']),
                                     key="individual_token_selector")
        
        if selected_token in results['token_data']:
            # Get token data
            df = results['token_data'][selected_token]
            cluster_id = results['cluster_labels'][results['token_names'].index(selected_token)]
            
            # Display token info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cluster", cluster_id)
            
            with col2:
                st.metric("Data Points", len(df))
            
            with col3:
                price_change = (df['price'][-1] - df['price'][0]) / df['price'][0] * 100
                st.metric("Price Change", f"{price_change:.2f}%")
            
            with col4:
                returns = np.diff(df['price'].to_numpy()) / df['price'].to_numpy()[:-1]
                volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
                st.metric("Volatility", f"{volatility:.2f}%")
            
            # Price chart
            st.subheader("Price Chart")
            
            price_col = 'log_price' if results['use_log_price'] else 'price'
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[price_col],
                mode='lines',
                name=price_col.replace('_', ' ').title()
            ))
            
            fig.update_layout(
                title=f"{selected_token} - {price_col.replace('_', ' ').title()}",
                xaxis_title="Time",
                yaxis_title=price_col.replace('_', ' ').title(),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ACF details
            if selected_token in results.get('acf_results', {}):
                st.subheader("Autocorrelation Details")
                
                acf_data = results['acf_results'][selected_token]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Significant Lags", len(acf_data['significant_lags']))
                    st.metric("First Zero Crossing", acf_data['first_zero_crossing'])
                
                with col2:
                    if not np.isnan(acf_data['decay_rate']):
                        st.metric("Decay Rate", f"{acf_data['decay_rate']:.4f}")
                    st.metric("ACF at Lag 1", f"{acf_data['acf'][1]:.4f}" if len(acf_data['acf']) > 1 else "N/A")
            
            # Similar tokens in cluster
            st.subheader("Similar Tokens (Same Cluster)")
            
            mask = results['cluster_labels'] == cluster_id
            similar_tokens = [results['token_names'][i] for i in np.where(mask)[0] if results['token_names'][i] != selected_token]
            
            if similar_tokens:
                # Show first 10 similar tokens
                st.write(", ".join(similar_tokens[:10]))
    
    else:  # Cluster Analysis mode
        st.subheader("Cluster-Based Token Analysis")
        
        # Cluster selection
        selected_cluster = st.selectbox("Select Cluster", sorted(np.unique(results['cluster_labels'])),
                                        key="cluster_analysis_selector")
        
        # Get tokens in selected cluster
        mask = results['cluster_labels'] == selected_cluster
        cluster_tokens = [results['token_names'][i] for i in np.where(mask)[0]]
        
        st.info(f"Cluster {selected_cluster} contains {len(cluster_tokens)} tokens")
        
        # Show cluster characteristics
        if selected_cluster in results['cluster_stats']:
            stats = results['cluster_stats'][selected_cluster]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tokens in Cluster", stats['n_tokens'])
            with col2:
                st.metric("Avg Length", f"{stats['avg_length']:.0f} min")
            with col3:
                st.metric("Avg Volatility", f"{stats['price_characteristics']['avg_volatility']:.4f}")
        
        # Select tokens to display
        num_tokens_display = st.slider("Number of tokens to display", min_value=1, max_value=len(cluster_tokens), value=min(10, len(cluster_tokens)), key="cluster_tokens_slider")
        
        # Option to select specific tokens or use first N
        use_custom_selection = st.checkbox("Select specific tokens", value=False, key="cluster_explorer_custom")
        
        if use_custom_selection:
            display_tokens = st.multiselect(
                "Select tokens to display", 
                cluster_tokens, 
                default=cluster_tokens[:num_tokens_display],
                key="cluster_explorer_tokens"
            )
        else:
            display_tokens = cluster_tokens[:num_tokens_display]
        
        if display_tokens:
            # Create price charts for selected tokens
            st.subheader(f"Price Charts for {len(display_tokens)} Tokens")
            
            price_col = 'log_price' if results['use_log_price'] else 'price'
            
            # Create subplots for price charts
            n_tokens = len(display_tokens)
            n_cols = min(2, n_tokens)  # Max 2 columns for better visibility
            n_rows = (n_tokens + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f"{token[:20]}..." if len(token) > 20 else token for token in display_tokens],
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            for idx, token in enumerate(display_tokens):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                if token in results['token_data']:
                    df = results['token_data'][token]
                    
                    fig.add_trace(go.Scatter(
                        x=df['datetime'],
                        y=df[price_col],
                        mode='lines',
                        name=f'{token[:15]}...' if len(token) > 15 else token,
                        showlegend=False
                    ), row=row, col=col)
            
            # Update layout
            fig.update_layout(height=400*n_rows, showlegend=False)
            
            # Update axes labels
            for row in range(1, n_rows + 1):
                for col in range(1, n_cols + 1):
                    fig.update_xaxes(title_text="Time", row=row, col=col)
                    fig.update_yaxes(title_text=price_col.replace('_', ' ').title(), row=row, col=col)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics table
            st.subheader("Token Statistics Summary")
            
            summary_data = []
            for token in display_tokens:
                if token in results['token_data']:
                    df = results['token_data'][token]
                    
                    # Calculate basic stats
                    returns = np.diff(df['price'].to_numpy()) / df['price'].to_numpy()[:-1]
                    price_change = (df['price'][-1] - df['price'][0]) / df['price'][0] * 100
                    volatility = np.std(returns) * 100  # As percentage
                    
                    summary_entry = {
                        'Token': token[:25] + '...' if len(token) > 25 else token,
                        'Data Points': len(df),
                        'Price Change (%)': f"{price_change:.2f}",
                        'Volatility (%)': f"{volatility:.2f}"
                    }
                    
                    # Add ACF data if available
                    if token in results.get('acf_results', {}):
                        acf_res = results['acf_results'][token]
                        summary_entry.update({
                            'Significant Lags': len(acf_res['significant_lags']),
                            'Decay Rate': f"{acf_res['decay_rate']:.4f}" if not np.isnan(acf_res['decay_rate']) else "N/A",
                            'First Zero': acf_res['first_zero_crossing']
                        })
                    
                    summary_data.append(summary_entry)
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Additional cluster analysis
            st.subheader("Cluster Pattern Analysis")
            
            # Calculate cluster-wide statistics
            all_returns = []
            all_volatilities = []
            all_decay_rates = []
            
            for token in display_tokens:
                if token in results['token_data']:
                    df = results['token_data'][token]
                    
                    returns = np.diff(df['price'].to_numpy()) / df['price'].to_numpy()[:-1]
                    volatility = np.std(returns)
                    
                    all_returns.extend(returns.tolist())
                    all_volatilities.append(volatility)
                    
                    # Add decay rate if ACF data is available
                    if token in results.get('acf_results', {}):
                        acf_res = results['acf_results'][token]
                        if not np.isnan(acf_res['decay_rate']):
                            all_decay_rates.append(acf_res['decay_rate'])
            
            if all_returns and all_volatilities:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig = go.Figure(data=[go.Histogram(x=all_returns, nbinsx=30)])
                    fig.update_layout(title="Return Distribution", xaxis_title="Returns", yaxis_title="Frequency", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure(data=[go.Histogram(x=all_volatilities, nbinsx=20)])
                    fig.update_layout(title="Volatility Distribution", xaxis_title="Volatility", yaxis_title="Frequency", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    if all_decay_rates:
                        fig = go.Figure(data=[go.Histogram(x=all_decay_rates, nbinsx=20)])
                        fig.update_layout(title="Decay Rate Distribution", xaxis_title="Decay Rate", yaxis_title="Frequency", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid decay rates in this cluster")


# ================================
# MULTI-RESOLUTION ANALYSIS DISPLAY FUNCTIONS
# ================================

def display_multi_resolution_overview(results: Dict):
    """Display overview of multi-resolution analysis results"""
    st.header("📊 Multi-Resolution Analysis Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tokens Analyzed", format_large_number(results.get('total_tokens_analyzed', 0)))
        
    with col2:
        st.metric("Lifespan Categories", len(results.get('categories', {})))
        
    with col3:
        analysis_method = results.get('analysis_method', 'multi_resolution')
        st.metric("Analysis Method", analysis_method.replace('_', ' ').title())
        
    with col4:
        if 'acf_comparison' in results:
            st.metric("ACF Comparison", "✅ Available")
        else:
            st.metric("ACF Comparison", "❌ Not computed")
    
    # Category summary
    st.subheader("📈 Lifespan Category Distribution")
    
    if 'category_summary' in results:
        summary_data = []
        for category, info in results['category_summary'].items():
            summary_data.append({
                'Category': category,
                'Tokens': info['n_tokens'],
                'Clusters': info['n_clusters'],
                'Lifespan Range': info['lifespan_range']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Category distribution chart
        fig = px.bar(summary_df, x='Category', y='Tokens', 
                    title="Token Distribution by Lifespan Category",
                    color='Category')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🎯 Key Insights")
    categories = results.get('categories', {})
    
    if categories:
        insights = []
        
        for category_name, category_results in categories.items():
            n_tokens = len(category_results.get('token_data', {}))
            n_clusters = category_results.get('n_clusters', 0)
            lifespan_range = category_results.get('lifespan_range', 'Unknown')
            
            insights.append(f"**{category_name}**: {n_tokens} tokens ({lifespan_range}) → {n_clusters} behavioral clusters")
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    # Analysis method info
    st.subheader("⚙️ Analysis Configuration")
    st.markdown(f"""
    - **Price Transformation**: {results.get('analysis_method', 'unknown').replace('multi_resolution_', '')}
    - **Categories Analyzed**: {len(categories)}
    - **Total Clusters Found**: {sum(cat.get('n_clusters', 0) for cat in categories.values())}
    """)


def display_category_comparison(results: Dict):
    """Display comparison between lifespan categories"""
    st.header("🏃 Category Comparison Analysis")
    
    categories = results.get('categories', {})
    
    if not categories:
        st.error("No category data available")
        return
    
    # Create comparison metrics
    comparison_data = []
    for category_name, category_results in categories.items():
        n_tokens = len(category_results.get('token_data', {}))
        n_clusters = category_results.get('n_clusters', 0)
        lifespan_range = category_results.get('lifespan_range', 'Unknown')
        
        # Calculate average cluster size
        avg_cluster_size = n_tokens / n_clusters if n_clusters > 0 else 0
        
        comparison_data.append({
            'Category': category_name,
            'Tokens': n_tokens,
            'Clusters': n_clusters,
            'Avg Cluster Size': round(avg_cluster_size, 1),
            'Lifespan Range': lifespan_range
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.subheader("📊 Category Metrics Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison_df, x='Category', y='Tokens', 
                    title="Tokens per Category", color='Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison_df, x='Category', y='Clusters', 
                    title="Clusters per Category", color='Category')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed category analysis
    st.subheader("🔍 Detailed Category Analysis")
    
    selected_category = st.selectbox("Select category for detailed analysis:", 
                                   list(categories.keys()))
    
    if selected_category in categories:
        category_results = categories[selected_category]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tokens in Category", len(category_results.get('token_data', {})))
            
        with col2:
            st.metric("Clusters Found", category_results.get('n_clusters', 0))
            
        with col3:
            st.metric("Lifespan Range", category_results.get('lifespan_range', 'Unknown'))
        
        # Show cluster characteristics for selected category
        if 'cluster_stats' in category_results:
            st.subheader(f"📈 {selected_category} Cluster Characteristics")
            
            cluster_data = []
            for cluster_id, stats in category_results['cluster_stats'].items():
                cluster_data.append({
                    'Cluster': f"Cluster {cluster_id}",
                    'Tokens': stats.get('n_tokens', 0),
                    'Avg Return': f"{stats.get('price_characteristics', {}).get('avg_return', 0)*100:.1f}%",
                    'Avg Volatility': f"{stats.get('price_characteristics', {}).get('avg_volatility', 0)*100:.1f}%"
                })
            
            if cluster_data:
                cluster_df = pd.DataFrame(cluster_data)
                st.dataframe(cluster_df, use_container_width=True)


def display_cross_category_acf(results: Dict):
    """Display cross-category ACF comparison"""
    st.header("📈 Cross-Category ACF Analysis")
    
    if 'acf_comparison' not in results:
        st.warning("ACF comparison not available. Enable 'Cross-category ACF comparison' in the sidebar and rerun analysis.")
        return
    
    acf_comparison = results['acf_comparison']
    
    # Category ACF means comparison
    if 'category_acf_means' in acf_comparison:
        st.subheader("🎯 ACF Patterns by Lifespan Category")
        
        fig = go.Figure()
        
        for category_name, acf_mean in acf_comparison['category_acf_means'].items():
            lags = list(range(len(acf_mean)))
            fig.add_trace(go.Scatter(
                x=lags, y=acf_mean,
                mode='lines+markers',
                name=f"{category_name}",
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Average ACF Patterns by Lifespan Category",
            xaxis_title="Lag (minutes)",
            yaxis_title="Autocorrelation",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cross-category correlations
    if 'cross_category_correlations' in acf_comparison:
        st.subheader("🔗 Cross-Category ACF Correlations")
        
        corr_data = []
        for comparison, correlation in acf_comparison['cross_category_correlations'].items():
            cat1, cat2 = comparison.split('_vs_')
            corr_data.append({
                'Category 1': cat1,
                'Category 2': cat2,
                'ACF Correlation': round(correlation, 3)
            })
        
        corr_df = pd.DataFrame(corr_data)
        st.dataframe(corr_df, use_container_width=True)
        
        # Correlation heatmap
        categories = list(results.get('categories', {}).keys())
        if len(categories) > 1:
            # Create correlation matrix
            corr_matrix = np.eye(len(categories))
            
            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if i != j:
                        key1 = f"{cat1}_vs_{cat2}"
                        key2 = f"{cat2}_vs_{cat1}"
                        
                        if key1 in acf_comparison['cross_category_correlations']:
                            corr_matrix[i, j] = acf_comparison['cross_category_correlations'][key1]
                        elif key2 in acf_comparison['cross_category_correlations']:
                            corr_matrix[i, j] = acf_comparison['cross_category_correlations'][key2]
            
            fig = px.imshow(corr_matrix, 
                          x=categories, y=categories,
                          title="ACF Pattern Similarity Between Categories",
                          color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Distinctive patterns
    if 'distinctive_patterns' in acf_comparison:
        st.subheader("🎨 Category-Specific Distinctive Patterns")
        
        for category, patterns in acf_comparison['distinctive_patterns'].items():
            with st.expander(f"📊 {category} Distinctive Lags"):
                st.markdown(f"**Most distinctive lags for {category}:**")
                
                distinctive_data = []
                for i, lag in enumerate(patterns['distinctive_lags']):
                    difference = patterns['differences'][i]
                    distinctive_data.append({
                        'Lag': lag,
                        'Difference from Others': round(difference, 4)
                    })
                
                distinctive_df = pd.DataFrame(distinctive_data)
                st.dataframe(distinctive_df, use_container_width=True)


def display_category_clustering(results: Dict):
    """Display clustering results for each category"""
    st.header("🎯 Category-Specific Clustering Analysis")
    
    categories = results.get('categories', {})
    
    if not categories:
        st.error("No category data available")
        return
    
    # Category selection
    selected_category = st.selectbox("Select category for clustering analysis:", 
                                   list(categories.keys()))
    
    if selected_category not in categories:
        return
    
    category_results = categories[selected_category]
    
    # Display cluster information
    st.subheader(f"🔍 {selected_category} Category Clustering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tokens", len(category_results.get('token_data', {})))
        
    with col2:
        st.metric("Number of Clusters", category_results.get('n_clusters', 0))
        
    with col3:
        st.metric("Lifespan Range", category_results.get('lifespan_range', 'Unknown'))
    
    # t-SNE visualization for this category
    if 't_sne_2d' in category_results:
        st.subheader("🗺️ t-SNE Cluster Visualization")
        
        tsne_2d = category_results['t_sne_2d']
        cluster_labels = category_results.get('cluster_labels', [])
        
        if len(tsne_2d) > 0:
            fig = px.scatter(
                x=tsne_2d[:, 0], y=tsne_2d[:, 1],
                color=[f"Cluster {label}" for label in cluster_labels],
                title=f"{selected_category} Category: t-SNE Clustering Visualization",
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    if 'cluster_stats' in category_results:
        st.subheader("📊 Cluster Characteristics")
        
        cluster_stats = category_results['cluster_stats']
        
        for cluster_id, stats in cluster_stats.items():
            with st.expander(f"Cluster {cluster_id} Details"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tokens in Cluster", stats.get('n_tokens', 0))
                    
                with col2:
                    avg_return = stats.get('price_characteristics', {}).get('avg_return', 0)
                    st.metric("Avg Return", f"{avg_return*100:.1f}%")
                    
                with col3:
                    avg_volatility = stats.get('price_characteristics', {}).get('avg_volatility', 0)
                    st.metric("Avg Volatility", f"{avg_volatility*100:.1f}%")
    
    # DTW clustering results if available
    if 'dtw_clustering' in category_results:
        st.subheader("🔄 DTW Clustering Results")
        
        dtw_results = category_results['dtw_clustering']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("DTW Clusters Found", dtw_results.get('n_clusters', 0))
            
        with col2:
            st.metric("Tokens in DTW Analysis", len(dtw_results.get('token_names', [])))
        
        # DTW cluster statistics
        if 'cluster_stats' in dtw_results:
            st.markdown("**DTW Cluster Length Statistics:**")
            
            dtw_cluster_data = []
            for cluster_id, stats in dtw_results['cluster_stats'].items():
                dtw_cluster_data.append({
                    'DTW Cluster': f"Cluster {cluster_id}",
                    'Tokens': stats.get('n_tokens', 0),
                    'Avg Length': round(stats.get('avg_length', 0), 1),
                    'Min Length': stats.get('min_length', 0),
                    'Max Length': stats.get('max_length', 0)
                })
            
            if dtw_cluster_data:
                dtw_df = pd.DataFrame(dtw_cluster_data)
                st.dataframe(dtw_df, use_container_width=True)


def display_combined_tsne(results: Dict):
    """Display combined t-SNE visualization across all categories"""
    st.header("🗺️ Combined Multi-Resolution t-SNE")
    
    categories = results.get('categories', {})
    
    if not categories:
        st.error("No category data available")
        return
    
    # Combine t-SNE data from all categories
    combined_tsne_data = []
    combined_colors = []
    combined_labels = []
    
    color_map = {
        'Sprint': 'red',
        'Standard': 'blue', 
        'Marathon': 'green'
    }
    
    for category_name, category_results in categories.items():
        if 't_sne_2d' in category_results:
            tsne_2d = category_results['t_sne_2d']
            cluster_labels = category_results.get('cluster_labels', [])
            
            for i, row in enumerate(tsne_2d):
                x, y = row[0], row[1]
                combined_tsne_data.append([x, y])
                combined_colors.append(category_name)
                
                # Create combined label with category and cluster
                cluster_id = cluster_labels[i] if i < len(cluster_labels) else 0
                combined_labels.append(f"{category_name}-C{cluster_id}")
    
    if combined_tsne_data:
        combined_tsne_data = np.array(combined_tsne_data)
        
        # Create combined visualization
        fig = px.scatter(
            x=combined_tsne_data[:, 0], 
            y=combined_tsne_data[:, 1],
            color=combined_colors,
            hover_data={'Cluster': combined_labels},
            title="Combined Multi-Resolution t-SNE: All Categories",
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Lifespan Category'}
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Combined Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Data Points", format_data_points(len(combined_tsne_data)))
            
        with col2:
            st.metric("Categories Visualized", len(set(combined_colors)))
            
        with col3:
            st.metric("Unique Cluster-Category Combinations", len(set(combined_labels)))
        
        # Category distribution in combined view
        category_counts = pd.Series(combined_colors).value_counts()
        
        fig = px.pie(values=category_counts.values, 
                    names=category_counts.index,
                    title="Distribution of Tokens Across Categories")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No t-SNE data available for visualization")


def display_behavioral_archetypes(results: Dict):
    """Display behavioral archetype analysis interface"""
    st.header("🎭 Behavioral Archetype Analysis")
    
    st.markdown("""
    This section analyzes memecoin tokens to identify behavioral archetypes including death patterns.
    
    **Key Features:**
    - Death detection using robust algorithms for tokens with small values
    - Identification of 5-8 behavioral archetypes
    - Early detection rules using only first 5 minutes of data
    - Survival analysis and return profiles
    """)
    
    # Initialize behavioral analyzer
    if 'behavioral_analyzer' not in st.session_state:
        st.session_state.behavioral_analyzer = BehavioralArchetypeAnalyzer()
    
    behavioral_analyzer = st.session_state.behavioral_analyzer
    
    # Configuration section
    st.subheader("🛠️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Data source selection
        project_root = Path(__file__).parent.parent
        processed_dir = project_root / "data" / "processed"
        
        st.write("**Data Source**: Processed Categories")
        st.write(f"Directory: `{processed_dir}`")
        
        if processed_dir.exists():
            categories = [d.name for d in processed_dir.iterdir() if d.is_dir()]
            st.write(f"Available categories: {', '.join(categories)}")
        else:
            st.error("Processed data directory not found. Please run data analysis first.")
            return
    
    with col2:
        # Analysis parameters
        token_limit_input = st.text_input(
            "Tokens per category (or 'none' for unlimited)",
            value="1000",
            help="Limit tokens per category for faster analysis. Use 'none' for unlimited comprehensive analysis."
        )
        
        # Parse token limit
        try:
            if token_limit_input.lower() == 'none':
                token_limit = None
            else:
                token_limit = int(token_limit_input)
                if token_limit < 10:
                    st.error("Token limit must be at least 10 or 'none'")
                    token_limit = 1000
        except ValueError:
            st.error("Invalid input. Please enter a number or 'none'")
            token_limit = 1000
        
        n_clusters_range = st.multiselect(
            "Number of clusters to try",
            options=[5, 6, 7, 8, 9, 10],
            default=[6, 7, 8],
            help="Range of cluster numbers to test"
        )
    
    with col3:
        # Output options
        save_results = st.checkbox("Save results to files", value=True)
        
        if save_results:
            output_dir = project_root / "time_series" / "results"
            st.write(f"**Output directory**: `{output_dir}`")
            output_dir.mkdir(exist_ok=True)
    
    # Run analysis button
    if st.button("🚀 Run Behavioral Archetype Analysis", type="primary"):
        if not processed_dir.exists():
            st.error("Processed data directory not found")
            return
        
        if not n_clusters_range:
            st.error("Please select at least one cluster number")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("Loading categorized token data...")
            progress_bar.progress(10)
            
            token_data = behavioral_analyzer.load_categorized_tokens(
                processed_dir, limit=token_limit
            )
            
            if not token_data:
                st.error("No token data found")
                return
            
            # Step 2: Extract features
            status_text.text("Extracting features (death detection, lifecycle, early features)...")
            progress_bar.progress(30)
            
            features_df = behavioral_analyzer.extract_all_features(token_data)
            
            # Step 3: Perform clustering
            status_text.text("Performing clustering analysis...")
            progress_bar.progress(60)
            
            clustering_results = behavioral_analyzer.perform_clustering(
                features_df, n_clusters_range=n_clusters_range
            )
            
            # Step 4: Identify archetypes
            status_text.text("Identifying behavioral archetypes...")
            progress_bar.progress(80)
            
            archetypes = behavioral_analyzer.identify_archetypes(
                features_df, clustering_results
            )
            
            # Step 5: Create early detection rules (DISABLED - not requested)
            # status_text.text("Creating early detection rules...")
            # progress_bar.progress(90)
            
            # early_detection_model = behavioral_analyzer.create_early_detection_rules(
            #     features_df, archetypes
            # )
            early_detection_model = None
            
            # Step 6: Save results
            if save_results:
                status_text.text("Saving results...")
                timestamp = behavioral_analyzer.save_results(
                    features_df, clustering_results, archetypes, output_dir
                )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results in session state
            st.session_state.archetype_results = {
                'features_df': features_df,
                'clustering_results': clustering_results,
                'archetypes': archetypes,
                'early_detection_model': early_detection_model,
                'timestamp': timestamp if save_results else None
            }
            
            st.success(f"🎉 Behavioral archetype analysis complete! Found {len(archetypes)} archetypes.")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")
            return
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    # Display results if available
    if 'archetype_results' in st.session_state:
        results = st.session_state.archetype_results
        
        # Create sub-tabs for results
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
            "📊 Archetype Overview",
            "🎯 Archetype Details", 
            "📈 Survival Analysis",
            "⚡ Early Detection",
            "🗺️ Archetype t-SNE"
        ])
        
        with subtab1:
            display_archetype_overview(results)
        
        with subtab2:
            display_archetype_details(results)
        
        with subtab3:
            display_survival_analysis(results)
        
        with subtab4:
            # display_early_detection(results)  # DISABLED - not requested
            st.info("Early detection analysis not implemented yet.")
        
        with subtab5:
            display_archetype_tsne(results)


def display_archetype_overview(results: Dict):
    """Display archetype analysis overview"""
    st.subheader("📊 Behavioral Archetype Overview")
    
    features_df = results['features_df']
    archetypes = results['archetypes']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tokens", len(features_df))
    
    with col2:
        dead_pct = features_df['is_dead'].mean() * 100
        st.metric("Dead Tokens", f"{dead_pct:.1f}%")
    
    with col3:
        st.metric("Archetypes Found", len(archetypes))
    
    with col4:
        avg_lifespan = features_df['lifespan_minutes'].mean()
        st.metric("Avg Lifespan", f"{avg_lifespan:.0f} min")
    
    # Archetype distribution
    st.subheader("🎭 Archetype Distribution")
    
    archetype_counts = features_df['cluster'].value_counts().sort_index()
    archetype_names = [archetypes[i]['name'] for i in archetype_counts.index]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=archetype_names,
            values=archetype_counts.values,
            hole=0.3,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Tokens: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    fig.update_layout(
        title="Token Distribution by Behavioral Archetype",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Archetype characteristics table
    st.subheader("📋 Archetype Characteristics")
    
    table_data = []
    for cluster_id, archetype in archetypes.items():
        stats = archetype['stats']
        table_data.append({
            'Archetype': archetype['name'],
            'Tokens': stats['n_tokens'],
            'Percentage': f"{stats['pct_of_total']:.1f}%",
            'Death Rate': f"{stats['pct_dead']:.1f}%",
            'Avg Lifespan': f"{stats['avg_lifespan']:.0f} min",
            'Avg Max Return (5min)': f"{stats['avg_max_return']*100:.1f}%"
        })
    
    # Convert to polars DataFrame for consistency
    table_df = pl.DataFrame(table_data)
    st.dataframe(table_df.to_pandas(), use_container_width=True)


def display_archetype_details(results: Dict):
    """Display detailed archetype analysis"""
    st.subheader("🎯 Detailed Archetype Analysis")
    
    archetypes = results['archetypes']
    features_df = results['features_df']
    
    # Select archetype for detailed view
    archetype_options = {f"{arch['name']} (Cluster {cluster_id})": cluster_id 
                        for cluster_id, arch in archetypes.items()}
    
    selected_archetype = st.selectbox(
        "Select archetype for detailed analysis:",
        options=list(archetype_options.keys())
    )
    
    if selected_archetype:
        cluster_id = archetype_options[selected_archetype]
        archetype = archetypes[cluster_id]
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        
        # Archetype summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tokens", archetype['stats']['n_tokens'])
        
        with col2:
            st.metric("Death Rate", f"{archetype['stats']['pct_dead']:.1f}%")
        
        with col3:
            st.metric("Avg Lifespan", f"{archetype['stats']['avg_lifespan']:.0f} min")
        
        # Representative examples
        st.subheader(f"📋 Representative Examples: {archetype['name']}")
        
        examples_data = []
        for token in archetype['examples'][:10]:
            token_data = cluster_data[cluster_data['token'] == token]
            if not token_data.empty:
                row = token_data.iloc[0]
                examples_data.append({
                    'Token': token,
                    'Category': row['category'],
                    'Is Dead': '💀' if row['is_dead'] else '✅',
                    'Lifespan': f"{row['lifespan_minutes']:.0f} min",
                    'Final Price Ratio': f"{row['final_price_ratio']*100:.1f}%" if not np.isnan(row['final_price_ratio']) else 'N/A'
                })
        
        if examples_data:
            # Convert to polars DataFrame for consistency
            examples_df = pl.DataFrame(examples_data)
            st.dataframe(examples_df.to_pandas(), use_container_width=True)
        
        # ACF signature
        if archetype['acf_signature']:
            st.subheader(f"📈 ACF Signature: {archetype['name']}")
            
            acf_data = archetype['acf_signature']
            lags = [int(col.split('_')[-1]) for col in acf_data.keys()]
            acf_values = list(acf_data.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=lags,
                y=acf_values,
                mode='lines+markers',
                name=archetype['name'],
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title=f"Average ACF Pattern: {archetype['name']}",
                xaxis_title="Lag (minutes)",
                yaxis_title="Autocorrelation",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def display_survival_analysis(results: Dict):
    """Display survival analysis for archetypes"""
    st.subheader("📈 Survival Analysis by Archetype")
    
    st.markdown("""
    Survival analysis shows how long tokens of each archetype typically survive before "dying".
    """)
    
    # Placeholder for survival curves
    # This would require lifelines library for proper Kaplan-Meier curves
    st.info("📊 Survival curves visualization would be implemented here using lifelines library")
    
    # For now, show lifespan distribution
    features_df = results['features_df']
    archetypes = results['archetypes']
    
    # Create lifespan distribution by archetype
    archetype_names = {i: arch['name'] for i, arch in archetypes.items()}
    features_df['archetype_name'] = features_df['cluster'].map(archetype_names)
    
    fig = px.box(
        features_df,
        x='archetype_name',
        y='lifespan_minutes',
        title="Lifespan Distribution by Archetype",
        labels={'archetype_name': 'Archetype', 'lifespan_minutes': 'Lifespan (minutes)'}
    )
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)


def display_early_detection(results: Dict):
    """Display early detection rules and performance"""
    st.subheader("⚡ Early Detection Rules (First 5 Minutes)")
    
    early_detection_model = results.get('early_detection_model')
    
    if early_detection_model is None:
        st.warning("Early detection model not available")
        return
    
    st.markdown("""
    These rules can classify tokens into archetypes using only the first 5 minutes of trading data.
    """)
    
    # Feature importance
    features_df = results['features_df']
    early_feature_cols = [col for col in features_df.columns if '5min' in col]
    
    if early_feature_cols:
        feature_importance = sorted(
            zip(early_feature_cols, early_detection_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        
        # Top features chart
        top_features = feature_importance[:10]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[feat[1] for feat in top_features],
                y=[feat[0] for feat in top_features],
                orientation='h',
                text=[f"{feat[1]:.3f}" for feat in top_features],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Top 10 Most Important Early Detection Features",
            xaxis_title="Feature Importance",
            yaxis_title="Feature",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    X = features_df[early_feature_cols].fillna(0).values
    y = features_df['cluster'].values
    accuracy = early_detection_model.score(X, y)
    
    st.metric("Early Detection Accuracy", f"{accuracy:.1%}")
    
    # Test early detection
    st.subheader("🧪 Test Early Detection")
    
    # Allow user to select a token to test
    token_options = features_df['token'].unique()[:50]  # Limit for performance
    selected_token = st.selectbox("Select token to test:", token_options)
    
    if selected_token and st.button("Test Early Detection"):
        token_data = features_df[features_df['token'] == selected_token].iloc[0]
        
        # Get early features
        early_features = [token_data[col] for col in early_feature_cols]
        prediction = early_detection_model.predict([early_features])[0]
        
        # Show results
        actual_archetype = results['archetypes'][token_data['cluster']]['name']
        predicted_archetype = results['archetypes'][prediction]['name']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Actual Archetype", actual_archetype)
        
        with col2:
            st.metric("Predicted Archetype", predicted_archetype)
        
        if actual_archetype == predicted_archetype:
            st.success("✅ Correct prediction!")
        else:
            st.error("❌ Incorrect prediction")


def display_archetype_tsne(results: Dict):
    """Display t-SNE visualization colored by archetypes"""
    st.subheader("🗺️ Archetype t-SNE Visualization")
    
    clustering_results = results['clustering_results']
    
    if 'X_tsne' not in clustering_results:
        st.warning("t-SNE data not available")
        return
    
    features_df = results['features_df']
    archetypes = results['archetypes']
    X_tsne = clustering_results['X_tsne']
    
    # Create archetype names
    archetype_names = {i: arch['name'] for i, arch in archetypes.items()}
    features_df['archetype_name'] = features_df['cluster'].map(archetype_names)
    
    # Create t-SNE plot
    fig = px.scatter(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        color=features_df['archetype_name'],
        hover_data={
            'Token': features_df['token'],
            'Is Dead': features_df['is_dead'],
            'Lifespan': features_df['lifespan_minutes']
        },
        title="Behavioral Archetypes in t-SNE Space",
        labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Archetype'}
    )
    
    fig.update_layout(
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Visualization Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Points", len(X_tsne))
    
    with col2:
        st.metric("Archetypes", len(archetypes))
    
    with col3:
        n_dead = features_df['is_dead'].sum()
        st.metric("Dead Tokens", f"{n_dead} ({n_dead/len(features_df)*100:.1f}%)")


def display_baseline_cluster_overview(results: Dict):
    """Display baseline clustering overview"""
    st.subheader("📊 Baseline Clustering Overview")
    
    # Overall statistics
    total_tokens = results['total_tokens_analyzed']
    n_categories = len(results['categories'])
    stability_summary = results.get('stability_summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tokens", total_tokens)
    
    with col2:
        st.metric("Categories", n_categories)
    
    with col3:
        avg_ari = stability_summary.get('average_ari', 0)
        st.metric("Average ARI", f"{avg_ari:.3f}")
    
    with col4:
        min_ari = stability_summary.get('min_ari', 0)
        st.metric("Min ARI", f"{min_ari:.3f}")
    
    # Category breakdown
    st.subheader("📈 Category Breakdown")
    
    category_data = []
    for category_name, category_results in results['categories'].items():
        category_data.append({
            'Category': category_name,
            'Tokens': category_results['n_tokens'],
            'Optimal K': category_results['optimal_k'],
            'Average ARI': category_results['stability_results']['average_ari'],
            'Min ARI': category_results['stability_results']['min_ari']
        })
    
    if category_data:
        import pandas as pd
        df = pd.DataFrame(category_data)
        st.dataframe(df)
    
    # Success criteria check
    st.subheader("✅ CEO Requirements Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Features Used**: 14 essential features ✅")
        st.write("**Method**: Elbow method for K selection ✅")
        st.write("**Stability**: ARI calculation ✅")
    
    with col2:
        success_criteria = avg_ari > 0.7
        status = "✅ PASSED" if success_criteria else "❌ FAILED"
        st.write(f"**Success Criteria (ARI > 0.7)**: {status}")
        
        if success_criteria:
            st.success("🎉 Perfect clustering stability achieved!")
        else:
            st.warning("⚠️ Clustering stability below threshold")


def display_baseline_cluster_details(results: Dict):
    """Display detailed cluster characteristics"""
    st.subheader("🏷️ Cluster Details by Category")
    
    # Category selection
    categories = list(results['categories'].keys())
    selected_category = st.selectbox("Select Category:", categories)
    
    if selected_category:
        category_results = results['categories'][selected_category]
        features_df = category_results['features']
        clustering_results = category_results['clustering_results']
        
        st.subheader(f"📊 {selected_category} Category Details")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tokens", len(features_df))
        
        with col2:
            st.metric("Clusters", category_results['optimal_k'])
        
        with col3:
            st.metric("Silhouette Score", f"{clustering_results['silhouette_score']:.3f}")
        
        # Cluster breakdown
        st.subheader("🎯 Cluster Breakdown")
        
        cluster_stats = []
        for cluster_id in range(category_results['optimal_k']):
            cluster_mask = clustering_results['labels'] == cluster_id
            cluster_features = features_df[cluster_mask]
            
            if len(cluster_features) > 0:
                cluster_stats.append({
                    'Cluster': cluster_id,
                    'Tokens': len(cluster_features),
                    'Percentage': f"{len(cluster_features)/len(features_df)*100:.1f}%",
                    'Dead Tokens': f"{cluster_features['is_dead'].sum()}/{len(cluster_features)}",
                    'Avg Lifespan': f"{cluster_features['lifespan_minutes'].mean():.1f}",
                    'Avg Return': f"{cluster_features['mean_return'].mean():.4f}",
                    'Avg Volatility': f"{cluster_features['volatility_5min'].mean():.4f}"
                })
        
        if cluster_stats:
            import pandas as pd
            df = pd.DataFrame(cluster_stats)
            st.dataframe(df)
        
        # Feature importance visualization
        st.subheader("📈 Feature Importance by Cluster")
        
        # Create heatmap of feature values by cluster
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        cluster_means = []
        
        for cluster_id in range(category_results['optimal_k']):
            cluster_mask = clustering_results['labels'] == cluster_id
            cluster_features = features_df[cluster_mask]
            
            if len(cluster_features) > 0:
                cluster_mean = cluster_features[feature_cols].mean()
                cluster_mean['cluster'] = cluster_id
                cluster_means.append(cluster_mean)
        
        if cluster_means:
            import pandas as pd
            import plotly.express as px
            
            heatmap_df = pd.DataFrame(cluster_means)
            heatmap_df = heatmap_df.set_index('cluster')
            
            fig = px.imshow(
                heatmap_df.T,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"Feature Values by Cluster - {selected_category}"
            )
            
            fig.update_layout(
                xaxis_title="Cluster",
                yaxis_title="Feature",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)


def display_baseline_stability_analysis(results: Dict):
    """Display stability analysis results"""
    st.subheader("📈 Clustering Stability Analysis")
    
    # Overall stability summary
    stability_summary = results.get('stability_summary', {})
    
    if stability_summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average ARI", f"{stability_summary['average_ari']:.3f}")
        
        with col2:
            st.metric("Min ARI", f"{stability_summary['min_ari']:.3f}")
        
        with col3:
            st.metric("Max ARI", f"{stability_summary['max_ari']:.3f}")
        
        with col4:
            st.metric("Total Runs", stability_summary['n_runs'])
    
    # Per-category stability
    st.subheader("🎯 Stability by Category")
    
    stability_data = []
    for category_name, category_results in results['categories'].items():
        stability = category_results['stability_results']
        stability_data.append({
            'Category': category_name,
            'Average ARI': stability['average_ari'],
            'Min ARI': stability['min_ari'],
            'Max ARI': stability['max_ari'],
            'Runs': stability['n_runs'],
            'Status': '✅ STABLE' if stability['average_ari'] > 0.7 else '⚠️ UNSTABLE'
        })
    
    if stability_data:
        import pandas as pd
        df = pd.DataFrame(stability_data)
        st.dataframe(df)
    
    # ARI distribution visualization
    st.subheader("📊 ARI Score Distribution")
    
    all_ari_scores = []
    category_labels = []
    
    for category_name, category_results in results['categories'].items():
        ari_scores = category_results['stability_results']['ari_scores']
        all_ari_scores.extend(ari_scores)
        category_labels.extend([category_name] * len(ari_scores))
    
    if all_ari_scores:
        import pandas as pd
        import plotly.express as px
        
        ari_df = pd.DataFrame({
            'ARI Score': all_ari_scores,
            'Category': category_labels
        })
        
        fig = px.box(
            ari_df,
            x='Category',
            y='ARI Score',
            title="ARI Score Distribution by Category"
        )
        
        # Add success threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                     annotation_text="Success Threshold (0.7)")
        
        st.plotly_chart(fig, use_container_width=True)


def display_baseline_elbow_method(results: Dict):
    """Display elbow method results"""
    st.subheader("🎯 Elbow Method Analysis")
    
    st.info("**Note**: This visualization shows the elbow method results used for K selection. The elbow point indicates the optimal number of clusters.")
    
    # Show elbow results for each category
    for category_name, category_results in results['categories'].items():
        st.subheader(f"📈 {category_name} Category")
        
        optimal_k = category_results['optimal_k']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Optimal K", optimal_k)
        
        with col2:
            st.metric("Silhouette Score", f"{category_results['clustering_results']['silhouette_score']:.3f}")
        
        # Note about elbow visualization
        st.info(f"✅ Elbow method selected K={optimal_k} for {category_name} category")
        
        # Show cluster distribution
        features_df = category_results['features']
        clustering_results = category_results['clustering_results']
        
        cluster_sizes = []
        for cluster_id in range(optimal_k):
            cluster_mask = clustering_results['labels'] == cluster_id
            cluster_sizes.append(len(features_df[cluster_mask]))
        
        if cluster_sizes:
            import pandas as pd
            import plotly.express as px
            
            cluster_df = pd.DataFrame({
                'Cluster': list(range(optimal_k)),
                'Size': cluster_sizes
            })
            
            fig = px.bar(
                cluster_df,
                x='Cluster',
                y='Size',
                title=f"Cluster Sizes - {category_name}"
            )
            
            st.plotly_chart(fig, use_container_width=True)


def display_baseline_export_results(results: Dict):
    """Display export results and options"""
    st.subheader("💾 Export Results")
    
    # Export summary
    st.write("**Export Summary:**")
    st.write(f"- Total tokens analyzed: {results['total_tokens_analyzed']}")
    st.write(f"- Categories: {len(results['categories'])}")
    st.write(f"- Average stability (ARI): {results.get('stability_summary', {}).get('average_ari', 0):.3f}")
    
    # Show export files
    st.subheader("📁 Generated Files")
    
    export_info = []
    for category_name, category_results in results['categories'].items():
        export_info.append({
            'Category': category_name,
            'CSV File': f"baseline_{category_name.lower()}_[timestamp].csv",
            'Features': "14 essential features + cluster assignments",
            'Tokens': category_results['n_tokens']
        })
    
    export_info.append({
        'Category': 'Summary',
        'CSV File': 'baseline_clustering_summary_[timestamp].json',
        'Features': 'Overall results and stability metrics',
        'Tokens': results['total_tokens_analyzed']
    })
    
    if export_info:
        import pandas as pd
        df = pd.DataFrame(export_info)
        st.dataframe(df)
    
    # Export options
    st.subheader("🔄 Re-export Options")
    
    if st.button("📥 Export Results Again"):
        from pathlib import Path
        output_dir = Path("time_series/results")
        export_baseline_results(results, output_dir)
        st.success("Results exported successfully!")
    
    # Feature summary
    st.subheader("📊 14 Essential Features")
    
    features_info = [
        {"Category": "Death Features", "Features": "is_dead, death_minute, lifespan_minutes", "Count": 3},
        {"Category": "Core Statistics", "Features": "mean_return, std_return, volatility_5min, max_drawdown", "Count": 4},
        {"Category": "ACF Features", "Features": "acf_lag_1, acf_lag_5, acf_lag_10", "Count": 3},
        {"Category": "Early Detection", "Features": "return_magnitude_5min, trend_direction_5min, price_change_ratio_5min, autocorrelation_5min", "Count": 4}
    ]
    
    import pandas as pd
    features_df = pd.DataFrame(features_info)
    st.dataframe(features_df)
    
    total_features = sum(f["Count"] for f in features_info)
    st.success(f"✅ Total: {total_features} features (CEO requirement: ≤15)")


if __name__ == "__main__":
    main() 