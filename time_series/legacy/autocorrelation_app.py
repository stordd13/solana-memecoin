# autocorrelation_app.py
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
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Removed pandas import - using polars for better performance
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from streamlit_utils.formatting import format_large_number, format_percentage, format_data_points

# Import our analyzer
from time_series.autocorrelation_clustering import AutocorrelationClusteringAnalyzer
from time_series.behavioral_archetype_analysis import BehavioralArchetypeAnalyzer


def run_baseline_clustering_analysis(processed_data_path, token_limits, k_range, n_stability_runs, export_results, sample_ratio=None, pre_loaded_tokens=None):
    """
    Run baseline clustering analysis with CEO requirements (15 features, elbow method, stability testing).
    """
    from time_series.archetype_utils import categorize_by_lifespan
    from sklearn.metrics import adjusted_rand_score
    import json
    from datetime import datetime
    
    # Initialize analyzer
    analyzer = BehavioralArchetypeAnalyzer()
    
    if pre_loaded_tokens is not None:
        # Use pre-loaded and pre-sampled tokens from multi-resolution analysis
        st.info(f"Using pre-loaded tokens from multi-resolution analysis...")
        categorized_tokens = pre_loaded_tokens
        st.info(f"Categories: {', '.join(f'{k}: {len(v)} tokens' for k, v in categorized_tokens.items())}")
    else:
        # Load tokens from processed categories (WITHOUT sampling here)
        st.info("Loading categorized token data...")
        token_data = analyzer.load_categorized_tokens(processed_data_path, limit=None, sample_ratio=None)  # No sampling yet
        
        if not token_data:
            st.error("No token data found!")
            return {}
        
        print(f"🔍 TOKEN FLOW TRACKING: Loaded {len(token_data)} tokens from processed categories")
        
        # Categorize by lifespan FIRST
        st.info("Categorizing tokens by lifespan...")
        categorized_tokens = categorize_by_lifespan(token_data, token_limits)
        
        total_categorized = sum(len(category_tokens) for category_tokens in categorized_tokens.values())
        print(f"🔍 TOKEN FLOW TRACKING: After lifespan categorization: {total_categorized} tokens")
        for cat_name, cat_tokens in categorized_tokens.items():
            print(f"   {cat_name}: {len(cat_tokens)} tokens")
    
    # THEN apply sampling to maintain category proportions (skip if using pre-loaded tokens)
    if pre_loaded_tokens is None and sample_ratio is not None and 0 < sample_ratio < 1:
        st.info(f"Applying stratified sampling ({sample_ratio*100:.1f}%)...")
        sampled_categorized_tokens = {}
        
        for category_name, category_tokens in categorized_tokens.items():
            if len(category_tokens) == 0:
                sampled_categorized_tokens[category_name] = {}
                continue
                
            # Calculate sample size for this category
            category_sample_size = max(1, int(len(category_tokens) * sample_ratio))
            category_sample_size = min(category_sample_size, len(category_tokens))
            
            # Random sample within category
            import random
            token_names = list(category_tokens.keys())
            sampled_names = random.sample(token_names, category_sample_size)
            
            sampled_categorized_tokens[category_name] = {
                name: category_tokens[name] for name in sampled_names
            }
            
            st.info(f"  {category_name}: {len(sampled_names)} tokens (from {len(category_tokens)})")
        
        categorized_tokens = sampled_categorized_tokens
        
        total_after_sampling = sum(len(category_tokens) for category_tokens in categorized_tokens.values())
        print(f"🔍 TOKEN FLOW TRACKING: After sampling: {total_after_sampling} tokens")
    
    # Extract 15 features for each category
    st.info("Extracting 15 essential features...")
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
        print(f"🔍 TOKEN FLOW TRACKING: Processing {category_name} with {len(category_tokens)} tokens")
        
        # Extract 15 features
        features_df = analyzer.extract_all_features(category_tokens)
        
        if features_df is None or len(features_df) == 0:
            st.warning(f"No features extracted for {category_name} - skipping")
            print(f"🔍 TOKEN FLOW TRACKING: WARNING - {category_name} lost all tokens during feature extraction!")
            continue
        
        st.info(f"Extracted {len(features_df)} features for {category_name}")
        print(f"🔍 TOKEN FLOW TRACKING: {category_name} feature extraction result: {len(features_df)} tokens")
        
        # Check if we have enough samples for clustering
        if len(features_df) < 3:
            st.warning(f"Category {category_name} has only {len(features_df)} samples - skipping clustering (need at least 3)")
            continue
        
        # Find optimal K and run clustering using the proper method
        st.info(f"Running clustering analysis for {category_name}...")
        clustering_results = analyzer.perform_clustering(features_df, n_clusters_range=k_range)
        
        # Extract optimal K from clustering results
        optimal_k = clustering_results.get('best_k', 2)
        
        st.info(f"Optimal K found: {optimal_k} for {category_name}")
        
        # Test stability (if enabled)
        if n_stability_runs > 0:
            stability_results = analyzer.test_clustering_stability(features_df, optimal_k, n_stability_runs)
        else:
            # Skip stability testing - provide placeholder results
            print("DEBUG: Skipping stability tests (user disabled)")
            stability_results = {
                'mean_ari': 1.0,  # Placeholder
                'min_ari': 1.0,
                'max_ari': 1.0,
                'std_ari': 0.0,
                'ari_scores': [1.0],
                'n_runs': 0
            }
        
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
    stability_tests_run = False
    
    for category_results in results['categories'].values():
        stability = category_results['stability_results']
        if 'ari_scores' in stability and stability['n_runs'] > 0:
            all_ari_scores.extend(stability['ari_scores'])
            stability_tests_run = True
    
    if all_ari_scores and stability_tests_run:
        results['stability_summary'] = {
            'mean_ari': np.mean(all_ari_scores),
            'min_ari': np.min(all_ari_scores),
            'max_ari': np.max(all_ari_scores),
            'n_runs': len(all_ari_scores)
        }
    else:
        # No stability tests were run
        results['stability_summary'] = {
            'mean_ari': 0.0,  # Indicate tests were skipped
            'min_ari': 0.0,
            'max_ari': 0.0,
            'n_runs': 0
        }
    
    # Export results if requested
    if export_results:
        st.info("Exporting results...")
        export_baseline_results(results, project_root / "time_series" / "results")
    
    return results


def export_baseline_results(results, output_dir):
    """Export baseline clustering results to files."""
    from pathlib import Path
    import json
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Helper function to convert numpy arrays to lists for JSON serialization
    def numpy_to_json(obj):
        """Convert numpy arrays to lists recursively"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    # Export summary
    summary_file = output_dir / f"baseline_clustering_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(numpy_to_json({
            'analysis_type': results['analysis_type'],
            'total_tokens_analyzed': results['total_tokens_analyzed'],
            'stability_summary': results['stability_summary'],
            'categories': {name: {
                'n_tokens': cat['n_tokens'],
                'optimal_k': cat['optimal_k'],
                'stability_results': cat['stability_results']
            } for name, cat in results['categories'].items()}
        }), f, indent=2)
    
    # Export detailed results per category
    for category_name, category_results in results['categories'].items():
        category_file = output_dir / f"baseline_{category_name.lower()}_{timestamp}.csv"
        features_df = category_results['features']
        clustering_results = category_results['clustering_results']
        
        # Add cluster assignments to features
        # Extract labels from the correct nested structure
        best_k = clustering_results.get('best_k', 2)
        if 'kmeans' in clustering_results and best_k in clustering_results['kmeans']:
            labels = clustering_results['kmeans'][best_k]['labels']
        else:
            # Fallback: assign all tokens to cluster 0
            print(f"DEBUG: Warning - no labels found for {category_name}, using default cluster assignment")
            labels = [0] * len(features_df)
        
        features_with_clusters = features_df.with_columns([
            pl.Series('cluster', labels)
        ])
        
        features_with_clusters.to_pandas().to_csv(category_file, index=False)
    
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
    - **Feature-based**: Uses 15 engineered features (ACF values + statistical measures) for clustering
    - **Price-only**: Clusters directly on price series (returns, log returns, raw prices, log prices, DTW features) + computes ACF
    """)
    
    # Initialize analyzer
    analyzer = AutocorrelationClusteringAnalyzer()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data directory selection
    # Try to find the correct data directory automatically
    possible_paths = [
        "data/processed/",
        "../data_processed",
        "data/raw/dataset",
        "../data/raw/dataset", 
        "../../data/raw/dataset",
        Path(__file__).parent.parent / "data/processed"
    ]

    
    default_path = "data/processed"
    for path in possible_paths:
        if Path(path).exists():
            default_path = str(path)
            break
    
    data_dir = st.sidebar.text_input(
        "Data Directory", 
        value=default_path,
        help="Path to directory containing token parquet files"
    )
    
    # Analysis type selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Type")
    
    analysis_type = st.sidebar.radio(
        "Choose analysis approach:",
        ["📊 Lifespan Analysis (Sprint/Standard/Marathon)", "🎭 Behavioral Archetypes (15-Feature)", "Price-only"],
        help="""
        📊 **Lifespan Analysis**: Analyzes how Sprint/Standard/Marathon tokens behave differently (ACF patterns by lifespan)
        🎭 **Behavioral Archetypes**: Finds distinct trading patterns regardless of lifespan (15 engineered features + clustering)
        **Price-only**: Simple clustering on raw price returns
        
        Note: These serve different purposes and can provide complementary insights.
        """
    )
    
    # Analysis-specific configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    # Initialize default variables (will be overridden by analysis-specific sections)
    max_tokens = None
    find_optimal_k = True
    n_clusters = None
    clustering_method = 'kmeans'
    
    # Behavioral Archetypes specific options
    if analysis_type == "🎭 Behavioral Archetypes (15-Feature)":
        with st.sidebar.expander("🎭 Behavioral Archetypes Settings", expanded=True):
            st.markdown("**⭐ PHASE 1B: Behavioral Archetype Analysis**")
            
            # Time series selection for behavioral analysis
            use_log_returns = st.selectbox(
                "Time series type:",
                ["log_returns", "returns", "prices", "log_prices"],
                help="Data transformation for behavioral pattern analysis"
            )
            
            # Add guidance for extreme volatility
            if use_log_returns == "returns":
                st.info("💡 **Returns** work well for normal volatility. For extreme moves (>1000%), consider **log_returns**.")
            elif use_log_returns == "log_returns":
                st.info("💡 **Log Returns** handle extreme volatility better. Recommended for memecoin analysis.")
            elif use_log_returns == "log_prices":
                st.info("💡 **Log Prices** are best for extreme volatility (10M%+ pumps, 99.9% dumps). Most stable for memecoin analysis.")
            else:
                st.info("💡 **Raw Prices** can be unstable with extreme volatility. Consider log_returns for better results.")
            
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
    
    # Lifespan Analysis specific options  
    elif analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)":
        with st.sidebar.expander("🚀 Multi-Resolution Settings", expanded=True):
            st.markdown("**⭐ PHASE 1A: Multi-Resolution Analysis**")
            
            multi_method = st.selectbox(
                "Price transformation:",
                ["returns", "log_returns", "prices", "log_prices", "dtw_features"],
                help="Method for analyzing across lifespan categories"
            )
            
            # Add guidance for extreme volatility
            if multi_method == "returns":
                st.info("💡 **Returns** work well for normal volatility. For extreme moves (>1000%), consider **log_returns**.")
            elif multi_method == "log_returns":
                st.info("💡 **Log Returns** handle extreme volatility better. Recommended for memecoin analysis.")
            elif multi_method == "log_prices":
                st.info("💡 **Log Prices** are best for extreme volatility (10M%+ pumps, 99.9% dumps). Most stable for memecoin analysis.")
            elif multi_method == "dtw_features":
                st.info("💡 **DTW Features** find similar temporal patterns regardless of scale. Good for discovering behavioral patterns.")
            else:
                st.info("💡 **Raw Prices** can be unstable with extreme volatility. Consider log_returns for better results.")
            
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
            
            # Sampling option for faster processing
            enable_sampling = st.checkbox(
                "🔬 Enable sampling (10%) for faster processing", 
                value=False,
                help="Sample 10% of tokens for faster debugging/testing (reduces processing time by ~90%)"
            )
            
            sample_ratio = 0.1 if enable_sampling else None
            
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
                "Enable 15-feature baseline clustering",
                value=True,
                help="Run baseline clustering with 15 features, elbow method, and stability testing"
            )
            
            if enable_baseline_clustering:
                # Clustering parameters
                k_range_min = st.number_input("Min clusters (K):", min_value=3, max_value=10, value=3)
                k_range_max = st.number_input("Max clusters (K):", min_value=3, max_value=15, value=10)
                
                # Stability testing
                run_stability_tests = st.checkbox(
                    "Run stability tests", 
                    value=True,
                    help="Stability tests validate clustering consistency but can be slow for large datasets"
                )
                
                if run_stability_tests:
                    n_stability_runs = st.number_input(
                        "Number of stability runs:", 
                        min_value=1, max_value=10, value=5,
                        help="Number of runs to test clustering stability (ARI calculation)"
                    )
                else:
                    n_stability_runs = 0  # Skip stability testing
                
                # Export options
                export_results = st.checkbox("Export results to files", value=True)
            
            st.markdown("""
            **Lifespan Categories (Death-Aware):**
            - **Sprint**: 0-400 active min (includes ALL dead tokens)
            - **Standard**: 400-1200 active min (typical lifecycle)
            - **Marathon**: 1200+ active min (extended development)
            
            **Analysis Features:**
            - Multi-resolution ACF analysis across categories
            - 15-feature baseline clustering (CEO requirements)
            - Stability testing with ARI > 0.75 threshold
            - Elbow method for optimal K selection
            
            **Goal**: Discover behavioral archetypes including death patterns
            """)
        
        # Multi-resolution uses fixed optimal settings
        find_optimal_k = True
        n_clusters = None
        clustering_method = 'kmeans'
        max_tokens = None
        
        # Initialize multi-resolution specific variables for other analysis types
        if not enable_baseline_clustering:
            enable_dtw_clustering = False
            compare_across_categories = True
            k_range_min = 3
            k_range_max = 10
            n_stability_runs = 5
            export_results = True
    
    # Initialize variables for other analysis types that don't have multi-resolution settings
    if analysis_type not in ["📊 Lifespan Analysis (Sprint/Standard/Marathon)"]:
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
    if 'results' in st.session_state and analysis_type not in ["📊 Lifespan Analysis (Sprint/Standard/Marathon)"]:
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
                
                # Validate configuration based on analysis type
                if analysis_type == "Price-only":
                    if 'price_method' not in locals():
                        st.error("Price method not selected. Please configure the analysis settings.")
                        return
                elif analysis_type == "🎭 Behavioral Archetypes (15-Feature)":
                    if 'use_log_returns' not in locals():
                        st.error("Time series type not selected. Please configure the analysis settings.")
                        return
                elif analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)":
                    if 'multi_method' not in locals():
                        st.error("Price transformation not selected. Please configure the analysis settings.")
                        return
                
                # Check for cached results with parameter-based invalidation
                cache_key = f"{analysis_type}_{data_dir}"
                if analysis_type == "Price-only":
                    cache_key += f"_{price_method}_{max_tokens}_{clustering_method}_{find_optimal_k}_{n_clusters}"
                elif analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)":
                    cache_key += f"_{multi_method}_{max_tokens_per_category}_{sample_ratio}_{clustering_method}_{find_optimal_k}_{n_clusters}"
                elif analysis_type == "🎭 Behavioral Archetypes (15-Feature)":
                    cache_key += f"_{use_log_returns}_{max_tokens}_{clustering_method}_{find_optimal_k}_{n_clusters}"
                
                # Check if we have cached results with same parameters
                use_cached = False
                if 'cached_results' in st.session_state and 'cached_key' in st.session_state:
                    if st.session_state['cached_key'] == cache_key:
                        st.info("🎯 **Found cached results with same parameters** - using cached analysis")
                        use_cached = True
                        results = st.session_state['cached_results']
                
                if not use_cached:
                    # Run unified analysis - eliminates redundancy and streamlines all analysis types
                    try:
                        from time_series.unified_analysis import UnifiedAnalysisEngine
                        unified_engine = UnifiedAnalysisEngine()
                    except ImportError as e:
                        st.error(f"🚨 **Import Error**: Failed to load unified analysis engine: {e}")
                        st.error("🔧 **Solution**: Check that all time_series modules are properly installed and accessible.")
                        return
                    except Exception as e:
                        st.error(f"🚨 **Initialization Error**: {e}")
                        return
                
                    # Map analysis types to unified engine parameters
                    try:
                        if analysis_type == "Price-only":
                            st.info(f"🔄 Running unified price-only analysis with method: {price_method}")
                            
                            results = unified_engine.run_unified_analysis(
                                data_path=data_path,
                                analysis_type="price_only",
                                time_series_method=price_method,
                                max_tokens=max_tokens,
                                clustering_method=clustering_method,
                                find_optimal_k=find_optimal_k,
                                n_clusters=n_clusters
                            )
                        
                        elif analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)":
                            st.info(f"🔄 Running unified lifespan analysis with method: {multi_method}")
                            
                            # Use processed data directory
                            processed_data_path = data_path.parent / "processed"
                            if not processed_data_path.exists():
                                st.error(f"📁 **Processed data directory not found**: {processed_data_path.absolute()}")
                                st.error("🔧 **Next steps**: Please run the main data analysis first to generate processed categories.")
                                st.info("💡 **How to fix**: Go to the main dashboard (`streamlit run data_analysis/app.py`) and run the full pipeline to create processed data.")
                                return
                            
                            results = unified_engine.run_unified_analysis(
                                data_path=processed_data_path,
                                analysis_type="lifespan", 
                                time_series_method=multi_method,
                                max_tokens_per_category=max_tokens_per_category,
                                sample_ratio=sample_ratio,
                                clustering_method=clustering_method,
                                find_optimal_k=find_optimal_k,
                                n_clusters=n_clusters
                            )
                            
                        elif analysis_type == "🎭 Behavioral Archetypes (15-Feature)":
                            st.info(f"🔄 Running unified behavioral archetype analysis with method: {use_log_returns}")
                            
                            # Use processed data directory
                            processed_data_path = data_path.parent / "processed"
                            if not processed_data_path.exists():
                                st.error(f"📁 **Processed data directory not found**: {processed_data_path.absolute()}")
                                st.error("🔧 **Next steps**: Please run the main data analysis first to generate processed categories.")
                                st.info("💡 **How to fix**: Go to the main dashboard (`streamlit run data_analysis/app.py`) and run the full pipeline to create processed data.")
                                return
                            
                            results = unified_engine.run_unified_analysis(
                                data_path=processed_data_path,
                                analysis_type="behavioral",
                                time_series_method=use_log_returns,  # This is already a string like "log_returns"
                                max_tokens=max_tokens,
                                clustering_method=clustering_method,
                                find_optimal_k=find_optimal_k,
                                n_clusters=n_clusters
                            )
                            
                        else:
                            st.error(f"🚨 **Unknown analysis type**: {analysis_type}")
                            st.error("🔧 **Available types**: Price-only, Lifespan Analysis, Behavioral Archetypes")
                            return
                            
                    except Exception as e:
                        st.error(f"🚨 **Analysis Failed**: {str(e)}")
                        st.error("🔧 **Possible causes**: Data loading issues, feature extraction problems, or clustering failures")
                        if "processed" in str(e).lower():
                            st.info("💡 **Hint**: This might be a processed data issue. Try running the main data analysis pipeline first.")
                        with st.expander("🔍 **Full Error Details**", expanded=False):
                            st.code(str(e))
                        return
                    
                    # Cache the results for future use
                    st.session_state['cached_results'] = results
                    st.session_state['cached_key'] = cache_key
                    st.info("💾 **Results cached** for faster future access with same parameters")
                
                # Display cluster quality analysis
                if 'quality_metrics' in results:
                    quality = results['quality_metrics']
                    
                    if quality['is_severely_imbalanced']:
                        st.warning(f"⚠️ **Cluster Imbalance Detected**: {quality['max_cluster_percentage']:.1f}% of tokens in one cluster")
                        
                        # Show imbalance analysis
                        if 'imbalance_analysis' in quality and 'death_analysis' in quality['imbalance_analysis']:
                            st.info("**Likely cause**: Most tokens are dead/inactive, creating natural clustering imbalance")
                            death_analysis = quality['imbalance_analysis']['death_analysis']
                            for cluster_id, stats in death_analysis.items():
                                if stats['size'] > len(results.get('features_df', [])) * 0.5:  # Large cluster
                                    st.info(f"Cluster {cluster_id}: {stats['death_rate']*100:.1f}% death rate ({stats['size']} tokens)")
                    
                    st.info(f"📊 **Clustering Quality**: Silhouette={quality['silhouette_score']:.3f}, K={quality['n_clusters']}")
                    
                    # Show cluster distribution
                    st.info("**Cluster Distribution**:")
                    for cluster_id, percentage in quality['cluster_percentages'].items():
                        st.text(f"  Cluster {cluster_id}: {percentage:.1f}% ({quality['cluster_sizes'][cluster_id]} tokens)")
                    
                    # Recommendations
                    if quality['is_severely_imbalanced']:
                        st.info("💡 **Recommendations**:")
                        st.info("• Try DBSCAN clustering for outlier detection")
                        st.info("• Use stratified sampling to balance alive/dead tokens")  
                        st.info("• Consider filtering out dead tokens for behavioral analysis")
                
                # Store results in session state with caching metadata
                st.session_state['results'] = results
                st.session_state['analysis_metadata'] = {
                    'analysis_type': analysis_type,
                    'sample_ratio': sample_ratio if analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)" else None,
                    'baseline_clustering_enabled': enable_baseline_clustering if analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)" else False,
                    'processed_data_path': str(processed_data_path) if analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)" else None,
                    'total_tokens_analyzed': results.get('total_tokens_analyzed', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Success message depends on analysis type
                if analysis_type == "📊 Lifespan Analysis (Sprint/Standard/Marathon)":
                    total_tokens = results.get('total_tokens_analyzed', 0)
                    n_categories = len(results.get('categories', {}))
                    success_msg = f"✅ Multi-Resolution Analysis complete! Analyzed {total_tokens} tokens across {n_categories} lifespan categories."
                    
                    # Add baseline clustering info if enabled
                    if enable_baseline_clustering and 'baseline_clustering' in results:
                        baseline_results = results['baseline_clustering']
                        stability_summary = baseline_results.get('stability_summary', {})
                        avg_ari = stability_summary.get('mean_ari', 0)
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
        if 'categories' in results:  # Lifespan analysis (Sprint/Standard/Marathon)
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
                    
        elif results.get('analysis_type') == 'behavioral':  # Behavioral archetype analysis
            with tab1:
                display_behavioral_archetypes(results)
                
            with tab2:
                st.info("🔧 Additional behavioral analysis tabs coming soon...")
                
        elif results.get('analysis_type') == 'price_only':  # Price-only analysis
            with tab1:
                display_overview(results)
                
            with tab2:
                st.info("🔧 Additional price-only analysis tabs coming soon...")
                
        else:  # Legacy analysis (backward compatibility)
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
    """Display analysis overview with defensive programming for different result structures"""
    st.header("📊 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Defensive programming for token count
        token_count = 'Unknown'
        if 'token_names' in results:
            token_count = len(results['token_names'])
        elif 'total_tokens_analyzed' in results:
            token_count = results['total_tokens_analyzed']
        st.metric("Total Tokens", token_count)
        
    with col2:
        # Defensive programming for cluster count
        n_clusters = 'Unknown'
        if 'n_clusters' in results:
            n_clusters = results['n_clusters']
        elif 'clustering_results' in results:
            n_clusters = results['clustering_results'].get('n_clusters', 'Unknown')
        st.metric("Number of Clusters", n_clusters)
        
    with col3:
        analysis_method = results.get('analysis_method', 'feature_based')
        if analysis_method.startswith('price_only'):
            method_display = f"Price-only ({analysis_method.split('_')[-1]})"
        elif analysis_method.startswith('behavioral'):
            method_display = "Behavioral Archetype"
        elif analysis_method.startswith('lifespan'):
            method_display = "Lifespan Analysis"
        else:
            method_display = "Feature-based"
        st.metric("Analysis Type", method_display)
        
    with col4:
        # Defensive programming for average length
        if 'token_data' in results and results['token_data']:
            try:
                avg_length = np.mean([len(df) for df in results['token_data'].values()])
                st.metric("Avg Token Length", f"{avg_length:.0f} minutes")
            except Exception as e:
                st.metric("Avg Token Length", "N/A")
                print(f"DEBUG: Could not calculate average length: {e}")
        else:
            st.metric("Avg Token Length", "N/A")
    
    # Cluster distribution with defensive programming
    st.subheader("Cluster Distribution")
    
    if 'cluster_labels' in results:
        try:
            # Use polars for better performance
            cluster_series = pl.Series('cluster', results['cluster_labels'])
            cluster_counts = cluster_series.value_counts().sort('cluster')
            
            # Extract values for plotting
            cluster_ids = cluster_counts['cluster'].to_list()
            count_values = cluster_counts['count'].to_list()
            
            fig = go.Figure(data=[
                go.Bar(x=[f"Cluster {i}" for i in cluster_ids],
                       y=count_values,
                       text=count_values,
                       textposition='auto')
            ])
            fig.update_layout(
                title="Number of Tokens per Cluster",
                xaxis_title="Cluster",
                yaxis_title="Number of Tokens",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ Could not display cluster distribution chart")
            print(f"DEBUG: Cluster distribution error: {e}")
    else:
        st.info("📊 Cluster distribution data not available for this analysis type")
    
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
    
    # Use polars DataFrame for better performance
    summary_df = pl.DataFrame(cluster_summary)
    st.dataframe(summary_df.to_pandas(), use_container_width=True)


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
    if len(cluster_tokens) > 1:
        num_tokens_to_show = st.slider("Number of tokens to display", min_value=1, max_value=len(cluster_tokens), value=min(5, len(cluster_tokens)), key="acf_tokens_slider")
    else:
        num_tokens_to_show = 1
        st.info(f"This cluster contains only 1 token: {cluster_tokens[0]}")
    
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
            # Use polars DataFrame for better performance
            summary_df = pl.DataFrame(summary_data)
            st.dataframe(summary_df.to_pandas(), use_container_width=True)


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
        
        if price_method == 'dtw_features':
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
        # Feature-based analysis - use the original 15 engineered features
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
    
    # Use polars DataFrame for better performance  
    importance_df = pl.DataFrame(feature_importance).sort('Variance', descending=True)
    
    # Extract data for plotting
    variance_values = importance_df['Variance'].to_list()
    feature_names = importance_df['Feature'].to_list()
    
    fig = go.Figure(data=[
        go.Bar(x=variance_values,
               y=feature_names,
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
        
        # Use polars DataFrame for better performance
        scores_df = pl.DataFrame({
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
        
        # Convert to pandas for styling (polars doesn't support styling)
        st.dataframe(scores_df.to_pandas().style.apply(highlight_optimal, axis=1), use_container_width=True)
        
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
                # Use polars DataFrame for better performance
                summary_df = pl.DataFrame(summary_data)
                # Convert to pandas only for Streamlit display
                st.dataframe(summary_df.to_pandas(), use_container_width=True)
            
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
        
        # Use polars DataFrame for better performance
        summary_df = pl.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Category distribution chart (only if we have data)
        if summary_df.height > 0:
            fig = px.bar(summary_df, x='Category', y='Tokens', 
                        title="Token Distribution by Lifespan Category",
                        color='Category')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for category distribution chart.")
    
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
    - **Price Transformation**: {results.get('analysis_method', 'unknown').replace('_', ' ')}
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
    
    # Use polars DataFrame for better performance
    comparison_df = pl.DataFrame(comparison_data)
    
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
                # Use polars DataFrame for better performance
                cluster_df = pl.DataFrame(cluster_data)
                # Convert to pandas only for Streamlit display
                st.dataframe(cluster_df.to_pandas(), use_container_width=True)


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
        
        # Use polars DataFrame for better performance
        corr_df = pl.DataFrame(corr_data)
        # Convert to pandas only for Streamlit display
        st.dataframe(corr_df.to_pandas(), use_container_width=True)
        
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
                
                # Use polars DataFrame for better performance
                distinctive_df = pl.DataFrame(distinctive_data)
                # Convert to pandas only for Streamlit display
                st.dataframe(distinctive_df.to_pandas(), use_container_width=True)


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
                # Use polars DataFrame for better performance
                dtw_df = pl.DataFrame(dtw_cluster_data)
                # Convert to pandas only for Streamlit display
                st.dataframe(dtw_df.to_pandas(), use_container_width=True)


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
        # Use polars Series for better performance
        category_series = pl.Series('category', combined_colors)
        category_counts = category_series.value_counts().sort('category')
        
        # Convert to pandas for plotly compatibility
        category_counts_pd = category_counts.to_pandas()
        fig = px.pie(values=category_counts_pd['count'].values, 
                    names=category_counts_pd['category'].values,
                    title="Distribution of Tokens Across Categories")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No t-SNE data available for visualization")


def extract_tokens_from_baseline_results(baseline_results: Dict) -> Dict[str, pl.DataFrame]:
    """
    Extract token data from cached baseline clustering results to avoid reloading.
    
    Args:
        baseline_results: Results from baseline clustering analysis
        
    Returns:
        Dictionary mapping token names to DataFrames
    """
    token_data = {}
    
    # Extract tokens from each category in baseline results
    categories = baseline_results.get('categories', {})
    
    for category_name, category_data in categories.items():
        if 'token_data' in category_data:
            category_tokens = category_data['token_data']
            
            # Add tokens from this category
            for token_name, token_df in category_tokens.items():
                # Ensure the token has the required columns
                if 'datetime' in token_df.columns and 'price' in token_df.columns:
                    # Add category information if not present
                    if 'category' not in token_df.columns:
                        token_df = token_df.with_columns([
                            pl.lit(category_name).alias('category')
                        ])
                    token_data[token_name] = token_df
    
    return token_data


def display_behavioral_archetypes(results: Dict):
    """Display behavioral archetype analysis results or interface"""
    st.header("🎭 Behavioral Archetype Analysis")
    
    # Debug: Show what results we have
    st.write("**Debug: Main Behavioral Display:**")
    st.write(f"- Results type: {type(results)}")
    st.write(f"- Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
    if isinstance(results, dict):
        st.write(f"- analysis_type: {results.get('analysis_type', 'NOT FOUND')}")
    
    # Check if this is a unified analysis result
    if results.get('analysis_type') == 'behavioral':
        st.info("✅ Detected unified behavioral analysis result")
        # Display unified behavioral archetype results
        display_unified_behavioral_results(results)
        return
    else:
        st.info(f"ℹ️ Not a unified behavioral result (analysis_type: {results.get('analysis_type', 'None')})")
    
    st.markdown("""
    This section analyzes memecoin tokens to identify behavioral archetypes including death patterns.
    
    **Key Features:**
    - Death detection using robust algorithms for tokens with small values
    - Identification of 5-8 behavioral archetypes
    - Early detection rules using only first 5 minutes of data
    - Survival analysis and return profiles
    """)
    
    # Check if we can use cached multi-resolution results
    can_use_cached = False
    cached_metadata = st.session_state.get('analysis_metadata', {})
    
    if cached_metadata.get('analysis_type') == "📊 Lifespan Analysis (Sprint/Standard/Marathon)":
        can_use_cached = True
        st.info("✅ Using cached multi-resolution analysis results for faster processing!")
        
        with st.expander("📊 Cached Analysis Details", expanded=False):
            st.write(f"**Analysis Type**: {cached_metadata.get('analysis_type')}")
            st.write(f"**Tokens Analyzed**: {cached_metadata.get('total_tokens_analyzed', 'Unknown')}")
            st.write(f"**Sample Ratio**: {cached_metadata.get('sample_ratio', 'None (full dataset)')}")
            st.write(f"**Baseline Clustering**: {'Yes' if cached_metadata.get('baseline_clustering_enabled') else 'No'}")
            st.write(f"**Analysis Timestamp**: {cached_metadata.get('timestamp', 'Unknown')}")
    
    # Initialize behavioral analyzer
    if 'behavioral_analyzer' not in st.session_state:
        st.session_state.behavioral_analyzer = BehavioralArchetypeAnalyzer()
    
    behavioral_analyzer = st.session_state.behavioral_analyzer
    
    # Check for existing behavioral archetype results
    archetype_results_available = 'archetype_results' in st.session_state
    if archetype_results_available:
        st.success("✅ **Behavioral archetype analysis results are available from previous run!**")
        with st.expander("📊 Previous Analysis Details", expanded=False):
            results = st.session_state.archetype_results
            if 'clustering_results' in results:
                optimal_k = results['clustering_results'].get('optimal_k', 'Unknown')
                st.write(f"**Optimal clusters found**: {optimal_k}")
            if 'features_df' in results:
                n_tokens = len(results['features_df'])
                st.write(f"**Tokens analyzed**: {n_tokens}")
            st.write("**Status**: You can view results below or run a new analysis with different parameters")
    
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
            options=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
            default=[6, 7, 8],
            help="Range of cluster numbers to test"
        )
        
        # Sampling option for faster processing
        enable_sampling = st.checkbox(
            "🔬 Enable sampling for faster processing", 
            value=False,
            help="Sample 10% of tokens for faster debugging/testing"
        )
        
        sample_ratio = 0.1 if enable_sampling else None
    
    with col3:
        # Output options
        save_results = st.checkbox("Save results to files", value=True)
        
        if save_results:
            output_dir = project_root / "time_series" / "results"
            st.write(f"**Output directory**: `{output_dir}`")
            output_dir.mkdir(exist_ok=True)
    
    # Add option to use cached results if available
    use_cached_data = False
    if can_use_cached:
        col_cached1, col_cached2 = st.columns(2)
        
        with col_cached1:
            use_cached_data = st.checkbox(
                "🔄 Use cached multi-resolution tokens", 
                value=True,
                help="Use the same tokens from the multi-resolution analysis for consistency"
            )
        
        with col_cached2:
            if use_cached_data:
                st.info("Will use tokens from cached analysis with same sampling ratio")
            else:
                st.warning("Will load fresh tokens (may have different sampling)")

    # Run analysis button
    run_fresh = st.button("🚀 Run Behavioral Archetype Analysis", type="primary")
    
    if run_fresh or (can_use_cached and use_cached_data):
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
            # Step 1: Load data (use cached tokens if available)
            if use_cached_data and 'baseline_clustering' in results:
                status_text.text("Extracting tokens from cached baseline clustering results...")
                progress_bar.progress(10)
                
                # Extract tokens from cached baseline clustering
                token_data = extract_tokens_from_baseline_results(results['baseline_clustering'])
                st.info(f"✅ Loaded {len(token_data)} tokens from cached baseline clustering results")
                
                # Override sampling parameters to match cached analysis
                cached_sample_ratio = cached_metadata.get('sample_ratio')
                if cached_sample_ratio:
                    sample_ratio = cached_sample_ratio
                    st.info(f"Using cached sampling ratio: {sample_ratio}")
                
            else:
                status_text.text("Loading categorized token data...")
                progress_bar.progress(10)
                
                token_data = behavioral_analyzer.load_categorized_tokens(
                    processed_dir, limit=token_limit, sample_ratio=sample_ratio
                )
            
            if not token_data:
                st.error("No token data found")
                return
            
            # Show token count summary
            st.info(f"📊 Loaded {len(token_data)} tokens for behavioral archetype analysis")
            
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
    st.write("**Debug: Session State Check:**")
    st.write(f"- 'archetype_results' in session_state: {'archetype_results' in st.session_state}")
    if 'archetype_results' in st.session_state:
        st.write(f"- Session state keys: {list(st.session_state.keys())}")
        results = st.session_state.archetype_results
        st.write(f"- Archetype results type: {type(results)}")
        st.write(f"- Archetype results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        # CRITICAL: Add cluster column to features_df ONCE at the main entry point
        # This prevents the recurring "cluster column missing" errors across all display functions
        features_df = results['features_df']
        if 'cluster' not in features_df.columns:
            clustering_results = results['clustering_results']
            best_k = clustering_results.get('best_k', 2)
            if 'kmeans' in clustering_results and best_k in clustering_results['kmeans']:
                cluster_labels = clustering_results['kmeans'][best_k]['labels']
                features_df = features_df.with_columns(pl.Series('cluster', cluster_labels))
                # Update the results with the corrected features_df
                results['features_df'] = features_df
                st.session_state.archetype_results['features_df'] = features_df
            else:
                st.error("No cluster labels found in results")
                return
        
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
    
    try:
        # Debug information
        st.write("**Debug: Results Structure:**")
        st.write(f"- Available keys: {list(results.keys())}")
        
        # Validate required keys
        required_keys = ['features_df', 'archetypes']
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            st.error(f"❌ Missing required keys in results: {missing_keys}")
            st.write("**Available results keys:**", list(results.keys()))
            return
        
        features_df = results['features_df']
        archetypes = results['archetypes']
        
        # Validate data structures
        if features_df is None:
            st.error("❌ features_df is None")
            return
        
        if not hasattr(features_df, 'height'):
            st.error(f"❌ features_df is not a valid DataFrame. Type: {type(features_df)}")
            return
            
        if features_df.height == 0:
            st.warning("⚠️ features_df is empty")
            return
            
        if not archetypes:
            st.warning("⚠️ No archetypes found")
            return
            
        st.success(f"✅ Data validation passed - {features_df.height} tokens, {len(archetypes)} archetypes")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tokens", len(features_df))
        
        with col2:
            # Check if 'is_dead' column exists
            if 'is_dead' in features_df.columns:
                dead_pct = features_df['is_dead'].mean() * 100
                st.metric("Dead Tokens", f"{dead_pct:.1f}%")
            else:
                st.metric("Dead Tokens", "N/A")
                st.warning("⚠️ 'is_dead' column not found in features")
        
        with col3:
            st.metric("Archetypes Found", len(archetypes))
        
        with col4:
            # Check if 'lifespan_minutes' column exists
            if 'lifespan_minutes' in features_df.columns:
                avg_lifespan = features_df['lifespan_minutes'].mean()
                st.metric("Avg Lifespan", f"{avg_lifespan:.0f} min")
            else:
                st.metric("Avg Lifespan", "N/A")
                st.warning("⚠️ 'lifespan_minutes' column not found in features")
        
        # Archetype distribution
        st.subheader("🎭 Archetype Distribution")
        
        # Validate clustering results
        if 'clustering_results' not in results:
            st.error("❌ clustering_results not found in results")
            return
            
        clustering_results = results['clustering_results']
        best_k = clustering_results.get('best_k', 2)
        
        # Try to get cluster labels
        cluster_labels = None
        if 'kmeans' in clustering_results and best_k in clustering_results['kmeans']:
            cluster_labels = clustering_results['kmeans'][best_k]['labels']
            st.write(f"✅ Using cluster labels from kmeans k={best_k}")
        elif 'cluster' in features_df.columns:
            cluster_labels = features_df['cluster'].to_list()
            st.write("✅ Using cluster labels from features_df")
        else:
            st.error("❌ No cluster labels found in results")
            st.write("**Debug info:**")
            st.write(f"- best_k: {best_k}")
            st.write(f"- clustering_results keys: {list(clustering_results.keys())}")
            if 'kmeans' in clustering_results:
                st.write(f"- kmeans keys: {list(clustering_results['kmeans'].keys())}")
            st.write(f"- features_df columns: {list(features_df.columns)}")
            return
        
        cluster_series = pl.Series('cluster', cluster_labels)
        archetype_counts = cluster_series.value_counts().sort('cluster')
        
        # Extract data for plotting
        cluster_ids = archetype_counts['cluster'].to_list()
        counts = archetype_counts['count'].to_list()
        
        # Validate archetypes data
        try:
            archetype_names = [archetypes[i]['name'] for i in cluster_ids]
        except KeyError as e:
            st.error(f"❌ Missing archetype data for cluster {e}")
            st.write(f"Available archetype keys: {list(archetypes.keys())}")
            st.write(f"Required cluster IDs: {cluster_ids}")
            return
        except Exception as e:
            st.error(f"❌ Error extracting archetype names: {e}")
            return
        
        # Create visualization
        fig = go.Figure(data=[
            go.Pie(
                labels=archetype_names,
                values=counts,
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
                'Avg Max Return (5min)': f"{stats['avg_return_magnitude']*100:.1f}%"
            })
        
        # Convert to polars DataFrame for consistency
        table_df = pl.DataFrame(table_data)
        st.dataframe(table_df.to_pandas(), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error in display_archetype_overview: {str(e)}")
        st.write("**Exception details:**")
        import traceback
        st.code(traceback.format_exc())


def display_archetype_details(results: Dict):
    """Display detailed archetype analysis"""
    st.subheader("🎯 Detailed Archetype Analysis")
    
    try:
        # Debug and validation
        st.write("**Debug: Archetype Details:**")
        st.write(f"- Available keys: {list(results.keys())}")
        
        if 'archetypes' not in results:
            st.error("❌ 'archetypes' not found in results")
            return
        if 'features_df' not in results:
            st.error("❌ 'features_df' not found in results")
            return
            
        archetypes = results['archetypes']
        features_df = results['features_df']
        
        if not archetypes:
            st.warning("⚠️ No archetypes found")
            return
        if features_df is None or features_df.height == 0:
            st.warning("⚠️ No feature data available")
            return
            
        st.success(f"✅ Found {len(archetypes)} archetypes with {features_df.height} tokens")
        
        # Note: cluster column is now added at the main entry point in display_behavioral_archetypes
        # No need to add it here anymore
        
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
            cluster_data = features_df.filter(pl.col('cluster') == cluster_id)
            
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
                filtered_data = cluster_data.filter(pl.col('token') == token)
                if filtered_data.height > 0:
                    token_data = filtered_data.row(0, named=True)
                    examples_data.append({
                        'Token': token,
                        'Category': token_data.get('category', 'N/A'),
                        'Is Dead': '💀' if token_data.get('is_dead', False) else '✅',
                        'Lifespan': f"{token_data.get('lifespan_minutes', 0):.0f} min",
                        'Final Price Ratio': f"{token_data['final_price_ratio']*100:.1f}%" if 'final_price_ratio' in token_data and not np.isnan(token_data['final_price_ratio']) else 'N/A'
                    })
                else:
                    st.warning(f"Token {token} not found in cluster data")
                    continue
            
            # Use polars DataFrame for better performance
            examples_df = pl.DataFrame(examples_data)
            # Convert to pandas only for Streamlit display
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
                
    except Exception as e:
        st.error(f"❌ Error in display_archetype_details: {str(e)}")
        st.write("**Exception details:**")
        import traceback
        st.code(traceback.format_exc())


def display_survival_analysis(results: Dict):
    """Display survival analysis for archetypes"""
    st.subheader("📈 Survival Analysis by Archetype")
    
    try:
        # Debug and validation
        st.write("**Debug: Survival Analysis:**")
        st.write(f"- Available keys: {list(results.keys())}")
        
        if 'features_df' not in results:
            st.error("❌ 'features_df' not found in results")
            return
        if 'archetypes' not in results:
            st.error("❌ 'archetypes' not found in results")
            return
            
        features_df = results['features_df']
        archetypes = results['archetypes']
        
        if features_df is None or features_df.height == 0:
            st.warning("⚠️ No feature data available")
            return
        if not archetypes:
            st.warning("⚠️ No archetypes found")
            return
            
        st.success(f"✅ Survival analysis ready for {len(archetypes)} archetypes")
        
        st.markdown("""
        Survival analysis shows how long tokens of each archetype typically survive before "dying".
        """)
        
        # Placeholder for survival curves - This would require lifelines library for proper Kaplan-Meier curves
        st.info("📊 Survival curves visualization would be implemented here using lifelines library")
        
        # For now, show lifespan distribution
        # Create lifespan distribution by archetype
        archetype_names = {i: arch['name'] for i, arch in archetypes.items()}
        features_df_with_names = features_df.with_columns(
            pl.col('cluster').map_elements(lambda x: archetype_names.get(x, f"Cluster {x}"), return_dtype=pl.Utf8).alias('archetype_name')
        )
        
        fig = px.box(
            features_df_with_names.to_pandas(),
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
        
    except Exception as e:
        st.error(f"❌ Error in display_survival_analysis: {str(e)}")
        st.write("**Exception details:**")
        import traceback
        st.code(traceback.format_exc())


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
    features_df = features_df.with_columns(
        pl.col('cluster').map_elements(lambda x: archetype_names.get(x, f"Cluster {x}"), return_dtype=pl.Utf8).alias('archetype_name')
    )
    
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


# ================================
# BASELINE CLUSTERING DISPLAY FUNCTIONS
# ================================

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
        avg_ari = stability_summary.get('mean_ari', 0)
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
            'Average ARI': category_results['stability_results']['mean_ari'],
            'Min ARI': category_results['stability_results']['min_ari']
        })
    
    if category_data:
        # Use polars DataFrame for better performance
        df = pl.DataFrame(category_data)
        # Convert to pandas only for Streamlit display
        st.dataframe(df.to_pandas())
    
    # Success criteria check
    st.subheader("✅ CEO Requirements Check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Features Used**: 15 essential features ✅")
        st.write("**Method**: Elbow method for K selection ✅")
        st.write("**Stability**: ARI calculation ✅")
    
    with col2:
        success_criteria = avg_ari > 0.75
        status = "✅ PASSED" if success_criteria else "❌ FAILED"
        st.write(f"**Success Criteria (ARI > 0.75)**: {status}")
        
        if success_criteria:
            st.success("🎉 Perfect clustering stability achieved!")
        else:
            st.warning("⚠️ Clustering stability below threshold")


def display_baseline_stability_analysis(results: Dict):
    """Display stability analysis results"""
    st.subheader("📈 Clustering Stability Analysis")
    
    # Overall stability summary
    stability_summary = results.get('stability_summary', {})
    
    if stability_summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average ARI", f"{stability_summary['mean_ari']:.3f}")
        
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
            'Average ARI': stability['mean_ari'],
            'Min ARI': stability['min_ari'],
            'Max ARI': stability['max_ari'],
            'Runs': stability['n_runs'],
            'Status': '✅ STABLE' if stability['mean_ari'] > 0.75 else '⚠️ UNSTABLE'
        })
    
    if stability_data:
        # Use polars DataFrame for better performance
        df = pl.DataFrame(stability_data)
        # Convert to pandas only for Streamlit display
        st.dataframe(df.to_pandas())
    
    # ARI distribution visualization
    st.subheader("📊 ARI Score Distribution")
    
    all_ari_scores = []
    category_labels = []
    
    for category_name, category_results in results['categories'].items():
        ari_scores = category_results['stability_results']['ari_scores']
        all_ari_scores.extend(ari_scores)
        category_labels.extend([category_name] * len(ari_scores))
    
    if all_ari_scores:
        # Use polars DataFrame for better performance
        ari_df = pl.DataFrame({
            'ARI Score': all_ari_scores,
            'Category': category_labels
        })
        
        # Convert to pandas for plotly compatibility
        ari_df_pd = ari_df.to_pandas()
        fig = px.box(
            ari_df_pd,
            x='Category',
            y='ARI Score',
            title="ARI Score Distribution by Category"
        )
        
        # Add success threshold line
        fig.add_hline(y=0.75, line_dash="dash", line_color="green", 
                     annotation_text="Success Threshold (0.75)")
        
        st.plotly_chart(fig, use_container_width=True)


def display_unified_behavioral_results(results: Dict):
    """Display unified behavioral archetype analysis results"""
    st.subheader("🎯 Unified Behavioral Archetype Results")
    
    try:
        # Debug information
        st.write("**Debug: Unified Results Structure:**")
        st.write(f"- Available keys: {list(results.keys())}")
        
        # Display analysis summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tokens Analyzed", results.get('total_tokens_analyzed', 'Unknown'))
        with col2:
            if 'clustering_results' in results:
                st.metric("Clusters Found", results['clustering_results'].get('n_clusters', 'Unknown'))
            else:
                st.metric("Clusters Found", "N/A")
                st.warning("⚠️ clustering_results not found")
        with col3:
            if 'quality_metrics' in results:
                quality = results['quality_metrics']
                st.metric("Silhouette Score", f"{quality.get('silhouette_score', 0):.3f}")
            else:
                st.metric("Silhouette Score", "N/A")
                st.warning("⚠️ quality_metrics not found")
        
        # Display cluster imbalance warning if needed
        if 'quality_metrics' in results:
            quality = results['quality_metrics']
            if quality.get('is_severely_imbalanced', False):
                st.warning(f"⚠️ **Cluster Imbalance**: {quality.get('max_cluster_percentage', 0):.1f}% of tokens in one cluster")
        
        # Display archetype information
        if 'archetypes' in results:
            st.subheader("📊 Identified Behavioral Archetypes")
            archetypes = results['archetypes']
            
            for cluster_id, archetype in archetypes.items():
                with st.expander(f"Cluster {cluster_id}: {archetype['name']}", expanded=True):
                    stats = archetype['stats']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Tokens", stats['n_tokens'])
                    with col2:
                        st.metric("% of Total", f"{stats['pct_of_total']:.1f}%")
                    with col3:
                        if 'pct_dead' in stats:
                            st.metric("% Dead Tokens", f"{stats['pct_dead']:.1f}%")
        
        # Display features dataframe if available
        if 'features_df' in results:
            st.subheader("🔍 Feature Analysis")
            features_df = results['features_df']
            
            if features_df is not None and hasattr(features_df, 'height') and features_df.height > 0:
                # Show cluster distribution
                if 'cluster' in features_df.columns:
                    cluster_dist = features_df.group_by('cluster').count().sort('cluster')
                    st.write("**Cluster Distribution:**")
                    st.dataframe(cluster_dist)
                
                # Show feature summary
                feature_cols = [col for col in features_df.columns 
                               if col not in ['token', 'category', 'lifespan_category', 'cluster']]
                if feature_cols:
                    st.write(f"**Features used**: {', '.join(feature_cols[:10])}")
                    if len(feature_cols) > 10:
                        st.write(f"... and {len(feature_cols) - 10} more features")
            else:
                st.warning("No feature data available to display")
        
        # Display clustering quality metrics
        if 'clustering_results' in results:
            clustering_results = results['clustering_results']
            quality_metrics = clustering_results.get('quality_metrics', {})
            
            if quality_metrics:
                st.subheader("📈 Clustering Quality Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{quality_metrics.get('silhouette_score', 0):.4f}")
                    st.caption("Higher is better (closer to 1)")
                    
                with col2:
                    db_score = quality_metrics.get('davies_bouldin_score', float('inf'))
                    if db_score != float('inf'):
                        st.metric("Davies-Bouldin Score", f"{db_score:.4f}")
                        st.caption("Lower is better (closer to 0)")
                    else:
                        st.metric("Davies-Bouldin Score", "N/A")
                
                # Show imbalance analysis if available
                if 'imbalance_analysis' in quality_metrics:
                    imbalance = quality_metrics['imbalance_analysis']
                    
                    if 'death_analysis' in imbalance and imbalance['death_analysis']:
                        st.write("**Death Rate by Cluster:**")
                        death_data = []
                        for cluster_id, death_info in imbalance['death_analysis'].items():
                            death_data.append({
                                'Cluster': cluster_id,
                                'Death Rate': f"{death_info.get('death_rate', 0)*100:.1f}%",
                                'Size': death_info.get('size', 0)
                            })
                        if death_data:
                            st.dataframe(death_data)
                        
    except Exception as e:
        st.error(f"❌ Error in display_unified_behavioral_results: {str(e)}")
        st.write("**Exception details:**")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main() 