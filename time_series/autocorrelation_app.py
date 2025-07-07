"""
Streamlit App for Autocorrelation and Time Series Clustering Analysis
"""

import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Import our analyzer
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


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
        ["Feature-based (ACF + Statistics)", "Price-only", "Multi-Resolution ACF"],
        help="Feature-based uses 16 engineered features. Price-only clusters on price series. Multi-Resolution compares Sprint/Standard/Marathon tokens."
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
                                       min_value=10, value=100,
                                       help="Limit for faster analysis")
            
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
            
            max_tokens = st.number_input("Max tokens to analyze:", 
                                       min_value=10, value=100,
                                       help="Limit for faster analysis")
            
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
    
    # Multi-Resolution ACF specific options
    elif analysis_type == "Multi-Resolution ACF":
        with st.sidebar.expander("🚀 Multi-Resolution Settings", expanded=True):
            st.markdown("**⭐ PHASE 1A: Multi-Resolution Analysis**")
            
            multi_method = st.selectbox(
                "Price transformation:",
                ["returns", "log_returns", "prices", "dtw_features"],
                help="Method for analyzing across lifespan categories"
            )
            
            max_tokens_per_category = st.number_input(
                "Max tokens per category:",
                min_value=10, max_value=1000, value=100,
                help="Limit per category for balanced analysis across Sprint/Standard/Marathon"
            )
            
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
            
            st.markdown("""
            **Lifespan Categories:**
            - **Sprint**: 200-400 min (fast pump/dump)
            - **Standard**: 400-1200 min (typical lifecycle)
            - **Marathon**: 1200+ min (extended development)
            
            **Why limit per category?** 
            Ensures balanced representation across categories.
            Without limits: Sprint=50, Standard=5000, Marathon=20 → imbalanced analysis.
            
            **Goal**: Discover behavioral archetypes for stable ML baseline
            """)
        
        # Multi-resolution uses fixed optimal settings
        use_log_price = True
        find_optimal_k = True
        n_clusters = None
        clustering_method = 'kmeans'
        max_tokens = None
        
        # Initialize multi-resolution specific variables for other analysis types
        multi_method = "returns"
        max_tokens_per_category = 100
        enable_dtw_clustering = False
        compare_across_categories = True
    
    # Initialize variables for other analysis types that don't have multi-resolution settings
    if analysis_type != "Multi-Resolution ACF":
        # Default values for multi-resolution variables not used in other types
        multi_method = "returns"
        max_tokens_per_category = 100
        enable_dtw_clustering = False
        compare_across_categories = True
    
    # Initialize price-only specific variables for other analysis types
    if analysis_type != "Price-only":
        price_method = "returns"
        max_sequence_length = 500
        use_max_length = True
    
    # Quick rerun option if results exist (only for non-multi-resolution analysis)
    if 'results' in st.session_state and analysis_type != "Multi-Resolution ACF":
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
                elif analysis_type == "Multi-Resolution ACF":
                    # Run multi-resolution lifespan category analysis
                    st.info("🚀 Running Phase 1A: Multi-Resolution ACF Analysis...")
                    
                    results = analyzer.analyze_by_lifespan_category(
                        data_path,
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
                if analysis_type == "Multi-Resolution ACF":
                    total_tokens = results.get('total_tokens_analyzed', 0)
                    n_categories = len(results.get('categories', {}))
                    st.success(f"✅ Multi-Resolution Analysis complete! Analyzed {total_tokens} tokens across {n_categories} lifespan categories.")
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
        if 'categories' in results:  # Multi-Resolution analysis
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Multi-Resolution Overview", 
                "🏃 Category Comparison",
                "📈 Cross-Category ACF",
                "🎯 Category Clustering",
                "🗺️ Combined t-SNE"
            ])
        else:  # Standard analysis
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", 
                "📈 Autocorrelation Analysis",
                "🎯 Clustering Results",
                "📈 Elbow Analysis",
                "🗺️ t-SNE Visualization",
                "🔍 Token Explorer"
            ])
        
        # Handle tab content based on analysis type
        if 'categories' in results:  # Multi-Resolution analysis
            with tab1:
                display_multi_resolution_overview(results)
                
            with tab2:
                display_category_comparison(results)
                
            with tab3:
                display_cross_category_acf(results)
                
            with tab4:
                display_category_clustering(results)
                
            with tab5:
                display_combined_tsne(results)
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
        st.metric("Total Tokens Analyzed", results.get('total_tokens_analyzed', 0))
        
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
                    'Avg Return': f"{stats.get('avg_return', 0)*100:.1f}%",
                    'Avg Volatility': f"{stats.get('avg_volatility', 0)*100:.1f}%"
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
                    avg_return = stats.get('avg_return', 0)
                    st.metric("Avg Return", f"{avg_return*100:.1f}%")
                    
                with col3:
                    avg_volatility = stats.get('avg_volatility', 0)
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
            
            for i, (x, y) in enumerate(tsne_2d):
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
            st.metric("Total Data Points", len(combined_tsne_data))
            
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


if __name__ == "__main__":
    main() 