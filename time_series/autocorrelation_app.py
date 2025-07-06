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
    - **Autocorrelation (ACF)** and **Partial Autocorrelation (PACF)**
    - **Time Series Clustering** to find similar patterns
    - **t-SNE Visualization** for dimensionality reduction
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
    
    # Analysis parameters
    use_log_price = st.sidebar.checkbox("Use Log Prices", value=True, 
                                       help="Use log-transformed prices for analysis")
    
    # Optional token limit
    limit_tokens = st.sidebar.checkbox("Limit Tokens", value=False,
                                      help="Check to limit number of tokens for faster testing")
    
    max_tokens = None
    if limit_tokens:
        max_tokens = st.sidebar.number_input("Max Tokens to Analyze", 
                                            min_value=10, value=100,
                                            help="Limit number of tokens for faster analysis (leave blank for no limit)")
    
    # Clustering parameters
    find_optimal_k = st.sidebar.checkbox("Find Optimal K", value=True,
                                        help="Automatically find optimal number of clusters using elbow method")
    
    n_clusters = None
    if not find_optimal_k:
        n_clusters = st.sidebar.slider("Number of Clusters", 
                                      min_value=2, max_value=20, value=5,
                                      help="Number of clusters for K-means")
    
    clustering_method = st.sidebar.selectbox("Clustering Method",
                                           ["kmeans", "hierarchical", "dbscan"])
    
    # Analysis type selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Type")
    
    analysis_type = st.sidebar.radio(
        "Choose analysis approach:",
        ["Feature-based (ACF + Statistics)", "Price-only"],
        help="Feature-based uses 16 engineered features. Price-only clusters directly on price series."
    )
    
    # Price-only specific options
    if analysis_type == "Price-only":
        price_method = st.sidebar.selectbox(
            "Price clustering method:",
            ["returns", "log_returns", "normalized_prices", "dtw_features"],
            help="Method for converting price series to features"
        )
        
        max_sequence_length = st.sidebar.number_input(
            "Max sequence length:",
            min_value=50, value=500,
            help="Maximum length of price sequences (None = use shortest)"
        )
        
        use_max_length = st.sidebar.checkbox("Use max length limit", value=True)
        
        st.sidebar.markdown("""
        **Price Methods:**
        - **returns**: Raw returns (more stationary)
        - **log_returns**: Log returns (better properties)  
        - **normalized_prices**: 0-1 scaled prices
        - **dtw_features**: Statistical features extracted from prices
        """)
    
    # Quick rerun option if results exist
    if 'results' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Quick Rerun Options:**")
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
                        max_tokens=max_tokens,
                        max_length=max_sequence_length if use_max_length else None
                    )
                else:
                    # Run complete feature-based analysis
                    results = analyzer.run_complete_analysis(
                        data_path,
                        use_log_price=use_log_price,
                        n_clusters=n_clusters,
                        find_optimal_k=find_optimal_k,
                        max_tokens=max_tokens
                    )
                
                # Store results in session state
                st.session_state['results'] = results
                st.success(f"Analysis complete! Analyzed {len(results['token_names'])} tokens.")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.error(f"Full traceback: {traceback.format_exc()}")
                return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "📈 Autocorrelation Analysis",
            "🎯 Clustering Results",
            "📈 Elbow Analysis",
            "🗺️ t-SNE Visualization",
            "🔍 Token Explorer"
        ])
        
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
            if token in results['acf_results']:
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
            if token in results['acf_results']:
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
            
            if token in results['acf_results'] and token in results['token_data']:
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
            if token in results['acf_results']:
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
            if selected_token in results['acf_results']:
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
                if token in results['token_data'] and token in results['acf_results']:
                    df = results['token_data'][token]
                    acf_res = results['acf_results'][token]
                    
                    # Calculate basic stats
                    returns = np.diff(df['price'].to_numpy()) / df['price'].to_numpy()[:-1]
                    price_change = (df['price'][-1] - df['price'][0]) / df['price'][0] * 100
                    volatility = np.std(returns) * 100  # As percentage
                    
                    summary_data.append({
                        'Token': token[:25] + '...' if len(token) > 25 else token,
                        'Data Points': len(df),
                        'Price Change (%)': f"{price_change:.2f}",
                        'Volatility (%)': f"{volatility:.2f}",
                        'Significant Lags': len(acf_res['significant_lags']),
                        'Decay Rate': f"{acf_res['decay_rate']:.4f}" if not np.isnan(acf_res['decay_rate']) else "N/A",
                        'First Zero': acf_res['first_zero_crossing']
                    })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Additional cluster analysis
            st.subheader("Cluster Pattern Analysis")
            
            # Calculate cluster-wide statistics
            all_returns = []
            all_volatilities = []
            all_decay_rates = []
            
            for token in display_tokens:
                if token in results['token_data'] and token in results['acf_results']:
                    df = results['token_data'][token]
                    acf_res = results['acf_results'][token]
                    
                    returns = np.diff(df['price'].to_numpy()) / df['price'].to_numpy()[:-1]
                    volatility = np.std(returns)
                    
                    all_returns.extend(returns.tolist())
                    all_volatilities.append(volatility)
                    
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


if __name__ == "__main__":
    main() 