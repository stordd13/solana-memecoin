# autocorrelation_streamlit_app.py
"""
Refactored Streamlit App for Autocorrelation Analysis
Focused on analyzing token autocorrelation patterns and distributions
to determine optimal prediction horizons for deep learning models
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional
import json

# Import our analyzer
from time_series.autocorrelation_analysis import AutocorrelationAnalyzer


# Streamlit page configuration
st.set_page_config(
    page_title="Autocorrelation Analysis for Deep Learning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e86de;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef9e7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📈 Autocorrelation Analysis for Deep Learning</h1>', unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = AutocorrelationAnalyzer()

if 'token_data' not in st.session_state:
    st.session_state.token_data = None

if 'distribution_results' not in st.session_state:
    st.session_state.distribution_results = None

if 'horizon_recommendations' not in st.session_state:
    st.session_state.horizon_recommendations = None

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Data loading section
st.sidebar.header("📁 Data Loading")
processed_data_path = st.sidebar.text_input(
    "Processed Data Directory",
    value="../data/processed",
    help="Path to the processed data directory containing categorized tokens"
)

# Token limit configuration
use_token_limit = st.sidebar.checkbox(
    "Limit Tokens per Category",
    value=True,
    help="Enable to limit the number of tokens loaded per category for faster processing"
)

if use_token_limit:
    max_tokens_per_category = st.sidebar.number_input(
        "Max Tokens per Category",
        min_value=1,
        max_value=10000,
        value=500,
        help="Maximum number of tokens to load per category"
    )
else:
    max_tokens_per_category = None
    st.sidebar.info("Loading ALL tokens (may be slow for large datasets)")

# Analysis configuration
st.sidebar.header("🔧 Analysis Settings")
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["returns", "log_returns", "prices", "log_prices"],
    index=0,  # Default to "returns" (more stable than log_returns)
    help="Type of time series to analyze for autocorrelation"
)

max_lag = st.sidebar.number_input(
    "Maximum Lag (minutes)",
    min_value=10,
    max_value=480,
    value=240,
    help="Maximum lag to compute autocorrelation for (240 = 4 hours, good for memecoin analysis)"
)

confidence_level = st.sidebar.slider(
    "Confidence Level",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.05,
    help="Confidence level for ACF confidence intervals. Higher values (e.g., 0.95) create wider bands, making it harder for lags to be considered 'significant'. Lower values (e.g., 0.70) create narrower bands, identifying more lags as significant. Recommended: 0.80 (balanced), 0.95 (conservative), 0.70 (liberal exploration)"
)

# Add explanation box for analysis types
with st.sidebar.expander("📊 Analysis Type Guide"):
    st.markdown("""
    **Choose the right analysis type:**
    
    🔹 **Returns** (recommended first):
    - Most stable for ACF analysis
    - Shows momentum patterns
    - Best for getting started
    
    🔹 **Log Returns**:
    - Better for extreme volatility
    - Handles large price swings
    - Use if returns give issues
    
    🔹 **Prices**:
    - Shows trend autocorrelation
    - Good for long-term patterns
    - May be non-stationary
    
    🔹 **Log Prices**:
    - Reveals multiplicative patterns
    - Good for growth rate analysis
    - Shows relative price changes
    """)

# Add explanation box for confidence level
with st.sidebar.expander("ℹ️ Understanding Confidence Level"):
    st.markdown("""
    **Confidence Level** determines the width of the confidence bands in ACF plots:
    
    - **0.80 (80%)**: Balanced choice - good for exploration
    - **0.95 (95%)**: Conservative - only strong correlations are significant  
    - **0.70 (70%)**: Liberal - more correlations considered significant
    - **0.50 (50%)**: Very liberal - maximum exploration mode
    
    **How it works:**
    - ACF values outside the confidence bands are statistically significant
    - Higher confidence = wider bands = fewer significant lags
    - Lower confidence = narrower bands = more significant lags
    
    **For Deep Learning:**
    - Use 0.95 for balanced lag selection
    - Use 0.99 if you want only the strongest patterns
    - Use 0.90 to explore more potential lags
    """)

# Update analyzer settings
st.session_state.analyzer.max_lag = max_lag
st.session_state.analyzer.confidence_level = confidence_level

# Load data button
if st.sidebar.button("🔄 Load Token Data", type="primary"):
    with st.spinner("Loading token data..."):
        try:
            processed_dir = Path(processed_data_path)
            if not processed_dir.exists():
                st.error(f"Directory not found: {processed_dir}")
            else:
                st.session_state.token_data = st.session_state.analyzer.load_processed_tokens(
                    processed_dir, max_tokens_per_category
                )
                st.success(f"Loaded {len(st.session_state.token_data)} tokens")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Main tabs
if st.session_state.token_data is not None:
    tab1, tab2, tab3 = st.tabs(["🔍 Individual Token Analysis", "📊 Distribution Analysis", "🎯 Prediction Horizons"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Individual Token Autocorrelation Analysis</h2>', unsafe_allow_html=True)
        
        # Token selection
        token_names = list(st.session_state.token_data.keys())
        selected_token = st.selectbox(
            "Select Token for Analysis",
            token_names,
            help="Choose a token to analyze its individual autocorrelation pattern"
        )
        
        if selected_token:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🔍 Analyze Token", type="primary"):
                    with st.spinner(f"Analyzing {selected_token}..."):
                        try:
                            # Get token data
                            token_df = st.session_state.token_data[selected_token]
                            
                            # Prepare data
                            from time_series.legacy.archetype_utils import prepare_token_data
                            prices, returns, death_minute = prepare_token_data(token_df)
                            
                            # Compute ACF
                            acf_result = st.session_state.analyzer.compute_token_autocorrelation(
                                prices, returns, selected_token, analysis_type
                            )
                            
                            if acf_result['success']:
                                # Create ACF plot
                                fig = go.Figure()
                                
                                lags = list(range(len(acf_result['acf_values'])))
                                
                                # Add ACF values
                                fig.add_trace(go.Scatter(
                                    x=lags,
                                    y=acf_result['acf_values'],
                                    mode='lines+markers',
                                    name='ACF',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Add confidence intervals if available
                                if acf_result['confidence_intervals']:
                                    conf_int = np.array(acf_result['confidence_intervals'])
                                    fig.add_trace(go.Scatter(
                                        x=lags,
                                        y=conf_int[:, 0],
                                        mode='lines',
                                        line=dict(color='red', dash='dash'),
                                        name='Lower CI',
                                        showlegend=False
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=lags,
                                        y=conf_int[:, 1],
                                        mode='lines',
                                        line=dict(color='red', dash='dash'),
                                        name='Upper CI',
                                        fill='tonexty',
                                        fillcolor='rgba(255,0,0,0.1)'
                                    ))
                                
                                # Add zero line
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                
                                # Highlight significant lags
                                for lag in acf_result['significant_lags'][:10]:  # Show first 10
                                    fig.add_vline(
                                        x=lag,
                                        line_dash="dot",
                                        line_color="green",
                                        annotation_text=f"Lag {lag}",
                                        annotation_position="top"
                                    )
                                
                                fig.update_layout(
                                    title=f"Autocorrelation Function - {selected_token}",
                                    xaxis_title="Lag",
                                    yaxis_title="Autocorrelation",
                                    height=500,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # PACF plot
                                if acf_result['pacf_values']:
                                    fig_pacf = go.Figure()
                                    pacf_lags = list(range(len(acf_result['pacf_values'])))
                                    
                                    fig_pacf.add_trace(go.Scatter(
                                        x=pacf_lags,
                                        y=acf_result['pacf_values'],
                                        mode='lines+markers',
                                        name='PACF',
                                        line=dict(color='orange', width=2)
                                    ))
                                    
                                    fig_pacf.add_hline(y=0, line_dash="dash", line_color="gray")
                                    
                                    fig_pacf.update_layout(
                                        title=f"Partial Autocorrelation Function - {selected_token}",
                                        xaxis_title="Lag",
                                        yaxis_title="Partial Autocorrelation",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_pacf, use_container_width=True)
                                
                                # Store result for display
                                st.session_state.current_token_result = acf_result
                                
                            else:
                                st.error(f"Failed to analyze {selected_token}: {acf_result['failure_reason']}")
                                
                        except Exception as e:
                            st.error(f"Error analyzing token: {str(e)}")
            
            with col2:
                # Display token statistics if available
                if 'current_token_result' in st.session_state and st.session_state.current_token_result['success']:
                    result = st.session_state.current_token_result
                    
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**📊 Token Statistics**")
                    st.metric("Series Length", result['series_length'])
                    st.metric("Max ACF Lag", f"{result['max_acf_lag']}")
                    st.metric("Max ACF Value", f"{result['max_acf_value']:.3f}")
                    st.metric("First Zero Crossing", f"{result['first_zero_crossing']}")
                    st.metric("Decay Rate", f"{result['decay_rate']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Series statistics
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.markdown("**📈 Series Statistics**")
                    series_stats = result['series_stats']
                    st.metric("Mean", f"{series_stats['mean']:.6f}")
                    st.metric("Std Dev", f"{series_stats['std']:.6f}")
                    st.metric("Skewness", f"{series_stats['skewness']:.3f}")
                    st.metric("Kurtosis", f"{series_stats['kurtosis']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Significant lags
                    if result['significant_lags']:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("**⚡ Significant Lags**")
                        sig_lags_str = ", ".join(map(str, result['significant_lags'][:10]))
                        st.write(f"First 10: {sig_lags_str}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Autocorrelation Distribution Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔄 Compute Distribution Analysis", type="primary"):
                with st.spinner("Computing autocorrelation distributions..."):
                    try:
                        st.session_state.distribution_results = st.session_state.analyzer.compute_autocorrelation_distributions(
                            st.session_state.token_data,
                            analysis_type=analysis_type,
                            n_jobs=-1
                        )
                        st.success("Distribution analysis complete!")
                    except Exception as e:
                        st.error(f"Error computing distributions: {str(e)}")
        
        with col2:
            if st.session_state.distribution_results is not None:
                results = st.session_state.distribution_results
                st.metric("Total Tokens", results['total_tokens'])
                st.metric("Successful", results['successful_tokens'])
                st.metric("Failed", results['failed_tokens'])
                success_rate = (results['successful_tokens'] / results['total_tokens']) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Display distribution results
        if st.session_state.distribution_results is not None and st.session_state.distribution_results['success']:
            results = st.session_state.distribution_results
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"**Successful tokens:** {results['successful_tokens']}")
                st.write(f"**Failed tokens:** {results['failed_tokens']}")
                st.write(f"**Total lag statistics:** {len(results['distribution_stats']['lag_statistics'])}")
                st.write(f"**Individual results:** {len(results['individual_results'])}")
                
                # Show sample of lag statistics
                lag_stats = results['distribution_stats']['lag_statistics']
                if lag_stats:
                    sample_lag = list(lag_stats.keys())[0]
                    st.write(f"**Sample lag stats ({sample_lag}):** {lag_stats[sample_lag]}")
                else:
                    st.error("No lag statistics found! This explains why plots are empty.")
            
            # Overall statistics
            st.markdown('<h3 class="sub-header">📊 Overall Statistics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            overall_stats = results['distribution_stats']['overall_statistics']
            
            with col1:
                st.metric(
                    "Mean Optimal Lag",
                    f"{overall_stats['max_acf_lag_distribution']['mean']:.1f}",
                    f"±{overall_stats['max_acf_lag_distribution']['std']:.1f}"
                )
            
            with col2:
                st.metric(
                    "Mean Max ACF",
                    f"{overall_stats['max_acf_value_distribution']['mean']:.3f}",
                    f"±{overall_stats['max_acf_value_distribution']['std']:.3f}"
                )
            
            with col3:
                st.metric(
                    "Mean Decay Rate",
                    f"{overall_stats['decay_rate_distribution']['mean']:.3f}",
                    f"±{overall_stats['decay_rate_distribution']['std']:.3f}"
                )
            
            with col4:
                st.metric(
                    "Mean Zero Crossing",
                    f"{overall_stats['first_zero_crossing_distribution']['mean']:.1f}",
                    f"±{overall_stats['first_zero_crossing_distribution']['std']:.1f}"
                )
            
            # Add Average ACF Plot (NEW)
            st.markdown('<h3 class="sub-header">📈 Average ACF Across All Tokens</h3>', unsafe_allow_html=True)
            
            lag_stats = results['distribution_stats']['lag_statistics']
            
            # Prepare data for average ACF plot
            lags = []
            means = []
            stds = []
            medians = []
            q25s = []
            q75s = []
            
            for lag_key in sorted(lag_stats.keys(), key=lambda x: int(x.split('_')[1])):
                lag_num = int(lag_key.split('_')[1])
                if lag_num <= max_lag:  # Show all computed lags
                    lags.append(lag_num)
                    means.append(lag_stats[lag_key]['mean'])
                    stds.append(lag_stats[lag_key]['std'])
                    medians.append(lag_stats[lag_key]['median'])
                    q25s.append(lag_stats[lag_key]['q25'])
                    q75s.append(lag_stats[lag_key]['q75'])
            
            # Create average ACF plot with confidence bands
            fig_avg = go.Figure()
            
            # Add mean ACF line
            fig_avg.add_trace(go.Scatter(
                x=lags,
                y=means,
                mode='lines+markers',
                name='Mean ACF',
                line=dict(color='blue', width=3),
                marker=dict(size=4)
            ))
            
            # Add confidence bands (mean ± std)
            fig_avg.add_trace(go.Scatter(
                x=lags,
                y=[m + s for m, s in zip(means, stds)],
                mode='lines',
                line=dict(color='blue', dash='dash', width=1),
                name='Mean + Std',
                showlegend=False
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=lags,
                y=[m - s for m, s in zip(means, stds)],
                mode='lines',
                line=dict(color='blue', dash='dash', width=1),
                name='Mean - Std',
                fill='tonexty',
                fillcolor='rgba(0, 100, 255, 0.1)',
                showlegend=True
            ))
            
            # Add median line
            fig_avg.add_trace(go.Scatter(
                x=lags,
                y=medians,
                mode='lines',
                name='Median ACF',
                line=dict(color='red', width=2, dash='dot')
            ))
            
            # Add quartile bands
            fig_avg.add_trace(go.Scatter(
                x=lags,
                y=q75s,
                mode='lines',
                line=dict(color='gray', width=1),
                name='75th Percentile',
                showlegend=False
            ))
            
            fig_avg.add_trace(go.Scatter(
                x=lags,
                y=q25s,
                mode='lines',
                line=dict(color='gray', width=1),
                name='25th Percentile',
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=True
            ))
            
            # Add zero line
            fig_avg.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Add significance threshold lines
            significance_threshold = 0.1
            fig_avg.add_hline(y=significance_threshold, line_dash="dot", line_color="green", 
                            annotation_text="Significance threshold", annotation_position="right")
            fig_avg.add_hline(y=-significance_threshold, line_dash="dot", line_color="green")
            
            fig_avg.update_layout(
                title="Average Autocorrelation Function Across All Tokens",
                xaxis_title="Lag (minutes)",
                yaxis_title="Autocorrelation",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_avg, use_container_width=True)
            
            # Add interpretation
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **📊 Key Insights from Average ACF:**
                - First zero crossing at lag ~{next((i for i, m in enumerate(means) if m < 0), len(means))}
                - Strongest correlation at lag {lags[means.index(max(means[1:]))] if len(means) > 1 else 0}
                - Mean ACF at lag 1: {means[1] if len(means) > 1 else 0:.3f}
                - Decay rate indicates {'strong' if means[1] > 0.5 else 'moderate' if means[1] > 0.2 else 'weak'} persistence
                """)
            
            with col2:
                st.warning(f"""
                **🎯 Recommended Sequence Lengths:**
                - Conservative: {next((i for i, m in enumerate(means) if abs(m) < 0.1), 10)} lags
                - Balanced: {next((i for i, m in enumerate(means) if m < 0), 20)} lags
                - Aggressive: {min(30, len([m for m in means if abs(m) > 0.05]))} lags
                """)
            
            # Distribution visualizations
            st.markdown('<h3 class="sub-header">🔥 ACF Distribution Analysis</h3>', unsafe_allow_html=True)
            
            # Create detailed distribution plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ACF Distribution by Lag', 'ACF Heatmap', 
                               'Decay Pattern', 'Significant Lags Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Box plot of ACF distributions for key time horizons
            key_lags = [1, 2, 5, 10, 15, 30, 60, 120, 180, 240]  # 1min, 2min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h
            lag_labels = ['1min', '2min', '5min', '10min', '15min', '30min', '1h', '2h', '3h', '4h']
            
            box_data = []
            for i, lag in enumerate(key_lags):
                if lag <= max_lag:  # Only show lags within our range
                    lag_key = f'lag_{lag}'
                    if lag_key in lag_stats:
                        # Get individual ACF values from results
                        acf_values = [r['acf_values'][lag] for r in results['individual_results'] 
                                    if len(r['acf_values']) > lag]
                        if acf_values and len(acf_values) > 5:  # Only show if we have enough data
                            box_data.append(go.Box(
                                y=acf_values,
                                name=lag_labels[i],
                                boxpoints='outliers',
                                marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
                            ))
            
            for trace in box_data:
                fig.add_trace(trace, row=1, col=1)
            
            # 2. Lag Statistics Heatmap (interpretable)
            if lag_stats:
                # Create meaningful statistics heatmap
                selected_lags = [lag for lag in key_lags if lag <= max_lag and f'lag_{lag}' in lag_stats]
                stat_names = ['Mean', 'Std', 'Median', '% Significant']
                heatmap_data = []
                
                for stat_name in stat_names:
                    row_data = []
                    for lag in selected_lags:
                        lag_key = f'lag_{lag}'
                        if stat_name == 'Mean':
                            row_data.append(lag_stats[lag_key]['mean'])
                        elif stat_name == 'Std':
                            row_data.append(lag_stats[lag_key]['std'])
                        elif stat_name == 'Median':
                            row_data.append(lag_stats[lag_key]['median'])
                        elif stat_name == '% Significant':
                            pct = (lag_stats[lag_key]['significant_tokens'] / results['successful_tokens']) * 100
                            row_data.append(pct)
                    heatmap_data.append(row_data)
                
                if heatmap_data and selected_lags:
                    heatmap = go.Heatmap(
                        z=heatmap_data,
                        x=[f'{lag}min' if lag < 60 else f'{lag//60}h' for lag in selected_lags],
                        y=stat_names,
                        colorscale='RdYlBu_r',
                        text=[[f'{val:.3f}' for val in row] for row in heatmap_data],
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title='Value')
                    )
                    fig.add_trace(heatmap, row=1, col=2)
            
            # 3. Decay pattern analysis
            if means and len(means) > 5:
                # Use actual lag values, not indices
                decay_lags = lags[:min(60, len(lags))]  # First 60 lags or 1 hour
                decay_means = means[:len(decay_lags)]
                
                if decay_lags and decay_means:
                    fig.add_trace(
                        go.Scatter(
                            x=decay_lags,
                            y=decay_means,
                            mode='lines+markers',
                            name='Mean ACF Decay',
                            line=dict(color='purple', width=2),
                            marker=dict(size=3)
                        ),
                        row=2, col=1
                    )
                    
                    # Add simple trend line
                    if len(decay_means) > 10:
                        try:
                            # Linear trend line
                            x_vals = np.array(decay_lags)
                            y_vals = np.array(decay_means)
                            coeffs = np.polyfit(x_vals, y_vals, 1)
                            y_trend = coeffs[0] * x_vals + coeffs[1]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=decay_lags,
                                    y=y_trend,
                                    mode='lines',
                                    name='Linear Trend',
                                    line=dict(color='orange', dash='dash', width=2)
                                ),
                                row=2, col=1
                            )
                        except Exception as e:
                            st.warning(f"Could not fit trend line: {e}")
            
            # 4. Significant lags histogram
            if results['individual_results']:
                sig_lag_counts = {}
                max_display_lag = min(max_lag, 120)  # Don't show beyond 120 for clarity
                
                for r in results['individual_results']:
                    if 'significant_lags' in r and r['significant_lags']:
                        for lag in r['significant_lags']:
                            if lag <= max_display_lag:
                                sig_lag_counts[lag] = sig_lag_counts.get(lag, 0) + 1
                
                if sig_lag_counts and len(sig_lag_counts) > 0:
                    # Sort by lag for better visualization
                    sorted_lags = sorted(sig_lag_counts.keys())
                    sorted_counts = [sig_lag_counts[lag] for lag in sorted_lags]
                    
                    fig.add_trace(
                        go.Bar(
                            x=sorted_lags,
                            y=sorted_counts,
                            name='Significant Count',
                            marker_color='green',
                            opacity=0.7
                        ),
                        row=2, col=2
                    )
                else:
                    # Add placeholder if no significant lags found
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text="No significant lags found<br>Try lowering confidence level",
                        showarrow=False,
                        xref="x4", yref="y4",
                        font=dict(size=12, color="gray")
                    )
            
            # Update layout
            fig.update_xaxes(title_text="", row=1, col=1)
            fig.update_xaxes(title_text="Lag", row=1, col=2)
            fig.update_xaxes(title_text="Lag", row=2, col=1)
            fig.update_xaxes(title_text="Lag", row=2, col=2)
            
            fig.update_yaxes(title_text="ACF Value", row=1, col=1)
            fig.update_yaxes(title_text="Token", row=1, col=2)
            fig.update_yaxes(title_text="Mean ACF", row=2, col=1)
            fig.update_yaxes(title_text="# Significant", row=2, col=2)
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Category analysis if available
            if 'category_analysis' in results and results['category_analysis']:
                st.markdown('<h3 class="sub-header">📋 Category Analysis</h3>', unsafe_allow_html=True)
                
                category_stats = results['category_analysis']
                
                # Create category comparison
                categories = list(category_stats.keys())
                category_data = []
                
                for category in categories:
                    stats = category_stats[category]
                    category_data.append({
                        'Category': category,
                        'Token Count': stats['token_count'],
                        'Mean Max ACF Lag': stats['mean_max_acf_lag'],
                        'Mean Max ACF Value': stats['mean_max_acf_value'],
                        'Mean Decay Rate': stats['mean_decay_rate'],
                        'Mean Zero Crossing': stats['mean_first_zero_crossing']
                    })
                
                category_df = pd.DataFrame(category_data)
                st.dataframe(category_df, use_container_width=True)
                
                # Category comparison chart
                fig_cat = go.Figure()
                
                for metric in ['Mean Max ACF Lag', 'Mean Max ACF Value', 'Mean Decay Rate']:
                    fig_cat.add_trace(go.Bar(
                        name=metric,
                        x=category_df['Category'],
                        y=category_df[metric],
                        text=category_df[metric].round(3),
                        textposition='auto'
                    ))
                
                fig_cat.update_layout(
                    title="Category Comparison",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_cat, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Prediction Horizon Optimization</h2>', unsafe_allow_html=True)
        
        if st.session_state.distribution_results is not None:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                significance_threshold = st.slider(
                    "Significance Threshold",
                    min_value=0.05,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    help="Threshold for considering ACF values significant"
                )
                
                if st.button("🎯 Generate Horizon Recommendations", type="primary"):
                    with st.spinner("Generating recommendations..."):
                        try:
                            st.session_state.horizon_recommendations = st.session_state.analyzer.identify_optimal_prediction_horizons(
                                st.session_state.distribution_results,
                                significance_threshold
                            )
                            st.success("Recommendations generated!")
                        except Exception as e:
                            st.error(f"Error generating recommendations: {str(e)}")
            
            with col2:
                if st.session_state.horizon_recommendations is not None:
                    recommendations = st.session_state.horizon_recommendations
                    
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**⚙️ Analysis Summary**")
                    summary = recommendations['analysis_summary']
                    st.metric("Tokens Analyzed", summary['total_analyzed_tokens'])
                    st.metric("Strong Lags Found", summary['strong_lags_found'])
                    st.metric("Mean Optimal Lag", f"{summary['mean_optimal_lag']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Display recommendations
            if st.session_state.horizon_recommendations is not None and st.session_state.horizon_recommendations['success']:
                recommendations = st.session_state.horizon_recommendations
                
                # General recommendations with insights
                st.markdown('<h3 class="sub-header">🎯 Data-Driven Recommendations</h3>', unsafe_allow_html=True)
                
                general_rec = recommendations['general_recommendations']
                insights = general_rec.get('insights', {})
                
                # Display insights first
                if insights:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        persistence = insights.get('persistence', 'unknown')
                        color = 'green' if persistence == 'high' else 'orange' if persistence == 'medium' else 'red'
                        st.markdown(f"**Persistence:** <span style='color:{color}'>{persistence.title()}</span>", unsafe_allow_html=True)
                    
                    with col2:
                        window = insights.get('predictability_window', 'unknown')
                        st.markdown(f"**Predictability Window:** {window} minutes")
                    
                    with col3:
                        decay = insights.get('decay_type', 'unknown')
                        st.markdown(f"**Decay Type:** {decay.title()}")
                    
                    with col4:
                        feature_lags = insights.get('optimal_feature_lags', [])
                        st.markdown(f"**Key Feature Lags:** {len(feature_lags)} found")
                
                st.markdown("---")
                
                # Horizon recommendations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.markdown("**🔥 Short Term Horizon**")
                    st.metric("Recommended Lag", f"{general_rec['short_term_horizon']} minutes")
                    
                    # Add reasoning based on data
                    if insights.get('persistence') == 'high':
                        st.markdown("✅ High persistence detected - excellent for short-term predictions")
                    elif insights.get('persistence') == 'medium':
                        st.markdown("⚠️ Medium persistence - good for immediate predictions")
                    else:
                        st.markdown("❌ Low persistence - challenging for predictions")
                    
                    st.markdown("Best for: High-frequency trading, immediate predictions")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.markdown("**⚡ Medium Term Horizon**")
                    st.metric("Recommended Lag", f"{general_rec['medium_term_horizon']} minutes")
                    
                    # Add reasoning
                    decay_type = insights.get('decay_type', '')
                    if decay_type == 'slow':
                        st.markdown("✅ Slow decay - good medium-term predictability")
                    elif decay_type == 'moderate':
                        st.markdown("⚠️ Moderate decay - reasonable medium-term patterns")
                    else:
                        st.markdown("❌ Fast decay - limited medium-term predictability")
                    
                    st.markdown("Best for: Intraday trading, pattern recognition")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.markdown("**📈 Long Term Horizon**")
                    st.metric("Recommended Lag", f"{general_rec['long_term_horizon']} minutes")
                    
                    # Add reasoning
                    window = insights.get('predictability_window', 0)
                    if window > 20:
                        st.markdown("✅ Long predictability window - supports long-term models")
                    elif window > 10:
                        st.markdown("⚠️ Medium window - limited long-term predictability")
                    else:
                        st.markdown("❌ Short window - avoid long-term predictions")
                    
                    st.markdown("Best for: Trend following, position sizing")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Custom horizons
                if general_rec['custom_horizons']:
                    st.markdown('<h3 class="sub-header">🔧 Custom Horizons (Top 5)</h3>', unsafe_allow_html=True)
                    
                    custom_data = []
                    for i, horizon in enumerate(general_rec['custom_horizons']):
                        custom_data.append({
                            'Rank': i + 1,
                            'Lag (minutes)': horizon['lag'],
                            'Mean ACF': f"{horizon['mean_acf']:.3f}",
                            'Std ACF': f"{horizon['std_acf']:.3f}",
                            'Significant Tokens': horizon['significant_tokens'],
                            'Strength': 'High' if abs(horizon['mean_acf']) > 0.15 else 'Medium' if abs(horizon['mean_acf']) > 0.1 else 'Low'
                        })
                    
                    custom_df = pd.DataFrame(custom_data)
                    st.dataframe(custom_df, use_container_width=True)
                
                # Category-specific recommendations
                if 'category_recommendations' in recommendations and recommendations['category_recommendations']:
                    st.markdown('<h3 class="sub-header">📋 Category-Specific Recommendations</h3>', unsafe_allow_html=True)
                    
                    category_rec = recommendations['category_recommendations']
                    
                    for category, rec in category_rec.items():
                        st.markdown(f"**{category}**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Recommended Horizon", f"{rec['recommended_horizon']} minutes")
                        with col2:
                            st.metric("Confidence", f"{rec['confidence']:.3f}")
                        with col3:
                            st.metric("Decay Rate", f"{rec['decay_rate']:.3f}")
                
                # Export recommendations
                st.markdown('<h3 class="sub-header">💾 Export Results</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Export Distribution Results", type="secondary"):
                        try:
                            output_path = Path("time_series/results/autocorrelation_distributions.json")
                            st.session_state.analyzer.export_results(
                                st.session_state.distribution_results,
                                output_path
                            )
                            st.success(f"Distribution results exported to {output_path}")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                
                with col2:
                    if st.button("🎯 Export Horizon Recommendations", type="secondary"):
                        try:
                            output_path = Path("time_series/results/horizon_recommendations.json")
                            st.session_state.analyzer.export_results(
                                st.session_state.horizon_recommendations,
                                output_path
                            )
                            st.success(f"Recommendations exported to {output_path}")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                
                # Deep Learning Configuration
                st.markdown('<h3 class="sub-header">🤖 Deep Learning Configuration</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**💡 Recommended Model Configurations**")
                
                general_rec = recommendations['general_recommendations']
                
                config_text = f"""
                **LSTM/GRU Configuration:**
                ```python
                # Short-term model
                sequence_length = {general_rec['short_term_horizon']}
                prediction_horizon = 1
                
                # Medium-term model  
                sequence_length = {general_rec['medium_term_horizon']}
                prediction_horizon = 5
                
                # Long-term model
                sequence_length = {general_rec['long_term_horizon']}
                prediction_horizon = 15
                ```
                
                **Transformer Configuration:**
                ```python
                # Multi-horizon model
                context_length = {max(general_rec['short_term_horizon'], general_rec['medium_term_horizon'])}
                prediction_horizons = [{general_rec['short_term_horizon']}, {general_rec['medium_term_horizon']}, {general_rec['long_term_horizon']}]
                ```
                """
                
                st.markdown(config_text)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ Please run distribution analysis first to generate horizon recommendations.")

else:
    st.info("👆 Please load token data using the sidebar to begin analysis.")
    
    # Instructions
    st.markdown("""
    ## 📚 Instructions
    
    1. **Configure Settings**: Use the sidebar to set the processed data directory and analysis parameters
    2. **Load Data**: Click "Load Token Data" to load tokens from the processed categories
    3. **Individual Analysis**: Analyze specific tokens to understand their autocorrelation patterns
    4. **Distribution Analysis**: Compute autocorrelation distributions across all tokens
    5. **Horizon Optimization**: Generate recommendations for optimal prediction horizons
    
    ## 🎯 Purpose
    
    This tool helps determine optimal prediction horizons for deep learning models by analyzing autocorrelation patterns in memecoin time series data. The recommendations can be directly used to configure LSTM, GRU, or Transformer models for time series prediction.
    
    ## 📊 Analysis Types
    
    - **Returns**: Percentage price changes (most stable, recommended first)
    - **Log Returns**: Logarithmic returns (handles extreme volatility better)
    - **Prices**: Raw price values (shows trend autocorrelation)
    - **Log Prices**: Log of price values (reveals multiplicative patterns)
    """)
    
    # Add autocorrelation explanation
    with st.expander("🎓 Understanding Autocorrelation - A Technical Deep Dive"):
        st.markdown("""
        ## What is Autocorrelation?
        
        **Autocorrelation** (also called serial correlation) measures how correlated a time series is with itself at different time lags. It's essentially the correlation between observations at different distances apart.
        
        ### Mathematical Definition
        
        For a time series $X_t$, the autocorrelation at lag $k$ is:
        
        $$\\rho_k = \\frac{\\text{Cov}(X_t, X_{t-k})}{\\text{Var}(X_t)}$$
        
        Where:
        - $\\rho_k$ is the autocorrelation at lag $k$
        - $X_t$ is the value at time $t$
        - $X_{t-k}$ is the value at time $t-k$
        
        ### Interpretation
        
        - **ACF = 1**: Perfect positive correlation (identical patterns repeat)
        - **ACF = 0**: No correlation (past values don't predict future)
        - **ACF = -1**: Perfect negative correlation (patterns invert)
        
        ### Why It Matters for Deep Learning
        
        1. **Feature Engineering**: Identifies which past lags contain predictive information
        2. **Model Architecture**: Determines optimal sequence lengths for RNNs/LSTMs
        3. **Prediction Horizons**: Shows how far ahead we can reliably predict
        4. **Pattern Detection**: Reveals hidden periodicities and cycles
        
        ### Example Patterns
        
        **High Autocorrelation (ACF > 0.5)**:
        - Strong momentum/trending behavior
        - Past values strongly predict future
        - Good for trend-following strategies
        
        **Low Autocorrelation (ACF ≈ 0)**:
        - Random walk behavior
        - Difficult to predict
        - May need external features
        
        **Negative Autocorrelation (ACF < 0)**:
        - Mean-reverting behavior
        - Oscillating patterns
        - Good for contrarian strategies
        
        ### ACF vs PACF
        
        - **ACF (Autocorrelation Function)**: Shows total correlation including indirect effects
        - **PACF (Partial ACF)**: Shows direct correlation, removing indirect effects through intermediate lags
        
        ### For Memecoin Analysis
        
        Memecoins often show:
        - **Quick decay**: Autocorrelation drops rapidly (unpredictable)
        - **Volatility clustering**: High ACF in squared returns (GARCH effects)
        - **Pump patterns**: Temporary high ACF during pump phases
        
        ### Practical Usage
        
        ```python
        # If ACF at lag 5 = 0.3, it means:
        # "The price 5 minutes ago explains 30% of the variance in current price"
        
        # For LSTM configuration:
        if max_significant_lag == 15:
            sequence_length = 15  # Use past 15 time steps
            prediction_horizon = 1  # Predict 1 step ahead
        ```
        
        ### Statistical Significance
        
        The confidence bands show which autocorrelations are statistically significant:
        - Values outside the bands = significant correlation
        - Values inside the bands = could be random noise
        
        ### Common Pitfalls
        
        1. **Non-stationarity**: Trends can create spurious autocorrelation
        2. **Outliers**: Extreme values can distort ACF
        3. **Short series**: Need sufficient data for reliable estimates
        
        ### Deep Learning Applications
        
        1. **Sequence Length Selection**:
           - Use lags where ACF is significant
           - Typically up to first zero crossing
        
        2. **Multi-Horizon Prediction**:
           - Different models for different horizons
           - Based on ACF decay pattern
        
        3. **Feature Lags**:
           - Include lags with high PACF values
           - These have direct predictive power
        """)
    
    with st.expander("🔍 Quick ACF Interpretation Guide"):
        st.markdown("""
        ### Reading ACF Plots
        
        | Pattern | Interpretation | Action |
        |---------|---------------|--------|
        | Slow decay | Strong trend/momentum | Use longer sequences |
        | Fast decay | Weak memory | Short sequences sufficient |
        | Oscillating | Cyclical patterns | Capture full cycle |
        | All near zero | Random/unpredictable | Need external features |
        | Spikes at specific lags | Periodic patterns | Include those specific lags |
        
        ### For Your Models
        
        - **LSTM/GRU**: Set `sequence_length` to where ACF becomes insignificant
        - **CNN**: Use kernel sizes that match significant lag patterns
        - **Transformer**: Set `context_length` to capture all significant correlations
        - **Ensemble**: Different models for different ACF patterns in your data
        """)

# Footer
st.markdown("---")
st.markdown("🤖 **Autocorrelation Analysis Tool** | Built for Deep Learning Model Optimization")