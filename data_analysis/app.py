"""
Streamlit app for memecoin data analysis
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit_utils.formatting import format_large_number, format_file_count, format_data_points, format_percentage, format_currency
from streamlit_utils.components import DataSourceManager, TokenSelector, NavigationManager
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import shutil
import plotly.express as px
import os

from data_analysis.data_loader import DataLoader
from data_analysis.data_quality import DataQualityAnalyzer
from data_analysis.price_analysis import PriceAnalyzer
from data_analysis.export_utils import export_parquet_files
from data_cleaning.clean_tokens import CategoryAwareTokenCleaner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("Memecoin Data Analysis Dashboard")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = None
    if 'quality_analyzer' not in st.session_state:
        st.session_state.quality_analyzer = None
    if 'price_analyzer' not in st.session_state:
        st.session_state.price_analyzer = None
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = None
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Initialize shared components
    data_source_manager = DataSourceManager()
    navigation_manager = NavigationManager()
    
    # Data source selection (before loading)
    if not st.session_state.data_loaded:
        result = data_source_manager.render_data_source_selection()
        if result:
            selected_subfolder, file_count = result
            if data_source_manager.render_load_button(selected_subfolder, data_loader_class=DataLoader):
                return
        else:
            return
    
    # Sidebar navigation after loading
    page = st.sidebar.radio("Go to", ["Data Quality", "Price Analysis", "Pattern Detection", "Price Distribution", "Variability Analysis"])
    
    # Common navigation controls
    navigation_manager.render_common_sidebar_controls([
        'quality_analyzer', 
        'price_analyzer'
    ])
        
    # Token selection using shared component (only after data is loaded)
    if st.session_state.data_loader is not None:
        token_selector = TokenSelector(st.session_state.data_loader, key_prefix="dq_")
        selected_token_dicts = token_selector.render_token_selection()
        token_selector.display_selection_summary(selected_token_dicts)
        selected_tokens = [t['symbol'] for t in selected_token_dicts]
    else:
        st.sidebar.error("Data loader not initialized. Please reload the page.")
        selected_tokens = []
    # Main content based on selected page
    if page == "Data Quality":
        show_data_quality(selected_tokens)
    elif page == "Price Analysis":
        show_price_analysis(selected_tokens)
    elif page == "Pattern Detection":
        show_pattern_detection(selected_tokens)
    elif page == "Price Distribution":
        show_price_distribution(selected_tokens)
    elif page == "Variability Analysis":
        show_variability_analysis(selected_tokens)

def show_data_quality(selected_tokens):
    """Display data quality analysis"""
    st.header("Data Quality Analysis")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return

    # Analyze selected tokens
    quality_reports = {}
    for token in selected_tokens:
        df = st.session_state.data_loader.get_token_data(token)
        if df is not None and not df.is_empty():
            if st.session_state.quality_analyzer is None:
                st.session_state.quality_analyzer = DataQualityAnalyzer()
            quality_reports[token] = st.session_state.quality_analyzer.analyze_single_file(df, token)
            
    if quality_reports:
        # Display quality summary
        st.session_state.quality_analyzer.display_quality_summary(quality_reports)
        
        # Count and display the number of dead tokens
        dead_tokens_count = sum(1 for report in quality_reports.values() if report['is_dead'])
        st.subheader(f"Number of Dead Tokens: {dead_tokens_count}")
        
        # Calculate and display dead tokens per hour for multiple tokens
        if len(selected_tokens) > 1:
            # Group tokens by the duration of their "death"
            death_durations = [report.get('death_duration_hours', 0) for report in quality_reports.values()]
            duration_counts = Counter(death_durations)
            
            # Filter out tokens that are not dead (duration > 0)
            dead_duration_counts = {duration: count for duration, count in duration_counts.items() if duration > 0}

            st.subheader("Dead Token Analysis: At Which Hour Did They Die?")
            
            if dead_duration_counts:
                # Prepare data for plotting
                durations = sorted(dead_duration_counts.keys())
                counts = [dead_duration_counts[d] for d in durations]

                fig = go.Figure(data=[go.Bar(
                    x=durations,
                    y=counts,
                    text=counts,
                    textposition='auto',
                )])
                fig.update_layout(
                    title_text="Distribution of Token Deaths by Hour",
                    xaxis_title="Hour at Which Price Became Constant",
                    yaxis_title="Number of Tokens",
                    xaxis=dict(tickmode='linear', dtick=1)
                )
                st.plotly_chart(fig)
            else:
                st.info("No tokens with a constant price at the end were found.")
        
        # List of dead tokens
        dead_tokens = [token for token, report in quality_reports.items() if report['is_dead']]
        if dead_tokens:
            st.subheader("Dead Tokens")
            dead_tokens_df = pl.DataFrame({'Token': dead_tokens})
            st.dataframe(dead_tokens_df)
            if st.button("Download Dead Tokens List as CSV", key="download_dead_csv"):
                st.download_button(
                    label="Download as CSV",
                    data=dead_tokens_df.to_csv(),
                    file_name="dead_tokens.csv",
                    mime="text/csv",
                    key="download_dead_csv_file"
                )
            if st.button("Export Dead Token Parquet Files to processed/", key="export_dead_parquet"):
                st.info('Export button clicked! (Dead Tokens)')
                export_parquet_files(dead_tokens, "Dead Tokens")
        
        # Add new "Export All Categories" button at the top
        st.subheader("🚀 Smart Category Export (Mutually Exclusive)")
        st.info("📊 **NEW**: Export all categories with strict hierarchy - each token appears in EXACTLY ONE category!")
        
        # Debug gap detection button
        if st.button("🔍 Debug Gap Detection", key="debug_gaps", help="Analyze why tokens with gaps might not be detected"):
            debug_results = st.session_state.quality_analyzer.debug_gap_detection(quality_reports, min_gap_size=10.0)
            
            # Show debug results in the app
            if debug_results['tokens_with_large_gaps']:
                st.subheader("Debug Results: Tokens with Large Gaps")
                debug_df_data = []
                for token, info in debug_results['tokens_with_large_gaps'].items():
                    debug_df_data.append({
                        'Token': token,
                        'Max Gap (min)': f"{info['max_gap_minutes']:.1f}",
                        'Total Gaps': info['total_gaps'],
                        'Large Gaps (≥10min)': info['large_gaps_count'],
                        'Quality Score': f"{info['quality_score']:.1f}",
                        'Meets Current Threshold (>5 gaps)': '✅' if info['total_gaps'] > 5 else '❌'
                    })
                
                debug_df = pl.DataFrame(debug_df_data)
                st.dataframe(debug_df, use_container_width=True)
                
                # Show threshold analysis
                current_threshold_count = debug_results['current_threshold_count']
                large_gaps_count = len(debug_results['tokens_with_large_gaps'])
                
                st.info(f"""
                **Analysis Summary:**
                - Tokens with gaps ≥10 minutes: **{large_gaps_count}**
                - Tokens meeting current threshold (>5 total gaps): **{current_threshold_count}**
                - Your expected ~20 tokens with 61-minute gaps should appear above
                """)
                
                # Suggest threshold adjustment if needed
                if large_gaps_count > current_threshold_count:
                    st.warning(f"""
                    **Potential Issue**: {large_gaps_count} tokens have large gaps (≥10min) but only {current_threshold_count} meet the current threshold (>5 total gaps).
                    
                    Consider adjusting the threshold or gap detection criteria.
                    """)
            else:
                st.warning("No tokens with gaps ≥10 minutes found. This might indicate an issue with gap detection.")
        
        # NEW: Investigate tokens with gaps button
        if st.button("🔬 Investigate Tokens with Gaps", key="investigate_gaps", help="Comprehensive analysis of tokens with gaps to decide keep vs remove"):
            with st.spinner("Investigating tokens with gaps..."):
                investigation_results = st.session_state.quality_analyzer.investigate_tokens_with_gaps(quality_reports)
                
                # Display results in Streamlit
                st.subheader("🔬 Gap Investigation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tokens Analyzed", investigation_results['tokens_analyzed'])
                    
                with col2:
                    keep_count = len(investigation_results['recommendations']['keep_and_clean'])
                    st.metric("Keep & Clean", keep_count, delta="Recommended")
                    
                with col3:
                    remove_count = len(investigation_results['recommendations']['remove_completely'])
                    st.metric("Remove Completely", remove_count, delta="Exclude")
                
                # Recommendations breakdown
                st.subheader("📋 Recommendations Breakdown")
                
                for action, tokens in investigation_results['recommendations'].items():
                    if tokens:
                        action_name = action.replace('_', ' ').title()
                        
                        if action == 'keep_and_clean':
                            st.success(f"✅ **{action_name}** ({len(tokens)} tokens)")
                            st.write("These tokens have minor gaps that can be filled effectively")
                            with st.expander(f"View {action_name} tokens"):
                                st.write(", ".join(tokens))
                                
                        elif action == 'remove_completely':
                            st.error(f"❌ **{action_name}** ({len(tokens)} tokens)")
                            st.write("These tokens have too many/large gaps for reliable analysis")
                            with st.expander(f"View {action_name} tokens"):
                                st.write(", ".join(tokens))
                                
                        elif action == 'needs_manual_review':
                            st.warning(f"🤔 **{action_name}** ({len(tokens)} tokens)")
                            st.write("These tokens are borderline cases - examine individually")
                            with st.expander(f"View {action_name} tokens"):
                                st.write(", ".join(tokens))
                
                # Gap severity analysis
                st.subheader("📊 Gap Severity Analysis")
                severity_data = []
                for severity, tokens in investigation_results['gap_analysis'].items():
                    severity_data.append({
                        'Severity': severity.replace('_', ' ').title(),
                        'Count': len(tokens),
                        'Tokens': ', '.join(tokens[:3]) + ('...' if len(tokens) > 3 else '')
                    })
                
                if severity_data:
                    st.dataframe(severity_data, use_container_width=True)
                
                st.info("💡 **Next Steps**: Check the terminal output for detailed recommendations and then proceed with data cleaning or token removal as suggested.")
        
        if st.button("🔄 Export All Categories (Mutually Exclusive)", key="export_all_categories"):
            st.info('Exporting all categories with mutual exclusivity enforcement...')
            try:
                exported_results = st.session_state.quality_analyzer.export_all_categories_mutually_exclusive(quality_reports)
                
                # Display results
                total_exported = sum(len(tokens) for tokens in exported_results.values())
                st.success(f'✅ Successfully exported {format_large_number(total_exported)} tokens across all categories!')
                
                # Show breakdown
                st.subheader("Export Summary")
                for category, tokens in exported_results.items():
                    if tokens:
                        st.write(f"✅ **{category}**: {format_large_number(len(tokens))} tokens exported")
                    else:
                        st.write(f"⚠️ **{category}**: No tokens found")
                        
                st.info("💡 **No overlaps**: Each token now appears in exactly one category based on hierarchy: gaps > normal > extremes > dead")
                
            except Exception as e:
                st.error(f"Export failed: {e}")
        
        st.divider()
        
        # List of tokens with gaps (for information only, not export)
        tokens_with_gaps = [token for token, report in quality_reports.items() if report['gaps']['total_gaps'] > 0]
        if tokens_with_gaps:
            st.subheader("Tokens with Gaps (Analysis)")
            st.caption("Note: Use 'Export All Categories' above for mutually exclusive exports")
            
            # Calculate gap statistics (only from tokens that have gaps)
            all_gaps = []
            gap_counts = []
            gap_table_data = []
            
            for token, report in quality_reports.items():
                if report['gaps']['total_gaps'] > 0:
                    gap_details = report['gaps']['gap_details']
                    all_gaps.extend([gap['size_minutes'] for gap in gap_details])
                    gap_counts.append(report['gaps']['total_gaps'])
                    
                    # Prepare data for detailed gap table
                    for i, gap in enumerate(gap_details):
                        gap_table_data.append({
                            'Token': token,
                            'Gap #': i + 1,
                            'Start (min after launch)': f"{gap.get('start_minutes_after_launch', 0):.1f}",
                            'Gap Size (minutes)': f"{gap.get('size_minutes', 0):.1f}",
                            'Interpolation Type': gap.get('interpolation_type', 'N/A'),
                            'Start Time': gap.get('start_time', 'N/A')
                        })
            
            if all_gaps:
                # Display gap statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tokens with Gaps", len(tokens_with_gaps))
                with col2:
                    st.metric("Min Gap (minutes)", f"{min(all_gaps):.1f}")
                with col3:
                    st.metric("Max Gap (minutes)", f"{max(all_gaps):.1f}")
                with col4:
                    st.metric("Avg Gap (minutes)", f"{sum(all_gaps)/len(all_gaps):.1f}")
                
                # Additional statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Gaps", sum(gap_counts))
                with col2:
                    import statistics
                    st.metric("Median Gap (minutes)", f"{statistics.median(all_gaps):.1f}")
                with col3:
                    st.metric("Avg Gaps per Token", f"{sum(gap_counts)/len(gap_counts):.1f}")
                with col4:
                    st.metric("Max Gaps in Token", max(gap_counts))
            
            # Enhanced gap table with detailed information
            if gap_table_data:
                st.subheader("Detailed Gap Information")
                gap_details_df = pl.DataFrame(gap_table_data)
                st.dataframe(gap_details_df, use_container_width=True, height=400)
                
                if len(gap_table_data) > 20:
                    st.info(f"Showing all {len(gap_table_data)} gaps from {len(tokens_with_gaps)} tokens with gaps.")
        
        # List of tokens with extreme movements (for information only)
        tokens_with_extremes = [token for token, report in quality_reports.items() if report.get('is_extreme_token', False)]
        if tokens_with_extremes:
            st.subheader("Tokens with Extreme Movements (Analysis)")
            st.caption("Note: Use 'Export All Categories' above for mutually exclusive exports")
            st.write(f"Found {len(tokens_with_extremes)} tokens with extreme price movements (>1M% returns or >10k% minute jumps)")
            tokens_with_extremes_df = pl.DataFrame({'Token': tokens_with_extremes})
            st.dataframe(tokens_with_extremes_df)
        
        # Launch context analysis
        st.subheader("Launch Context Analysis")
        
        # Get market eras and count them
        market_eras = [report['launch_context'].get('market_era', 'unknown') 
                      for report in quality_reports.values()]
        era_counts = Counter(market_eras)
        
        # Plot distribution
        fig = go.Figure(data=[go.Bar(
            x=list(era_counts.keys()),
            y=list(era_counts.values())
        )])
        fig.update_layout(title="Token Distribution by Market Era",
                         xaxis_title="Market Era",
                         yaxis_title="Number of Tokens")
        st.plotly_chart(fig)

        # Normal behavior tokens (for information only)
        normal_behavior_tokens = []
        for token, report in quality_reports.items():
            is_dead = report.get('is_dead', False)
            is_extreme = report.get('is_extreme_token', False)
            # UPDATED: Check for significant gaps (many gaps OR large gaps)
            total_gaps = report.get('gaps', {}).get('total_gaps', 0)
            max_gap = report.get('gaps', {}).get('max_gap', 0)
            has_many_gaps = total_gaps > 5  # More than 5 gaps
            has_large_gap = max_gap > 30    # Any gap larger than 30 minutes
            has_significant_gaps = has_many_gaps or has_large_gap
            
            # Only include tokens with normal behavior (not dead, not extreme, minimal gaps)
            if not (is_dead or is_extreme or has_significant_gaps):
                normal_behavior_tokens.append(token)
        
        if normal_behavior_tokens:
            st.subheader("Normal Behavior Tokens (Analysis)")
            st.caption("Note: Use 'Export All Categories' above for mutually exclusive exports")
            st.write(f"Found {len(normal_behavior_tokens)} tokens with normal behavior (not dead, not extreme, minimal gaps)")
            normal_behavior_df = pl.DataFrame({'Token': normal_behavior_tokens})
            st.dataframe(normal_behavior_df)
        else:
            st.info("No tokens with normal behavior found in this sample.")
    else:
        st.warning("No data available for the selected tokens")
        
def show_price_analysis(selected_tokens):
    st.title("Price Analysis")
    if not selected_tokens:
        st.warning("Please select at least one token")
        return

    # Initialize analyzer if not in session state
    if st.session_state.price_analyzer is None:
        st.session_state.price_analyzer = PriceAnalyzer()

    # Single or multiple token analysis
    if len(selected_tokens) == 1:
        token = selected_tokens[0]
        df = st.session_state.data_loader.get_token_data(token)
        if df is not None and not df.is_empty():
            price_metrics = st.session_state.price_analyzer.analyze_prices(df, token)
            st.session_state.price_analyzer.display_price_metrics(price_metrics)
            st.session_state.price_analyzer.display_patterns(price_metrics.get('patterns', {}))
        else:
            st.warning(f"No data for {token}")
    else:
        st.subheader("Selected Tokens Analysis")
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_metrics = {}
        for i, token in enumerate(selected_tokens):
            try:
                status_text.text(f"Analyzing {token}...")
                df = st.session_state.data_loader.get_token_data(token)
                if df is not None and not df.is_empty():
                    metrics = st.session_state.price_analyzer.analyze_prices(df, token)
                    all_metrics[token] = metrics
            except Exception as e:
                st.warning(f"Error analyzing {token}: {str(e)}")
            progress_bar.progress((i + 1) / len(selected_tokens))
        status_text.text("Analysis complete!")
        if all_metrics:
            st.session_state.price_analyzer.display_aggregated_metrics(all_metrics)
        else:
            st.warning("No tokens were successfully analyzed")

def show_pattern_detection(selected_tokens):
    st.header("Pattern Detection")
    if not selected_tokens:
        st.warning("Please select at least one token")
        return
    
    if st.session_state.price_analyzer is None:
        st.session_state.price_analyzer = PriceAnalyzer()

    for token in selected_tokens:
        st.subheader(f"Patterns for {token}")
        df = st.session_state.data_loader.get_token_data(token)
        if df is not None and not df.is_empty():
            try:
                metrics = st.session_state.price_analyzer.analyze_prices(df, token)
                st.session_state.price_analyzer.display_patterns(metrics)
            except Exception as e:
                st.error(f"Error analyzing patterns for {token}: {str(e)}")

def show_price_distribution(selected_tokens):
    st.header("Price Distribution with Gap Analysis")
    if not selected_tokens:
        st.info("No tokens selected.")
        return
    
    for token in selected_tokens:
        df = st.session_state.data_loader.get_token_data(token)
        if df is None or df.is_empty():
            st.warning(f"No data for {token}")
            continue
        
        st.subheader(f"Analysis for {token}")
        
        # Analyze gaps first
        if st.session_state.quality_analyzer is None:
            st.session_state.quality_analyzer = DataQualityAnalyzer()
        
        quality_report = st.session_state.quality_analyzer.analyze_single_file(df, token)
        gaps_info = quality_report.get('gaps', {})
        
        # Display gap information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Gaps", gaps_info.get('total_gaps', 0))
        with col2:
            st.metric("Max Gap (min)", f"{gaps_info.get('max_gap', 0):.1f}")
        with col3:
            st.metric("Avg Gap (min)", f"{gaps_info.get('avg_gap', 0):.1f}")
        
        # Keep all operations in Polars
        df = df.filter(pl.col('price') > 0)
        if df.height == 0:
            st.warning(f"No positive prices for {token}")
            continue

        df = df.with_columns(
            pl.col('price').log().alias('log_price')
        ).filter(
            pl.col('log_price').is_finite()
        )
        
        if df.height == 0:
            st.warning(f"No finite log_price values for {token}")
            continue

        # Sort and remove duplicates
        df = df.sort('datetime').unique(subset='datetime', keep='first')
        
        # Create single plot with integrated gap visualization
        fig = go.Figure()
        
        # Convert to pandas for easier plotting
        df_pd = df.to_pandas()
        
        # Get price range for gap visualization
        min_log_price = df_pd['log_price'].min()
        max_log_price = df_pd['log_price'].max()
        
        # Add gap zones directly to the price chart
        if gaps_info.get('total_gaps', 0) > 0:
            gap_details = gaps_info.get('gap_details', [])
            
            if gap_details:
                for i, gap in enumerate(gap_details):
                    gap_start_time = gap.get('start_time')
                    gap_size_minutes = gap.get('size_minutes', 0)
                    
                    if gap_start_time and gap_size_minutes > 0:
                        # Calculate gap end time
                        gap_end_time = gap_start_time + timedelta(minutes=gap_size_minutes)
                        
                        # Add vertical gap zone spanning the entire price range
                        fig.add_shape(
                            type="rect",
                            x0=gap_start_time,
                            x1=gap_end_time,
                            y0=min_log_price,
                            y1=max_log_price,
                            fillcolor="orange",
                            opacity=0.3,
                            line=dict(color="red", width=1),
                            layer="below"  # Put gaps behind the price line
                        )
                        
                        # Add gap annotation
                        fig.add_annotation(
                            x=gap_start_time + timedelta(minutes=gap_size_minutes/2),
                            y=max_log_price - (max_log_price - min_log_price) * 0.1,
                            text=f"Gap: {gap_size_minutes:.1f}min",
                            showarrow=False,
                            font=dict(size=10, color="red"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="red",
                            borderwidth=1
                        )
        
        # Plot price line on top of gap zones
        fig.add_trace(
            go.Scatter(
                x=df_pd['datetime'], 
                y=df_pd['log_price'],
                mode='lines',
                name='Log Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            title_text=f"Price Distribution with Gap Analysis for {token}",
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Log(Price)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display gap details if any exist
        if gaps_info.get('total_gaps', 0) > 0:
            st.subheader("Gap Details")
            gap_details = gaps_info.get('gap_details', [])
            if gap_details:
                gap_data = []
                for i, gap in enumerate(gap_details[:10]):  # Show first 10 gaps
                    gap_data.append({
                        'Gap #': i + 1,
                        'Position': gap.get('position', 'N/A'),
                        'Start (min after launch)': f"{gap.get('start_minutes_after_launch', 0):.1f}",
                        'Size (minutes)': f"{gap.get('size_minutes', 0):.1f}",
                        'Start Time': gap.get('start_time', 'N/A'),
                        'Interpolation Type': gap.get('interpolation_type', 'N/A')
                    })
                
                gap_df = pl.DataFrame(gap_data)
                st.dataframe(gap_df, use_container_width=True)
                
                if len(gap_details) > 10:
                    st.info(f"Showing first 10 gaps out of {len(gap_details)} total gaps.")
        
        st.divider()

def show_variability_analysis(selected_tokens):
    """Display token variability analysis to distinguish real variations from straight-line patterns"""
    st.header("Token Variability Analysis")
    st.info("🔍 Analyze price patterns to distinguish real market variations from 'straight line' tokens")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Initialize cleaner for variability analysis
    if 'token_cleaner' not in st.session_state:
        st.session_state.token_cleaner = CategoryAwareTokenCleaner()
    
    # Analysis mode selection
    analysis_mode = st.radio(
        "Analysis Mode:",
        ["Individual Token Analysis", "Batch Comparison", "Distribution Analysis"],
        help="Choose how to analyze token variability"
    )
    
    if analysis_mode == "Individual Token Analysis":
        show_individual_variability_analysis(selected_tokens)
    elif analysis_mode == "Batch Comparison":
        show_batch_variability_comparison(selected_tokens)
    elif analysis_mode == "Distribution Analysis":
        show_variability_distribution_analysis(selected_tokens)

def show_individual_variability_analysis(selected_tokens):
    """Detailed analysis of individual tokens"""
    st.subheader("Individual Token Variability Analysis")
    
    for token in selected_tokens:
        df = st.session_state.data_loader.get_token_data(token)
        if df is None or df.is_empty():
            st.warning(f"No data for {token}")
            continue
            
        st.markdown(f"### 📊 Analysis for **{token}**")
        
        # Add returns if missing
        if 'returns' not in df.columns:
            df = st.session_state.token_cleaner._calculate_returns(df)
        
        # Get variability analysis
        try:
            result_df, modifications = st.session_state.token_cleaner._check_price_variability_graduated(df, token, "medium_term")
            
            # Extract metrics from modifications
            metrics = None
            for mod in modifications:
                if 'metrics' in mod:
                    metrics = mod['metrics']
                    break
            
            if not metrics:
                st.error(f"Could not analyze variability for {token}")
                continue
                
            # Ensure all required metrics exist with default values
            required_metrics = ['price_cv', 'log_price_cv', 'flat_periods_fraction', 'range_efficiency', 'normalized_entropy']
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    st.warning(f"Missing metric '{metric_name}' for {token}, using default value 0.0")
                    metrics[metric_name] = 0.0
            
            # Ensure new metric keys exist
            if 'max_flat_minutes' not in metrics:
                metrics['max_flat_minutes'] = 0
            if 'tick_frequency' not in metrics:
                metrics['tick_frequency'] = metrics.get('change_ratio', 0)
            
            # Add is_low_variability key if missing (use is_straight_line as fallback)
            if 'is_low_variability' not in metrics:
                metrics['is_low_variability'] = metrics.get('is_straight_line', False)
                
            # Display enhanced metrics in columns with new thresholds
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                recent_cv = metrics.get('price_cv', 0.0)  # Now shows recent CV
                recent_cv_pass = metrics.get('recent_cv_pass', False)
                cv_status = "🟢" if recent_cv_pass else "🔴"
                st.metric("Recent Activity CV", f"{recent_cv:.4f}", help="Recent price variability (last 50% of data)")
                st.write(f"{cv_status} Threshold: > 0.02")
            
            with col2:
                log_price_cv = metrics.get('log_price_cv', 0.0)
                log_cv_status = "🟢" if log_price_cv > 0.05 else "🔴"
                st.metric("Log Price CV", f"{log_price_cv:.4f}", help="Stable variability measure")
                st.write(f"{log_cv_status} Threshold: > 0.05")
            
            with col3:
                activity_dist = 1 - metrics.get('flat_periods_fraction', 0.0)  # Invert back
                activity_dist_pass = metrics.get('activity_dist_pass', False)
                activity_status = "🟢" if activity_dist_pass else "🔴"
                st.metric("Activity Distribution", f"{activity_dist:.3f}", help="How evenly spread price changes are over time")
                st.write(f"{activity_status} Threshold: > 0.3")
            
            with col4:
                movement_eff = metrics.get('range_efficiency', 0.0)  # Now shows movement efficiency
                movement_eff_pass = metrics.get('movement_eff_pass', False)
                movement_status = "🟢" if movement_eff_pass else "🔴"
                st.metric("Movement Efficiency", f"{movement_eff:.3f}", help="Price movement vs activity level")
                st.write(f"{movement_status} Threshold: > 0.1")
            
            with col5:
                pattern_complex = metrics.get('normalized_entropy', 0.0)  # Now shows pattern complexity
                pattern_complex_pass = metrics.get('pattern_complex_pass', False)
                complexity_status = "🟢" if pattern_complex_pass else "🔴"
                st.metric("Pattern Complexity", f"{pattern_complex:.3f}", help="Unpredictability of price patterns")
                st.write(f"{complexity_status} Threshold: > 0.2")
            
            # Overall decision with explanation
            is_filtered = metrics.get('is_low_variability', False)
            passes_count = metrics.get('passes_count', 0)
            recent_cv_pass = metrics.get('recent_cv_pass', False)
            
            if is_filtered:
                decision = "🔴 FILTERED (Low Variability)"
                explanation = f"Passes {passes_count}/5 criteria. Recent Activity CV: {'✓ Pass' if recent_cv_pass else '✗ Fail'}"
            else:
                decision = "🟢 PASSED"
                explanation = f"Passes {passes_count}/5 criteria with required Recent Activity CV ✓"
            
            st.markdown(f"**Final Decision:** {decision}")
            st.markdown(f"**Criteria:** {explanation}")
            st.info("**Filtering Rule:** Must pass Recent Activity CV + at least 3 out of 5 total criteria")
            
            # Price plot
            prices = df['price'].to_numpy()
            returns = df['returns'].to_numpy()
            
            # Create price plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Raw Price', 'Log Price', 'Returns Over Time', 'Rolling CV'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Raw price plot
            fig.add_trace(
                go.Scatter(y=prices, mode='lines', name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Log price plot (if all prices > 0)
            if np.all(prices > 0):
                log_prices = np.log(prices)
                fig.add_trace(
                    go.Scatter(y=log_prices, mode='lines', name='Log Price', line=dict(color='green')),
                    row=1, col=2
                )
            else:
                fig.add_trace(
                    go.Scatter(y=prices, mode='lines', name='Price (Linear)', line=dict(color='green')),
                    row=1, col=2
                )
            
            # Returns over time (time series)
            datetime_col = df['datetime'].to_numpy()
            finite_mask = np.isfinite(returns)
            finite_returns = returns[finite_mask]
            finite_datetime = datetime_col[1:][finite_mask[1:]]  # Skip first return (NaN)
            
            fig.add_trace(
                go.Scatter(x=finite_datetime, y=finite_returns, mode='markers+lines', 
                          name='Returns Over Time', line=dict(color='orange'), 
                          marker=dict(size=3)),
                row=2, col=1
            )
            
            # Rolling CV
            window_size = min(60, len(prices) // 4)
            if window_size >= 5:
                rolling_cv = []
                x_positions = []
                
                for i in range(0, len(prices) - window_size + 1, window_size // 2):
                    window_prices = prices[i:i + window_size]
                    cv = np.std(window_prices) / np.mean(window_prices) if np.mean(window_prices) > 0 else 0
                    rolling_cv.append(cv)
                    x_positions.append(i + window_size // 2)
                
                fig.add_trace(
                    go.Scatter(x=x_positions, y=rolling_cv, mode='lines+markers', 
                             name='Rolling CV', line=dict(color='purple')),
                    row=2, col=2
                )
                
                # Add threshold line
                fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                             annotation_text="Filter threshold", row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False, title_text=f"Variability Analysis: {token}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic statistics
            with st.expander("📈 Detailed Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Price Statistics:**")
                    st.write(f"• Length: {len(prices)} data points")
                    st.write(f"• Min Price: {np.min(prices):.2e}")
                    st.write(f"• Max Price: {np.max(prices):.2e}")
                    st.write(f"• Price Ratio: {np.max(prices)/np.min(prices):.1f}x")
                    st.write(f"• Mean Price: {np.mean(prices):.2e}")
                
                with col2:
                    st.write("**Returns Statistics:**")
                    finite_returns = returns[np.isfinite(returns)]
                    st.write(f"• Max |Return|: {np.max(np.abs(finite_returns)):.6f}")
                    st.write(f"• Mean |Return|: {np.mean(np.abs(finite_returns)):.6f}")
                    st.write(f"• Std Returns: {np.std(finite_returns):.6f}")
                    st.write(f"• Finite Returns: {len(finite_returns)}/{len(returns)}")
            
        except Exception as e:
            st.error(f"Error analyzing {token}: {e}")
        
        st.divider()

def show_batch_variability_comparison(selected_tokens):
    """Compare variability metrics across multiple tokens"""
    st.subheader("Batch Variability Comparison")
    
    if len(selected_tokens) < 2:
        st.warning("Please select at least 2 tokens for comparison.")
        return
    
    # Analyze all tokens
    results = []
    progress_bar = st.progress(0)
    
    for i, token in enumerate(selected_tokens):
        df = st.session_state.data_loader.get_token_data(token)
        if df is not None and not df.is_empty():
            # Add returns if missing
            if 'returns' not in df.columns:
                df = st.session_state.token_cleaner._calculate_returns(df)
            
            try:
                result_df, modifications = st.session_state.token_cleaner._check_price_variability_graduated(df, token, "medium_term")
                
                # Extract metrics
                metrics = None
                for mod in modifications:
                    if 'metrics' in mod:
                        metrics = mod['metrics'].copy()
                        break
                        
                if metrics:
                    # Ensure all required metrics exist with default values
                    required_metrics = ['price_cv', 'log_price_cv', 'flat_periods_fraction', 'range_efficiency', 'normalized_entropy']
                    for metric_name in required_metrics:
                        if metric_name not in metrics:
                            metrics[metric_name] = 0.0
                    
                    # Ensure new metric keys exist
                    if 'max_flat_minutes' not in metrics:
                        metrics['max_flat_minutes'] = 0
                    if 'tick_frequency' not in metrics:
                        metrics['tick_frequency'] = metrics.get('change_ratio', 0)
                    
                    # Add is_low_variability key if missing (use is_straight_line as fallback)
                    if 'is_low_variability' not in metrics:
                        metrics['is_low_variability'] = metrics.get('is_straight_line', False)
                    
                    metrics['token'] = token
                    results.append(metrics)
            except Exception as e:
                st.warning(f"Error analyzing {token}: {e}")
        
        progress_bar.progress((i + 1) / len(selected_tokens))
    
    if not results:
        st.error("No tokens could be analyzed.")
        return
    
    # Create comparison dataframe
    comparison_df = pl.DataFrame(results)
    
    # Display summary table
    st.subheader("📊 Comparison Table")
    display_df = comparison_df.select([
        'token',
        pl.col('price_cv').round(4).alias('Price CV'),
        pl.col('log_price_cv').round(4).alias('Log Price CV'),
        pl.col('flat_periods_fraction').round(3).alias('Flat Periods'),
        pl.col('range_efficiency').round(3).alias('Range Efficiency'),
        pl.col('normalized_entropy').round(3).alias('Entropy'),
        pl.col('is_low_variability').alias('Filtered?')
    ])
    
    st.dataframe(display_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filtered_count = comparison_df['is_low_variability'].sum()
        st.metric("Tokens Filtered", f"{filtered_count}/{len(results)}")
    
    with col2:
        avg_cv = comparison_df['price_cv'].mean()
        st.metric("Average Price CV", f"{avg_cv:.4f}")
    
    with col3:
        avg_flat = comparison_df['flat_periods_fraction'].mean()
        st.metric("Avg Flat Periods", f"{avg_flat:.3f}")
    
    # Visualization
    st.subheader("📈 Variability Metrics Visualization")
    
    # Convert to pandas for plotting
    plot_df = comparison_df.to_pandas()
    
    # Scatter plot: CV vs Flat Periods
    fig = px.scatter(
        plot_df, 
        x='price_cv', 
        y='flat_periods_fraction',
        color='is_low_variability',
        hover_data=['token', 'normalized_entropy', 'range_efficiency'],
        title='Price CV vs Flat Periods',
        color_discrete_map={True: 'red', False: 'green'},
        labels={'is_low_variability': 'Filtered'}
    )
    
    # Add threshold lines
    fig.add_vline(x=0.05, line_dash="dash", line_color="red", annotation_text="CV threshold")
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Flat periods threshold")
    
    st.plotly_chart(fig, use_container_width=True)

def show_variability_distribution_analysis(selected_tokens):
    """Show distribution of variability metrics"""
    st.subheader("Variability Distribution Analysis")
    st.info("📊 Compare your selected tokens against filtering thresholds")
    
    # Analyze all tokens
    results = []
    progress_bar = st.progress(0)
    
    for i, token in enumerate(selected_tokens):
        df = st.session_state.data_loader.get_token_data(token)
        if df is not None and not df.is_empty():
            # Add returns if missing
            if 'returns' not in df.columns:
                df = st.session_state.token_cleaner._calculate_returns(df)
            
            try:
                result_df, modifications = st.session_state.token_cleaner._check_price_variability_graduated(df, token, "medium_term")
                
                # Extract metrics
                metrics = None
                for mod in modifications:
                    if 'metrics' in mod:
                        metrics = mod['metrics'].copy()
                        break
                        
                if metrics:
                    # Ensure all required metrics exist with default values
                    required_metrics = ['price_cv', 'log_price_cv', 'flat_periods_fraction', 'range_efficiency', 'normalized_entropy']
                    for metric_name in required_metrics:
                        if metric_name not in metrics:
                            metrics[metric_name] = 0.0
                    
                    # Ensure new metric keys exist
                    if 'max_flat_minutes' not in metrics:
                        metrics['max_flat_minutes'] = 0
                    if 'tick_frequency' not in metrics:
                        metrics['tick_frequency'] = metrics.get('change_ratio', 0)
                    
                    # Add is_low_variability key if missing (use is_straight_line as fallback)
                    if 'is_low_variability' not in metrics:
                        metrics['is_low_variability'] = metrics.get('is_straight_line', False)
                    
                    metrics['token'] = token
                    results.append(metrics)
            except Exception as e:
                st.warning(f"Error analyzing {token}: {e}")
        
        progress_bar.progress((i + 1) / len(selected_tokens))
    
    if not results:
        st.error("No tokens could be analyzed.")
        return
    
    # Convert to Polars DataFrame for plotting
    plot_df = pl.DataFrame(results).to_pandas()
    
    # Create distribution plots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Price CV Distribution', 'Log Price CV Distribution', 'Flat Periods Distribution',
                       'Range Efficiency Distribution', 'Entropy Distribution', 'Filtering Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price CV distribution
    fig.add_trace(
        go.Histogram(x=plot_df['price_cv'], name='Price CV', nbinsx=20, opacity=0.7),
        row=1, col=1
    )
    fig.add_vline(x=0.05, line_dash="dash", line_color="red", row=1, col=1)
    
    # Log Price CV distribution
    fig.add_trace(
        go.Histogram(x=plot_df['log_price_cv'], name='Log Price CV', nbinsx=20, opacity=0.7),
        row=1, col=2
    )
    fig.add_vline(x=0.1, line_dash="dash", line_color="red", row=1, col=2)
    
    # Flat periods distribution
    fig.add_trace(
        go.Histogram(x=plot_df['flat_periods_fraction'], name='Flat Periods', nbinsx=20, opacity=0.7),
        row=1, col=3
    )
    fig.add_vline(x=0.8, line_dash="dash", line_color="red", row=1, col=3)
    
    # Range efficiency distribution
    fig.add_trace(
        go.Histogram(x=plot_df['range_efficiency'], name='Range Efficiency', nbinsx=20, opacity=0.7),
        row=2, col=1
    )
    fig.add_vline(x=0.1, line_dash="dash", line_color="red", row=2, col=1)
    
    # Entropy distribution
    fig.add_trace(
        go.Histogram(x=plot_df['normalized_entropy'], name='Entropy', nbinsx=20, opacity=0.7),
        row=2, col=2
    )
    fig.add_vline(x=0.3, line_dash="dash", line_color="red", row=2, col=2)
    
    # Filtering summary
    filter_counts = plot_df['is_low_variability'].value_counts()
    fig.add_trace(
        go.Bar(x=['Passed', 'Filtered'], 
               y=[filter_counts.get(False, 0), filter_counts.get(True, 0)],
               name='Filtering Results',
               marker_color=['green', 'red']),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Variability Metrics Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display filtering statistics
    st.subheader("🎯 Filtering Statistics")
    
    filtered_count = plot_df['is_low_variability'].sum()
    total_count = len(plot_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tokens", total_count)
    
    with col2:
        st.metric("Filtered", filtered_count, delta=format_percentage(filtered_count/total_count))
    
    with col3:
        st.metric("Passed", total_count - filtered_count, delta=format_percentage((total_count-filtered_count)/total_count))
    
    with col4:
        avg_metrics_passed = plot_df[~plot_df['is_low_variability']]['price_cv'].mean() if (total_count - filtered_count) > 0 else 0
        st.metric("Avg CV (Passed)", f"{avg_metrics_passed:.4f}")
    
    # Current thresholds info
    st.info("""
    **🎯 Current Filter Thresholds:**
    - Price CV < 0.05 (coefficient of variation)
    - Log Price CV < 0.1
    - Flat periods > 0.8 (80% of periods show minimal change)
    - Range efficiency < 0.1 (few meaningful price moves)
    - Normalized entropy < 0.3 (low price movement diversity)
    
    *All conditions must be met for a token to be filtered.*
    """)

if __name__ == "__main__":
    main()