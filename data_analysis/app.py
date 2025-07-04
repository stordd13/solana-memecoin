"""
Streamlit app for memecoin data analysis
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
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import random
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

def main():
    st.title("Memecoin Data Analysis Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    # Data folder selection (before loading)
    if not st.session_state.data_loaded:
        st.sidebar.subheader("Select Data Source")
        
        # Common data subfolders
        common_subfolders = [
            "raw/dataset",
            "processed",
            "cleaned",
            "features"
        ]
        
        # Find available subfolders with parquet files
        project_root = Path(__file__).resolve().parent.parent
        data_root = project_root / "data"
        available_subfolders = []
        
        for subfolder in common_subfolders:
            subfolder_path = data_root / subfolder
            if subfolder_path.exists():
                parquet_files = list(subfolder_path.rglob('*.parquet'))
                if parquet_files:
                    available_subfolders.append((subfolder, len(parquet_files)))
        
        # Add custom option for other subfolders
        for root, dirs, files in os.walk(data_root):
            if any(f.endswith('.parquet') for f in files):
                rel = os.path.relpath(root, data_root)
                if rel not in common_subfolders and (rel, len([f for f in files if f.endswith('.parquet')])) not in available_subfolders:
                    available_subfolders.append((rel, len([f for f in files if f.endswith('.parquet')])))
        
        if not available_subfolders:
            st.sidebar.error("No parquet files found in data directory!")
            return
        
        # Create selectbox with subfolder info
        subfolder_options = [f"{sf} ({count:,} files)" for sf, count in available_subfolders]
        if 'selected_subfolder_idx' not in st.session_state:
            st.session_state.selected_subfolder_idx = 0
        
        selected_idx = st.sidebar.selectbox(
            "Choose data subfolder:",
            range(len(subfolder_options)),
            format_func=lambda x: subfolder_options[x],
            index=st.session_state.selected_subfolder_idx,
            key="subfolder_select"
        )
        
        selected_subfolder = available_subfolders[selected_idx][0]
        file_count = available_subfolders[selected_idx][1]
        
        st.sidebar.info(f"Selected: `data/{selected_subfolder}`\n{file_count:,} parquet files")
        
        if st.sidebar.button("Load Data", type="primary"):
            try:
                st.session_state.data_loader = DataLoader(subfolder=selected_subfolder)
                # Pre-cache the tokens
                st.session_state.data_loader.get_available_tokens()
                st.session_state.data_loaded = True
                st.session_state.selected_subfolder_idx = selected_idx
                st.success(f"Data loaded from data/{selected_subfolder} successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data from data/{selected_subfolder}: {e}")
                logger.error(f"Error loading data from data/{selected_subfolder}: {e}")
        return
    # Sidebar navigation after loading
    page = st.sidebar.radio("Go to", ["Data Quality", "Price Analysis", "Pattern Detection", "Price Distribution", "Variability Analysis"])
    if st.sidebar.button("Change Data Source"):
        st.session_state.data_loaded = False
        st.session_state.data_loader = None
        st.session_state.pop('selected_subfolder_idx', None)
        st.session_state.pop('dq_selected_datasets', None)
        st.rerun()
    
    # Add a button to refresh analyzers
    if st.sidebar.button("Refresh Analyzers"):
        st.session_state.quality_analyzer = None
        st.session_state.price_analyzer = None
        st.success("Analyzers refreshed!")
        st.rerun()
        
    # Token selection mode in sidebar
    selection_mode = st.sidebar.radio(
        "Token Selection Mode",
        ["Single Token", "Multiple Tokens", "Random Tokens", "All Tokens"],
        help="Choose how to select tokens for analysis"
    )
    # Get available tokens from loaded folder
    available_tokens = st.session_state.data_loader.get_available_tokens()
    token_symbols = sorted([t['symbol'] for t in available_tokens])
    # Token selection widgets in sidebar
    selected_tokens = []
    if selection_mode == "Single Token":
        if 'dq_single_token' not in st.session_state or st.session_state.dq_single_token not in token_symbols:
            st.session_state.dq_single_token = token_symbols[0] if token_symbols else None
        st.sidebar.selectbox("Select Token", token_symbols, key="dq_single_token")
        selected_tokens = [st.session_state.dq_single_token] if st.session_state.dq_single_token else []
    elif selection_mode == "Multiple Tokens":
        if 'dq_multi_tokens' not in st.session_state:
            st.session_state.dq_multi_tokens = []
        st.sidebar.multiselect("Select Tokens", token_symbols, key="dq_multi_tokens")
        selected_tokens = st.session_state.dq_multi_tokens
    elif selection_mode == "Random Tokens":
        if 'dq_random_tokens' not in st.session_state:
            st.session_state.dq_random_tokens = []
        num_tokens = st.sidebar.number_input(
            "Number of Random Tokens", min_value=1, max_value=len(token_symbols), value=5, key="dq_random_num"
        )
        if st.sidebar.button("Select Random Tokens", key="dq_random_btn"):
            st.session_state.dq_random_tokens = random.sample(token_symbols, min(num_tokens, len(token_symbols)))
        selected_tokens = st.session_state.dq_random_tokens
    else:  # All Tokens
        selected_tokens = token_symbols.copy()
    # Main content based on selected page
    if page == "Data Quality":
        show_data_quality(selected_tokens, selection_mode)
    elif page == "Price Analysis":
        show_price_analysis(selected_tokens, selection_mode)
    elif page == "Pattern Detection":
        show_pattern_detection(selected_tokens, selection_mode)
    elif page == "Price Distribution":
        show_price_distribution(selected_tokens, selection_mode)
    elif page == "Variability Analysis":
        show_variability_analysis(selected_tokens, selection_mode)

def show_data_quality(selected_tokens, selection_mode):
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
        
        # Calculate and display dead tokens per hour only for multiple, random, or all tokens
        if selection_mode in ["Multiple Tokens", "Random Tokens", "All Tokens"]:
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
                st.success(f'✅ Successfully exported {total_exported:,} tokens across all categories!')
                
                # Show breakdown
                st.subheader("Export Summary")
                for category, tokens in exported_results.items():
                    if tokens:
                        st.write(f"✅ **{category}**: {len(tokens):,} tokens exported")
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
        
def show_price_analysis(selected_tokens, selection_mode):
    st.title("Price Analysis")
    if not selected_tokens:
        st.warning("Please select at least one token")
        return

    # Initialize analyzer if not in session state
    if st.session_state.price_analyzer is None:
        st.session_state.price_analyzer = PriceAnalyzer()

    # Single or multiple token analysis
    if selection_mode == "Single Token":
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

def show_pattern_detection(selected_tokens, selection_mode):
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

def show_price_distribution(selected_tokens, selection_mode):
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

def show_variability_analysis(selected_tokens, selection_mode):
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
            result_df, modifications = st.session_state.token_cleaner._check_price_variability(df, token)
            
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
                
            # Display metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                price_cv = metrics.get('price_cv', 0.0)
                cv_status = "🔴" if price_cv < 0.05 else "🟢"
                st.metric("Price CV", f"{price_cv:.4f}", help="Coefficient of Variation")
                st.write(f"{cv_status} Threshold: < 0.05")
            
            with col2:
                log_price_cv = metrics.get('log_price_cv', 0.0)
                log_cv_status = "🔴" if log_price_cv < 0.1 else "🟢"
                st.metric("Log Price CV", f"{log_price_cv:.4f}")
                st.write(f"{log_cv_status} Threshold: < 0.1")
            
            with col3:
                flat_periods = metrics.get('flat_periods_fraction', 0.0)
                flat_status = "🔴" if flat_periods > 0.8 else "🟢"
                st.metric("Flat Periods", f"{flat_periods:.3f}")
                st.write(f"{flat_status} Threshold: > 0.8")
            
            with col4:
                range_eff = metrics.get('range_efficiency', 0.0)
                range_status = "🔴" if range_eff < 0.1 else "🟢"
                st.metric("Range Efficiency", f"{range_eff:.3f}")
                st.write(f"{range_status} Threshold: < 0.1")
            
            with col5:
                norm_entropy = metrics.get('normalized_entropy', 0.0)
                entropy_status = "🔴" if norm_entropy < 0.3 else "🟢"
                st.metric("Entropy", f"{norm_entropy:.3f}")
                st.write(f"{entropy_status} Threshold: < 0.3")
            
            # Overall decision
            decision = "🔴 FILTERED (Low Variability)" if metrics['is_low_variability'] else "🟢 PASSED"
            st.markdown(f"**Final Decision:** {decision}")
            
            # Price plot
            prices = df['price'].to_numpy()
            returns = df['returns'].to_numpy()
            
            # Create price plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Raw Price', 'Log Price', 'Returns Distribution', 'Rolling CV'),
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
            
            # Returns distribution
            finite_returns = returns[np.isfinite(returns)]
            fig.add_trace(
                go.Histogram(x=finite_returns, name='Returns', nbinsx=50, opacity=0.7),
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
                result_df, modifications = st.session_state.token_cleaner._check_price_variability(df, token)
                
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
                result_df, modifications = st.session_state.token_cleaner._check_price_variability(df, token)
                
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
        st.metric("Filtered", filtered_count, delta=f"{filtered_count/total_count*100:.1f}%")
    
    with col3:
        st.metric("Passed", total_count - filtered_count, delta=f"{(total_count-filtered_count)/total_count*100:.1f}%")
    
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