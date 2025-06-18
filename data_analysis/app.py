"""
Streamlit app for memecoin data analysis
"""

import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import random
import shutil
import plotly.express as px
import os

from data_loader import DataLoader
from data_quality import DataQualityAnalyzer
from price_analysis import PriceAnalyzer
from export_utils import export_parquet_files

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
        project_root = Path(__file__).resolve().parent.parent
        data_root = project_root / "data"
        subdirs = []
        for root, dirs, files in os.walk(data_root):
            if any(f.endswith('.parquet') for f in files):
                rel = os.path.relpath(root, data_root)
                subdirs.append(rel)
        subdirs = sorted(subdirs)
        if 'selected_data_root' not in st.session_state:
            st.session_state.selected_data_root = subdirs[0] if subdirs else ''
        selected_root = st.sidebar.selectbox("Select data folder:", subdirs, index=subdirs.index(st.session_state.selected_data_root))
        st.session_state.selected_data_root = selected_root
        data_dir = str(data_root / selected_root)
        
        # Display the number of files in the selected folder
        try:
            num_files = len([f for f in Path(data_dir).rglob('*.parquet')])
            st.sidebar.info(f"{num_files} parquet files found.")
        except Exception as e:
            st.sidebar.warning(f"Could not count files: {e}")

        if st.sidebar.button("Load Data"):
            try:
                st.session_state.data_loader = DataLoader(data_dir)
                # Pre-cache the tokens
                st.session_state.data_loader.get_available_tokens()
                st.session_state.data_loaded = True
                st.success(f"Data loaded from {selected_root} successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data from {data_dir}: {e}")
                logger.error(f"Error loading data from {data_dir}: {e}")
        return
    # Sidebar navigation after loading
    page = st.sidebar.radio("Go to", ["Data Quality", "Price Analysis", "Pattern Detection", "Price Distribution"])
    if st.sidebar.button("Change Data Source"):
        st.session_state.data_loaded = False
        st.session_state.data_loader = None
        st.session_state.pop('selected_data_root', None)
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
        
        # List of tokens with gaps
        tokens_with_gaps = [token for token, report in quality_reports.items() if report['gaps']['total_gaps'] > 0]
        if tokens_with_gaps:
            st.subheader("Tokens with Gaps")
            
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
            
            # Simple token list for download/export (unchanged functionality)
            st.subheader("Token List for Export")
            tokens_with_gaps_df = pl.DataFrame({'Token': tokens_with_gaps})
            st.dataframe(tokens_with_gaps_df)
            if st.button("Download Tokens with Gaps List as CSV", key="download_gaps_csv"):
                st.download_button(
                    label="Download as CSV",
                    data=tokens_with_gaps_df.to_csv(),
                    file_name="tokens_with_gaps.csv",
                    mime="text/csv",
                    key="download_gaps_csv_file"
                )
            if st.button("Export Gap Token Parquet Files to processed/", key="export_gaps_parquet"):
                st.info('Export button clicked! (Tokens with Gaps)')
                export_parquet_files(tokens_with_gaps, "Tokens with Gaps")
        
        # List of tokens with extreme movements (consolidated category)
        tokens_with_extremes = [token for token, report in quality_reports.items() if report.get('is_extreme_token', False)]
        if tokens_with_extremes:
            st.subheader("Tokens with Extreme Movements")
            st.write(f"Found {len(tokens_with_extremes)} tokens with extreme price movements (>1M% returns or >10k% minute jumps)")
            tokens_with_extremes_df = pl.DataFrame({'Token': tokens_with_extremes})
            st.dataframe(tokens_with_extremes_df)
            if st.button("Download Tokens with Extremes List as CSV", key="download_extremes_csv"):
                st.download_button(
                    label="Download as CSV",
                    data=tokens_with_extremes_df.to_csv(),
                    file_name="tokens_with_extremes.csv",
                    mime="text/csv",
                    key="download_extremes_csv_file"
                )
            if st.button("Export Extreme Token Parquet Files to processed/", key="export_extremes_parquet"):
                st.info('Export button clicked! (Tokens with Extremes)')
                export_parquet_files(tokens_with_extremes, "Tokens with Extremes")
        
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

        # Normal behavior tokens (exclusive category)
        normal_behavior_tokens = []
        for token, report in quality_reports.items():
            is_dead = report.get('is_dead', False)
            is_extreme = report.get('is_extreme_token', False)
            has_significant_gaps = report.get('gaps', {}).get('total_gaps', 0) > 5
            
            # Only include tokens with normal behavior (not dead, not extreme, minimal gaps)
            if not (is_dead or is_extreme or has_significant_gaps):
                normal_behavior_tokens.append(token)
        
        if normal_behavior_tokens:
            st.subheader("Normal Behavior Tokens")
            st.write(f"Found {len(normal_behavior_tokens)} tokens with normal behavior (not dead, not extreme, minimal gaps)")
            normal_behavior_df = pl.DataFrame({'Token': normal_behavior_tokens})
            st.dataframe(normal_behavior_df)
            
            # Add download button for normal behavior tokens
            if st.button("Download Normal Behavior Tokens List", key="download_normal_behavior_csv_main"):
                st.download_button(
                    label="Download as CSV",
                    data="\n".join(normal_behavior_tokens),
                    file_name="normal_behavior_tokens.csv",
                    mime="text/csv",
                    key="download_normal_behavior_csv_file_main"
                )
            # Add export button for normal behavior tokens to processed folder
            if st.button("Export Normal Behavior Token Parquet Files to processed/", key="export_normal_behavior_parquet_main"):
                st.info('Export button clicked! (Normal Behavior Tokens)')
                try:
                    export_parquet_files(normal_behavior_tokens, "Normal Behavior Tokens")
                    st.success(f'Exported {len(normal_behavior_tokens)} normal behavior token parquet files to data/processed/normal_behavior_tokens/')
                except Exception as e:
                    st.error(f"Export failed: {e}")
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

if __name__ == "__main__":
    main()