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
        
        # List of tokens with issues (extreme jumps)
        tokens_with_issues = [token for token, report in quality_reports.items() if report.get('has_extreme_jump', False)]
        if tokens_with_issues:
            st.subheader("Tokens with Issues (Extreme Price Jumps)")
            tokens_with_issues_df = pl.DataFrame({'Token': tokens_with_issues})
            st.dataframe(tokens_with_issues_df)
            if st.button("Download Tokens with Issues List as CSV", key="download_issues_csv"):
                st.download_button(
                    label="Download as CSV",
                    data=tokens_with_issues_df.to_csv(),
                    file_name="tokens_with_issues.csv",
                    mime="text/csv",
                    key="download_issues_csv_file"
                )
            if st.button("Export Issue Token Parquet Files to processed/", key="export_issues_parquet"):
                st.info('Export button clicked! (Tokens with Issues)')
                export_parquet_files(tokens_with_issues, "Tokens with Issues")
        
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

        # Reconstruct high_quality dict for export and download
        high_quality = {token: report for token, report in quality_reports.items() if report['quality_score'] >= 80}
        if high_quality:
            # Add download button for high quality tokens
            if st.button("Download High Quality Tokens List", key="download_high_quality_csv"):
                high_quality_tokens = list(high_quality.keys())
                st.download_button(
                    label="Download as CSV",
                    data="\n".join(high_quality_tokens),
                    file_name="high_quality_tokens.csv",
                    mime="text/csv",
                    key="download_high_quality_csv_file"
                )
            # Add export button for high quality tokens to processed folder
            if st.button("Export High Quality Token Parquet Files to processed/", key="export_high_quality_parquet"):
                st.info('Export button clicked! (High Quality Tokens)')
                try:
                    export_parquet_files(list(high_quality.keys()), "High Quality Tokens")
                except Exception as e:
                    st.error(f"Export failed: {e}")
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
    st.header("Price Distribution (Log-Scale)")
    if not selected_tokens:
        st.info("No tokens selected.")
        return
    for token in selected_tokens:
        df = st.session_state.data_loader.get_token_data(token)
        if df is None or df.is_empty():
            st.warning(f"No data for {token}")
            continue
        
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

        # Clipping extreme values in Polars
        q_low = df.select(pl.col('log_price').quantile(0.01, "lower")).item()
        q_high = df.select(pl.col('log_price').quantile(0.99, "higher")).item()
        
        # df = df.with_columns(
        #     pl.col('log_price').clip(lower_bound=q_low, upper_bound=q_high).alias('log_price_clipped')
        # )
        
        # if (df['log_price'] != df['log_price_clipped']).any():
        #     st.info(f"Clipped extreme log(price) values for {token} to improve plot visibility.")

        # Sort and remove duplicates
        df = df.sort('datetime').unique(subset='datetime', keep='first')
        
        # Use Polars DataFrame directly with Plotly Express
        fig = px.line(df, x='datetime', y='log_price', title=f'Log(Price) for {token} (clipped)')
        fig.update_layout(yaxis_autorange=True, xaxis_autorange=True)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()