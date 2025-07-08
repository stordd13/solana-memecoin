"""
Streamlit app for quantitative memecoin analysis
Professional financial market visualizations
"""

import streamlit as st
import polars as pl
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import logging
import numpy as np
import random
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from data_analysis.data_loader import DataLoader
from quant_analysis.quant_analysis import QuantAnalysis
from quant_analysis.quant_viz import QuantVisualizations
from streamlit_utils.formatting import format_large_number, format_percentage, format_data_points

# Page config
st.set_page_config(
    page_title="Memecoin Quant Analysis",
    page_icon="📊",
    layout="wide"
)

# Define time windows for analysis (replacing undefined extended_windows)
# Note: 1430 instead of 1440 due to data buffer/padding constraints
DEFAULT_WINDOWS = [1, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1430]
COMMON_WINDOWS = [5, 10, 15, 30, 60, 120, 240]
EXTENDED_WINDOWS = [1, 2, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1430]

def main():
    st.title("📊 Professional Quantitative Analysis - Memecoin Trading")
    st.markdown("Advanced financial market analysis using price action only")
    st.markdown("---")
    
    # Initialize session state for data loading (same as data_analysis)
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = None
    if 'quant_analyzer' not in st.session_state:
        st.session_state.quant_analyzer = None
    if 'quant_viz' not in st.session_state:
        st.session_state.quant_viz = None
    
    # Data source selection (same logic as data_analysis)
    if not st.session_state.data_loaded:
        st.header("Select Data Source")
        
        # Get project root and data directory
        project_root = Path(__file__).parent.parent
        data_dir_root = project_root / "data"
        
        # Get all subdirectories recursively
        def get_all_subdirs(base_dir):
            subdirs = []
            if not base_dir.exists():
                return subdirs
            for root, dirs, files in os.walk(base_dir):
                for d in dirs:
                    full_path = Path(root) / d
                    rel_path = full_path.relative_to(base_dir)
                    subdirs.append(str(rel_path))
            return sorted(set(subdirs))
        
        all_data_roots = get_all_subdirs(data_dir_root)
        
        if not all_data_roots:
            st.error("No data directories found. Please ensure data exists in the data/ folder.")
            return
        
        # Data root selection
        if 'selected_data_root' not in st.session_state:
            st.session_state.selected_data_root = all_data_roots[0] if all_data_roots else ''
        
        selected_data_root = st.selectbox(
            "Select Data Root Directory",
            all_data_roots,
            index=all_data_roots.index(st.session_state.selected_data_root) if st.session_state.selected_data_root in all_data_roots else 0,
            help="Choose the data directory containing your token files"
        )
        st.session_state.selected_data_root = selected_data_root
        
        # Load data button
        if st.button("Load Data", type="primary"):
            try:
                base_path = data_dir_root / selected_data_root
                data_loader = DataLoader(base_path=str(base_path))
                
                # Check if data exists
                available_tokens = data_loader.get_available_tokens()
                if not available_tokens:
                    st.error(f"No token data found in {base_path}")
                    return
                
                # Store in session state
                st.session_state.data_loader = data_loader
                st.session_state.quant_analyzer = QuantAnalysis()
                st.session_state.quant_viz = QuantVisualizations()
                st.session_state.data_loaded = True
                
                st.success(f"Loaded {len(available_tokens)} tokens from {selected_data_root}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar navigation after loading (same as data_analysis)
    st.sidebar.title("Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        [
            "🔥 Multi-Token Risk Metrics", 
            "🔄 24-Hour Lifecycle Analysis",
            "Entry/Exit Matrix Analysis",
            "Entry/Exit Moment Matrix",
            "Volatility Surface",
            "Microstructure Analysis",
            "Price Distribution Evolution",
            "Optimal Holding Period",
            "Market Regime Analysis",
            "Multi-Token Correlation",
            "Comprehensive Report",
            "Trade Timing Heatmap"
        ]
    )
    
    # Data source controls
    if st.sidebar.button("Change Data Source"):
        st.session_state.data_loaded = False
        st.session_state.data_loader = None
        st.session_state.pop('selected_data_root', None)
        st.rerun()
    
    if st.sidebar.button("Refresh Analyzers"):
        st.session_state.quant_analyzer = QuantAnalysis()
        st.session_state.quant_viz = QuantVisualizations()
        st.success("Analyzers refreshed!")
        st.rerun()
    
    # Token selection (same logic as data_analysis)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Token Selection")
    
    selection_mode = st.sidebar.radio(
        "Token Selection Mode",
        ["Single Token", "Multiple Tokens", "Random Tokens", "All Tokens"],
        help="Choose how to select tokens for analysis"
    )
    
    # Get available tokens
    available_tokens = st.session_state.data_loader.get_available_tokens()
    token_symbols = sorted([t['symbol'] for t in available_tokens])
    
    # Token selection widgets
    selected_tokens = []
    if selection_mode == "Single Token":
        if 'qa_single_token' not in st.session_state or st.session_state.qa_single_token not in token_symbols:
            st.session_state.qa_single_token = token_symbols[0] if token_symbols else None
        st.sidebar.selectbox("Select Token", token_symbols, key="qa_single_token")
        selected_tokens = [st.session_state.qa_single_token] if st.session_state.qa_single_token else []
    elif selection_mode == "Multiple Tokens":
        if 'qa_multi_tokens' not in st.session_state:
            st.session_state.qa_multi_tokens = []
        st.sidebar.multiselect("Select Tokens", token_symbols, key="qa_multi_tokens")
        selected_tokens = st.session_state.qa_multi_tokens
    elif selection_mode == "Random Tokens":
        if 'qa_random_tokens' not in st.session_state:
            st.session_state.qa_random_tokens = []
        num_tokens = st.sidebar.number_input(
            "Number of Random Tokens", min_value=1, max_value=len(token_symbols), value=10, key="qa_random_num"
        )
        if st.sidebar.button("Select Random Tokens", key="qa_random_btn"):
            st.session_state.qa_random_tokens = random.sample(token_symbols, min(num_tokens, len(token_symbols)))
        selected_tokens = st.session_state.qa_random_tokens
    else:  # All Tokens
        selected_tokens = token_symbols.copy()
    
    st.sidebar.info(f"Selected: {len(selected_tokens)} tokens")
    
    # Main analysis content
    if analysis_type == "🔥 Multi-Token Risk Metrics":
        show_multi_token_risk_metrics(selected_tokens)
    elif analysis_type == "🔄 24-Hour Lifecycle Analysis":
        show_24_hour_lifecycle_analysis(selected_tokens)
    elif analysis_type == "Entry/Exit Matrix Analysis":
        show_entry_exit_matrix(selected_tokens, selection_mode)
    elif analysis_type == "Entry/Exit Moment Matrix":
        show_entry_exit_moment_matrix(selected_tokens)
    elif analysis_type == "Volatility Surface":
        show_volatility_surface(selected_tokens, selection_mode)
    elif analysis_type == "Microstructure Analysis":
        show_microstructure_analysis(selected_tokens, selection_mode)
    elif analysis_type == "Price Distribution Evolution":
        show_price_distribution_evolution(selected_tokens, selection_mode)
    elif analysis_type == "Optimal Holding Period":
        show_optimal_holding_period(selected_tokens, selection_mode)
    elif analysis_type == "Market Regime Analysis":
        show_market_regime_analysis(selected_tokens, selection_mode)
    elif analysis_type == "Multi-Token Correlation":
        show_multi_token_correlation(selected_tokens)
    elif analysis_type == "Comprehensive Report":
        show_comprehensive_report(selected_tokens)
    elif analysis_type == "Trade Timing Heatmap":
        show_trade_timing_heatmap(selected_tokens)



def show_multi_token_risk_metrics(selected_tokens):
    """Multi-Token Risk Metrics Analysis"""
    st.header("🔥 Multi-Token Risk Metrics Analysis")
    st.markdown("Compare risk metrics across all tokens in the dataset")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Parameters
    time_horizons = st.multiselect(
        "Time Horizons (minutes)",
        DEFAULT_WINDOWS,
        default=[60, 240, 1430]
    )
    
    if not time_horizons:
        st.warning("Please select at least one time horizon.")
        return
    
    if st.button("🚀 Calculate Risk Metrics", type="primary"):
        with st.spinner(f"Calculating risk metrics for {len(selected_tokens)} tokens..."):
            # Load and analyze data
            token_results = []
            progress_bar = st.progress(0)
            
            for i, token_name in enumerate(selected_tokens):
                try:
                    df = st.session_state.data_loader.get_token_data(token_name)
                    if df is not None and len(df) > 100:
                        results = st.session_state.quant_analyzer.temporal_risk_reward_analysis(df, time_horizons)
                        
                        # Ensure consistent DataFrame structure
                        if not results.is_empty() and len(results.columns) > 1:
                            results = results.with_columns([pl.lit(token_name).alias('Token')])
                            
                            # Verify we have the expected columns
                            expected_columns = ['horizon_minutes', 'win_rate', 'avg_gain_%', 'avg_loss_%', 
                                              'risk_reward_ratio', 'expected_value_%', 'sharpe_ratio', 'Token']
                            
                            # Only add if we have all expected columns or at least the core ones
                            if all(col in results.columns for col in ['horizon_minutes', 'win_rate', 'sharpe_ratio']):
                                token_results.append(results)
                            else:
                                st.warning(f"{token_name}: Incomplete risk metrics (missing core columns)")
                        else:
                            st.warning(f"{token_name}: No valid risk metrics calculated (insufficient data)")
                    else:
                        st.warning(f"{token_name}: Insufficient data ({len(df) if df is not None else 0} rows, need >100)")
                    progress_bar.progress((i + 1) / len(selected_tokens))
                except Exception as e:
                    st.warning(f"Error analyzing {token_name}: {str(e)}")
                    continue
            
            progress_bar.empty()
            
            if token_results:
                # Debug: Check DataFrame structures before concatenation
                st.info(f"Successfully analyzed {len(token_results)} tokens for concatenation")
                
                try:
                    # Combine all results
                    combined_results = pl.concat(token_results)
                except Exception as concat_error:
                    st.error(f"Error concatenating results: {str(concat_error)}")
                    
                    # Debug information
                    st.error("**Debug Information:**")
                    for i, result_df in enumerate(token_results):
                        st.error(f"DataFrame {i}: {result_df.shape} shape, columns: {list(result_df.columns)}")
                    
                    # Try to fix by ensuring all DataFrames have the same columns
                    st.info("Attempting to fix column mismatches...")
                    
                    # Get all unique columns
                    all_columns = set()
                    for result_df in token_results:
                        all_columns.update(result_df.columns)
                    
                    # Standardize all DataFrames
                    standardized_results = []
                    for result_df in token_results:
                        # Add missing columns with null values
                        missing_cols = all_columns - set(result_df.columns)
                        for col in missing_cols:
                            result_df = result_df.with_columns([pl.lit(None).alias(col)])
                        
                        # Reorder columns consistently
                        result_df = result_df.select(sorted(all_columns))
                        standardized_results.append(result_df)
                    
                    # Try concatenation again
                    combined_results = pl.concat(standardized_results)
                    st.success("✅ Successfully fixed and concatenated results!")
                
                # Display summary statistics
                st.subheader("📊 Risk Metrics Summary")
                summary_stats = combined_results.group_by('horizon_minutes').agg([
                    pl.col('sharpe_ratio').mean().alias('Avg Sharpe'),
                    pl.col('win_rate').mean().alias('Avg Win Rate (%)'),
                    pl.col('risk_reward_ratio').mean().alias('Avg Risk/Reward'),
                    pl.col('Token').count().alias('Token Count')
                ])
                st.dataframe(summary_stats, use_container_width=True)
                
                # Add visualizations
                st.subheader("📈 Risk Metrics Visualizations")
                try:
                    qv = st.session_state.quant_viz
                    if hasattr(qv, 'plot_multi_token_risk_metrics'):
                        risk_metrics_fig = qv.plot_multi_token_risk_metrics(combined_results)
                        st.plotly_chart(risk_metrics_fig, use_container_width=True)
                        
                        # Key insights
                        st.subheader("🔍 Key Insights")
                        
                        # Find best performing horizon for each metric
                        best_sharpe_horizon = summary_stats.sort('Avg Sharpe', descending=True).limit(1)
                        best_win_rate_horizon = summary_stats.sort('Avg Win Rate (%)', descending=True).limit(1)
                        best_risk_reward_horizon = summary_stats.sort('Avg Risk/Reward', descending=True).limit(1)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if len(best_sharpe_horizon) > 0:
                                horizon = best_sharpe_horizon['horizon_minutes'][0]
                                value = best_sharpe_horizon['Avg Sharpe'][0]
                                st.success(f"🏆 **Best Sharpe Ratio**\n\n{horizon} min horizon: {value:.3f}")
                        
                        with col2:
                            if len(best_win_rate_horizon) > 0:
                                horizon = best_win_rate_horizon['horizon_minutes'][0]
                                value = best_win_rate_horizon['Avg Win Rate (%)'][0]
                                st.success(f"🎯 **Best Win Rate**\n\n{horizon} min horizon: {value:.1f}%")
                        
                        with col3:
                            if len(best_risk_reward_horizon) > 0:
                                horizon = best_risk_reward_horizon['horizon_minutes'][0]
                                value = best_risk_reward_horizon['Avg Risk/Reward'][0]
                                st.success(f"⚖️ **Best Risk/Reward**\n\n{horizon} min horizon: {value:.2f}")
                        
                        # Performance trends
                        horizons = summary_stats['horizon_minutes'].to_list()
                        if len(horizons) > 1:
                            st.markdown("**📊 Performance Trends:**")
                            
                            # Analyze trends
                            sharpe_values = summary_stats['Avg Sharpe'].to_list()
                            win_rate_values = summary_stats['Avg Win Rate (%)'].to_list()
                            
                            if sharpe_values[-1] > sharpe_values[0]:
                                st.info("📈 **Sharpe Ratio**: Improves with longer time horizons")
                            else:
                                st.info("📉 **Sharpe Ratio**: Decreases with longer time horizons")
                            
                            if win_rate_values[-1] > win_rate_values[0]:
                                st.info("📈 **Win Rate**: Improves with longer time horizons")
                            else:
                                st.info("📉 **Win Rate**: Decreases with longer time horizons")
                    else:
                        st.warning("Visualization function not available. Please refresh analyzers.")
                except Exception as e:
                    st.warning(f"Could not generate visualizations: {str(e)}")
                
                # Detailed results with data quality warnings
                st.subheader("📈 Detailed Results")
                
                # Add data quality warnings
                high_win_rates = combined_results.filter(pl.col('win_rate') >= 95.0)
                if len(high_win_rates) > 0:
                    st.warning(f"⚠️ **Data Quality Alert**: {len(high_win_rates)} entries have win rates ≥95%. This often indicates:\n"
                             f"- Limited historical data (new tokens)\n"
                             f"- Tokens in early pump phase\n"
                             f"- Insufficient time horizons for meaningful analysis\n\n"
                             f"**Recommendation**: Focus on tokens with 60-85% win rates for more reliable analysis.")
                
                # Show results sorted by Sharpe ratio
                display_results = combined_results.sort('sharpe_ratio', descending=True)
                st.dataframe(display_results, use_container_width=True)
                
                # Add summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_win_rate = combined_results['win_rate'].mean()
                    st.metric("Average Win Rate", f"{avg_win_rate:.1f}%")
                
                with col2:
                    tokens_with_perfect_wr = len(combined_results.filter(pl.col('win_rate') == 100.0))
                    st.metric("100% Win Rate Count", f"{tokens_with_perfect_wr}")
                
                with col3:
                    reliable_tokens = len(combined_results.filter(
                        (pl.col('win_rate') >= 60.0) & 
                        (pl.col('win_rate') <= 85.0) &
                        (pl.col('sharpe_ratio') > 0.1)
                    ))
                    st.metric("Reliable Analysis Count", f"{reliable_tokens}")
                
                # Add interpretation guide for multi-token risk metrics
                with st.expander("📖 Multi-Token Risk Metrics Interpretation Guide"):
                    st.markdown("""
                    **How to Read the Risk Metrics Dashboard:**
                    
                    **📊 Win Rate by Time Horizon** (Top Left):
                    - **X-axis**: Different holding periods (5 minutes to ~24 hours)
                    - **Y-axis**: Percentage of profitable trades
                    - **Multiple lines**: Each token's win rate pattern
                    - **Target**: Look for tokens with consistently high win rates (60-85%) across time horizons
                    - **⚠️ 100% Win Rates**: Often indicate insufficient data or early pump phase - use caution
                    - **Quality Threshold**: Require minimum 50+ data points for reliable win rate calculation
                    - **Insight**: Tokens with declining win rates over time may be pump-and-dump patterns
                    
                    **📈 Sharpe Ratio by Time Horizon** (Top Right):
                    - **X-axis**: Different holding periods
                    - **Y-axis**: Risk-adjusted return (higher = better)
                    - **Sharpe > 0.5**: Excellent for crypto (non-annualized ratios)
                    - **Sharpe > 0.2**: Good performance, **Sharpe < 0**: Poor performance
                    - **Note**: Values are non-annualized for crypto trading relevance
                    - **Insight**: Peak Sharpe ratios indicate optimal holding periods for each token
                    
                    **⚖️ Risk vs Reward Scatter** (Bottom Left):
                    - **X-axis**: Risk (volatility) - lower is better
                    - **Y-axis**: Reward (average return) - higher is better
                    - **Top-left quadrant**: Best tokens (high return, low risk)
                    - **Bottom-right quadrant**: Worst tokens (low return, high risk)
                    - **Insight**: Distance from origin indicates overall performance quality
                    
                    **🕒 Performance Across Time Horizons** (Bottom Right):
                    - **Heatmap**: Color-coded performance for each token at each time horizon
                    - **Green**: Positive performance, **Red**: Negative performance
                    - **Bright colors**: Stronger performance (positive or negative)
                    - **Insight**: Identify tokens with consistent performance patterns
                    
                    **🎯 Trading Strategy Insights:**
                    - **Portfolio Construction**: Select tokens from top-left of risk/reward scatter
                    - **Optimal Timing**: Use Sharpe ratio peaks to determine best holding periods
                    - **Risk Management**: Avoid tokens with declining win rates over time
                    - **Diversification**: Choose tokens with different time horizon strengths
                    - **Entry Strategy**: Focus on tokens with high early-period win rates
                    
                    **⚠️ Warning Signs:**
                    - **Sharpe ratio < 0**: Consistent losses - avoid these tokens
                    - **Win rate < 40%**: Poor success probability
                    - **Win rate = 100%**: Often insufficient data - verify with longer time horizons
                    - **High volatility + low returns**: High risk without compensation
                    - **Declining performance over time**: Possible pump-and-dump behavior
                    
                    **📋 Data Quality Guidelines:**
                    - **Reliable Analysis**: Win rates 60-85% with positive Sharpe ratios
                    - **Minimum Data**: Requires 50+ historical data points per time horizon
                    - **New Token Caution**: Recently launched tokens may show inflated metrics
                    - **Cross-Validation**: Compare multiple time horizons for consistency
                    """)
            else:
                st.error("No valid data to analyze.")

def show_24_hour_lifecycle_analysis(selected_tokens):
    """24-Hour Lifecycle Analysis - Analyze patterns within token lifecycle"""
    st.header("🔄 24-Hour Lifecycle Analysis")
    st.markdown("""
    **24-Hour Lifecycle Analysis**: Analyze patterns and performance within the 24-hour token lifecycle.
    
    - **Early vs Late Performance**: Compare first hours vs final hours
    - **Hourly Volatility Patterns**: How volatility changes throughout the day
    - **Momentum Decay**: How initial momentum fades over time
    - **Lifecycle Stages**: Performance in different phases (launch, peak, decline)
    - **Optimal Trading Windows**: Best hours for different strategies
    """)
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        lifecycle_segments = st.selectbox(
            "Lifecycle Segments",
            ["4 Segments (6h each)", "6 Segments (4h each)", "8 Segments (3h each)", "12 Segments (2h each)", "24 Segments (1h each)"],
            index=1
        )
    with col2:
        analysis_metrics = st.multiselect(
            "Analysis Metrics",
            ["Returns", "Volatility", "Price Momentum", "Volume Proxy", "Trend Strength"],
            default=["Returns", "Volatility", "Price Momentum"]
        )
    
    if not analysis_metrics:
        st.warning("Please select at least one analysis metric.")
        return
    
    if st.button("🔄 Analyze Lifecycle Patterns", type="primary"):
        with st.spinner(f"Analyzing 24-hour lifecycle for {len(selected_tokens)} tokens..."):
            
            # Parse segments
            segments_map = {
                "4 Segments (6h each)": 4,
                "6 Segments (4h each)": 6, 
                "8 Segments (3h each)": 8,
                "12 Segments (2h each)": 12,
                "24 Segments (1h each)": 24
            }
            n_segments = segments_map[lifecycle_segments]
            
            all_lifecycle_data = []
            progress_bar = st.progress(0)
            successful_tokens = 0
            failed_tokens = 0
            
            for i, token_name in enumerate(selected_tokens):
                try:
                    df = st.session_state.data_loader.get_token_data(token_name)
                    
                    if df is not None and not df.is_empty() and len(df) >= n_segments * 10:  # Need at least 10 minutes per segment
                        segment_size = len(df) // n_segments
                        
                        for segment_idx in range(n_segments):
                            start_idx = segment_idx * segment_size
                            end_idx = (segment_idx + 1) * segment_size if segment_idx < n_segments - 1 else len(df)
                            
                            segment_data = df[start_idx:end_idx]
                            
                            if len(segment_data) > 5:  # Need minimum data points
                                # Calculate metrics for this segment
                                returns = segment_data['price'].pct_change().drop_nulls()
                                
                                segment_results = {
                                    'Token': token_name,
                                    'Lifecycle_Segment': f"Segment {segment_idx + 1}",
                                    'Hours_Into_Lifecycle': f"{(24/n_segments) * segment_idx:.1f}-{(24/n_segments) * (segment_idx + 1):.1f}h",
                                    'Segment_Minutes': len(segment_data),
                                    'Start_Price': segment_data['price'].first(),
                                    'End_Price': segment_data['price'].last(),
                                    'Min_Price': segment_data['price'].min(),
                                    'Max_Price': segment_data['price'].max()
                                }
                                
                                # Calculate requested metrics
                                if "Returns" in analysis_metrics and len(returns) > 0:
                                    # Mean return: average of minute-by-minute returns
                                    segment_results['Mean_Return_Pct'] = returns.mean() * 100
                                    
                                    # Cumulative return: start to end of segment
                                    segment_results['Cumulative_Return_Pct'] = ((segment_data['price'].last() / segment_data['price'].first()) - 1) * 100
                                    
                                    # Max return: highest point reached in segment (vs segment start)
                                    segment_results['Max_Return_Pct'] = ((segment_data['price'].max() / segment_data['price'].first()) - 1) * 100
                                    
                                    # Min return: lowest point reached in segment (vs segment start)
                                    segment_results['Min_Return_Pct'] = ((segment_data['price'].min() / segment_data['price'].first()) - 1) * 100
                                    
                                    # If this is the last segment, calculate 24h total return
                                    if segment_idx == n_segments - 1:
                                        # Get the very first price point of the token
                                        first_price = df['price'].first()
                                        final_price = segment_data['price'].last()
                                        segment_results['Total_24h_Return_Pct'] = ((final_price / first_price) - 1) * 100
                                        
                                        # Peak return during entire 24h lifecycle
                                        peak_price = df['price'].max()
                                        segment_results['Peak_24h_Return_Pct'] = ((peak_price / first_price) - 1) * 100
                                
                                if "Volatility" in analysis_metrics and len(returns) > 1:
                                    segment_results['Volatility_Pct'] = returns.std() * 100
                                    segment_results['Realized_Volatility_Pct'] = (returns.abs().mean()) * 100
                                
                                if "Price Momentum" in analysis_metrics and len(returns) > 0:
                                    # Simple momentum: average of positive returns
                                    positive_returns = returns.filter(returns > 0)
                                    segment_results['Positive_Momentum_Pct'] = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
                                    segment_results['Win_Rate_Pct'] = (len(positive_returns) / len(returns)) * 100
                                
                                if "Volume Proxy" in analysis_metrics:
                                    # Use price volatility as volume proxy
                                    segment_results['Volume_Proxy'] = returns.abs().sum() * 100
                                
                                if "Trend Strength" in analysis_metrics and len(returns) > 2:
                                    # Simple trend strength: correlation with time
                                    time_series = pl.Series(range(len(segment_data['price'])))
                                    try:
                                        correlation = segment_data['price'].corr(time_series)
                                        segment_results['Trend_Strength'] = correlation if not np.isnan(correlation) else 0
                                    except:
                                        segment_results['Trend_Strength'] = 0
                                
                                all_lifecycle_data.append(segment_results)
                        
                        successful_tokens += 1
                    else:
                        failed_tokens += 1
                        if len(df) < n_segments * 10:
                            st.warning(f"{token_name}: Insufficient data ({len(df)} minutes, need {n_segments * 10})")
                        
                    progress_bar.progress((i + 1) / len(selected_tokens))
                    
                except Exception as e:
                    failed_tokens += 1
                    st.warning(f"Error analyzing {token_name}: {str(e)}")
                    continue
            
            progress_bar.empty()
            
            # Show processing summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tokens", len(selected_tokens))
            with col2:
                st.metric("Successful", successful_tokens, delta=f"{successful_tokens/len(selected_tokens)*100:.1f}%")
            with col3:
                st.metric("Failed", failed_tokens, delta=f"-{failed_tokens/len(selected_tokens)*100:.1f}%")
            
            if all_lifecycle_data:
                lifecycle_df = pl.DataFrame(all_lifecycle_data)
                
                # Store results in session state to prevent reset on UI changes
                st.session_state.lifecycle_results = {
                    'lifecycle_df': lifecycle_df,
                    'analysis_metrics': analysis_metrics,
                    'successful_tokens': successful_tokens,
                    'total_segments': len(all_lifecycle_data)
                }
            else:
                st.error(f"❌ No valid lifecycle data to analyze. {failed_tokens} tokens failed processing.")
                st.info("💡 Try selecting tokens with more data or using fewer segments.")
                return

    # Display results (either from current analysis or session state)
    if 'lifecycle_results' in st.session_state:
        lifecycle_df = st.session_state.lifecycle_results['lifecycle_df']
        analysis_metrics = st.session_state.lifecycle_results['analysis_metrics']
        successful_tokens = st.session_state.lifecycle_results['successful_tokens']
        total_segments = st.session_state.lifecycle_results['total_segments']
        
        # Summary by lifecycle segment
        st.subheader("📊 Lifecycle Analysis Summary")
        
        # Aggregate by segment
        numeric_cols = [col for col in lifecycle_df.columns if col.endswith('_Pct') or col in ['Trend_Strength', 'Volume_Proxy']]
        if numeric_cols:
            summary_stats = lifecycle_df.group_by('Lifecycle_Segment').agg([
                pl.col(col).mean().alias(f'Avg_{col}') for col in numeric_cols if col in lifecycle_df.columns
            ] + [
                pl.col('Token').count().alias('Token_Count'),
                pl.col('Segment_Minutes').mean().alias('Avg_Minutes_Per_Segment')
            ])
            
            st.dataframe(summary_stats.sort('Lifecycle_Segment'), use_container_width=True)
        
        # Detailed results
        st.subheader("📈 Detailed Lifecycle Results")
        st.dataframe(lifecycle_df.sort(['Token', 'Lifecycle_Segment']), use_container_width=True)
        
        # Visualization options
        st.subheader("📊 Visualization Options")
        col1, col2 = st.columns(2)
        with col1:
            visualization_type = st.selectbox(
                "Select Visualization Type",
                ["Summary Charts", "Aggregated Analysis", "Token Ranking", "Early vs Late Comparison"],
                key="lifecycle_viz_type"
            )
        with col2:
            if visualization_type in ["Aggregated Analysis", "Token Ranking"]:
                heatmap_metric = st.selectbox(
                    "Select Metric for Analysis",
                    [col for col in lifecycle_df.columns if col.endswith('_Pct') or col in ['Trend_Strength', 'Volume_Proxy']],
                    key="lifecycle_analysis_metric"
                )
                show_comparison = st.checkbox("Show Early vs Late Comparison", value=False, key="lifecycle_show_comparison")
            elif visualization_type == "Early vs Late Comparison":
                show_comparison = True
            else:
                show_comparison = st.checkbox("Show Early vs Late Comparison", value=True, key="lifecycle_show_comparison")
        
        # Generate visualizations based on selection
        qv = st.session_state.quant_viz
        
        if visualization_type == "Summary Charts":
            st.subheader("📊 Lifecycle Summary Charts")
            if hasattr(qv, 'plot_lifecycle_summary_charts'):
                try:
                    summary_fig = qv.plot_lifecycle_summary_charts(lifecycle_df, analysis_metrics)
                    st.plotly_chart(summary_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate summary charts: {str(e)}")
        
        elif visualization_type == "Aggregated Analysis" and heatmap_metric:
            st.subheader(f"📈 {heatmap_metric} - Statistical Distribution Analysis")
            st.markdown(f"""
            **Statistical Analysis for {heatmap_metric}:**
            - **Central Tendency**: Mean vs Median across lifecycle segments
            - **Distribution Spread**: Standard deviation and Interquartile Range (IQR)  
            - **Percentile Bands**: 10th-90th percentile range with median
            - **Extreme Values**: Min/Max values with mean reference line
            
            This analysis works efficiently with any number of tokens ({len(selected_tokens)} selected).
            """)
            try:
                agg_fig = qv.plot_lifecycle_aggregated_analysis(lifecycle_df, heatmap_metric)
                st.plotly_chart(agg_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate aggregated analysis: {str(e)}")
        
        elif visualization_type == "Token Ranking" and heatmap_metric:
            st.subheader(f"🏆 Token Performance Ranking - {heatmap_metric}")
            
            # Allow user to adjust the number of top/bottom tokens to show
            top_n = st.slider("Number of top/bottom tokens to show", min_value=5, max_value=50, value=20, key="lifecycle_top_n")
            
            # Add explanation of the selected metric
            metric_explanations = {
                'Mean_Return_Pct': '**Mean Return**: Average of minute-by-minute percentage price changes within each segment',
                'Cumulative_Return_Pct': '**Cumulative Return**: Price change from start to end of each lifecycle segment',
                'Max_Return_Pct': '**Max Return**: Highest point reached within each segment (vs segment start)',
                'Min_Return_Pct': '**Min Return**: Lowest point reached within each segment (vs segment start)',
                'Total_24h_Return_Pct': '**24h Total Return**: Complete return from token launch to 24h end (single value per token)',
                'Peak_24h_Return_Pct': '**24h Peak Return**: Maximum return achieved during entire 24h lifecycle (single value per token)',
                'Volatility_Pct': '**Volatility**: Standard deviation of minute-by-minute returns within each segment',
                'Positive_Momentum_Pct': '**Positive Momentum**: Average of only the positive minute-by-minute returns',
                'Win_Rate_Pct': '**Win Rate**: Percentage of minutes with positive price changes',
                'Volume_Proxy': '**Volume Proxy**: Sum of absolute returns as trading activity indicator',
                'Trend_Strength': '**Trend Strength**: Correlation between price and time (trend consistency)'
            }
            
            metric_explanation = metric_explanations.get(heatmap_metric, f'**{heatmap_metric}**: Selected metric for analysis')
            
            st.markdown(f"""
            **Token Ranking Analysis:**
            
            {metric_explanation}
            
            **Analysis Components:**
            - **Top {top_n} Performers**: Best performing tokens for this metric
            - **Bottom {top_n} Performers**: Worst performing tokens for this metric
            - **Performance vs Risk**: Risk-return profile visualization
            - **Distribution**: Performance distribution across all {len(selected_tokens)} selected tokens
            
            **Ranking Method:** {"Single 24h value per token" if heatmap_metric in ['Total_24h_Return_Pct', 'Peak_24h_Return_Pct'] else "Average across all lifecycle segments"}
            """)
            try:
                ranking_fig = qv.plot_lifecycle_token_ranking(lifecycle_df, heatmap_metric, top_n=top_n)
                st.plotly_chart(ranking_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate token ranking: {str(e)}")
        
        elif visualization_type == "Early vs Late Comparison":
            show_comparison = True
        
        # Early vs Late comparison (can be combined with other visualizations)
        if show_comparison and hasattr(qv, 'plot_lifecycle_comparison'):
            st.subheader("⚖️ Early vs Late Lifecycle Comparison")
            try:
                comparison_fig = qv.plot_lifecycle_comparison(lifecycle_df, "early_vs_late")
                st.plotly_chart(comparison_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate comparison chart: {str(e)}")
        
        # Key insights
        st.subheader("🔍 Key Lifecycle Insights")
        
        if 'Cumulative_Return_Pct' in lifecycle_df.columns:
            best_segment = lifecycle_df.group_by('Lifecycle_Segment').agg([
                pl.col('Cumulative_Return_Pct').mean().alias('Avg_Return')
            ]).sort('Avg_Return', descending=True).limit(1)
            
            if len(best_segment) > 0:
                best_segment_name = best_segment['Lifecycle_Segment'][0]
                best_return = best_segment['Avg_Return'][0]
                st.success(f"🏆 Best Performing Segment: {best_segment_name} (Avg Return: {best_return:.2f}%)")
        
        if 'Volatility_Pct' in lifecycle_df.columns:
            most_volatile = lifecycle_df.group_by('Lifecycle_Segment').agg([
                pl.col('Volatility_Pct').mean().alias('Avg_Volatility')
            ]).sort('Avg_Volatility', descending=True).limit(1)
            
            if len(most_volatile) > 0:
                volatile_segment = most_volatile['Lifecycle_Segment'][0]
                volatile_value = most_volatile['Avg_Volatility'][0]
                st.info(f"⚡ Most Volatile Segment: {volatile_segment} (Avg Volatility: {volatile_value:.2f}%)")
        
        st.success(f"✅ Successfully analyzed {total_segments} lifecycle segments from {successful_tokens} tokens!")
        
        # Add interpretation guide for 24-hour lifecycle analysis
        with st.expander("📖 24-Hour Lifecycle Analysis Interpretation Guide"):
            st.markdown("""
            **How to Read the Lifecycle Analysis:**
            
            **📊 Summary Charts** (Top Row):
            - **Returns by Segment**: Shows average performance in each lifecycle phase
            - **Volatility by Segment**: Risk level changes throughout token lifecycle  
            - **Volume Proxy by Segment**: Trading activity patterns over time
            - **Insight**: Early segments often show highest returns but also highest volatility
            
            **🎯 Aggregated Analysis**:
            - **Distribution plots**: How selected metric varies across all tokens and segments
            - **Box plots**: Median, quartiles, and outliers for each lifecycle segment
            - **Violin plots**: Full distribution shape showing concentration areas
            - **Target**: Look for segments with consistently positive distributions
            
            **🏆 Token Ranking**:
            - **Top performers**: Tokens with best performance in selected metric
            - **Bottom performers**: Tokens with worst performance (avoid or short)
            - **Risk/Return scatter**: Balance between performance and volatility
            - **Insight**: Top performers in early segments may be good momentum plays
            
            **⚖️ Early vs Late Comparison**:
            - **First half vs Second half**: Performance comparison between lifecycle phases
            - **Statistical significance**: Whether differences are meaningful
            - **Distribution overlays**: Visual comparison of performance patterns
            - **Target**: Tokens with sustained performance throughout lifecycle
            
            **🕒 Lifecycle Segment Insights:**
            
            **Early Segments (0-6 hours)**:
            - **Highest volatility**: Maximum risk and reward potential
            - **Momentum effects**: Strong price movements from initial interest
            - **Strategy**: Momentum trading, quick scalps, high-risk/high-reward
            
            **Middle Segments (6-12 hours)**:
            - **Stabilization phase**: Volatility typically decreases
            - **Trend confirmation**: Sustainable vs unsustainable moves become clear
            - **Strategy**: Trend following, position building on confirmed moves
            
            **Late Segments (12-24 hours)**:
            - **Maturation phase**: Lower volatility, clearer trends
            - **Value discovery**: Price approaching fair value
            - **Strategy**: Mean reversion, value plays, lower-risk entries
            
            **🎯 Trading Strategy Applications:**
            
            **Segment-Based Entry Strategy**:
            - **High-risk traders**: Focus on early segments with highest return potential
            - **Conservative traders**: Wait for middle/late segments with lower volatility
            - **Diversified approach**: Allocate different position sizes by segment risk
            
            **Risk Management by Segment**:
            - **Early segments**: Wider stop-losses, smaller position sizes
            - **Middle segments**: Standard risk management rules
            - **Late segments**: Tighter stops, larger positions (if trend confirmed)
            
            **Portfolio Construction**:
            - **Mix lifecycle stages**: Combine tokens in different lifecycle phases
            - **Segment rotation**: Move from early to late segment tokens over time
            - **Risk balancing**: Use late-segment tokens to balance early-segment risk
            
            **⚠️ Key Warnings:**
            - **Declining performance over time**: May indicate pump-and-dump behavior
            - **Extreme early volatility**: High risk of total loss
            - **No clear patterns**: Random behavior makes prediction difficult
            - **Low sample sizes**: Results may not be statistically significant
            """)
        

def show_entry_exit_matrix(selected_tokens, selection_mode):
    st.header("⏱️ Entry/Exit Matrix Analysis")
    st.markdown("""
    **Entry/Exit Matrix Analysis**: Analyze optimal entry and exit timing combinations across multiple tokens.
    
    - **Entry Window**: Minutes after token launch to enter position
    - **Exit Window**: Minutes after entry to exit position  
    - **Multi-Token Support**: Aggregates results across all selected tokens
    - **Confidence Intervals**: Shows statistical confidence in results
    - **Efficient Algorithm**: Uses optimized calculations vs. brute-force rolling analysis
    """)

    # Use the tokens selected from the main token selection
    if not selected_tokens:
        st.warning("Please select at least one token from the sidebar.")
        return
    
    st.info(f"Analyzing {len(selected_tokens)} selected tokens")

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        entry_windows = st.multiselect(
            "Entry Windows (minutes)",
            EXTENDED_WINDOWS,
            default=[5, 10, 15, 30, 60, 120, 240],
            key="single_matrix_entry_windows"
        )
    with col2:
        exit_windows = st.multiselect(
            "Exit Windows (minutes)",
            EXTENDED_WINDOWS,
            default=[5, 10, 15, 30, 60, 120, 240],
            key="single_matrix_exit_windows"
        )

    if st.button("Generate Matrix", type="primary", key="single_matrix_generate_btn"):
        if selected_tokens:
            with st.spinner(f"Calculating entry/exit matrix for {len(selected_tokens)} tokens..."):
                # Load data for all selected tokens
                token_data = []
                for token_name in selected_tokens:
                    try:
                        df = st.session_state.data_loader.get_token_data(token_name)
                        if isinstance(df, pl.DataFrame) and not df.is_empty():
                            token_data.append((token_name, df))
                    except Exception as e:
                        continue
                if not token_data:
                    st.warning("No valid token data loaded.")
                else:
                    qv = st.session_state.quant_viz
                    # Use the same logic as the single-token view for each token, then average
                    # Use aggregate_entry_exit_matrices, but ensure it uses the correct logic
                    if hasattr(qv, 'aggregate_entry_exit_matrices'):
                        aggregated_matrix, confidence_matrix = qv.aggregate_entry_exit_matrices(token_data, entry_windows, exit_windows)
                        fig = qv.plot_multi_token_entry_exit_matrix(aggregated_matrix, confidence_matrix, len(token_data), entry_windows)
                        st.plotly_chart(fig, use_container_width=True)
                        # Best entry/exit
                        matrix_np = aggregated_matrix.to_numpy()
                        if matrix_np.size > 0 and not np.all(np.isnan(matrix_np)):
                            best_idx = np.nanargmax(matrix_np)
                            best_entry_idx, best_exit_idx = np.unravel_index(best_idx, matrix_np.shape)
                            best_entry = entry_windows[best_entry_idx]
                            best_exit = [int(col) for col in aggregated_matrix.columns][best_exit_idx]
                            st.success(f"Best Entry/Exit: {best_entry}/{best_exit} min, Avg Return: {matrix_np[best_entry_idx, best_exit_idx]:.2f}%")
                        else:
                            st.info("No valid data for best entry/exit.")
                        # Detailed results
                        st.subheader("📊 Detailed Results")
                        results_df = pl.DataFrame({
                            'Entry Window': [f"{e} min" for e in entry_windows for _ in exit_windows],
                            'Exit Window': [f"{e} min" for _ in entry_windows for e in exit_windows],
                            'Avg Return (%)': aggregated_matrix.to_numpy().flatten(),
                            '95% CI': confidence_matrix.to_numpy().flatten()
                        })
                        results_df = results_df.sort('Avg Return (%)', descending=True)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Add interpretation guide for entry/exit matrix
                        with st.expander("📖 Entry/Exit Matrix Interpretation Guide"):
                            st.markdown("""
                            **How to Read the Entry/Exit Matrix:**
                            
                            **🎯 Matrix Heatmap**:
                            - **X-axis**: Exit windows (time to hold position)
                            - **Y-axis**: Entry windows (time after launch to enter)
                            - **Colors**: Green = positive returns, Red = negative returns
                            - **Intensity**: Brighter colors = stronger performance
                            - **Target**: Look for consistently green areas with high values
                            
                            **📊 Statistical Confidence**:
                            - **95% Confidence Intervals**: Statistical reliability of results
                            - **Sample sizes**: More tokens = more reliable results
                            - **Significance**: Larger confidence intervals = less certain results
                            - **Target**: Narrow confidence intervals with positive returns
                            
                            **🎯 Optimal Strategy Identification:**
                            
                            **Best Entry/Exit Combinations**:
                            - **Top-performing cells**: Highest average returns across all tokens
                            - **Risk-adjusted performance**: Consider both return and confidence
                            - **Consistency**: Look for strategies that work across multiple tokens
                            
                            **Entry Window Patterns**:
                            - **Early entries (5-15 min)**: Capture initial momentum but higher risk
                            - **Medium entries (30-60 min)**: Balance between momentum and confirmation
                            - **Late entries (120+ min)**: Lower risk but may miss initial moves
                            
                            **Exit Window Patterns**:
                            - **Quick exits (5-15 min)**: Scalping strategies, capture immediate moves
                            - **Medium exits (30-60 min)**: Standard swing trading approaches
                            - **Long exits (120+ min)**: Position trading, ride longer trends
                            
                            **🎯 Trading Strategy Applications:**
                            
                            **Momentum Strategy**:
                            - **Pattern**: Early entry + Quick exit (top-left quadrant)
                            - **Logic**: Capture initial price spikes from launch hype
                            - **Risk**: High volatility, requires fast execution
                            - **Best for**: Experienced traders with fast execution systems
                            
                            **Swing Trading Strategy**:
                            - **Pattern**: Medium entry + Medium exit (center area)
                            - **Logic**: Wait for initial volatility to settle, ride confirmed trends
                            - **Risk**: Moderate, balanced approach
                            - **Best for**: Most traders, good risk/reward balance
                            
                            **Position Trading Strategy**:
                            - **Pattern**: Late entry + Long exit (bottom-right quadrant)
                            - **Logic**: Enter after trend confirmation, hold for major moves
                            - **Risk**: Lower but may miss best opportunities
                            - **Best for**: Conservative traders, larger position sizes
                            
                            **⚠️ Risk Management Guidelines:**
                            
                            **High-Return Strategies** (Bright green cells):
                            - Use smaller position sizes due to higher risk
                            - Implement tight stop-losses
                            - Monitor closely for quick exit if needed
                            
                            **Consistent Strategies** (Moderate green across many tokens):
                            - Can use larger position sizes
                            - More reliable for systematic trading
                            - Good for portfolio construction
                            
                            **Avoid Red Zones**:
                            - Combinations showing consistent losses
                            - May indicate poor timing or market inefficiencies
                            - Use for contrarian strategies only with extreme caution
                            
                            **📈 Performance Metrics:**
                            - **Average Return**: Expected profit/loss per trade
                            - **Win Rate**: Percentage of profitable trades (not shown but implied)
                            - **Risk-Adjusted Return**: Consider volatility and confidence intervals
                            - **Scalability**: How well strategy works across different tokens
                            """)
                    else:
                        st.error("aggregate_entry_exit_matrices not implemented in QuantVisualizations.")
        else:
            st.warning("Please select at least one token.")

def show_entry_exit_moment_matrix(selected_tokens):
    st.header("⏱️ Entry/Exit Moment Matrix")
    st.markdown("""
    **Entry/Exit Moment Matrix Analysis**: Visualize average returns for each (entry minute, exit minute) pair.
    
    - **Entry Minute**: Specific minute after token launch to enter position
    - **Exit Minute**: Specific minute after token launch to exit position  
    - **Heatmap**: Shows average return % across all selected tokens
    - **Optimization**: Choose between original or Polars-optimized computation
    """)

    # Use the tokens selected from the main token selection
    if not selected_tokens:
        st.warning("Please select at least one token from the sidebar.")
        return
    
    st.info(f"Analyzing {len(selected_tokens)} selected tokens")

    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_entry_minute = st.number_input("Max Entry Minute", min_value=10, max_value=1440, value=240, key="moment_matrix_max_entry")
    with col2:
        max_exit_minute = st.number_input("Max Exit Minute", min_value=10, max_value=1440, value=240, key="moment_matrix_max_exit")
    with col3:
        use_optimized = st.checkbox("Use Polars Optimization", value=True, 
                                   help="Faster computation using vectorized Polars operations vs nested loops")

    if st.button("Show Entry/Exit Moment Matrix", type="primary", key="moment_matrix_generate_btn"):
        if selected_tokens:
            optimization_text = "Polars-optimized" if use_optimized else "original"
            with st.spinner(f"Computing {optimization_text} entry/exit moment matrix for {len(selected_tokens)} tokens..."):
                # Load data for all selected tokens
                token_data = []
                for token_name in selected_tokens:
                    try:
                        df = st.session_state.data_loader.get_token_data(token_name)
                        if isinstance(df, pl.DataFrame) and not df.is_empty():
                            token_data.append(df)
                    except Exception as e:
                        continue
                if not token_data:
                    st.warning("No valid token data loaded.")
                else:
                    qv = st.session_state.quant_viz
                    
                    # Choose method based on user preference
                    if use_optimized:
                        fig = qv.plot_entry_exit_moment_matrix_optimized(
                            token_data, 
                            max_entry_minute=int(max_entry_minute), 
                            max_exit_minute=int(max_exit_minute)
                        )
                        st.success("✅ Used Polars-optimized computation (much faster for large datasets)")
                    else:
                        fig = qv.plot_entry_exit_moment_matrix(
                            token_data, 
                            max_entry_minute=int(max_entry_minute), 
                            max_exit_minute=int(max_exit_minute)
                        )
                        st.info("ℹ️ Used original computation (slower but proven)")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance note
                    if len(selected_tokens) > 100 and not use_optimized:
                        st.warning("💡 **Performance Tip**: With 100+ tokens, consider using Polars Optimization for much faster computation!")
                    
                    # Add interpretation guide for entry/exit moment matrix
                    with st.expander("📖 Entry/Exit Moment Matrix Interpretation Guide"):
                        st.markdown("""
                        **How to Read the Entry/Exit Moment Matrix:**
                        
                        **🎯 Matrix Heatmap Structure**:
                        - **X-axis**: Exit minute (specific time to exit position)
                        - **Y-axis**: Entry minute (specific time to enter position)
                        - **Color scale**: Return percentage (green = profit, red = loss)
                        - **Diagonal line**: Represents holding periods (entry to exit duration)
                        - **Below diagonal**: Invalid trades (exit before entry)
                        
                        **📊 Pattern Recognition:**
                        
                        **Horizontal Patterns** (Fixed Entry Time):
                        - **Green horizontal streaks**: Good entry times across multiple exit times
                        - **Red horizontal streaks**: Poor entry times regardless of exit
                        - **Analysis**: Identify consistently profitable entry moments
                        
                        **Vertical Patterns** (Fixed Exit Time):
                        - **Green vertical streaks**: Good exit times across multiple entry times
                        - **Red vertical streaks**: Poor exit times regardless of entry
                        - **Analysis**: Identify optimal exit moments
                        
                        **Diagonal Patterns** (Fixed Holding Period):
                        - **Green diagonal lines**: Profitable holding periods
                        - **Red diagonal lines**: Unprofitable holding periods
                        - **Thickness**: Width of profitable diagonals shows holding period flexibility
                        
                        **🔥 Hot Zones (High Returns)**:
                        - **Early entry + Quick exit**: Momentum capture strategies
                        - **Early entry + Medium exit**: Trend riding opportunities
                        - **Late entry + Long exit**: Value/recovery plays
                        - **Concentrated zones**: Specific time windows with exceptional returns
                        
                        **❄️ Cold Zones (Poor Returns)**:
                        - **Avoid these combinations**: Consistent loss-making patterns
                        - **Late entry + Quick exit**: Often poor risk/reward
                        - **Peak entry + Any exit**: Buying at local tops
                        - **Random scattered losses**: High volatility periods
                        
                        **🎯 Strategy Development:**
                        
                        **Momentum Scalping** (Top-left quadrant):
                        - **Entry**: Very early (0-30 minutes)
                        - **Exit**: Quick (5-60 minutes after entry)
                        - **Logic**: Capture immediate launch momentum
                        - **Risk**: High volatility, requires fast execution
                        
                        **Trend Following** (Upper-middle area):
                        - **Entry**: Early to medium (30-120 minutes)
                        - **Exit**: Medium to long (60-240 minutes after entry)
                        - **Logic**: Ride established trends
                        - **Risk**: Moderate, good risk/reward balance
                        
                        **Recovery Trading** (Lower-right quadrant):
                        - **Entry**: Late (120+ minutes)
                        - **Exit**: Much later (60+ minutes after entry)
                        - **Logic**: Buy dips and ride recoveries
                        - **Risk**: Lower but requires patience
                        
                        **📈 Optimization Techniques:**
                        
                        **Find Best Combinations**:
                        - **Brightest green cells**: Highest return combinations
                        - **Consistent green areas**: Reliable profit zones
                        - **Large profitable regions**: Flexible timing strategies
                        - **Avoid isolated hot spots**: May be statistical noise
                        
                        **Risk Assessment**:
                        - **Color intensity**: Darker colors = higher magnitude
                        - **Pattern consistency**: Repeating patterns more reliable
                        - **Size of profitable zones**: Larger zones = more forgiving timing
                        - **Proximity to losses**: Profitable zones near losses = higher risk
                        
                        **Execution Considerations**:
                        - **Market timing**: Matrix assumes token launch detection
                        - **Execution speed**: Early strategies require fast execution
                        - **Slippage**: Consider transaction costs for quick trades
                        - **Monitoring**: Some strategies require active monitoring
                        
                        **⚠️ Important Limitations:**
                        
                        **Statistical Warnings**:
                        - **Sample size**: Some combinations have few observations
                        - **Survivorship bias**: Only includes completed token lifecycles
                        - **Market regime dependency**: Patterns may change with market conditions
                        - **Overfitting risk**: Very specific timing may not generalize
                        
                        **Practical Constraints**:
                        - **Market hours**: Assumes 24/7 monitoring capability
                        - **Execution capacity**: Multiple simultaneous positions
                        - **Capital allocation**: Need sufficient capital for position sizing
                        - **Technology requirements**: Fast data feeds and execution systems
                        
                        **📊 Advanced Analysis:**
                        
                        **Pattern Evolution**:
                        - **Compare across tokens**: Find universal vs token-specific patterns
                        - **Time-based analysis**: How patterns change by launch time/day
                        - **Market condition correlation**: Patterns during different market regimes
                        
                        **Portfolio Application**:
                        - **Strategy diversification**: Use multiple entry/exit combinations
                        - **Risk spreading**: Avoid concentration in single time windows
                        - **Capital efficiency**: Optimize position timing for maximum utilization
                        - **Performance tracking**: Monitor actual vs predicted returns
                        
                        **Real-time Implementation**:
                        - **Live monitoring**: Track tokens from launch
                        - **Alert systems**: Notify when optimal entry/exit times approach
                        - **Dynamic adjustment**: Adapt to real-time market conditions
                        - **Performance feedback**: Update strategies based on live results
                        """)
        else:
            st.warning("Please select at least one token.")

def show_volatility_surface(selected_tokens, selection_mode):
    """Enhanced Volatility Surface Analysis"""
    st.header("📈 Volatility Surface Analysis")
    st.markdown("""
    **Enhanced Volatility Analysis**: Comprehensive volatility analysis with corrected calculations.
    
    - **Corrected Logic**: Proper rolling volatility calculation with accurate annualization
    - **Multiple Visualizations**: 3D surface, dashboard, and multi-token comparison
    - **Polars Optimized**: Fast computation using vectorized operations
    - **Professional Insights**: Statistical analysis and comparative metrics
    """)
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Single Token Analysis", "Multi-Token Comparison"],
        help="Choose between detailed single token analysis or comparative multi-token analysis"
    )
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        windows = st.multiselect(
            "Rolling Windows (minutes)",
            DEFAULT_WINDOWS,
            default=[5, 10, 30, 60, 240],
            help="Time windows for rolling volatility calculation"
        )
    with col2:
        if analysis_mode == "Single Token Analysis":
            percentiles = st.multiselect(
                "Percentiles for Surface",
                [5, 10, 25, 50, 75, 90, 95],
                default=[10, 25, 50, 75, 90],
                help="Percentiles to display in the 3D volatility surface"
            )
    
    if not windows:
        st.warning("Please select at least one time window.")
        return
    
    if analysis_mode == "Single Token Analysis":
        # Single token detailed analysis
        selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
        
        if selected_token and st.button("📊 Generate Volatility Analysis", type="primary"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            if df is None or df.is_empty():
                st.error("No data available for selected token.")
                return
            
            with st.spinner("Calculating comprehensive volatility analysis..."):
                qv = st.session_state.quant_viz
                
                # Generate visualizations
                st.subheader("📊 Volatility Analysis Dashboard")
                try:
                    dashboard_fig = qv.plot_volatility_analysis_dashboard(df, windows)
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating dashboard: {str(e)}")
                
                # 3D Volatility Surface
                if len(percentiles) >= 3:
                    st.subheader("🌊 3D Volatility Surface")
                    try:
                        surface_fig = qv.plot_volatility_surface(df, windows, percentiles)
                        st.plotly_chart(surface_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating surface: {str(e)}")
                
                # Statistical Summary
                st.subheader("📊 Volatility Statistics")
                try:
                    # Calculate summary statistics using Polars
                    df_analysis = df.with_columns([
                        df['price'].pct_change().alias('returns')
                    ])
                    
                    volatility_stats = []
                    for window in windows:
                        if len(df_analysis) >= window * 2:
                            rolling_vol = df_analysis.with_columns([
                                df_analysis['returns'].rolling_std(window_size=window, min_periods=max(1, window//2)).alias('rolling_vol')
                            ])['rolling_vol'].drop_nulls() * np.sqrt(525600 / window) * 100
                            
                            if len(rolling_vol) > 0:
                                volatility_stats.append({
                                    'Window (min)': window,
                                    'Mean Vol (%)': rolling_vol.mean(),
                                    'Median Vol (%)': rolling_vol.median(),
                                    'Std Dev (%)': rolling_vol.std(),
                                    'Min Vol (%)': rolling_vol.min(),
                                    'Max Vol (%)': rolling_vol.max(),
                                    'Observations': len(rolling_vol)
                                })
                    
                    if volatility_stats:
                        stats_df = pl.DataFrame(volatility_stats)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Key insights
                        st.subheader("🔍 Key Insights")
                        
                        # Find optimal volatility window
                        best_stability = stats_df.sort('Std Dev (%)').limit(1)
                        highest_vol = stats_df.sort('Mean Vol (%)', descending=True).limit(1)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if len(best_stability) > 0:
                                window = best_stability['Window (min)'][0]
                                std_val = best_stability['Std Dev (%)'][0]
                                st.success(f"🎯 **Most Stable**\n\n{window} min window\nStd Dev: {std_val:.2f}%")
                        
                        with col2:
                            if len(highest_vol) > 0:
                                window = highest_vol['Window (min)'][0]
                                vol_val = highest_vol['Mean Vol (%)'][0]
                                st.info(f"⚡ **Highest Volatility**\n\n{window} min window\nMean: {vol_val:.1f}%")
                        
                        with col3:
                            overall_avg = stats_df['Mean Vol (%)'].mean()
                            st.metric("📊 Overall Avg Volatility", f"{overall_avg:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
    
    else:
        # Multi-token comparison
        if len(selected_tokens) < 2:
            st.warning("Please select at least 2 tokens for comparison.")
            return
        
        # Limit tokens for performance
        max_tokens = st.slider("Max tokens to analyze", min_value=2, max_value=min(50, len(selected_tokens)), value=min(10, len(selected_tokens)))
        tokens_to_analyze = selected_tokens[:max_tokens]
        
        if st.button("📊 Compare Volatility Across Tokens", type="primary"):
            with st.spinner(f"Analyzing volatility for {len(tokens_to_analyze)} tokens..."):
                # Load data for all selected tokens
                token_data = []
                progress_bar = st.progress(0)
                
                for i, token_name in enumerate(tokens_to_analyze):
                    try:
                        df = st.session_state.data_loader.get_token_data(token_name)
                        if isinstance(df, pl.DataFrame) and not df.is_empty():
                            token_data.append((token_name, df))
                        progress_bar.progress((i + 1) / len(tokens_to_analyze))
                    except Exception as e:
                        continue
                
                progress_bar.empty()
                
                if len(token_data) < 2:
                    st.error("Need at least 2 valid tokens for comparison.")
                    return
                
                # Generate multi-token comparison
                st.subheader(f"📊 Multi-Token Volatility Comparison ({len(token_data)} tokens)")
                try:
                    qv = st.session_state.quant_viz
                    comparison_fig = qv.plot_multi_token_volatility_comparison(token_data, windows)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Volatility Rankings")
                    
                    # Calculate overall volatility for each token
                    token_volatilities = []
                    for token_name, df in token_data:
                        df_analysis = df.with_columns([
                            df['price'].pct_change().alias('returns')
                        ])
                        
                        token_vols = []
                        for window in windows:
                            if len(df_analysis) >= window * 2:
                                rolling_vol = df_analysis.with_columns([
                                    df_analysis['returns'].rolling_std(window_size=window, min_periods=max(1, window//2)).alias('rolling_vol')
                                ])['rolling_vol'].drop_nulls() * np.sqrt(525600 / window) * 100
                                
                                if len(rolling_vol) > 0:
                                    token_vols.append(rolling_vol.mean())
                        
                        if token_vols:
                            token_volatilities.append({
                                'Token': token_name,
                                'Avg Volatility (%)': np.mean(token_vols),
                                'Volatility Std (%)': np.std(token_vols),
                                'Windows Analyzed': len(token_vols)
                            })
                    
                    if token_volatilities:
                        rankings_df = pl.DataFrame(token_volatilities).sort('Avg Volatility (%)', descending=True)
                        st.dataframe(rankings_df, use_container_width=True)
                        
                        # Key insights
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            most_volatile = rankings_df.head(1)
                            if len(most_volatile) > 0:
                                token = most_volatile['Token'][0]
                                vol = most_volatile['Avg Volatility (%)'][0]
                                st.error(f"⚡ **Most Volatile**\n\n{token}\n{vol:.1f}% avg volatility")
                        
                        with col2:
                            least_volatile = rankings_df.tail(1)
                            if len(least_volatile) > 0:
                                token = least_volatile['Token'][0]
                                vol = least_volatile['Avg Volatility (%)'][0]
                                st.success(f"🎯 **Least Volatile**\n\n{token}\n{vol:.1f}% avg volatility")
                        
                        with col3:
                            overall_avg = rankings_df['Avg Volatility (%)'].mean()
                            st.info(f"📊 **Average**\n\n{overall_avg:.1f}% across all tokens")
                    
                    # Add interpretation guide for volatility surface
                    with st.expander("📖 Volatility Surface Analysis Interpretation Guide"):
                        st.markdown("""
                        **How to Read the Volatility Surface:**
                        
                        **📊 3D Volatility Surface** (Single Token):
                        - **X-axis**: Time progression (minutes since launch)
                        - **Y-axis**: Rolling window sizes (different time horizons)
                        - **Z-axis (color)**: Volatility level (% annualized)
                        - **Surface patterns**: How volatility changes over time and across windows
                        - **Hot spots**: Areas of extreme volatility (red/orange areas)
                        
                        **📈 Multi-Token Comparison**:
                        - **Multiple lines**: Each token's volatility evolution
                        - **Line patterns**: Volatility behavior across different window sizes
                        - **Convergence/Divergence**: How similar tokens behave
                        - **Outliers**: Tokens with unusual volatility patterns
                        
                        **🎯 Volatility Interpretation:**
                        
                        **High Volatility (>500% annualized)**:
                        - **Extreme risk**: Potential for large gains or losses
                        - **Timing critical**: Small timing differences = big impact
                        - **Strategy**: Scalping, very quick in/out, small position sizes
                        - **Warning**: High probability of stop-loss hits
                        
                        **Medium Volatility (100-500% annualized)**:
                        - **Standard memecoin range**: Typical for most tokens
                        - **Manageable risk**: Standard risk management applies
                        - **Strategy**: Swing trading, momentum plays, position trading
                        - **Optimal**: Good balance of opportunity and controllable risk
                        
                        **Low Volatility (<100% annualized)**:
                        - **Stable phase**: Lower risk but also lower reward potential
                        - **Trend following**: Better for trend-based strategies
                        - **Strategy**: Position trading, mean reversion, larger sizes
                        - **Warning**: May lack momentum for significant moves
                        
                        **🕒 Time-Based Patterns:**
                        
                        **Early Launch (0-30 minutes)**:
                        - **Typically highest volatility**: Maximum opportunity and risk
                        - **Pattern**: Sharp spike then gradual decline
                        - **Strategy**: Quick scalps, momentum trading
                        - **Risk management**: Wide stops, small positions
                        
                        **Middle Phase (30-120 minutes)**:
                        - **Volatility normalization**: Finding sustainable levels
                        - **Pattern**: Volatility clustering around mean
                        - **Strategy**: Trend confirmation, swing entries
                        - **Risk management**: Standard stops, moderate positions
                        
                        **Mature Phase (120+ minutes)**:
                        - **Lower, stable volatility**: Trend-following phase
                        - **Pattern**: Consistent, predictable volatility
                        - **Strategy**: Position trading, trend riding
                        - **Risk management**: Tight stops, larger positions
                        
                        **📊 Window Size Analysis:**
                        
                        **Short Windows (5-15 minutes)**:
                        - **High sensitivity**: Captures micro-movements
                        - **Use for**: Scalping, intraday timing
                        - **Pattern**: Most reactive, highest values
                        
                        **Medium Windows (30-60 minutes)**:
                        - **Balanced view**: Good for swing trading
                        - **Use for**: Position entry/exit timing
                        - **Pattern**: Smoothed but responsive
                        
                        **Long Windows (120+ minutes)**:
                        - **Trend perspective**: Overall volatility regime
                        - **Use for**: Portfolio risk management
                        - **Pattern**: Smooth, shows major regime changes
                        
                        **🎯 Trading Applications:**
                        
                        **Position Sizing**:
                        - **High volatility periods**: Reduce position size proportionally
                        - **Low volatility periods**: Can increase position size
                        - **Dynamic sizing**: Adjust based on current volatility level
                        
                        **Entry Timing**:
                        - **Volatility spikes**: Wait for normalization before entry
                        - **Volatility lulls**: Prepare for potential breakouts
                        - **Volatility trends**: Enter in direction of volatility change
                        
                        **Risk Management**:
                        - **Stop-loss width**: Wider stops during high volatility
                        - **Take-profit levels**: Closer targets during low volatility
                        - **Time stops**: Exit if volatility doesn't match expectations
                        
                        **⚠️ Warning Signals:**
                        - **Extreme volatility spikes**: May indicate manipulation or news
                        - **Volatility collapse**: Could signal loss of interest
                        - **Abnormal patterns**: Unusual surface shapes may indicate issues
                        - **No clear pattern**: Random volatility makes prediction difficult
                        
                        **📈 Advanced Strategies:**
                        - **Volatility arbitrage**: Trade volatility differences between tokens
                        - **Volatility timing**: Enter during low vol, exit during high vol
                        - **Surface trading**: Use surface patterns to predict volatility changes
                        - **Multi-timeframe**: Combine different window insights for better timing
                        """)
                
                except Exception as e:
                    st.error(f"Error generating comparison: {str(e)}")

def show_microstructure_analysis(selected_tokens, selection_mode):
    """Market Microstructure Analysis"""
    st.header("🔬 Market Microstructure Analysis")
    st.markdown("""
    **Analyze high-frequency market behavior and quality indicators**
    
    This analysis examines the market's internal structure and quality using price-only data,
    providing insights into liquidity, efficiency, and trading dynamics without relying on volume data.
    """)
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Analysis configuration
    st.sidebar.markdown("### 🔬 Analysis Configuration")
    show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=True)
    show_summary_dashboard = st.sidebar.checkbox("Show Summary Dashboard", value=True)
    analysis_window = st.sidebar.slider("Analysis Window (minutes)", 30, 240, 60, 
                                       help="Window size for rolling calculations")
    
    selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
    
    if selected_token:
        if st.button("🔬 Analyze Market Microstructure"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            with st.spinner("Analyzing market microstructure..."):
                # Perform microstructure analysis
                results = st.session_state.quant_analyzer.microstructure_analysis(df)
                
                # Store results in session state for persistence
                st.session_state.microstructure_results = results
                
                # Main metrics display
                st.subheader("📊 Key Microstructure Metrics")
                
                # Create metrics columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    vol_1h = results.get('avg_realized_volatility_1h', 0)
                    st.metric(
                        "Realized Volatility (1h)", 
                        f"{vol_1h * 100:.1f}%" if not np.isnan(vol_1h) else "N/A",
                        help="Annualized volatility calculated from 1-hour rolling windows"
                    )
                    
                    vol_4h = results.get('avg_realized_volatility_4h', 0)
                    st.metric(
                        "Realized Volatility (4h)", 
                        f"{vol_4h * 100:.1f}%" if not np.isnan(vol_4h) else "N/A",
                        help="Annualized volatility calculated from 4-hour rolling windows"
                    )
                
                with col2:
                    spread = results.get('bid_ask_spread_estimate', 0)
                    spread_conf = results.get('spread_confidence', 'Unknown')
                    st.metric(
                        "Bid-Ask Spread Estimate", 
                        f"{spread * 10000:.1f} bps" if not np.isnan(spread) else "N/A",
                        delta=f"Confidence: {spread_conf}",
                        help="Roll's estimator for effective bid-ask spread (basis points)"
                    )
                    
                    kyle_lambda = results.get('kyle_lambda', np.nan)
                    kyle_r2 = results.get('kyle_r_squared', 0)
                    st.metric(
                        "Kyle's Lambda", 
                        f"{kyle_lambda:.2e}" if not np.isnan(kyle_lambda) else "N/A",
                        delta=f"R²: {kyle_r2:.3f}" if kyle_r2 > 0 else "Low fit",
                        help="Price impact coefficient (higher = more impact per unit volume)"
                    )
                
                with col3:
                    amihud = results.get('avg_amihud_illiquidity', 0)
                    st.metric(
                        "Amihud Illiquidity", 
                        f"{amihud:.3f}" if not np.isnan(amihud) else "N/A",
                        help="Price impact per unit of trading activity (higher = less liquid)"
                    )
                    
                    efficiency = results.get('avg_price_efficiency', 0)
                    st.metric(
                        "Price Efficiency", 
                        f"{efficiency:.3f}" if not np.isnan(efficiency) else "N/A",
                        help="How directly price moves (higher = more efficient price discovery)"
                    )
                
                with col4:
                    autocorr = results.get('avg_return_autocorr', 0)
                    st.metric(
                        "Return Autocorrelation", 
                        f"{autocorr:.3f}" if not np.isnan(autocorr) else "N/A",
                        help="Predictability of returns (closer to 0 = more random)"
                    )
                    
                    vol_clustering = results.get('volatility_clustering', 0)
                    st.metric(
                        "Volatility Clustering", 
                        f"{vol_clustering:.3f}" if not np.isnan(vol_clustering) else "N/A",
                        help="Tendency for high volatility to follow high volatility"
                    )
                
                # Summary Dashboard
                if show_summary_dashboard:
                    st.subheader("🎯 Microstructure Summary Dashboard")
                    try:
                        summary_fig = st.session_state.quant_viz.plot_microstructure_summary_dashboard(results)
                        st.plotly_chart(summary_fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating summary dashboard: {str(e)}")
                
                # Detailed Time Series Analysis
                st.subheader("📈 Microstructure Time Series Analysis")
                try:
                    micro_fig = st.session_state.quant_viz.plot_microstructure_analysis(df, results, analysis_window)
                    st.plotly_chart(micro_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating microstructure analysis: {str(e)}")
                
                # Detailed Metrics Table
                if show_detailed_metrics:
                    st.subheader("📋 Detailed Metrics Breakdown")
                    
                    # Create detailed metrics dataframe
                    detailed_metrics = {
                        'Metric': [
                            'Average Realized Volatility (1h)',
                            'Average Realized Volatility (4h)', 
                            'Volatility of Volatility',
                            'Bid-Ask Spread Estimate',
                            'Spread Confidence Level',
                            "Kyle's Lambda (Price Impact)",
                            "Kyle's Lambda R-Squared",
                            'Amihud Illiquidity (Raw)',
                            'Amihud Illiquidity (Smoothed)',
                            'Average Price Efficiency',
                            'Average Return Autocorrelation',
                            'Volatility Clustering',
                            'Average Price Velocity',
                            'Price Velocity Volatility'
                        ],
                        'Value': [
                            f"{results.get('avg_realized_volatility_1h', 0) * 100:.2f}%" if not np.isnan(results.get('avg_realized_volatility_1h', np.nan)) else "N/A",
                            f"{results.get('avg_realized_volatility_4h', 0) * 100:.2f}%" if not np.isnan(results.get('avg_realized_volatility_4h', np.nan)) else "N/A",
                            f"{results.get('volatility_of_volatility', 0) * 100:.2f}%" if not np.isnan(results.get('volatility_of_volatility', np.nan)) else "N/A",
                            f"{results.get('bid_ask_spread_estimate', 0) * 10000:.2f} bps" if not np.isnan(results.get('bid_ask_spread_estimate', np.nan)) else "N/A",
                            results.get('spread_confidence', 'Unknown'),
                            f"{results.get('kyle_lambda', 0):.4e}" if not np.isnan(results.get('kyle_lambda', np.nan)) else "N/A",
                            f"{results.get('kyle_r_squared', 0):.4f}",
                            f"{results.get('avg_amihud_illiquidity', 0):.4f}" if not np.isnan(results.get('avg_amihud_illiquidity', np.nan)) else "N/A",
                            f"{results.get('avg_amihud_smooth', 0):.4f}" if not np.isnan(results.get('avg_amihud_smooth', np.nan)) else "N/A",
                            f"{results.get('avg_price_efficiency', 0):.4f}" if not np.isnan(results.get('avg_price_efficiency', np.nan)) else "N/A",
                            f"{results.get('avg_return_autocorr', 0):.4f}" if not np.isnan(results.get('avg_return_autocorr', np.nan)) else "N/A",
                            f"{results.get('volatility_clustering', 0):.4f}" if not np.isnan(results.get('volatility_clustering', np.nan)) else "N/A",
                            f"{results.get('avg_price_velocity', 0):.6f}" if not np.isnan(results.get('avg_price_velocity', np.nan)) else "N/A",
                            f"{results.get('price_velocity_volatility', 0):.6f}" if not np.isnan(results.get('price_velocity_volatility', np.nan)) else "N/A"
                        ],
                        'Description': [
                            'Short-term volatility measure (annualized)',
                            'Medium-term volatility measure (annualized)',
                            'Volatility of the volatility (instability measure)',
                            'Estimated transaction cost from price reversals',
                            'Reliability of the spread estimate',
                            'Market impact per unit of signed volume',
                            'Goodness of fit for Kyle lambda regression',
                            'Price impact per unit of activity (direct)',
                            'Price impact per unit of activity (smoothed)',
                            'How efficiently price moves without noise',
                            'Predictability of future returns from past returns',
                            'Tendency of volatility to persist',
                            'Average rate of absolute price change',
                            'Variability in the rate of price change'
                        ]
                    }
                    
                    metrics_df = pl.DataFrame(detailed_metrics)
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Interpretation Guide
                st.subheader("📚 Interpretation Guide")
                
                with st.expander("🔍 Understanding Market Microstructure Metrics"):
                    st.markdown("""
                    **Volatility Metrics:**
                    - **Realized Volatility**: Measures actual price volatility over time windows
                    - **Volatility of Volatility**: Higher values indicate unstable, unpredictable volatility
                    
                    **Liquidity & Transaction Cost Metrics:**
                    - **Bid-Ask Spread**: Estimated transaction cost (lower = cheaper to trade)
                    - **Kyle's Lambda**: Price impact coefficient (lower = less market impact)
                    - **Amihud Illiquidity**: Price impact per trading activity (lower = more liquid)
                    
                    **Market Quality Metrics:**
                    - **Price Efficiency**: How directly price moves (higher = less noise)
                    - **Return Autocorrelation**: Predictability (closer to 0 = more efficient)
                    - **Volatility Clustering**: GARCH-like effects (higher = more clustering)
                    
                    **Price Dynamics:**
                    - **Price Velocity**: Rate of price change (higher = more active)
                    - **Price Velocity Volatility**: Consistency of price movement
                    """)
                
                with st.expander("💡 Trading Implications"):
                    st.markdown("""
                    **High Liquidity Indicators:**
                    - Low bid-ask spread estimate
                    - Low Kyle's lambda
                    - Low Amihud illiquidity
                    - High price efficiency
                    
                    **Market Quality Indicators:**
                    - Return autocorrelation near zero
                    - Stable realized volatility
                    - Low volatility clustering
                    
                    **Risk Indicators:**
                    - High volatility of volatility
                    - High return autocorrelation (predictable reversals)
                    - High Kyle's lambda (high impact trades)
                    """)
                
                st.success("✅ Microstructure analysis completed successfully!")
                st.info("💡 **Tip**: Use this analysis to assess market quality, transaction costs, and optimal trading strategies for this token.")
                
                # Add comprehensive interpretation guide for microstructure analysis
                with st.expander("📖 Comprehensive Microstructure Analysis Interpretation Guide"):
                    st.markdown("""
                    **How to Read the Microstructure Analysis:**
                    
                    **🎯 Summary Dashboard** (6-Panel Analysis):
                    
                    **Panel 1 - Realized Volatility Evolution**:
                    - **Multiple lines**: Different rolling window volatilities over time
                    - **Y-axis**: Annualized volatility percentage
                    - **Pattern insights**: Volatility spikes indicate high-risk periods
                    - **Trading application**: Reduce position sizes during volatility spikes
                    
                    **Panel 2 - Bid-Ask Spread Estimate (Roll's Estimator)**:
                    - **Line trend**: Estimated transaction costs over time
                    - **Y-axis**: Basis points (1 bps = 0.01%)
                    - **Lower values**: Cheaper to trade, higher liquidity
                    - **Trading application**: Time trades when spreads are narrow
                    
                    **Panel 3 - Price Impact (Kyle's Lambda)**:
                    - **Line evolution**: How much each trade moves the price
                    - **Higher values**: Greater price impact per trade
                    - **Pattern**: Declining lambda = improving liquidity
                    - **Trading application**: Break large orders when lambda is high
                    
                    **Panel 4 - Amihud Illiquidity**:
                    - **Price impact per unit of activity**: Lower = more liquid
                    - **Trend analysis**: Declining trend = improving liquidity
                    - **Spikes**: Periods of poor liquidity
                    - **Trading application**: Avoid large trades during spikes
                    
                    **Panel 5 - Price Efficiency**:
                    - **Directness of price movement**: Higher = less noise
                    - **Values near 1**: Very efficient price discovery
                    - **Values near 0**: Noisy, inefficient price movement
                    - **Trading application**: Trust trends more when efficiency is high
                    
                    **Panel 6 - Return Autocorrelation**:
                    - **Predictability measure**: How much returns predict future returns
                    - **Near zero**: Efficient, unpredictable (good)
                    - **High positive**: Momentum effects (trend continuation)
                    - **High negative**: Mean reversion effects (reversal patterns)
                    
                    **🎯 Key Metric Interpretations:**
                    
                    **Liquidity Assessment:**
                    - **Excellent liquidity**: Spread < 10 bps, Kyle's λ < 1e-6, Amihud < 0.1
                    - **Good liquidity**: Spread < 50 bps, Kyle's λ < 1e-5, Amihud < 0.5
                    - **Poor liquidity**: Spread > 100 bps, Kyle's λ > 1e-4, Amihud > 1.0
                    - **Warning signs**: Rapidly increasing spreads or price impact
                    
                    **Market Quality Indicators:**
                    - **High quality**: Price efficiency > 0.7, |Autocorr| < 0.1, Low vol clustering
                    - **Medium quality**: Price efficiency 0.4-0.7, |Autocorr| < 0.3
                    - **Poor quality**: Price efficiency < 0.4, |Autocorr| > 0.5
                    - **Red flags**: High volatility clustering, extreme autocorrelation
                    
                    **Transaction Cost Analysis:**
                    - **Low cost environment**: Spread < 20 bps, Low price impact
                    - **Moderate costs**: Spread 20-100 bps, Moderate impact
                    - **High cost environment**: Spread > 100 bps, High impact
                    - **Cost timing**: Trade when all cost metrics are low
                    
                    **🎯 Trading Strategy Applications:**
                    
                    **Scalping Strategies**:
                    - **Requirements**: Low spreads (<20 bps), High efficiency (>0.7)
                    - **Optimal conditions**: Low price impact, Minimal autocorrelation
                    - **Risk management**: Exit quickly if conditions deteriorate
                    - **Position sizing**: Small sizes to minimize market impact
                    
                    **Momentum Trading**:
                    - **Look for**: Positive return autocorrelation, High efficiency
                    - **Avoid when**: High price impact, Poor liquidity
                    - **Entry timing**: During periods of low volatility clustering
                    - **Exit conditions**: When autocorrelation turns negative
                    
                    **Mean Reversion Strategies**:
                    - **Ideal conditions**: Negative return autocorrelation, Good liquidity
                    - **Entry signals**: After efficiency drops (noise increases)
                    - **Risk management**: Stop if autocorrelation becomes strongly positive
                    - **Position sizing**: Larger sizes when price impact is low
                    
                    **Position Trading**:
                    - **Focus on**: Long-term efficiency trends, Stable liquidity
                    - **Entry timing**: When short-term microstructure is favorable
                    - **Hold conditions**: Maintain while liquidity and efficiency remain good
                    - **Exit signals**: Deteriorating microstructure quality
                    
                    **⚠️ Risk Warning Signals:**
                    
                    **Immediate Exit Signals**:
                    - **Liquidity crisis**: Spreads >500 bps, Extreme price impact
                    - **Market breakdown**: Price efficiency <0.2, Extreme autocorrelation
                    - **Volatility explosion**: Vol clustering >0.8, Unstable patterns
                    
                    **Reduce Position Signals**:
                    - **Deteriorating liquidity**: Increasing spreads/impact over time
                    - **Efficiency decline**: Falling price efficiency trend
                    - **Increased predictability**: Rising |autocorrelation| values
                    
                    **Monitor Closely**:
                    - **Volatile microstructure**: Rapidly changing metrics
                    - **Regime changes**: Sudden shifts in relationship patterns
                    - **Low confidence**: High volatility of volatility
                    
                    **📊 Advanced Applications:**
                    
                    **Portfolio Risk Management**:
                    - **Diversification**: Mix tokens with different microstructure profiles
                    - **Position allocation**: Larger weights to higher-quality microstructure
                    - **Rebalancing timing**: Use microstructure signals for timing
                    
                    **Execution Optimization**:
                    - **Order sizing**: Adjust based on current price impact
                    - **Timing strategies**: Execute during favorable microstructure windows
                    - **Slippage control**: Monitor and adapt to changing transaction costs
                    
                    **Risk Model Enhancement**:
                    - **Dynamic volatility**: Use realized vol for position sizing
                    - **Liquidity-adjusted VaR**: Incorporate transaction costs in risk metrics
                    - **Regime detection**: Use microstructure changes to detect market shifts
                    """)
                

def show_price_distribution_evolution(selected_tokens, selection_mode):
    """Enhanced Price Distribution Evolution Analysis"""
    st.header("📊 Price Distribution Evolution Analysis")
    st.markdown("""
    **Comprehensive statistical analysis of how price return distributions change over time**
    
    This analysis divides the token's lifecycle into periods and examines:
    - **Distribution shape evolution** (skewness, kurtosis, normality)
    - **Statistical significance testing** (Shapiro-Wilk, Jarque-Bera tests)
    - **Distribution fitting** (normal, t-distribution for heavy tails)
    - **Q-Q plots** for visual normality assessment
    - **Outlier detection** and box plot analysis
    """)
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Single Token Analysis", "Multi-Token Aggregated Analysis"],
        help="Choose between detailed single token analysis or aggregated multi-token analysis"
    )
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    with col1:
        if analysis_mode == "Single Token Analysis":
            selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
        else:
            st.info(f"Analyzing {len(selected_tokens)} selected tokens")
            selected_token = None
    with col2:
        n_periods = st.selectbox("Number of Periods", [4, 6, 8, 12], index=1)
    with col3:
        show_summary = st.checkbox("Show Summary Dashboard", value=True)
    
    # Analysis execution
    period_stats = None  # Initialize variable
    
    if (analysis_mode == "Single Token Analysis" and selected_token) or (analysis_mode == "Multi-Token Aggregated Analysis" and selected_tokens):
        if st.button("📊 Analyze Distribution Evolution", type="primary"):
            
            if analysis_mode == "Single Token Analysis":
                # Single token analysis
                df = st.session_state.data_loader.get_token_data(selected_token)
                
                if df is None or df.is_empty():
                    st.error(f"No data available for {selected_token}")
                    return
                
                if df.height < n_periods * 20:
                    st.warning(f"Insufficient data points ({df.height}) for {n_periods} periods. Need at least {n_periods * 20} points.")
                    return
                
                with st.spinner("Performing comprehensive distribution analysis..."):
                    try:
                        qv = st.session_state.quant_viz
                        fig, period_stats = qv.plot_price_distribution_evolution(df, n_periods=n_periods)
                        
                        st.subheader(f"📈 Distribution Evolution Analysis - {selected_token}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during single token analysis: {str(e)}")
                        return
            
            else:
                # Multi-token aggregated analysis
                with st.spinner(f"Performing aggregated distribution analysis for {len(selected_tokens)} tokens..."):
                    try:
                        # Load data for all selected tokens
                        token_data = []
                        progress_bar = st.progress(0)
                        
                        for i, token_name in enumerate(selected_tokens):
                            try:
                                df = st.session_state.data_loader.get_token_data(token_name)
                                if df is not None and not df.is_empty() and df.height >= n_periods * 20:
                                    token_data.append((token_name, df))
                                progress_bar.progress((i + 1) / len(selected_tokens))
                            except Exception:
                                continue
                        
                        progress_bar.empty()
                        
                        if len(token_data) < 2:
                            st.error(f"Need at least 2 valid tokens for aggregated analysis. Only {len(token_data)} tokens have sufficient data.")
                            return
                        
                        st.info(f"Successfully loaded {len(token_data)} tokens with sufficient data")
                        
                        # Perform multi-token aggregated analysis
                        qv = st.session_state.quant_viz
                        fig, period_stats = qv.plot_multi_token_distribution_evolution(token_data, n_periods=n_periods)
                        
                        st.subheader(f"📈 Aggregated Distribution Evolution Analysis - {len(token_data)} Tokens")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display aggregated statistics table
                        st.subheader("📊 Aggregated Period Statistics")
                        
                        # Convert period_stats to DataFrame for display
                        aggregated_df = pl.DataFrame(period_stats)
                        
                        # Format for better display
                        display_df = aggregated_df.select([
                            pl.col('period').alias('Period'),
                            pl.col('total_tokens').alias('Tokens'),
                            pl.col('avg_mean_return').round(3).alias('Avg Return (%)'),
                            pl.col('std_mean_return').round(3).alias('Return Std (%)'),
                            pl.col('avg_volatility').round(3).alias('Avg Volatility (%)'),
                            pl.col('avg_skewness').round(3).alias('Avg Skewness'),
                            pl.col('avg_kurtosis').round(3).alias('Avg Kurtosis'),
                            pl.col('normality_percentage').round(1).alias('Normal %')
                        ])
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Key insights for aggregated analysis
                        st.subheader("🔍 Aggregated Key Insights")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_normal_pct = aggregated_df['normality_percentage'].mean()
                            st.metric("Avg Normal %", format_percentage(avg_normal_pct))
                        with col2:
                            best_period = aggregated_df.sort('avg_mean_return', descending=True).limit(1)
                            if len(best_period) > 0:
                                period_num = best_period['period'][0]
                                return_val = best_period['avg_mean_return'][0]
                                st.metric("Best Period", f"Period {period_num}", f"{return_val:.2f}%")
                        with col3:
                            most_stable = aggregated_df.sort('std_mean_return').limit(1)
                            if len(most_stable) > 0:
                                period_num = most_stable['period'][0]
                                std_val = most_stable['std_mean_return'][0]
                                st.metric("Most Stable", f"Period {period_num}", f"±{std_val:.2f}%")
                        with col4:
                            total_observations = aggregated_df['total_tokens'].sum()
                            st.metric("Total Observations", format_data_points(total_observations))
                        
                        # Interpretation for aggregated analysis
                        st.markdown("#### 📖 **Aggregated Analysis Interpretation:**")
                        
                        if avg_normal_pct >= 70:
                            st.success("🎯 **Strong Distributional Consistency**: Most tokens show normal distributions across periods")
                        elif avg_normal_pct >= 30:
                            st.warning("⚠️ **Mixed Distributional Behavior**: Some periods/tokens deviate from normality")
                        else:
                            st.error("🚨 **Non-Normal Behavior Dominant**: Most periods show non-normal distributions")
                        
                        # Trading implications for aggregated analysis
                        st.markdown("#### 💡 **Multi-Token Trading Implications:**")
                        
                        implications = []
                        
                        # Check for consistent patterns
                        return_trend = aggregated_df['avg_mean_return'].to_list()
                        if len(return_trend) > 1:
                            if return_trend[-1] < return_trend[0]:
                                implications.append("📉 **Declining Returns Pattern**: Consider shorter holding periods across token portfolio")
                        
                        # Check volatility patterns
                        vol_trend = aggregated_df['avg_volatility'].to_list()
                        if len(vol_trend) > 1:
                            if vol_trend[-1] > vol_trend[0]:
                                implications.append("📊 **Rising Volatility Pattern**: Increase caution in later periods")
                        
                        # Check consistency
                        return_dispersion = aggregated_df['std_mean_return'].mean()
                        if return_dispersion > 2.0:
                            implications.append("🎯 **High Token Variability**: Individual token selection critical - not all tokens follow pattern")
                        
                        if avg_normal_pct < 50:
                            implications.append("⚠️ **Non-Normal Risk**: Use robust risk management across entire portfolio")
                        
                        if implications:
                            for implication in implications:
                                st.markdown(f"- {implication}")
                        else:
                            st.success("✅ **Stable Multi-Token Patterns**: Consistent behavior across token portfolio")
                        
                        # Set period_stats for the shared display code
                        # Note: period_stats is now aggregated stats, not individual token stats
                    
                    except Exception as e:
                        st.error(f"Error during aggregated analysis: {str(e)}")
                        return
            
            # Shared display code for results (only runs for single token analysis)
            if period_stats is not None and analysis_mode == "Single Token Analysis":
                # Show summary dashboard if requested
                if show_summary:
                    st.subheader("📊 Evolution Summary Dashboard")
                    try:
                        qv = st.session_state.quant_viz
                        summary_fig = qv.plot_distribution_evolution_summary(period_stats)
                        st.plotly_chart(summary_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate summary dashboard: {str(e)}")
                
                # Display detailed statistics table
                st.subheader("📋 Detailed Period Statistics")
                
                stats_data = []
                for stat in period_stats:
                    stats_data.append({
                        'Period': stat['period'],
                        'Start Time': stat['start_time'].strftime('%H:%M') if stat['start_time'] else 'N/A',
                        'End Time': stat['end_time'].strftime('%H:%M') if stat['end_time'] else 'N/A',
                        'Sample Size': stat['count'],
                        'Mean Return (%)': f"{stat['mean']:.3f}",
                        'Volatility (%)': f"{stat['std']:.3f}",
                        'Median (%)': f"{stat['median']:.3f}",
                        'Skewness': f"{stat['skewness']:.3f}",
                        'Excess Kurtosis': f"{stat['kurtosis']:.3f}",
                        'Min Return (%)': f"{stat['min']:.3f}",
                        'Max Return (%)': f"{stat['max']:.3f}",
                        'IQR (%)': f"{stat['q75'] - stat['q25']:.3f}",
                        'Shapiro p-value': f"{stat['shapiro_p_value']:.4f}" if stat['shapiro_p_value'] > 0 else 'N/A',
                        'Is Normal?': '✅ Yes' if stat['is_normal'] else '❌ No',
                        'JB Statistic': f"{stat['jb_statistic']:.2f}"
                    })
                
                stats_df = pl.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Key insights and interpretation
                st.subheader("🔍 Key Insights & Interpretation")
                
                mean_trend = "increasing" if period_stats[-1]['mean'] > period_stats[0]['mean'] else "decreasing"
                vol_trend = "increasing" if period_stats[-1]['std'] > period_stats[0]['std'] else "decreasing"
                normal_periods = sum(1 for stat in period_stats if stat['is_normal'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Normal Periods", f"{normal_periods}/{len(period_stats)}")
                with col2:
                    avg_skew = np.mean([stat['skewness'] for stat in period_stats])
                    st.metric("Avg Skewness", f"{avg_skew:.3f}")
                with col3:
                    avg_kurtosis = np.mean([stat['kurtosis'] for stat in period_stats])
                    st.metric("Avg Excess Kurtosis", f"{avg_kurtosis:.3f}")
                with col4:
                    volatility_stability = np.std([stat['std'] for stat in period_stats])
                    st.metric("Volatility Stability", f"{volatility_stability:.3f}")
                
                # Interpretation guide
                st.markdown("#### 📖 **Interpretation Guide:**")
                
                if normal_periods >= len(period_stats) * 0.7:
                    st.success("🎯 **Mostly Normal Distributions**: Returns follow normal distribution in most periods - suitable for standard risk models")
                elif normal_periods >= len(period_stats) * 0.3:
                    st.warning("⚠️ **Mixed Distribution Behavior**: Some periods deviate from normality - use robust risk measures")
                else:
                    st.error("🚨 **Non-Normal Distributions**: Most periods show non-normal behavior - standard risk models may be inadequate")
                
                if abs(avg_skew) > 0.5:
                    skew_direction = "positive" if avg_skew > 0 else "negative"
                    st.info(f"📊 **Significant {skew_direction.title()} Skewness**: Returns show {skew_direction} asymmetry - tail risk considerations important")
                
                if avg_kurtosis > 1:
                    st.warning("📈 **Heavy Tails Detected**: Higher probability of extreme returns - increase risk buffer")
                elif avg_kurtosis < -0.5:
                    st.info("📉 **Light Tails**: Lower probability of extreme returns than normal distribution")
                
                # Trading implications
                st.markdown("#### 💡 **Trading Implications:**")
                
                implications = []
                if mean_trend == "decreasing":
                    implications.append("📉 **Declining Returns**: Consider shorter holding periods or exit strategies")
                if vol_trend == "increasing":
                    implications.append("📊 **Rising Volatility**: Increase position sizing caution and stop-loss levels")
                if normal_periods < len(period_stats) * 0.5:
                    implications.append("🎯 **Non-Normal Risk**: Use Value-at-Risk models designed for non-normal distributions")
                if avg_kurtosis > 2:
                    implications.append("⚠️ **Extreme Events**: High probability of large moves - consider wider stop-losses")
                
                if implications:
                    for implication in implications:
                        st.markdown(f"- {implication}")
                else:
                    st.success("✅ **Stable Distribution Characteristics**: Standard risk management approaches should be effective")
    
    # Educational content
    with st.expander("📚 Learn More About Distribution Analysis"):
        st.markdown("""
        **Key Concepts:**
        
        **Skewness**: Measures asymmetry of the distribution
        - **Positive skewness**: More extreme positive returns (right tail)
        - **Negative skewness**: More extreme negative returns (left tail)
        - **Zero skewness**: Symmetric distribution
        
        **Kurtosis**: Measures tail heaviness compared to normal distribution
        - **Positive excess kurtosis**: Heavier tails, more extreme events
        - **Negative excess kurtosis**: Lighter tails, fewer extreme events
        - **Zero excess kurtosis**: Normal distribution tails
        
        **Normality Tests:**
        - **Shapiro-Wilk**: Tests if data comes from normal distribution (p > 0.05 = normal)
        - **Jarque-Bera**: Tests normality based on skewness and kurtosis
        
        **Q-Q Plots**: Visual test for normality
        - Points on diagonal line = normal distribution
        - Deviations show non-normal behavior
        
        **Distribution Evolution**: How these characteristics change over time
        - Early periods often show different behavior than later periods
        - Important for risk management and strategy adaptation
        """)

def show_optimal_holding_period(selected_tokens, selection_mode):
    """Enhanced Optimal Holding Period Analysis"""
    st.header("⏰ Optimal Holding Period Analysis")
    st.markdown("""
    **Find the optimal holding period based on risk-adjusted returns**
    
    🎯 **How it works**: Tests different holding times (5min, 10min, 30min, etc.) by simulating:
    - Buying at EVERY possible moment in each token's lifecycle
    - Holding for the exact time period being tested
    - Selling automatically after that time
    - Measuring which holding time gives the best risk-adjusted returns
    
    💡 **Key insight**: Finds the sweet spot between:
    - **Return potential**: Longer holds may capture bigger moves
    - **Risk management**: Shorter holds reduce exposure to volatility
    - **Practical execution**: Realistic monitoring requirements
    """)
    
    # Add explanation box
    with st.expander("🤔 How does this analysis work? (Click to expand)"):
        st.markdown("""
        **Step-by-step process:**
        
        1. **Choose a holding period** (e.g., 30 minutes)
        2. **Simulate thousands of trades**:
           - Buy at minute 1 → Hold 30min → Sell at minute 31
           - Buy at minute 2 → Hold 30min → Sell at minute 32
           - Buy at minute 3 → Hold 30min → Sell at minute 33
           - ...continue for entire token lifecycle
        3. **Calculate performance metrics** for all these simulated trades
        4. **Repeat for all holding periods** (5min, 10min, 15min, etc.)
        5. **Find the optimal period** with best risk-adjusted returns
        
        **Important**: This is NOT about entry timing - it tests EVERY possible entry point!
        
        **Analysis Granularity** controls how many holding periods to test:
        - **1-2 min intervals**: Tests 1min, 2min, 3min, 4min... (very detailed)
        - **5 min intervals**: Tests 5min, 10min, 15min, 20min... (faster)
        - **10 min intervals**: Tests 10min, 20min, 30min... (quickest overview)
        """)
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Single Token Analysis", "Multi-Token Aggregated Analysis"],
        help="Choose between detailed single token analysis or aggregated multi-token analysis"
    )
    
    # Parameters with better explanations
    col1, col2 = st.columns(2)
    with col1:
        max_period = st.slider("Max Holding Period (minutes)", min_value=30, max_value=480, value=240, step=30,
                              help="Maximum time to hold positions in the analysis")
    with col2:
        step_size = st.selectbox("Analysis Granularity", [1, 2, 5, 10], index=2, 
                                format_func=lambda x: f"{x} min intervals" + (" (detailed)" if x <= 2 else " (fast)" if x >= 5 else ""),
                                help="How finely to test holding periods:\n• 1-2 min: Very detailed but slower\n• 5-10 min: Faster analysis, good for patterns")
    
    if analysis_mode == "Single Token Analysis":
        # Single token detailed analysis
        selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
        
        if selected_token and st.button("⏰ Find Optimal Holding Period", type="primary"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            if df is None or df.is_empty():
                st.error(f"No data available for {selected_token}")
                return
            
            with st.spinner("Calculating optimal holding periods..."):
                # Test different holding periods
                periods = list(range(step_size, min(max_period + 1, len(df)//2), step_size))
                results = []
                
                for period in periods:
                    returns = df['price'].pct_change(period).drop_nulls()
                    
                    if len(returns) > 0:
                        mean_return = returns.mean() * 100
                        volatility = returns.std() * 100
                        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(525600 / period)
                        
                        # Calculate win rate
                        win_rate = (returns > 0).mean() * 100
                        
                        # Calculate max drawdown for this period
                        cumulative = (1 + returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        max_drawdown = drawdown.min() * 100
                        
                        results.append({
                            'Holding Period (min)': period,
                            'Mean Return (%)': mean_return,
                            'Volatility (%)': volatility,
                            'Sharpe Ratio': sharpe,
                            'Win Rate (%)': win_rate,
                            'Max Drawdown (%)': max_drawdown,
                            'Sample Size': len(returns)
                        })
                
                if results:
                    results_df = pl.DataFrame(results)
                    
                    # Display results
                    st.subheader("📊 Holding Period Analysis Results")
                    st.dataframe(results_df.sort('Sharpe Ratio', descending=True), use_container_width=True)
                    
                    # Find optimal periods
                    optimal_sharpe_idx = results_df['Sharpe Ratio'].arg_max()
                    optimal_return_idx = results_df['Mean Return (%)'].arg_max()
                    optimal_winrate_idx = results_df['Win Rate (%)'].arg_max()
                    
                    # Key insights
                    st.subheader("🎯 Key Insights")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if optimal_sharpe_idx is not None:
                            optimal_stats = results_df[optimal_sharpe_idx]
                            period = optimal_stats['Holding Period (min)'][0]
                            sharpe = optimal_stats['Sharpe Ratio'][0]
                            st.success(f"🏆 **Best Sharpe Ratio**\n\n{period} min period\nSharpe: {sharpe:.3f}")
                    
                    with col2:
                        if optimal_return_idx is not None:
                            return_stats = results_df[optimal_return_idx]
                            period = return_stats['Holding Period (min)'][0]
                            ret = return_stats['Mean Return (%)'][0]
                            st.info(f"📈 **Highest Return**\n\n{period} min period\nReturn: {ret:.2f}%")
                    
                    with col3:
                        if optimal_winrate_idx is not None:
                            winrate_stats = results_df[optimal_winrate_idx]
                            period = winrate_stats['Holding Period (min)'][0]
                            wr = winrate_stats['Win Rate (%)'][0]
                            st.info(f"🎯 **Best Win Rate**\n\n{period} min period\nWin Rate: {wr:.1f}%")
                    
                    # Visualization with explanation
                    st.subheader("📈 Performance Visualization")
                    
                    st.info("""
                    **📊 How to read the chart:**
                    - **X-axis**: How long to hold positions (in minutes)
                    - **Y-axis**: Performance metrics (return, risk, success rate)
                    - **Peak points**: Optimal holding periods for each metric
                    - **Look for**: High Sharpe ratio (best risk-adjusted returns)
                    """)
                    
                    try:
                        qv = st.session_state.quant_viz
                        fig = qv.plot_optimal_holding_period(df, max_period=max_period, step=step_size)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate visualization: {str(e)}")
                else:
                    st.error("No valid holding periods could be analyzed.")
    
    else:
        # Multi-token aggregated analysis
        if len(selected_tokens) < 2:
            st.warning("Please select at least 2 tokens for multi-token analysis.")
            return
        
        # Limit tokens for performance
        max_tokens = st.slider("Max tokens to analyze", min_value=2, max_value=min(50, len(selected_tokens)), value=min(20, len(selected_tokens)))
        tokens_to_analyze = selected_tokens[:max_tokens]
        
        if st.button("⏰ Analyze Optimal Periods Across Tokens", type="primary"):
            with st.spinner(f"Analyzing optimal holding periods for {len(tokens_to_analyze)} tokens..."):
                # Load data for all selected tokens
                token_results = []
                progress_bar = st.progress(0)
                
                for i, token_name in enumerate(tokens_to_analyze):
                    try:
                        df = st.session_state.data_loader.get_token_data(token_name)
                        if df is not None and not df.is_empty() and len(df) > max_period:
                            
                            # Test different holding periods for this token
                            periods = list(range(step_size, min(max_period + 1, len(df)//2), step_size))
                            
                            for period in periods:
                                returns = df['price'].pct_change(period).drop_nulls()
                                
                                if len(returns) > 10:  # Minimum sample size
                                    mean_return = returns.mean() * 100
                                    volatility = returns.std() * 100
                                    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(525600 / period)
                                    win_rate = (returns > 0).mean() * 100
                                    
                                    token_results.append({
                                        'Token': token_name,
                                        'Holding Period (min)': period,
                                        'Mean Return (%)': mean_return,
                                        'Volatility (%)': volatility,
                                        'Sharpe Ratio': sharpe,
                                        'Win Rate (%)': win_rate,
                                        'Sample Size': len(returns)
                                    })
                        
                        progress_bar.progress((i + 1) / len(tokens_to_analyze))
                    except Exception as e:
                        continue
                
                progress_bar.empty()
                
                if token_results:
                    # Convert to DataFrame for analysis
                    all_results_df = pl.DataFrame(token_results)
                    
                    # Calculate aggregated statistics by holding period
                    aggregated_stats = all_results_df.group_by('Holding Period (min)').agg([
                        pl.col('Mean Return (%)').mean().alias('Avg Return (%)'),
                        pl.col('Mean Return (%)').std().alias('Return Std (%)'),
                        pl.col('Volatility (%)').mean().alias('Avg Volatility (%)'),
                        pl.col('Sharpe Ratio').mean().alias('Avg Sharpe Ratio'),
                        pl.col('Sharpe Ratio').std().alias('Sharpe Std'),
                        pl.col('Win Rate (%)').mean().alias('Avg Win Rate (%)'),
                        pl.col('Token').count().alias('Token Count'),
                        pl.col('Sample Size').mean().alias('Avg Sample Size')
                    ]).sort('Holding Period (min)')
                    
                    # Display aggregated results
                    st.subheader(f"📊 Aggregated Optimal Holding Period Analysis ({len(tokens_to_analyze)} tokens)")
                    st.dataframe(aggregated_stats.sort('Avg Sharpe Ratio', descending=True), use_container_width=True)
                    
                    # Find optimal periods across all tokens
                    best_sharpe_period = aggregated_stats.sort('Avg Sharpe Ratio', descending=True).limit(1)
                    best_return_period = aggregated_stats.sort('Avg Return (%)', descending=True).limit(1)
                    best_winrate_period = aggregated_stats.sort('Avg Win Rate (%)', descending=True).limit(1)
                    
                    # Key insights for multi-token analysis
                    st.subheader("🔍 Multi-Token Key Insights")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if len(best_sharpe_period) > 0:
                            period = best_sharpe_period['Holding Period (min)'][0]
                            sharpe = best_sharpe_period['Avg Sharpe Ratio'][0]
                            st.success(f"🏆 **Best Avg Sharpe**\n\n{period} min period\nSharpe: {sharpe:.3f}")
                    
                    with col2:
                        if len(best_return_period) > 0:
                            period = best_return_period['Holding Period (min)'][0]
                            ret = best_return_period['Avg Return (%)'][0]
                            st.info(f"📈 **Highest Avg Return**\n\n{period} min period\nReturn: {ret:.2f}%")
                    
                    with col3:
                        if len(best_winrate_period) > 0:
                            period = best_winrate_period['Holding Period (min)'][0]
                            wr = best_winrate_period['Avg Win Rate (%)'][0]
                            st.info(f"🎯 **Best Avg Win Rate**\n\n{period} min period\nWin Rate: {wr:.1f}%")
                    
                    with col4:
                        total_observations = aggregated_stats['Token Count'].sum()
                        st.metric("Total Observations", format_data_points(total_observations))
                    
                    # Multi-token visualization with explanation
                    st.subheader("📈 Multi-Token Performance Visualization")
                    
                    # Add plot explanation
                    st.info("""
                    **📊 How to read these charts:**
                    
                    **Top Left** - Risk-Adjusted Performance: Higher line = better holding periods
                    **Top Right** - Risk vs Return: Look for points in top-left (high return, low risk)  
                    **Bottom Left** - Success Rate: Higher bars = more profitable trades
                    **Bottom Right** - Token Heatmap: Green = good performance, Red = poor performance
                    """)
                    
                    try:
                        qv = st.session_state.quant_viz
                        fig = qv.plot_multi_token_optimal_holding_period(all_results_df, aggregated_stats)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate multi-token visualization: {str(e)}")
                    
                    # Detailed token-by-token results
                    with st.expander("📋 Detailed Token-by-Token Results"):
                        # Find best period for each token
                        token_optimal = all_results_df.group_by('Token').agg([
                            pl.col('Sharpe Ratio').max().alias('Best Sharpe'),
                            pl.col('Holding Period (min)').filter(pl.col('Sharpe Ratio') == pl.col('Sharpe Ratio').max()).first().alias('Optimal Period (min)'),
                            pl.col('Mean Return (%)').filter(pl.col('Sharpe Ratio') == pl.col('Sharpe Ratio').max()).first().alias('Optimal Return (%)'),
                            pl.col('Win Rate (%)').filter(pl.col('Sharpe Ratio') == pl.col('Sharpe Ratio').max()).first().alias('Optimal Win Rate (%)')
                        ]).sort('Best Sharpe', descending=True)
                        
                        st.dataframe(token_optimal, use_container_width=True)
                    
                    # Multi-token interpretation
                    st.subheader("🎯 Multi-Token Trading Implications")
                    
                    # Analyze consistency across tokens
                    period_consistency = aggregated_stats['Sharpe Std'].mean()
                    optimal_period_range = aggregated_stats.filter(pl.col('Avg Sharpe Ratio') > aggregated_stats['Avg Sharpe Ratio'].max() * 0.9)
                    
                    implications = []
                    
                    if period_consistency < 0.5:
                        implications.append("✅ **High Consistency**: Most tokens show similar optimal periods - good for systematic strategies")
                    else:
                        implications.append("⚠️ **High Variability**: Tokens have different optimal periods - individual optimization recommended")
                    
                    if len(optimal_period_range) > 1:
                        min_period = optimal_period_range['Holding Period (min)'].min()
                        max_period = optimal_period_range['Holding Period (min)'].max()
                        implications.append(f"🎯 **Optimal Range**: {min_period}-{max_period} minutes shows consistently good performance")
                    
                    # Check for clear patterns
                    best_periods = aggregated_stats.sort('Avg Sharpe Ratio', descending=True).head(3)['Holding Period (min)'].to_list()
                    if all(p <= 30 for p in best_periods):
                        implications.append("⚡ **Short-Term Advantage**: Quick scalping strategies (≤30 min) show best risk-adjusted returns")
                    elif all(p >= 60 for p in best_periods):
                        implications.append("🕐 **Long-Term Advantage**: Position trading (≥60 min) shows best risk-adjusted returns")
                    else:
                        implications.append("⚖️ **Mixed Strategies**: Both short and long holding periods can be effective")
                    
                    for implication in implications:
                        st.markdown(f"- {implication}")
                
                else:
                    st.error("No valid data to analyze across the selected tokens.")
    
    # Add interpretation guide for optimal holding period
    with st.expander("📖 Optimal Holding Period Interpretation Guide"):
        st.markdown("""
        **How to Read the Optimal Holding Period Analysis:**
        
        **📊 Results Table**:
        - **Holding Period**: Time to hold position (in minutes)
        - **Mean Return**: Average profit/loss per trade
        - **Volatility**: Risk level for each holding period
        - **Sharpe Ratio**: Risk-adjusted return (higher = better)
        - **Sample Size**: Number of observations (higher = more reliable)
        
        **🎯 Optimal Period Selection:**
        - **Best Sharpe Ratio**: Highest risk-adjusted returns
        - **Balancing act**: Consider both return and risk
        - **Sample size**: Ensure sufficient data for reliability
        - **Practical constraints**: Account for execution and monitoring
        
        **📈 Pattern Analysis:**
        
        **Short Periods (1-15 minutes)**:
        - **High frequency trading**: Quick in/out strategies
        - **Noise sensitivity**: More affected by market microstructure
        - **High monitoring**: Requires constant attention
        - **Strategy**: Scalping, arbitrage, momentum capture
        
        **Medium Periods (15-60 minutes)**:
        - **Swing trading**: Balance between noise and signal
        - **Manageable monitoring**: Reasonable attention requirements
        - **Good risk/reward**: Often optimal for most traders
        - **Strategy**: Trend following, breakout trading
        
        **Long Periods (60+ minutes)**:
        - **Position trading**: Lower frequency, higher conviction
        - **Trend dependence**: Relies on sustained price movements
        - **Lower monitoring**: Less hands-on management
        - **Strategy**: Trend riding, fundamental-based trades
        
        **🎯 Strategy Implementation:**
        
        **Risk Management:**
        - **Stop-loss placement**: Set based on optimal period volatility
        - **Position sizing**: Inverse to volatility of optimal period
        - **Time stops**: Exit if not profitable within optimal window
        
        **Entry/Exit Timing:**
        - **Entry confirmation**: Wait for signal strength appropriate to period
        - **Exit discipline**: Close positions at optimal time regardless of P&L
        - **Rolling optimization**: Periodically update optimal periods
        
        **Portfolio Application:**
        - **Multiple timeframes**: Use different optimal periods for different strategies
        - **Token-specific**: Each token may have different optimal periods
        - **Market adaptation**: Optimal periods may change with market conditions
        
        **⚠️ Important Considerations:**
        - **Sample size bias**: Longer periods have fewer observations
        - **Market regime changes**: Optimal periods may shift over time
        - **Transaction costs**: Consider execution costs for shorter periods
        - **Survivorship bias**: Analysis only includes completed token lifecycle
        """)

def show_market_regime_analysis(selected_tokens, selection_mode):
    """Market Regime Analysis"""
    st.header("🌊 Market Regime Analysis")
    st.markdown("Identify different market regimes (trending, ranging, volatile)")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
    
    if selected_token:
        if st.button("🌊 Detect Market Regimes"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            with st.spinner("Detecting market regimes..."):
                regime_df = st.session_state.quant_analyzer.market_regime_detection(df)
                
                # Show regime distribution
                regime_counts = regime_df.group_by('regime').count().sort('count', descending=True)
                st.subheader("📊 Regime Distribution")
                st.dataframe(regime_counts, use_container_width=True)
                
                # Add interpretation guide for market regime analysis
                with st.expander("📖 Market Regime Analysis Interpretation Guide"):
                    st.markdown("""
                    **How to Read the Market Regime Analysis:**
                    
                    **🌊 Regime Types:**
                    
                    **Trending Regime**:
                    - **Characteristics**: Sustained directional price movement
                    - **Strategy**: Momentum trading, trend following
                    - **Risk management**: Trend-aligned stops, position pyramiding
                    - **Exit signals**: Regime change to ranging or high volatility
                    
                    **Ranging Regime**:
                    - **Characteristics**: Price oscillates within defined bounds
                    - **Strategy**: Mean reversion, range trading, contrarian plays
                    - **Risk management**: Support/resistance stops, position sizing at extremes
                    - **Exit signals**: Breakout from range (regime change)
                    
                    **High Volatility Regime**:
                    - **Characteristics**: Extreme price movements, high uncertainty
                    - **Strategy**: Option strategies, volatility trading, reduced position size
                    - **Risk management**: Wide stops, small positions, quick exits
                    - **Entry timing**: Wait for volatility normalization
                    
                    **Low Volatility Regime**:
                    - **Characteristics**: Stable, predictable price movements
                    - **Strategy**: Larger positions, trend following, breakout preparation
                    - **Risk management**: Tighter stops, higher conviction trades
                    - **Warning**: Often precedes volatility expansion
                    
                    **📊 Regime Distribution Analysis:**
                    - **Dominant regime**: Most common market state
                    - **Regime transitions**: How frequently market changes character
                    - **Regime persistence**: Average duration of each regime
                    - **Trading implications**: Adapt strategy mix to regime distribution
                    
                    **🎯 Trading Strategy Adaptation:**
                    
                    **Regime-Aware Position Sizing**:
                    - **Trending regimes**: Larger positions with trend
                    - **Ranging regimes**: Moderate positions at extremes
                    - **High volatility**: Significantly reduced position sizes
                    - **Low volatility**: Larger positions with tight stops
                    
                    **Strategy Selection by Regime**:
                    - **Identify current regime**: Use real-time regime detection
                    - **Strategy switching**: Change approach based on regime
                    - **Multi-strategy portfolio**: Allocate based on regime probabilities
                    - **Regime timing**: Enter positions during favorable regimes
                    
                    **Risk Management by Regime**:
                    - **Stop-loss adjustment**: Wider stops in volatile regimes
                    - **Time stops**: Shorter holding periods in unstable regimes
                    - **Portfolio heat**: Reduce overall exposure during risky regimes
                    - **Correlation effects**: Regimes affect token correlations
                    
                    **⚠️ Regime Change Signals:**
                    - **Volatility spikes**: May signal regime transition
                    - **Trend breaks**: Trending to ranging regime change
                    - **Range breaks**: Ranging to trending regime change
                    - **Volume patterns**: Often accompany regime changes
                    
                    **📈 Advanced Applications:**
                    - **Regime forecasting**: Predict likely next regime
                    - **Cross-token regimes**: Compare regimes across tokens
                    - **Regime clustering**: Group tokens by similar regimes
                    - **Dynamic allocation**: Adjust portfolio based on regime mix
                    """)
                

def show_multi_token_correlation(selected_tokens):
    """Multi-Token Correlation Analysis"""
    st.header("🔗 Multi-Token Correlation Analysis")
    st.markdown("Analyze correlations between selected tokens")
    
    if len(selected_tokens) < 2:
        st.warning("Please select at least 2 tokens for correlation analysis")
        return
    
    if st.button("🔗 Calculate Correlations"):
        with st.spinner(f"Calculating correlations for {len(selected_tokens)} tokens..."):
            token_data = {}
            
            for token_name in selected_tokens:
                try:
                    df = st.session_state.data_loader.get_token_data(token_name)
                    if df is not None and not df.is_empty():
                        # Use returns for correlation
                        returns = df['price'].pct_change().drop_nulls()
                        token_data[token_name] = returns
                except Exception as e:
                    st.warning(f"Error loading {token_name}: {str(e)}")
                    continue
            
            if len(token_data) >= 2:
                st.success(f"Loaded {len(token_data)} tokens for correlation analysis")
                # For now, just show that data was loaded
                # Full correlation matrix implementation would require aligning time series
                st.info("Correlation matrix calculation requires time series alignment - implementation pending")
            else:
                st.error("Need at least 2 valid tokens for correlation analysis")

def show_comprehensive_report(selected_tokens):
    """Comprehensive Analysis Report"""
    st.header("📑 Comprehensive Analysis Report")
    st.markdown("Generate a complete analysis report for selected tokens")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    if st.button("📑 Generate Report"):
        st.info("Comprehensive report generation - implementation pending")
        st.markdown("""
        **Planned Report Sections:**
        - Executive Summary
        - Risk Metrics Analysis
        - Entry/Exit Optimization
        - Market Regime Analysis
        - Volatility Analysis
        - Recommendations
        """)

def show_trade_timing_heatmap(selected_tokens):
    """Trade Timing Heatmap"""
    st.header("Trade Timing Heatmap")
    st.markdown("Visualize the average return for each (entry minute, exit lag) pair across all selected tokens.")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    max_entry_minute = st.number_input("Max Entry Minute", min_value=10, max_value=1440, value=240)
    max_exit_lag = st.number_input("Max Exit Lag (minutes)", min_value=1, max_value=240, value=60)
    
    if st.button("Show Trade Timing Heatmap", type="primary"):
        with st.spinner(f"Computing trade timing heatmap for {len(selected_tokens)} tokens..."):
            # Load data for all selected tokens
            token_data = []
            for token_name in selected_tokens:
                df = st.session_state.data_loader.get_token_data(token_name)
                if isinstance(df, pl.DataFrame) and not df.is_empty():
                    token_data.append(df)
            if token_data:
                try:
                    qv = st.session_state.quant_viz
                    fig = qv.plot_trade_timing_heatmap(token_data, max_entry_minute=max_entry_minute, max_exit_lag=max_exit_lag)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating heatmap: {str(e)}")
                    st.info("Trade timing heatmap - implementation may need adjustment")
            else:
                st.warning("No valid token data loaded.")

if __name__ == "__main__":
    main()