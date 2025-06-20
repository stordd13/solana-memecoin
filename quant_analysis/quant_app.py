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

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from data_analysis.data_loader import DataLoader
from quant_analysis import QuantAnalysis
from quant_viz import QuantVisualizations

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
                        results = results.with_columns([pl.lit(token_name).alias('Token')])
                        token_results.append(results)
                    progress_bar.progress((i + 1) / len(selected_tokens))
                except Exception as e:
                    st.warning(f"Error analyzing {token_name}: {str(e)}")
                    continue
            
            progress_bar.empty()
            
            if token_results:
                # Combine all results
                combined_results = pl.concat(token_results)
                
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
                
                # Detailed results
                st.subheader("📈 Detailed Results")
                st.dataframe(combined_results.sort('sharpe_ratio', descending=True), use_container_width=True)
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
                                    segment_results['Mean_Return_Pct'] = returns.mean() * 100
                                    segment_results['Cumulative_Return_Pct'] = ((segment_data['price'].last() / segment_data['price'].first()) - 1) * 100
                                
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
            
            st.markdown(f"""
            **Token Ranking Analysis for {heatmap_metric}:**
            - **Top {top_n} Performers**: Best performing tokens across all lifecycle segments
            - **Bottom {top_n} Performers**: Worst performing tokens  
            - **Performance vs Volatility**: Risk-return profile of top performers
            - **Distribution**: Overall performance distribution across all {len(selected_tokens)} tokens
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
        else:
            st.warning("Please select at least one token.")

def show_volatility_surface(selected_tokens, selection_mode):
    """Volatility Surface Analysis"""
    st.header("📈 Volatility Surface Analysis")
    st.markdown("Analyze how volatility changes across different time windows")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
    
    if selected_token:
        # Parameters
        windows = st.multiselect(
            "Rolling Windows (minutes)",
            DEFAULT_WINDOWS,
            default=[5, 10, 30, 60, 240]
        )
        
        if st.button("📊 Generate Volatility Surface"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            with st.spinner("Calculating volatility surface..."):
                volatilities = []
                
                for window in windows:
                    rolling_vol = df['price'].pct_change().rolling_std(window, min_periods=1) * 100
                    volatilities.append({
                        'window': window,
                        'avg_volatility': rolling_vol.mean(),
                        'max_volatility': rolling_vol.max(),
                        'min_volatility': rolling_vol.min(),
                        'std_volatility': rolling_vol.std()
                    })
                
                vol_df = pl.DataFrame(volatilities)
                st.dataframe(vol_df, use_container_width=True)

def show_microstructure_analysis(selected_tokens, selection_mode):
    """Market Microstructure Analysis"""
    st.header("🔬 Market Microstructure Analysis")
    st.markdown("Analyze high-frequency market behavior patterns")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
    
    if selected_token:
        if st.button("🔬 Analyze Microstructure"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            with st.spinner("Analyzing market microstructure..."):
                results = st.session_state.quant_analyzer.microstructure_analysis(df)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Realized Volatility", f"{results['avg_realized_volatility']:.4f}")
                with col2:
                    st.metric("Bid-Ask Spread Estimate", f"{results['bid_ask_spread_estimate']:.6f}")
                with col3:
                    st.metric("Kyle's Lambda", f"{results['kyle_lambda']:.6f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Amihud Illiquidity", f"{results['avg_amihud_illiquidity']:.4f}")
                with col2:
                    st.metric("Volatility of Volatility", f"{results['volatility_of_volatility']:.4f}")
                
                st.info("Microstructure analysis provides insights into market quality and trading conditions")

def show_price_distribution_evolution(selected_tokens, selection_mode):
    """Price Distribution Evolution Analysis"""
    st.header("📊 Price Distribution Evolution")
    st.markdown("Analyze how price distributions change over time")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
    
    if selected_token:
        if st.button("📊 Analyze Distribution Evolution"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            with st.spinner("Analyzing price distributions..."):
                # Simple analysis - divide into periods and show basic stats
                n_periods = 6
                period_size = len(df) // n_periods
                
                period_stats = []
                for i in range(n_periods):
                    start_idx = i * period_size
                    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df)
                    
                    period_data = df[start_idx:end_idx]
                    period_returns = period_data['price'].pct_change().drop_nulls() * 100
                    
                    if len(period_returns) > 0:
                        period_stats.append({
                            'Period': f"Period {i+1}",
                            'Mean Return (%)': period_returns.mean(),
                            'Std Dev (%)': period_returns.std(),
                            'Skewness': period_returns.skew() if hasattr(period_returns, 'skew') else 0,
                            'Sample Size': len(period_returns)
                        })
                
                if period_stats:
                    stats_df = pl.DataFrame(period_stats)
                    st.dataframe(stats_df, use_container_width=True)

def show_optimal_holding_period(selected_tokens, selection_mode):
    """Optimal Holding Period Analysis"""
    st.header("⏰ Optimal Holding Period Analysis")
    st.markdown("Find the optimal holding period based on risk-adjusted returns")
    
    if not selected_tokens:
        st.warning("Please select at least one token.")
        return
    
    selected_token = st.selectbox("Select Token for Analysis", selected_tokens)
    
    if selected_token:
        if st.button("⏰ Find Optimal Holding Period"):
            df = st.session_state.data_loader.get_token_data(selected_token)
            
            with st.spinner("Calculating optimal holding periods..."):
                # Test different holding periods
                periods = list(range(1, min(241, len(df)//2), 5))
                results = []
                
                for period in periods:
                    returns = df['price'].pct_change(period).drop_nulls()
                    
                    if len(returns) > 0:
                        results.append({
                            'Holding Period (min)': period,
                            'Mean Return (%)': returns.mean() * 100,
                            'Volatility (%)': returns.std() * 100,
                            'Sharpe Ratio': returns.mean() / (returns.std() + 1e-10) * np.sqrt(525600 / period),
                            'Sample Size': len(returns)
                        })
                
                if results:
                    results_df = pl.DataFrame(results)
                    st.dataframe(results_df.sort('Sharpe Ratio', descending=True), use_container_width=True)
                    
                    # Find optimal
                    optimal_idx = results_df['Sharpe Ratio'].arg_max()
                    if optimal_idx is not None:
                        optimal_stats = results_df[optimal_idx]
                        st.success(f"Optimal Holding Period: {optimal_stats['Holding Period (min)'][0]} minutes")

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