"""
Streamlit app for quantitative memecoin analysis
Professional financial market visualizations
"""

import streamlit as st
import polars as pl
from pathlib import Path
import plotly.graph_objects as go
import sys
import os
import logging
import numpy as np

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

# Select data root (raw, processed, etc.) under project data folder
project_root = Path(__file__).parent.parent
data_dir_root = project_root / "data"

def get_all_subdirs(base_dir):
    """Recursively get all subdirectories under base_dir, including base_dir itself."""
    subdirs = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            full_path = Path(root) / d
            rel_path = full_path.relative_to(base_dir)
            subdirs.append(str(rel_path))
    # Also include the top-level dirs
    for d in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        rel_path = d.relative_to(base_dir)
        if str(rel_path) not in subdirs:
            subdirs.append(str(rel_path))
    return sorted(set(subdirs))

# Use recursive subdirectory selection
all_data_roots = get_all_subdirs(data_dir_root)
if 'qa_selected_data_root' not in st.session_state:
    st.session_state.qa_selected_data_root = all_data_roots[0] if all_data_roots else ''
selected_data_root = st.sidebar.selectbox(
    "Select Data Root (recursive)", all_data_roots,
    index=all_data_roots.index(st.session_state.qa_selected_data_root) if st.session_state.qa_selected_data_root in all_data_roots else 0
)
st.session_state.qa_selected_data_root = selected_data_root
base_path = data_dir_root / selected_data_root

# Initialize components
@st.cache_resource
def init_components(base_path_str):
    # Initialize DataLoader with user-selected base path
    return DataLoader(base_path=base_path_str), QuantAnalysis(), QuantVisualizations()

data_loader, quant_analyzer, quant_viz = init_components(str(base_path))

# Title
st.title("📊 Professional Quantitative Analysis - Memecoin Trading")
st.markdown("Advanced financial market analysis using price action only")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Analysis Type")
analysis_type = st.sidebar.selectbox(
    "Select Analysis",
    [
        "🔥 Multi-Token Entry/Exit Matrix",
        "🔥 Multi-Token Risk Metrics",
        "🔥 Multi-Token Temporal Analysis",
        "Entry/Exit Matrix (Single Token)",
        "Entry/Exit Moment Matrix",
        "Temporal Risk/Reward (Single Token)",
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

# Data selection
st.sidebar.markdown("---")
st.sidebar.subheader("Data Selection")
selected_tokens = []

# Use the selected data root as the folder to look for parquet files
folder_path = base_path
parquet_files = []
if folder_path.exists() and folder_path.is_dir():
    parquet_files = [p for p in folder_path.rglob("*.parquet")]

st.sidebar.info(f"Found {len(parquet_files)} tokens")

# Build a robust token-to-file mapping for all multi-token views
# This should be placed after parquet_files is defined

# Build token_file_map: {symbol: parquet_file}
token_file_map = {data_loader.get_token_info(pf)['symbol']: pf for pf in parquet_files}

# Main content area
if analysis_type == "Trade Timing Heatmap":
    st.header("Trade Timing Heatmap")
    st.markdown("Visualize the average return for each (entry minute, exit lag) pair across all selected tokens.")

    max_entry_minute = st.number_input("Max Entry Minute", min_value=10, max_value=1440, value=240)
    max_exit_lag = st.number_input("Max Exit Lag (minutes)", min_value=1, max_value=240, value=60)

    if st.button("Show Trade Timing Heatmap", type="primary"):
        with st.spinner(f"Computing trade timing heatmap for {len(selected_tokens)} tokens..."):
            # Load data for all selected tokens
            token_data = []
            for token_name in selected_tokens:
                token_file = token_file_map[token_name]
                df = data_loader.load_token_data(token_file)
                if isinstance(df, pl.DataFrame) and not df.is_empty():
                    token_data.append(df)
            if token_data:
                qv = QuantVisualizations()
                fig = qv.plot_trade_timing_heatmap(token_data, max_entry_minute=max_entry_minute, max_exit_lag=max_exit_lag)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid token data loaded.")

elif analysis_type == "🔥 Multi-Token Entry/Exit Matrix":
    st.header("🔥 Multi-Token Price Movement Analysis")
    st.markdown("Analyze average price movements across different time windows for ALL tokens")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        entry_windows = st.multiselect(
            "Entry Windows (minutes)",
            extended_windows,
            default=[5, 10, 15, 30, 60, 120, 240]
        )
    with col2:
        exit_windows = st.multiselect(
            "Exit Windows (minutes)",
            extended_windows,
            default=[5, 10, 15, 30, 60, 120, 240]
        )
    
    if st.button("🚀 Analyze All Tokens", type="primary"):
        if 'selected_tokens' in locals() and selected_tokens:
            with st.spinner(f"Analyzing {len(selected_tokens)} tokens... This may take a moment."):
                # Load data for all selected tokens
                token_data = []
                progress_bar = st.progress(0)
                
                for i, token_name in enumerate(selected_tokens):
                    try:
                        token_file = token_file_map[token_name]
                        df = data_loader.load_token_data(token_file)
                        token_data.append((token_name, df))
                        progress_bar.progress((i + 1) / len(selected_tokens))
                    except:
                        continue
                
                progress_bar.empty()
                
                # Calculate aggregated matrix
                all_returns = {}
                for entry in entry_windows:
                    for exit in exit_windows:
                        all_returns[f"{entry}_{exit}"] = []
                
                # Process each token
                for token_name, df in token_data:
                    # Analyze price movements for each window combination
                    for entry_window in entry_windows:
                        for exit_window in exit_windows:
                            # Calculate all possible returns for this window combination
                            # --- DEBUG: scale prices to avoid tiny float issues ---
                            price_scale = 1000.0
                            for i in range(entry_window, len(df) - exit_window):
                                entry_price = df['price'].to_numpy()[i] * price_scale
                                exit_price = df['price'].to_numpy()[i + exit_window] * price_scale
                                # Only skip if entry_price is exactly zero or NaN
                                if np.isnan(entry_price) or entry_price == 0:
                                    continue
                                trade_return = (exit_price / entry_price - 1) * 100
                                # Undo the effect of scaling (should be neutral for %)
                                all_returns[f"{entry_window}_{exit_window}"].append(trade_return)
                                # Print a few sample trades for sanity check
                                if i % 500 == 0 and entry_window == entry_windows[0] and exit_window == exit_windows[0]:
                                    print(f"Sample trade: entry={entry_price/price_scale:.8f}, exit={exit_price/price_scale:.8f}, return={trade_return:.2f}%")
                
                # Debug: print summary stats for entry prices and returns
                for key, returns in all_returns.items():
                    if returns:
                        logging.info(f"{key}: min={np.min(returns):.2f}, max={np.max(returns):.2f}, median={np.median(returns):.2f}, count={len(returns)}")
                
                # Build the matrix as a list of lists
                matrix_data = []
                trade_counts_data = []
                for entry_window in entry_windows:
                    row = []
                    count_row = []
                    for exit_window in exit_windows:
                        returns = all_returns[f"{entry_window}_{exit_window}"]
                        if returns:
                            row.append(np.mean(returns))
                            count_row.append(len(returns))
                        else:
                            row.append(0.0)
                            count_row.append(0)
                    matrix_data.append(row)
                    trade_counts_data.append(count_row)
                
                # Create polars DataFrames by transposing the data
                aggregated_matrix = pl.DataFrame({str(col): [row[i] for row in matrix_data] for i, col in enumerate(exit_windows)})
                trade_counts = pl.DataFrame({str(col): [row[i] for row in trade_counts_data] for i, col in enumerate(exit_windows)})
                
                # Create visualization with subplots
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Average Returns (%)', 'Number of Trades'),
                    horizontal_spacing=0.15
                )
                
                # Returns heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=aggregated_matrix.to_numpy().astype(float),
                        x=[f"{w}min" for w in exit_windows],
                        y=[f"{w}min" for w in entry_windows],
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(aggregated_matrix.to_numpy().astype(float), 2),
                        texttemplate='%{text}%',
                        textfont={"size": 10},
                        colorbar=dict(title="Avg Return %", x=0.45),
                        hovertemplate='Entry: %{y}<br>Exit: %{x}<br>Avg Return: %{z:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Trade count heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=trade_counts.to_numpy().astype(int),
                        x=[f"{w}min" for w in exit_windows],
                        y=[f"{w}min" for w in entry_windows],
                        colorscale='Blues',
                        text=trade_counts.to_numpy().astype(int),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Trades", x=1.02),
                        hovertemplate='Entry: %{y}<br>Exit: %{x}<br>Trades: %{z}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Exit Window", row=1, col=1)
                fig.update_xaxes(title_text="Exit Window", row=1, col=2)
                fig.update_yaxes(title_text="Entry Window", row=1, col=1)
                fig.update_yaxes(title_text="Entry Window", row=1, col=2)
                
                fig.update_layout(
                    title=f"Multi-Token Price Movement Analysis - {len(token_data)} tokens analyzed",
                    height=500,
                    width=1200
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Analyzed", len(token_data))
                with col2:
                    # Find the best (entry, exit) combination using numpy
                    matrix_np = aggregated_matrix.to_numpy()
                    if matrix_np.size > 0 and not np.all(np.isnan(matrix_np)):
                        best_idx = np.nanargmax(matrix_np)
                        best_entry_idx, best_exit_idx = np.unravel_index(best_idx, matrix_np.shape)
                        best_entry = entry_windows[best_entry_idx]
                        best_exit = [int(col) for col in aggregated_matrix.columns][best_exit_idx]
                        st.metric("Best Entry/Exit", f"{best_entry}/{best_exit} min", f"{matrix_np[best_entry_idx, best_exit_idx]:.2f}%")
                    else:
                        st.metric("Best Entry/Exit", "N/A", "No valid data")
                with col3:
                    total_trades = sum(all_returns[k].__len__() for k in all_returns)
                    st.metric("Total Trades Analyzed", f"{total_trades:,}")
                
                # Detailed results
                st.subheader("📊 Detailed Results")
                results_df = pl.DataFrame({
                    'Entry Window': [f"{e} min" for e in entry_windows for _ in exit_windows],
                    'Exit Window': [f"{e} min" for _ in entry_windows for e in exit_windows],
                    'Avg Return (%)': aggregated_matrix.to_numpy().flatten(),
                    'Trade Count': trade_counts.to_numpy().flatten()
                })
                results_df = results_df.sort('Avg Return (%)', descending=True)
                st.dataframe(results_df, use_container_width=True)

elif analysis_type == "Entry/Exit Matrix (Single Token)":
    st.header("⏱️ Optimal Entry/Exit Timing Matrix")
    st.markdown("Analyze average returns for different entry and exit window combinations (multi-token supported)")

    # --- Token selection UI (like multi-token views) ---
    token_selection_mode = st.sidebar.radio(
        "Token Selection",
        ["All Tokens", "Select Specific Tokens", "Random Sample"],
        key="single_matrix_token_mode"
    )
    token_names = [data_loader.get_token_info(pf)['symbol'] for pf in parquet_files]
    if token_selection_mode == "All Tokens":
        selected_tokens_matrix = token_names
        st.sidebar.info(f"Analyzing all {len(selected_tokens_matrix)} tokens")
    elif token_selection_mode == "Select Specific Tokens":
        selected_tokens_matrix = st.sidebar.multiselect(
            "Select tokens",
            token_names,
            default=token_names[:10] if len(token_names) >= 10 else token_names,
            key="single_matrix_token_multiselect"
        )
    else:  # Random Sample
        max_sample = max(1, min(100, len(token_names)))
        sample_size = st.sidebar.slider("Sample Size", 1, max_sample, min(50, max_sample), key="single_matrix_token_sample_size")
        if sample_size >= len(token_names):
            selected_tokens_matrix = token_names
        else:
            selected_tokens_matrix = list(np.random.choice(token_names, sample_size, replace=False))
        st.sidebar.info(f"Random sample of {len(selected_tokens_matrix)} tokens")

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        entry_windows = st.multiselect(
            "Entry Windows (minutes)",
            extended_windows,
            default=[5, 10, 15, 30, 60, 120, 240],
            key="single_matrix_entry_windows"
        )
    with col2:
        exit_windows = st.multiselect(
            "Exit Windows (minutes)",
            extended_windows,
            default=[5, 10, 15, 30, 60, 120, 240],
            key="single_matrix_exit_windows"
        )

    if st.button("Generate Matrix", type="primary", key="single_matrix_generate_btn"):
        if selected_tokens_matrix:
            with st.spinner(f"Calculating entry/exit matrix for {len(selected_tokens_matrix)} tokens..."):
                # Load data for all selected tokens
                token_data = []
                for token_name in selected_tokens_matrix:
                    try:
                        token_file = token_file_map[token_name]
                        df = data_loader.load_token_data(token_file)
                        if isinstance(df, pl.DataFrame) and not df.is_empty():
                            token_data.append((token_name, df))
                    except Exception as e:
                        continue
                if not token_data:
                    st.warning("No valid token data loaded.")
                else:
                    qv = QuantVisualizations()
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

elif analysis_type == "🔥 Multi-Token Risk Metrics":
    st.header("🔥 Multi-Token Risk Metrics Analysis")
    st.markdown("Compare risk metrics across all tokens in the dataset")
    
    if st.button("🚀 Analyze All Tokens", type="primary"):
        if 'selected_tokens' in locals() and selected_tokens:
            with st.spinner(f"Calculating risk metrics for {len(selected_tokens)} tokens..."):
                # Load data for all selected tokens
                token_data = []
                progress_bar = st.progress(0)
                
                for i, token_name in enumerate(selected_tokens):  # Process all selected tokens
                    try:
                        token_file = token_file_map[token_name]
                        df = data_loader.load_token_data(token_file)
                        if len(df) > 100:  # Only include tokens with sufficient data
                            token_data.append((token_name, df))
                        progress_bar.progress((i + 1) / len(selected_tokens))
                    except:
                        continue
                
                progress_bar.empty()
                
                # Calculate risk metrics for each token
                results = []
                for token_name, df in token_data:
                    returns = df['price'].pct_change().dropna()
                    
                    # Calculate metrics
                    metrics = {
                        'Token': token_name,
                        'Total Return (%)': (df['price'].to_numpy()[-1] / df['price'].to_numpy()[0] - 1) * 100,
                        'Volatility (%)': returns.std() * 100,  # Simple volatility, not annualized
                        'Sharpe Ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,  # Simple Sharpe
                        'Win Rate (%)': (returns > 0).sum() / len(returns) * 100,
                        'Avg Win (%)': returns[returns > 0].mean() * 100 if (returns > 0).any() else 0,
                        'Avg Loss (%)': abs(returns[returns < 0].mean()) * 100 if (returns < 0).any() else 0,
                        'Data Points': len(df)
                    }
                    results.append(metrics)
                
                results_df = pl.DataFrame(results)
                
                # Display summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tokens Analyzed", len(results_df))
                with col2:
                    st.metric("Avg Return", f"{results_df['Total Return (%)'].mean():.2f}%")
                with col3:
                    st.metric("Avg Sharpe", f"{results_df['Sharpe Ratio'].mean():.2f}")
                with col4:
                    st.metric("Avg Win Rate", f"{results_df['Win Rate (%)'].mean():.1f}%")
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot: Return vs Volatility
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Volatility (%)'],
                        y=results_df['Total Return (%)'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=results_df['Sharpe Ratio'],
                            colorscale='RdBu',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        text=results_df['Token'],
                        hovertemplate='Token: %{text}<br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>'
                    ))
                    fig.update_layout(
                        title="Risk-Return Profile",
                        xaxis_title="Volatility (% per minute)",
                        yaxis_title="Total Return (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distribution of Sharpe ratios
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=results_df['Sharpe Ratio'],
                        nbinsx=30,
                        name='Sharpe Ratio Distribution'
                    ))
                    fig.update_layout(
                        title="Sharpe Ratio Distribution",
                        xaxis_title="Sharpe Ratio",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top performers table
                st.subheader("🏆 Top Performers")
                top_performers = results_df.nlargest(10, 'Sharpe Ratio')[['Token', 'Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Win Rate (%)']]
                st.dataframe(top_performers, use_container_width=True)
                
                # Full results table
                st.subheader("📋 All Results")
                st.dataframe(results_df.sort('Sharpe Ratio', descending=True), use_container_width=True)

elif analysis_type == "🔥 Multi-Token Temporal Analysis":
    st.header("🔥 Multi-Token Temporal Analysis")
    st.markdown("Analyze how returns vary across different time horizons for ALL tokens")
    
    # Parameters
    time_horizons = st.multiselect(
        "Time Horizons (minutes)",
        [1, 5, 10, 15, 30, 60, 120, 240, 480],
        default=[5, 15, 30, 60, 120, 240]
    )
    
    if st.button("🚀 Analyze All Tokens", type="primary"):
        if 'selected_tokens' in locals() and selected_tokens and time_horizons:
            with st.spinner(f"Analyzing temporal patterns for {len(selected_tokens)} tokens..."):
                # Load data for all selected tokens
                all_horizon_returns = {horizon: [] for horizon in time_horizons}
                token_count = 0
                progress_bar = st.progress(0)
                
                for i, token_name in enumerate(selected_tokens):  # Process all selected tokens
                    try:
                        token_file = token_file_map[token_name]
                        df = data_loader.load_token_data(token_file)
                        
                        if len(df) > max(time_horizons) * 2:  # Ensure enough data
                            token_count += 1
                            
                            # Calculate returns for each horizon
                            for horizon in time_horizons:
                                returns = df['price'].pct_change(horizon).dropna()
                                all_horizon_returns[horizon].extend(returns.tolist())
                        
                        progress_bar.progress((i + 1) / len(selected_tokens))
                    except:
                        continue
                
                progress_bar.empty()
                
                # Analyze aggregated results
                results = []
                for horizon in time_horizons:
                    horizon_returns = pl.Series(all_horizon_returns[horizon])
                    
                    if len(horizon_returns) > 0:
                        positive_returns = horizon_returns[horizon_returns > 0]
                        negative_returns = horizon_returns[horizon_returns < 0]
                        
                        results.append({
                            'Horizon (min)': horizon,
                            'Win Rate (%)': len(positive_returns) / len(horizon_returns) * 100 if len(horizon_returns) > 0 else 0,
                            'Avg Gain (%)': positive_returns.mean() * 100 if len(positive_returns) > 0 else 0,
                            'Avg Loss (%)': abs(negative_returns.mean()) * 100 if len(negative_returns) > 0 else 0,
                            'Expected Value (%)': horizon_returns.mean() * 100,
                            'Sharpe Ratio': horizon_returns.mean() / horizon_returns.std() if horizon_returns.std() > 0 else 0,
                            'Total Observations': len(horizon_returns)
                        })
                
                results_df = pl.DataFrame(results)
                
                # Display summary
                st.subheader("📊 Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Analyzed", token_count)
                with col2:
                    if results_df['Expected Value (%)'].to_numpy().size > 0:
                        best_horizon_idx = np.nanargmax(results_df['Expected Value (%)'].to_numpy())
                        best_horizon = results_df['Horizon (min)'].to_numpy()[best_horizon_idx]
                    else:
                        best_horizon = None
                    st.metric("Best Horizon", f"{best_horizon} min")
                with col3:
                    total_obs = results_df['Total Observations'].sum()
                    st.metric("Total Observations", f"{total_obs:,}")
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Expected value by horizon
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=results_df['Horizon (min)'],
                        y=results_df['Expected Value (%)'],
                        text=results_df['Expected Value (%)'].round(2),
                        textposition='outside',
                        marker_color=results_df['Expected Value (%)'].apply(lambda x: 'green' if x > 0 else 'red')
                    ))
                    fig.update_layout(
                        title="Expected Value by Time Horizon",
                        xaxis_title="Time Horizon (minutes)",
                        yaxis_title="Expected Value (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Win rate by horizon
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Horizon (min)'],
                        y=results_df['Win Rate (%)'],
                        mode='lines+markers',
                        line=dict(width=3),
                        marker=dict(size=10)
                    ))
                    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                 annotation_text="50% Win Rate")
                    fig.update_layout(
                        title="Win Rate by Time Horizon",
                        xaxis_title="Time Horizon (minutes)",
                        yaxis_title="Win Rate (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk/Reward comparison
                st.subheader("⚖️ Risk/Reward Analysis")
                fig = go.Figure()
                
                # Add bars for gains and losses
                fig.add_trace(go.Bar(
                    name='Avg Gain',
                    x=results_df['Horizon (min)'],
                    y=results_df['Avg Gain (%)'],
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    name='Avg Loss',
                    x=results_df['Horizon (min)'],
                    y=-results_df['Avg Loss (%)'],
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title="Average Gains vs Losses by Time Horizon",
                    xaxis_title="Time Horizon (minutes)",
                    yaxis_title="Return (%)",
                    barmode='relative',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Key insights
                st.subheader("💡 Key Insights")
                if results_df['Expected Value (%)'].to_numpy().size > 0:
                    best_row_idx = np.nanargmax(results_df['Expected Value (%)'].to_numpy())
                    best_row = results_df.row(best_row_idx)
                else:
                    best_row = None
                if best_row is not None:
                    st.success(f"""
                    **Optimal Holding Period**: {best_row['Horizon (min)']} minutes
                    - Expected Value: {best_row['Expected Value (%)']:.2f}%
                    - Win Rate: {best_row['Win Rate (%)']:.1f}%
                    - Sharpe Ratio: {best_row['Sharpe Ratio']:.2f}
                    """)

elif analysis_type == "Temporal Risk/Reward (Single Token)":
    st.header("⚖️ Temporal Risk/Reward Analysis")
    st.markdown("Analyze how risk and reward change with different holding periods (multi-token supported)")

    # --- Token selection UI (like multi-token views) ---
    token_selection_mode = st.sidebar.radio(
        "Token Selection",
        ["All Tokens", "Select Specific Tokens", "Random Sample"],
        key="temporal_risk_token_mode"
    )
    token_names = [data_loader.get_token_info(pf)['symbol'] for pf in parquet_files]
    if token_selection_mode == "All Tokens":
        selected_tokens_temporal = token_names
        st.sidebar.info(f"Analyzing all {len(selected_tokens_temporal)} tokens")
    elif token_selection_mode == "Select Specific Tokens":
        selected_tokens_temporal = st.sidebar.multiselect(
            "Select tokens",
            token_names,
            default=token_names[:10] if len(token_names) >= 10 else token_names,
            key="temporal_risk_token_multiselect"
        )
    else:  # Random Sample
        max_sample = max(1, min(100, len(token_names)))
        sample_size = st.sidebar.slider("Sample Size", 1, max_sample, min(50, max_sample), key="temporal_risk_token_sample_size")
        if sample_size >= len(token_names):
            selected_tokens_temporal = token_names
        else:
            selected_tokens_temporal = list(np.random.choice(token_names, sample_size, replace=False))
        st.sidebar.info(f"Random sample of {len(selected_tokens_temporal)} tokens")

    # Parameters
    time_horizons = st.multiselect(
        "Time Horizons (minutes)",
        [1, 5, 10, 15, 30, 60, 120, 240, 480],
        default=[5, 15, 30, 60, 120, 240],
        key="temporal_risk_time_horizons"
    )

    if st.button("Analyze Risk/Reward", type="primary", key="temporal_risk_generate_btn"):
        if selected_tokens_temporal and time_horizons:
            # Load data for all selected tokens
            token_results = []
            for token_name in selected_tokens_temporal:
                try:
                    token_file = token_file_map[token_name]
                    df = data_loader.load_token_data(token_file)
                    if isinstance(df, pl.DataFrame) and not df.is_empty():
                        results = quant_analyzer.temporal_risk_reward_analysis(df, time_horizons)
                        results = results.with_columns([pl.lit(token_name).alias('Token')])
                        token_results.append(results)
                except Exception as e:
                    continue
            if not token_results:
                st.warning("No valid token data loaded.")
            else:
                # Concatenate all results
                all_results = pl.concat(token_results)
                # Aggregate by time horizon
                agg_results = all_results.groupby('horizon_minutes').agg([
                    pl.col('win_rate').mean().alias('Win Rate (%)'),
                    pl.col('avg_gain_%').mean().alias('Avg Gain (%)'),
                    pl.col('avg_loss_%').mean().alias('Avg Loss (%)'),
                    pl.col('expected_value_%').mean().alias('Expected Value (%)'),
                    pl.col('risk_reward_ratio').mean().alias('Risk/Reward Ratio'),
                    pl.col('sharpe_ratio').mean().alias('Sharpe Ratio'),
                    pl.col('Token').count().alias('Token Count')
                ]).sort('horizon_minutes')

                # Display summary
                st.subheader("📊 Aggregated Results (Mean Across Tokens)")
                st.dataframe(agg_results, use_container_width=True)

                # Per-token results
                st.subheader("📋 Per-Token Results")
                st.dataframe(all_results, use_container_width=True)
        else:
            st.warning("Please select at least one token and one time horizon.")

elif analysis_type == "Volatility Surface":
    st.header("📈 Volatility Surface Analysis")
    st.markdown("Analyze how volatility changes across different time windows")
    
    if st.button("Generate Volatility Surface", type="primary"):
        if 'selected_token' in locals():
            # Load data
            token_file = token_file_map[selected_token]
            df = data_loader.load_token_data(token_file)
            
            with st.spinner("Calculating volatility surface..."):
                # Calculate volatility for different windows
                windows = [5, 10, 15, 30, 60, 120, 240]
                volatilities = []
                
                for window in windows:
                    rolling_vol = df['price'].pct_change().rolling(window).std() * 100
                    volatilities.append({
                        'window': window,
                        'mean_vol': rolling_vol.mean(),
                        'median_vol': rolling_vol.median(),
                        'max_vol': rolling_vol.max(),
                        'min_vol': rolling_vol.min()
                    })
                
                vol_df = pl.DataFrame(volatilities)
                
                # Create visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vol_df['window'],
                    y=vol_df['mean_vol'],
                    mode='lines+markers',
                    name='Mean Volatility',
                    line=dict(width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=vol_df['window'],
                    y=vol_df['median_vol'],
                    mode='lines+markers',
                    name='Median Volatility',
                    line=dict(width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=vol_df['window'],
                    y=vol_df['max_vol'],
                    mode='lines',
                    name='Max Volatility',
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(
                    title='Volatility Surface Analysis',
                    xaxis_title='Time Window (minutes)',
                    yaxis_title='Volatility (%)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Volatility Statistics")
                st.dataframe(vol_df, use_container_width=True)

elif analysis_type == "Microstructure Analysis":
    st.header("🔬 Market Microstructure Analysis")
    st.markdown("Analyze high-frequency market behavior patterns")
    
    if st.button("Analyze Microstructure", type="primary"):
        if 'selected_token' in locals():
            # Load data
            token_file = token_file_map[selected_token]
            df = data_loader.load_token_data(token_file)
            
            with st.spinner("Analyzing market microstructure..."):
                # Calculate microstructure metrics
                results = quant_analyzer.microstructure_analysis(df)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Realized Volatility", f"{results['avg_realized_volatility']:.4f}")
                with col2:
                    st.metric("Bid-Ask Spread Estimate", f"{results['bid_ask_spread_estimate']:.4f}")
                with col3:
                    st.metric("Avg Illiquidity", f"{results['avg_amihud_illiquidity']:.4f}")
                
                st.info("Microstructure analysis provides insights into market quality and trading conditions")

elif analysis_type == "Price Distribution Evolution":
    st.header("📊 Price Distribution Evolution")
    st.markdown("Analyze how price distributions change over time")
    
    if st.button("Analyze Distribution", type="primary"):
        if 'selected_token' in locals():
            # Load data
            token_file = token_file_map[selected_token]
            df = data_loader.load_token_data(token_file)
            
            with st.spinner("Analyzing price distributions..."):
                # Divide data into time periods
                n_periods = 6
                period_size = len(df) // n_periods
                
                fig = go.Figure()
                
                for i in range(n_periods):
                    start_idx = i * period_size
                    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df)
                    
                    period_data = df.iloc[start_idx:end_idx]
                    period_returns = period_data['price'].pct_change().dropna() * 100
                    
                    # Add violin plot for this period
                    fig.add_trace(go.Violin(
                        y=period_returns,
                        name=f'Period {i+1}',
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                fig.update_layout(
                    title='Evolution of Return Distributions Over Time',
                    yaxis_title='Returns (%)',
                    xaxis_title='Time Period',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics by period
                st.subheader("📈 Period Statistics")
                period_stats = []
                for i in range(n_periods):
                    start_idx = i * period_size
                    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df)
                    period_returns = df.iloc[start_idx:end_idx]['price'].pct_change().dropna() * 100
                    
                    period_stats.append({
                        'Period': f'Period {i+1}',
                        'Mean Return (%)': period_returns.mean(),
                        'Volatility (%)': period_returns.std(),
                        'Skewness': period_returns.skew(),
                        'Kurtosis': period_returns.kurtosis()
                    })
                
                stats_df = pl.DataFrame(period_stats)
                st.dataframe(stats_df, use_container_width=True)

elif analysis_type == "Optimal Holding Period":
    st.header("⏰ Optimal Holding Period Analysis")
    st.markdown("Find the optimal holding period based on risk-adjusted returns")
    
    if st.button("Find Optimal Period", type="primary"):
        if 'selected_token' in locals():
            # Load data
            token_file = token_file_map[selected_token]
            df = data_loader.load_token_data(token_file)
            
            with st.spinner("Calculating optimal holding periods..."):
                # Test different holding periods
                periods = list(range(1, min(241, len(df)//2), 5))
                results = []
                
                for period in periods:
                    returns = df['price'].pct_change(period).dropna()
                    
                    results.append({
                        'Period': period,
                        'Mean Return (%)': returns.mean() * 100,
                        'Win Rate (%)': (returns > 0).sum() / len(returns) * 100,
                        'Sharpe Ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                        'Max Drawdown (%)': (returns.cumsum().expanding().max() - returns.cumsum()).max() * 100
                    })
                
                results_df = pl.DataFrame(results)
                
                # Find optimal based on Sharpe ratio
                if results_df['Sharpe Ratio'].to_numpy().size > 0:
                    optimal_idx = np.nanargmax(results_df['Sharpe Ratio'].to_numpy())
                    optimal_period = results_df['Period'].to_numpy()[optimal_idx]
                else:
                    optimal_idx = None
                    optimal_period = None
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Period'],
                        y=results_df['Sharpe Ratio'],
                        mode='lines+markers',
                        name='Sharpe Ratio'
                    ))
                    if optimal_period is not None:
                        fig.add_vline(x=optimal_period, line_dash="dash", line_color="red",
                                     annotation_text=f"Optimal: {optimal_period} min")
                    fig.update_layout(
                        title='Sharpe Ratio by Holding Period',
                        xaxis_title='Holding Period (minutes)',
                        yaxis_title='Sharpe Ratio',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['Period'],
                        y=results_df['Win Rate (%)'],
                        mode='lines+markers',
                        name='Win Rate'
                    ))
                    fig.update_layout(
                        title='Win Rate by Holding Period',
                        xaxis_title='Holding Period (minutes)',
                        yaxis_title='Win Rate (%)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                if optimal_period is not None:
                    st.success(f"**Optimal Holding Period: {optimal_period} minutes**")
                    optimal_stats = results_df.loc[optimal_idx]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Return", f"{optimal_stats['Mean Return (%)']:.2f}%")
                    with col2:
                        st.metric("Win Rate", f"{optimal_stats['Win Rate (%)']:.1f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{optimal_stats['Sharpe Ratio']:.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{optimal_stats['Max Drawdown (%)']:.2f}%")

elif analysis_type == "Market Regime Analysis":
    st.header("🌊 Market Regime Analysis")
    st.markdown("Identify different market regimes (trending, ranging, volatile)")
    
    if st.button("Analyze Regimes", type="primary"):
        if 'selected_token' in locals():
            # Load data
            token_file = token_file_map[selected_token]
            df = data_loader.load_token_data(token_file)
            
            with st.spinner("Detecting market regimes..."):
                # Detect regimes
                regime_df = quant_analyzer.market_regime_detection(df)
                
                # Create visualization
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=('Price and Regimes', 'Volatility'),
                                   row_heights=[0.7, 0.3])
                
                # Price chart with regime coloring
                for regime in regime_df['regime'].unique():
                    regime_data = regime_df[regime_df['regime'] == regime]
                    color = {'uptrend': 'green', 'downtrend': 'red', 
                            'ranging': 'blue', 'high_volatility': 'orange'}.get(regime, 'gray')
                    
                    fig.add_trace(go.Scatter(
                        x=regime_data['datetime'],
                        y=regime_data['price'],
                        mode='markers',
                        marker=dict(color=color, size=3),
                        name=regime.title(),
                        showlegend=True
                    ), row=1, col=1)
                
                # Volatility
                fig.add_trace(go.Scatter(
                    x=regime_df['datetime'],
                    y=regime_df['volatility'] * 100,
                    mode='lines',
                    name='Volatility',
                    showlegend=False
                ), row=2, col=1)
                
                fig.update_xaxes(title_text="Time", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
                
                fig.update_layout(height=800, title="Market Regime Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Regime statistics
                st.subheader("📊 Regime Statistics")
                regime_stats = regime_df.groupby('regime').agg({
                    'price': ['count', 'mean'],
                    'volatility': 'mean'
                }).round(4)
                regime_stats.columns = ['Duration (minutes)', 'Avg Price', 'Avg Volatility']
                regime_stats['Frequency (%)'] = (regime_stats['Duration (minutes)'] / len(regime_df) * 100).round(2)
                st.dataframe(regime_stats)

elif analysis_type == "Multi-Token Correlation":
    st.header("🔗 Multi-Token Correlation Analysis")
    st.markdown("Analyze correlations between selected tokens")
    
    if st.button("Analyze Correlations", type="primary"):
        if 'selected_tokens' in locals() and len(selected_tokens) >= 2:
            with st.spinner(f"Analyzing correlations between {len(selected_tokens)} tokens..."):
                # Load data for selected tokens
                token_data = {}
                for token_name in selected_tokens:
                    try:
                        token_file = token_file_map[token_name]
                        df = data_loader.load_token_data(token_file)
                        token_data[token_name] = df.set_index('datetime')['price']
                    except:
                        continue
                
                # Align all series to same time index
                price_df = pl.DataFrame(token_data)
                returns_df = price_df.pct_change().dropna()
                
                # Calculate correlation matrix
                corr_matrix = returns_df.corr()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.to_numpy(),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.to_numpy(), 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title="Token Correlation Matrix",
                    height=600,
                    width=700
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling correlation analysis
                if len(selected_tokens) == 2:
                    st.subheader("📈 Rolling Correlation")
                    window = st.slider("Rolling Window (minutes)", 30, 240, 60)
                    
                    rolling_corr = returns_df.iloc[:, 0].rolling(window).corr(returns_df.iloc[:, 1])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=returns_df.index,
                        y=rolling_corr,
                        mode='lines',
                        name=f'{window}-min Rolling Correlation'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title=f"Rolling Correlation: {selected_tokens[0]} vs {selected_tokens[1]}",
                        xaxis_title="Time",
                        yaxis_title="Correlation",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least 2 tokens for correlation analysis")

elif analysis_type == "Comprehensive Report":
    st.header("📑 Comprehensive Analysis Report")
    st.markdown("Generate a complete analysis report for selected tokens")
    
    if st.button("Generate Report", type="primary"):
        if 'max_tokens' in locals():
            with st.spinner(f"Generating comprehensive report for {max_tokens} tokens..."):
                # This would generate a comprehensive report
                st.info("Comprehensive report generation would include:")
                st.markdown("""
                - Executive Summary
                - Individual Token Analysis
                - Comparative Analysis
                - Risk Metrics Summary
                - Optimal Trading Windows
                - Market Regime Analysis
                - Correlation Analysis
                - Recommendations
                """)
                st.warning("Full implementation pending...")

elif analysis_type == "Entry/Exit Moment Matrix":
    st.header("⏱️ Entry/Exit Moment Matrix")
    st.markdown("Visualize the average return for each (entry minute, exit minute) pair across all selected tokens.")

    # Token selection UI (like other multi-token views)
    token_selection_mode = st.sidebar.radio(
        "Token Selection",
        ["All Tokens", "Select Specific Tokens", "Random Sample"],
        key="moment_matrix_token_mode"
    )
    token_names = [data_loader.get_token_info(pf)['symbol'] for pf in parquet_files]
    if token_selection_mode == "All Tokens":
        selected_tokens_moment = token_names
        st.sidebar.info(f"Analyzing all {len(selected_tokens_moment)} tokens")
    elif token_selection_mode == "Select Specific Tokens":
        selected_tokens_moment = st.sidebar.multiselect(
            "Select tokens",
            token_names,
            default=token_names[:10] if len(token_names) >= 10 else token_names,
            key="moment_matrix_token_multiselect"
        )
    else:  # Random Sample
        max_sample = max(1, min(100, len(token_names)))
        sample_size = st.sidebar.slider("Sample Size", 1, max_sample, min(50, max_sample), key="moment_matrix_token_sample_size")
        if sample_size >= len(token_names):
            selected_tokens_moment = token_names
        else:
            selected_tokens_moment = list(np.random.choice(token_names, sample_size, replace=False))
        st.sidebar.info(f"Random sample of {len(selected_tokens_moment)} tokens")

    # Parameters
    max_entry_minute = st.number_input("Max Entry Minute", min_value=10, max_value=1440, value=240, key="moment_matrix_max_entry")
    max_exit_minute = st.number_input("Max Exit Minute", min_value=10, max_value=1440, value=240, key="moment_matrix_max_exit")

    if st.button("Show Entry/Exit Moment Matrix", type="primary", key="moment_matrix_generate_btn"):
        if selected_tokens_moment:
            with st.spinner(f"Computing entry/exit moment matrix for {len(selected_tokens_moment)} tokens..."):
                # Load data for all selected tokens
                token_data = []
                for token_name in selected_tokens_moment:
                    try:
                        token_file = token_file_map[token_name]
                        df = data_loader.load_token_data(token_file)
                        if isinstance(df, pl.DataFrame) and not df.is_empty():
                            token_data.append(df)
                    except Exception as e:
                        continue
                if not token_data:
                    st.warning("No valid token data loaded.")
                else:
                    qv = QuantVisualizations()
                    fig = qv.plot_entry_exit_moment_matrix(token_data, max_entry_minute=int(max_entry_minute), max_exit_minute=int(max_exit_minute))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one token.")