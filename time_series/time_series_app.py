"""
Streamlit app for token time series analysis and modeling
"""

# FULLY POLARS MIGRATED
import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import random

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data_analysis.data_loader import DataLoader
from time_series_models import TimeSeriesModeler
import plotly.express as px
from time_series_analyzer import TimeSeriesAnalyzer

def render_time_series_page():
    st.title("Token Time Series Analysis")
    
    # Initialize data loader, modeler, and analyzer
    data_loader = DataLoader()
    modeler = TimeSeriesModeler()
    analyzer = TimeSeriesAnalyzer()
    
    # Sidebar for folder selection
    st.sidebar.header("Data Selection")
    # Let user select base directory (default to 'data/')
    base_dir = st.sidebar.text_input("Base data directory", value="data/")
    base_path = Path(base_dir)
    if not base_path.exists():
        st.sidebar.warning(f"Base directory {base_dir} does not exist.")
        return
    # Find all subfolders (recursively) with .parquet files
    all_subfolders = sorted({str(p.parent.relative_to(base_path)) for p in base_path.rglob("*.parquet")})
    if not all_subfolders:
        st.sidebar.warning("No subfolders with parquet files found.")
        return
    selected_folders = st.sidebar.multiselect(
        "Select subfolder(s) (recursive)",
        all_subfolders,
        default=[all_subfolders[0]]
    )
    # Initialize DataLoader with the chosen base path
    data_loader = DataLoader(base_path)
    # Load data
    with st.spinner("Loading data..."):
        df_polars = data_loader.load_data(selected_datasets=selected_folders)
        if df_polars is None or df_polars.is_empty():
            st.error("No data found in selected folder(s).")
            return
        df = df_polars
    
    # Unified analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Token", "Random N Tokens", "All Tokens", "Multi-Token Comparison"]
    )
    tokens = df['token'].unique().to_list()
    selected_tokens = []
    if analysis_mode == "Single Token":
        selected_token = st.sidebar.selectbox("Select Token", tokens)
        selected_tokens = [selected_token]
        for token in selected_tokens:
            with st.expander(f"Analysis for {token}", expanded=True):
                render_single_token_analysis(df.filter(pl.col('token') == token), modeler, analyzer, token_override=token)
    elif analysis_mode == "Random N Tokens":
        n = st.sidebar.number_input("Number of random tokens", min_value=1, max_value=len(tokens), value=min(5, len(tokens)))
        if st.sidebar.button("Sample Random Tokens"):
            selected_tokens = random.sample(list(tokens), int(n))
            st.session_state['random_tokens'] = selected_tokens
        else:
            selected_tokens = st.session_state.get('random_tokens', list(tokens)[:int(n)])
        st.sidebar.write(f"Selected: {selected_tokens}")
        for token in selected_tokens:
            with st.expander(f"Analysis for {token}", expanded=False):
                render_single_token_analysis(df.filter(pl.col('token') == token), modeler, analyzer, token_override=token)
    elif analysis_mode == "All Tokens":
        selected_tokens = list(tokens)
        st.sidebar.write(f"Selected all {len(selected_tokens)} tokens.")
        for token in selected_tokens:
            with st.expander(f"Analysis for {token}", expanded=False):
                render_single_token_analysis(df.filter(pl.col('token') == token), modeler, analyzer, token_override=token)
    elif analysis_mode == "Multi-Token Comparison":
        render_multi_token_analysis(df, modeler, analyzer)

def render_single_token_analysis(df: pl.DataFrame, modeler: TimeSeriesModeler, analyzer: TimeSeriesAnalyzer, token_override=None):
    """Render single token analysis page (now supports token_override for batch mode)"""
    if token_override is not None:
        selected_token = token_override
    else:
        st.header("Single Token Analysis")
        tokens = df['token'].unique().to_list()
        selected_token = st.selectbox("Select Token", tokens)
    
    # Filter data for selected token
    token_data = df.filter(pl.col('token') == selected_token).sort(pl.col('datetime'))
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Analysis", "Model Predictions", "Trading Signals", "Strategy Performance"
    ])
    
    with tab1:
        st.subheader("Price Analysis")
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=token_data['datetime'],
            y=token_data['price'],
            mode='lines',
            name='Price'
        ))
        fig.update_layout(
            title=f"{selected_token} Price",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${token_data['price'][-1]:.4f}")
        with col2:
            price_change = (token_data['price'][-1] - token_data['price'][0]) / token_data['price'][0] * 100
            st.metric("Price Change", f"{price_change:.2f}%")
        with col3:
            st.metric("Data Points", len(token_data))
        
        # Returns distribution
        returns = token_data['price'].pct_change().drop_nulls()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns'
        ))
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Model Predictions")
        
        # Model selection
        model_type = st.radio("Select Model", ["LSTM", "SARIMA"])
        
        if st.button("Generate Predictions"):
            try:
                if model_type == "LSTM":
                    model_results = modeler.fit_lstm(token_data, selected_token)
                    if model_results is None:
                        st.error("Failed to generate LSTM predictions. Not enough data points.")
                    else:
                        # Plot predictions
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=token_data['datetime'],
                            y=token_data['price'],
                            mode='lines',
                            name='Actual'
                        ))
                        fig.add_trace(go.Scatter(
                            x=token_data['datetime'],
                            y=model_results['train_predictions'],
                            mode='lines',
                            name='Train Predictions'
                        ))
                        fig.add_trace(go.Scatter(
                            x=token_data['datetime'],
                            y=model_results['test_predictions'],
                            mode='lines',
                            name='Test Predictions'
                        ))
                        fig.update_layout(
                            title=f"{selected_token} LSTM Predictions",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Train RMSE", f"{model_results['train_rmse']:.4f}")
                        with col2:
                            st.metric("Test RMSE", f"{model_results['test_rmse']:.4f}")
                else:
                    model_results = modeler.fit_sarima(token_data, selected_token)
                    if model_results is None:
                        st.error("Failed to generate SARIMA predictions. Not enough data points.")
                    else:
                        # Plot predictions
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=token_data['datetime'],
                            y=token_data['price'],
                            mode='lines',
                            name='Actual'
                        ))
                        fig.add_trace(go.Scatter(
                            x=token_data['datetime'],
                            y=model_results['forecast'],
                            mode='lines',
                            name='Forecast'
                        ))
                        fig.update_layout(
                            title=f"{selected_token} SARIMA Forecast",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("AIC", f"{model_results['aic']:.2f}")
                        with col2:
                            st.metric("BIC", f"{model_results['bic']:.2f}")
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    with tab3:
        st.subheader("Trading Signals")
        
        # Signal parameters
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.radio("Model Type", ["LSTM", "SARIMA"])
        with col2:
            threshold = st.slider("Signal Threshold", 0.01, 0.1, 0.02, 0.01)
        
        if st.button("Generate Signals"):
            try:
                signals = modeler.generate_trading_signals(
                    token_data, selected_token,
                    model_type=model_type.lower(),
                    threshold=threshold
                )
                
                if signals.is_empty():
                    st.error("Failed to generate trading signals. Not enough data points.")
                else:
                    # Plot signals
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=signals['datetime'],
                        y=signals['actual_price'],
                        mode='lines',
                        name='Price'
                    ))
                    
                    # Add buy signals
                    buy_signals = signals[signals['signal'] == 1]
                    if not buy_signals.is_empty():
                        fig.add_trace(go.Scatter(
                            x=buy_signals['datetime'],
                            y=buy_signals['actual_price'],
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='green'),
                            name='Buy Signal'
                        ))
                    
                    # Add sell signals
                    sell_signals = signals[signals['signal'] == -1]
                    if not sell_signals.is_empty():
                        fig.add_trace(go.Scatter(
                            x=sell_signals['datetime'],
                            y=sell_signals['actual_price'],
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=10, color='red'),
                            name='Sell Signal'
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_token} Trading Signals",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display signals table
                    st.dataframe(signals[['datetime', 'actual_price', 'predicted_price', 'signal']])
            except Exception as e:
                st.error(f"Error generating signals: {str(e)}")
    
    with tab4:
        st.subheader("Strategy Performance")
        
        if st.button("Evaluate Strategy"):
            try:
                signals = modeler.generate_trading_signals(
                    token_data, selected_token,
                    model_type=model_type.lower(),
                    threshold=threshold
                )
                
                if signals.is_empty():
                    st.error("Failed to evaluate strategy. Not enough data points.")
                else:
                    performance = modeler.evaluate_strategy(signals)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{performance['total_return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Win Rate", f"{performance['win_rate']:.2%}")
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")
                    with col5:
                        st.metric("Number of Trades", performance['num_trades'])
                    
                    # Plot cumulative returns
                    signals['cumulative_return'] = (1 + signals['strategy_return']).cumprod() - 1
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=signals['datetime'],
                        y=signals['cumulative_return'],
                        mode='lines',
                        name='Cumulative Return'
                    ))
                    fig.update_layout(
                        title="Cumulative Strategy Returns",
                        xaxis_title="Time",
                        yaxis_title="Cumulative Return",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error evaluating strategy: {str(e)}")

    # Extract features and classify pattern
    features = analyzer.extract_features(token_data)
    pattern = analyzer.classify_pattern(features)
    st.markdown(f"**Pattern Type:** `{pattern}`")
    st.dataframe(pd.DataFrame([features]).T.rename(columns={0: selected_token}))
    st.markdown("**STL Decomposition**")
    stl = analyzer.decompose_series(token_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=token_data['datetime'], y=stl['trend'], name='Trend'))
    fig.add_trace(go.Scatter(x=token_data['datetime'], y=stl['seasonal'], name='Seasonal'))
    fig.add_trace(go.Scatter(x=token_data['datetime'], y=stl['resid'], name='Residual'))
    fig.update_layout(title="STL Decomposition", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # SARIMA fit (single token only)
    sarima_clicked = st.button(f"Fit SARIMA for {selected_token}")
    if sarima_clicked:
        try:
            sarima = modeler.fit_sarima(token_data, selected_token)
            if sarima is not None:
                st.write(f"AIC: {sarima['aic']:.2f}, BIC: {sarima['bic']:.2f}, Stationary: {sarima['stationarity']['is_stationary']}")
                forecast = sarima['forecast']
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=token_data['datetime'], y=token_data['price'], name='Actual'))
                future_idx = pd.date_range(token_data['datetime'][-1], periods=len(forecast)+1, freq='T')[1:]
                fig2.add_trace(go.Scatter(x=future_idx, y=forecast, name='Forecast'))
                fig2.update_layout(title="SARIMA Forecast", height=350)
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"SARIMA error: {e}")

def render_multi_token_analysis(df: pl.DataFrame, modeler: TimeSeriesModeler, analyzer: TimeSeriesAnalyzer):
    """Render multi-token analysis page"""
    st.header("Multi-Token Analysis")
    
    # Show which folders are currently loaded
    st.info(f"Loaded data from: {df['dataset'].unique().to_list()}")
    
    # Dataset folder selection
    dataset_folders = sorted([f.name for f in Path("data/raw").iterdir() if f.is_dir() and f.name.startswith('dataset')])
    selected_dataset = st.selectbox("Select Dataset Folder", dataset_folders)
    
    # Filter data for selected dataset
    dataset_data = df[df['token'].isin([f.name.split('_')[0] for f in (Path("data/raw") / selected_dataset).glob("*.parquet")])].copy()
    
    if dataset_data.is_empty():
        st.error(f"No data found in {selected_dataset}")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "Price Comparison", "Volatility Analysis", "Correlation Analysis"
    ])
    
    with tab1:
        st.subheader("Price Comparison")
        
        # Normalize prices to starting value
        normalized_prices = pl.DataFrame()
        for token in dataset_data['token'].unique():
            token_data = dataset_data[dataset_data['token'] == token].sort(pl.col('datetime'))
            if not token_data.is_empty():
                normalized_prices[token] = token_data['price'] / token_data['price'][0]
                normalized_prices.index = token_data['datetime']
        
        # Plot normalized prices
        fig = go.Figure()
        for token in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices['datetime'],
                y=normalized_prices[token],
                mode='lines',
                name=token
            ))
        fig.update_layout(
            title="Normalized Price Comparison",
            xaxis_title="Time",
            yaxis_title="Normalized Price",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display price statistics
        st.subheader("Price Statistics")
        stats_df = pl.DataFrame()
        for token in dataset_data['token'].unique():
            token_data = dataset_data[dataset_data['token'] == token]
            stats_df[token] = {
                'Start Price': token_data['price'][0],
                'End Price': token_data['price'][-1],
                'Price Change': f"{((token_data['price'][-1] / token_data['price'][0] - 1) * 100):.2f}%",
                'Data Points': len(token_data)
            }
        st.dataframe(stats_df)
    
    with tab2:
        st.subheader("Volatility Analysis")
        
        # Calculate rolling volatility
        volatility_df = pl.DataFrame()
        window = st.slider("Volatility Window (minutes)", 5, 60, 15)
        
        for token in dataset_data['token'].unique():
            token_data = dataset_data[dataset_data['token'] == token].sort(pl.col('datetime'))
            if not token_data.is_empty():
                returns = token_data['price'].pct_change()
                volatility = returns.rolling(window=window).std() * np.sqrt(window)
                volatility_df[token] = volatility
                volatility_df.index = token_data['datetime']
        
        # Plot volatility
        fig = go.Figure()
        for token in volatility_df.columns:
            fig.add_trace(go.Scatter(
                x=volatility_df['datetime'],
                y=volatility_df[token],
                mode='lines',
                name=token
            ))
        fig.update_layout(
            title=f"Rolling Volatility ({window}-minute window)",
            xaxis_title="Time",
            yaxis_title="Volatility",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display volatility statistics
        st.subheader("Volatility Statistics")
        vol_stats = pl.DataFrame()
        for token in volatility_df.columns:
            vol_stats[token] = {
                'Mean Volatility': f"{volatility_df[token].mean():.4f}",
                'Max Volatility': f"{volatility_df[token].max():.4f}",
                'Min Volatility': f"{volatility_df[token].min():.4f}"
            }
        st.dataframe(vol_stats)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Calculate returns correlation
        returns_df = pl.DataFrame()
        for token in dataset_data['token'].unique():
            token_data = dataset_data[dataset_data['token'] == token].sort(pl.col('datetime'))
            if not token_data.is_empty():
                returns_df[token] = token_data['price'].pct_change()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Plot correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title="Returns Correlation Matrix",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation statistics
        st.subheader("Correlation Statistics")
        st.dataframe(corr_matrix)
        
        # Find most correlated and anti-correlated pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                    'Correlation': corr_matrix.iloc[i,j]
                })
        
        corr_pairs_df = pl.DataFrame(corr_pairs)
        corr_pairs_df = corr_pairs_df.sort(pl.col('Correlation'), descending=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Correlated Pairs")
            st.dataframe(corr_pairs_df.head())
        with col2:
            st.subheader("Most Anti-Correlated Pairs")
            st.dataframe(corr_pairs_df.tail())

    # Extract features and classify pattern for each token
    tokens = dataset_data['token'].unique().to_list()
    summary = []
    for token in tokens:
        token_data = dataset_data[dataset_data['token'] == token].sort(pl.col('datetime'))
        features = analyzer.extract_features(token_data)
        pattern = analyzer.classify_pattern(features)
        summary.append({**features, 'token': token, 'pattern': pattern})
    summary_df = pl.DataFrame(summary)
    st.dataframe(summary_df.set_index('token'))
    st.markdown("**Pattern Type Counts**")
    st.dataframe(summary_df['pattern'].value_counts().to_frame('count'))
    st.markdown("**Overlay: Normalized Price and Trend**")
    fig = go.Figure()
    for token in tokens:
        token_data = dataset_data[dataset_data['token'] == token].sort(pl.col('datetime'))
        norm_price = token_data['price'] / token_data['price'][0]
        stl = analyzer.decompose_series(token_data)
        fig.add_trace(go.Scatter(x=token_data['datetime'], y=norm_price, name=f'{token} Price', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=token_data['datetime'], y=stl['trend']/stl['trend'][0], name=f'{token} Trend', line=dict(dash='dot', width=1)))
    fig.update_layout(title="Normalized Price and Trend Overlay", height=400)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render_time_series_page() 