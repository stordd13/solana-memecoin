import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from analyze_tokens import TokenAnalyzer
import plotly.io as pio
pio.templates.default = "plotly_white"

# Set page config
st.set_page_config(
    page_title="Token Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = TokenAnalyzer()
if 'results' not in st.session_state:
    st.session_state.results = None

def load_data():
    """Load and analyze token data"""
    with st.spinner('Analyzing tokens... This might take a few minutes.'):
        st.session_state.results = st.session_state.analyzer.analyze_multiple_tokens()
        return st.session_state.analyzer.generate_summary_report(st.session_state.results)

def plot_price_comparison(df, selected_tokens):
    """Create price comparison plot"""
    fig = go.Figure()
    
    for token in selected_tokens:
        if token in st.session_state.results:
            data = st.session_state.results[token]['data']
            if data is not None:
                # Normalize prices to start at 1
                normalized_price = data['price'] / data['price'].iloc[0]
                fig.add_trace(go.Scatter(
                    x=data['datetime'],
                    y=normalized_price,
                    name=token.split('_')[0],
                    mode='lines'
                ))
    
    fig.update_layout(
        title='Normalized Price Comparison',
        xaxis_title='Time',
        yaxis_title='Normalized Price (Starting at 1)',
        hovermode='x unified',
        height=600
    )
    return fig

def plot_volatility_comparison(df, selected_tokens):
    """Create volatility comparison plot"""
    fig = go.Figure()
    
    for token in selected_tokens:
        if token in st.session_state.results:
            data = st.session_state.results[token]['data']
            if data is not None:
                fig.add_trace(go.Scatter(
                    x=data['datetime'],
                    y=data['rolling_std'],
                    name=token.split('_')[0],
                    mode='lines'
                ))
    
    fig.update_layout(
        title='Rolling Volatility Comparison (1-hour window)',
        xaxis_title='Time',
        yaxis_title='Volatility',
        hovermode='x unified',
        height=600
    )
    return fig

def plot_anomaly_distribution(df):
    """Create anomaly distribution plot"""
    fig = px.histogram(
        df,
        x='anomaly_count',
        nbins=20,
        title='Distribution of Price Anomalies',
        labels={'anomaly_count': 'Number of Anomalies', 'count': 'Number of Tokens'}
    )
    return fig

def plot_volatility_vs_anomalies(df):
    """Create scatter plot of volatility vs anomalies"""
    fig = px.scatter(
        df,
        x='price_volatility',
        y='anomaly_count',
        hover_data=['token'],
        title='Volatility vs Number of Anomalies',
        labels={
            'price_volatility': 'Price Volatility',
            'anomaly_count': 'Number of Anomalies'
        }
    )
    return fig

def main():
    st.title("Token Analysis Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    if st.sidebar.button("Load/Refresh Data"):
        df = load_data()
    else:
        if st.session_state.results is None:
            df = load_data()
        else:
            df = st.session_state.analyzer.generate_summary_report(st.session_state.results)
    
    # Token selection
    all_tokens = list(st.session_state.results.keys()) if st.session_state.results else []
    selected_tokens = st.sidebar.multiselect(
        "Select Tokens to Analyze",
        all_tokens,
        default=all_tokens[:5] if all_tokens else []
    )
    
    # Main content
    if not selected_tokens:
        st.warning("Please select at least one token to analyze")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Analysis", 
        "Volatility Analysis", 
        "Anomaly Analysis",
        "Summary Statistics"
    ])
    
    with tab1:
        st.plotly_chart(plot_price_comparison(df, selected_tokens), use_container_width=True)
        
        # Price statistics
        st.subheader("Price Statistics")
        price_stats = df[df['token'].isin([t.split('_')[0] for t in selected_tokens])]
        st.dataframe(price_stats[['token', 'mean_price', 'price_volatility', 'max_price_change', 'min_price_change']])
    
    with tab2:
        st.plotly_chart(plot_volatility_comparison(df, selected_tokens), use_container_width=True)
        
        # Volatility statistics
        st.subheader("Volatility Statistics")
        vol_stats = df[df['token'].isin([t.split('_')[0] for t in selected_tokens])]
        st.dataframe(vol_stats[['token', 'price_volatility', 'time_span_hours', 'data_points']])
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_anomaly_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_volatility_vs_anomalies(df), use_container_width=True)
        
        # Anomaly statistics
        st.subheader("Anomaly Statistics")
        anomaly_stats = df[df['token'].isin([t.split('_')[0] for t in selected_tokens])]
        st.dataframe(anomaly_stats[['token', 'anomaly_count', 'max_price_change', 'min_price_change']])
    
    with tab4:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            labels=dict(color="Correlation")
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 