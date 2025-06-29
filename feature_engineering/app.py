"""
Enhanced Feature Engineering & Analysis Application
Combines feature engineering with improved correlation analysis and FFT capabilities
"""

import streamlit as st
import polars as pl
import numpy as np
from pathlib import Path
import sys
import os
import warnings
import random
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
try:
    from .advanced_feature_engineering import AdvancedFeatureEngineer, create_rolling_features_safe
    from .correlation_analysis import TokenCorrelationAnalyzer, load_tokens_for_correlation
except ImportError:
    # Fallback for direct execution
    from advanced_feature_engineering import AdvancedFeatureEngineer, create_rolling_features_safe
    from correlation_analysis import TokenCorrelationAnalyzer, load_tokens_for_correlation

# Import price analysis for on-demand global features
from data_analysis.price_analysis import PriceAnalyzer

# Page configuration
st.set_page_config(
    page_title="🚀 Enhanced Feature Engineering Suite",
    page_icon="📊",
    layout="wide"
)

class EnhancedDataLoader:
    """Enhanced data loader that can browse any folder in data/"""
    
    def __init__(self, base_path: Path = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent / "data"
        else:
            self.base_path = Path(base_path)
    
    def get_all_folders(self) -> List[Path]:
        """Get all folders containing parquet files in data/"""
        folders = []
        for root, dirs, files in os.walk(self.base_path):
            if any(f.endswith('.parquet') for f in files):
                folders.append(Path(root))
        return sorted(folders)
    
    def get_relative_path(self, full_path: Path) -> str:
        """Get path relative to data/ folder"""
        try:
            return str(full_path.relative_to(self.base_path))
        except:
            return str(full_path)
    
    def get_tokens_in_folder(self, folder_path: Path) -> List[Dict]:
        """Get all token files in a specific folder"""
        tokens = []
        for file_path in folder_path.glob("*.parquet"):
            tokens.append({
                'symbol': file_path.stem.replace('_features', ''),
                'file': str(file_path),
                'folder': self.get_relative_path(folder_path)
            })
        return sorted(tokens, key=lambda x: x['symbol'])
    
    def load_token_data(self, file_path: str) -> Optional[pl.DataFrame]:
        """Load a single token file"""
        try:
            df = pl.read_parquet(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_multiple_tokens(self, token_infos: List[Dict]) -> Dict[str, pl.DataFrame]:
        """Load multiple tokens for correlation analysis"""
        token_data = {}
        progress_bar = st.progress(0)
        
        for i, token_info in enumerate(token_infos):
            progress_bar.progress((i + 1) / len(token_infos))
            
            df = self.load_token_data(token_info['file'])
            if df is not None:
                token_data[token_info['symbol']] = df
        
        progress_bar.empty()
        return token_data

def main():
    """Main function to run the Streamlit app"""
    
    # Initialize data loader and analyzers
    # These are initialized once and stored in session state
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = EnhancedDataLoader()
    
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = AdvancedFeatureEngineer()
        
    if 'correlation_analyzer' not in st.session_state:
        st.session_state.correlation_analyzer = TokenCorrelationAnalyzer()

    data_loader = st.session_state.data_loader
    feature_engineer = st.session_state.feature_engineer
    correlation_analyzer = st.session_state.correlation_analyzer
    
    st.sidebar.title("🔬 Analysis Dashboard")

    analysis_type = st.sidebar.radio(
        "Select Analysis Type:",
        ("🧮 Feature Engineering", "🔗 Correlation Analysis", "📊 FFT Analysis", "⚙️ Batch Processing", "📋 Implementation Report")
    )

    # NEW: Data source folder selection --------------------------------------
    all_folders = data_loader.get_all_folders()
    if not all_folders:
        st.sidebar.error("No folders with parquet files found in data/")
        st.stop()

    # Create human-readable labels (relative paths)
    folder_labels = [data_loader.get_relative_path(p) for p in all_folders]

    # Persist selection across reruns
    if 'selected_data_folder' not in st.session_state or st.session_state.selected_data_folder not in folder_labels:
        st.session_state.selected_data_folder = folder_labels[0]

    selected_folder_label = st.sidebar.selectbox(
        "Data Source Folder:",
        folder_labels,
        index=folder_labels.index(st.session_state.selected_data_folder)
    )

    st.session_state.selected_data_folder = selected_folder_label

    # Map label back to Path object
    selected_folder = all_folders[folder_labels.index(selected_folder_label)]

    st.sidebar.info(f"Using data source: `data/{selected_folder_label}`")

    # -----------------------------------------------------------------------

    available_tokens = data_loader.get_tokens_in_folder(selected_folder)

    if analysis_type == "🧮 Feature Engineering":
        run_feature_engineering(data_loader, feature_engineer, available_tokens)
    elif analysis_type == "🔗 Correlation Analysis":
        run_correlation_analysis(data_loader, correlation_analyzer, available_tokens)
    elif analysis_type == "📊 FFT Analysis":
        run_fft_analysis(data_loader, correlation_analyzer, available_tokens)
    elif analysis_type == "⚙️ Batch Processing":
        run_batch_processing(data_loader, feature_engineer, available_tokens)
    elif analysis_type == "📋 Implementation Report":
        show_implementation_report()

def run_feature_engineering(data_loader, feature_engineer, available_tokens):
    """Run feature engineering analysis"""
    st.header("🧮 Advanced Feature Engineering")
    
    if not available_tokens:
        st.error("❌ No tokens found")
        return
    
    # Token selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selection_mode = st.radio(
            "Token Selection Mode:",
            ["Single Token", "Multiple Tokens", "Random Selection"],
            horizontal=True
        )
    
    selected_tokens = []
    
    if selection_mode == "Single Token":
        selected_symbol = st.selectbox(
            "Select token:",
            options=[t['symbol'] for t in available_tokens]
        )
        selected_tokens = [t for t in available_tokens if t['symbol'] == selected_symbol]
    
    elif selection_mode == "Multiple Tokens":
        selected_symbols = st.multiselect(
            "Select tokens:",
            options=[t['symbol'] for t in available_tokens],
            default=[t['symbol'] for t in available_tokens[:3]]
        )
        selected_tokens = [t for t in available_tokens if t['symbol'] in selected_symbols]
    
    else:  # Random Selection
        num_random = st.number_input(
            "Number of tokens:", 
            min_value=2, 
            max_value=min(50, len(available_tokens)), 
            value=15,
            help="Enter the number of random tokens to select"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎲 Select Random"):
                # Store in session state to persist across reruns
                st.session_state.fft_random_selected_tokens = random.sample(
                    available_tokens, 
                    min(num_random, len(available_tokens))
                )
                st.rerun()  # Refresh to show the selection
        
        with col2:
            if 'fft_random_selected_tokens' in st.session_state:
                selected_tokens = st.session_state.fft_random_selected_tokens
                st.success(f"✅ Selected {len(selected_tokens)} random tokens")
                
                # Show selected token names
                token_names = [t['symbol'] for t in selected_tokens]
                st.write(f"**Selected tokens**: {', '.join(token_names[:10])}" + 
                        (f" and {len(token_names)-10} more..." if len(token_names) > 10 else ""))
                
                # Option to clear selection
                if st.button("🗑️ Clear Selection", key="fft_clear"):
                    del st.session_state.fft_random_selected_tokens
                    st.rerun()
            else:
                selected_tokens = []
                st.info("Click 'Select Random' to choose tokens")
    
    # Feature configuration
    with st.expander("⚙️ Feature Configuration"):
        col1, col2 = st.columns(2)
        with col1:
            use_technical = st.checkbox("Technical Indicators", value=True)
            use_statistical = st.checkbox("Statistical Features", value=True)
            use_volume = st.checkbox("Volume Features", value=True)
        with col2:
            window_sizes = st.multiselect(
                "Window Sizes:",
                options=[5, 10, 20, 30, 60, 120],
                default=[20, 60]
            )
    
    if selected_tokens and st.button("🚀 Engineer Features", type="primary"):
        progress_bar = st.progress(0)
        
        for i, token_info in enumerate(selected_tokens):
            progress_bar.progress((i + 1) / len(selected_tokens))
            
            df = data_loader.load_token_data(token_info['file'])
            if df is not None:
                with st.expander(f"📊 {token_info['symbol']}", expanded=(len(selected_tokens) == 1)):
                    analyze_token_features(df, token_info['symbol'], feature_engineer)
        
        progress_bar.empty()

def analyze_token_features(df, token_symbol, feature_engineer):
    """Analyze features for a single token with on-demand global features"""
    st.subheader(f"📊 Feature Analysis: {token_symbol}")
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_rolling_features = st.checkbox("🔄 Show Rolling Features", value=True)
    with col2:
        show_global_features = st.checkbox("📊 Show Global Features", value=False,
                                         help="Compute global analysis features on-demand")
    with col3:
        show_technical_indicators = st.checkbox("📈 Show Technical Indicators", value=True)
    
    if not any([show_rolling_features, show_global_features, show_technical_indicators]):
        st.warning("⚠️ Please select at least one feature type to display")
        return
    
    # Create comprehensive features
    with st.spinner("🔬 Engineering features..."):
        features = feature_engineer.create_comprehensive_features(df, token_symbol)
    
    if features['status'] != 'success':
        st.error(f"❌ Feature engineering failed: {features.get('reason', 'Unknown error')}")
        return
    
    # Display results in tabs
    tabs = []
    if show_rolling_features:
        tabs.append("🔄 Rolling Features")
    if show_global_features:
        tabs.append("📊 Global Features")
    if show_technical_indicators:
        tabs.append("📈 Technical Indicators")
    
    tab_objects = st.tabs(tabs)
    tab_index = 0
    
    # Rolling Features Tab
    if show_rolling_features:
        with tab_objects[tab_index]:
            st.subheader("🔄 Rolling/ML-Safe Features")
            st.info("💡 These features use only historical data and are safe for ML training")
            
            # Create rolling features using the safe function
            rolling_features_df = create_rolling_features_safe(df, token_symbol)
            
            if not rolling_features_df.is_empty():
                # Display rolling features summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Data Points", rolling_features_df.height)
                with col2:
                    st.metric("Rolling Features", len([c for c in rolling_features_df.columns if c not in ['datetime', 'price']]))
                with col3:
                    time_span = (rolling_features_df['datetime'].max() - rolling_features_df['datetime'].min()).total_seconds() / 3600
                    st.metric("Time Span (hours)", f"{time_span:.1f}")
                
                # Show sample of rolling features
                st.subheader("📋 Rolling Features Sample")
                feature_cols = [c for c in rolling_features_df.columns if c not in ['datetime', 'price']]
                if feature_cols:
                    display_df = rolling_features_df.select(['datetime'] + feature_cols[:10]).tail(10)
                    st.dataframe(display_df.to_pandas(), use_container_width=True)
                    
                    if len(feature_cols) > 10:
                        st.info(f"📊 Showing 10 of {len(feature_cols)} rolling features")
                
                # Plot key rolling features
                if 'log_returns' in rolling_features_df.columns:
                    st.subheader("📈 Rolling Features Visualization")
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Log Returns', 'Rolling Volatility', 'Cumulative Returns', 'Rolling Sharpe'),
                        vertical_spacing=0.08
                    )
                    
                    data = rolling_features_df.to_pandas()
                    
                    # Log returns
                    if 'log_returns' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data['datetime'], y=data['log_returns'],
                            name='Log Returns', line=dict(color='blue', width=1)
                        ), row=1, col=1)
                    
                    # Rolling volatility
                    if 'rolling_volatility' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data['datetime'], y=data['rolling_volatility'],
                            name='Rolling Vol', line=dict(color='red', width=1)
                        ), row=1, col=2)
                    
                    # Cumulative returns
                    if 'cumulative_log_returns' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data['datetime'], y=data['cumulative_log_returns'],
                            name='Cumulative Returns', line=dict(color='green', width=1)
                        ), row=2, col=1)
                    
                    # Rolling Sharpe
                    if 'rolling_sharpe' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data['datetime'], y=data['rolling_sharpe'],
                            name='Rolling Sharpe', line=dict(color='purple', width=1)
                        ), row=2, col=2)
                    
                    fig.update_layout(height=600, showlegend=False, title="Rolling Features Over Time")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No rolling features generated")
        
        tab_index += 1
    
    # Global Features Tab (computed on-demand)
    if show_global_features:
        with tab_objects[tab_index]:
            st.subheader("📊 Global Analysis Features")
            st.info("🧠 These features are computed on-demand for analysis (not stored)")
            
            # Compute global features on-demand
            with st.spinner("🔬 Computing global features on-demand..."):
                global_features = compute_global_features_on_demand(df, token_symbol)
            
            # Display global features
            display_global_features(global_features, token_symbol)
        
        tab_index += 1
    
    # Technical Indicators Tab
    if show_technical_indicators:
        with tab_objects[tab_index]:
            st.subheader("📈 Technical Indicators")
            
            if features['technical_features']:
                tech_features = features['technical_features']
                
                # MACD
                if tech_features.get('macd'):
                    st.subheader("📊 MACD")
                    macd = tech_features['macd']
                    
                    # MACD visualization
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
                        vertical_spacing=0.1
                    )
                    
                    x_axis = list(range(len(macd['macd_line'])))
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=macd['macd_line'],
                        name='MACD Line', line=dict(color='blue')
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis, y=macd['signal_line'],
                        name='Signal Line', line=dict(color='red')
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Bar(
                        x=x_axis, y=macd['histogram'],
                        name='Histogram', marker_color='green'
                    ), row=2, col=1)
                    
                    fig.update_layout(height=500, title="MACD Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Current Position", macd['current_position'])
                
                # Bollinger Bands
                if tech_features.get('bollinger_bands'):
                    st.subheader("📊 Bollinger Bands")
                    bb = tech_features['bollinger_bands']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Position", f"{bb['current_position']:.3f}")
                    with col2:
                        st.metric("Squeeze Periods", bb['squeeze_periods'])
                    with col3:
                        if abs(bb['current_position']) > 0.8:
                            st.warning("⚠️ Near band extremes")
                        else:
                            st.success("✅ Normal range")
                
                # RSI
                if tech_features.get('enhanced_rsi') and isinstance(tech_features['enhanced_rsi'], dict):
                    st.subheader("📊 Enhanced RSI")
                    rsi = tech_features['enhanced_rsi']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current RSI", f"{rsi['current_rsi']:.1f}")
                    with col2:
                        st.metric("Signal", rsi['signal'].title())
                    with col3:
                        if rsi.get('divergence', {}).get('has_divergence', False):
                            div_type = rsi['divergence']['type']
                            st.metric("Divergence", div_type.title())
                        else:
                            st.metric("Divergence", "None")
            else:
                st.warning("⚠️ No technical indicators available")
    
    # Summary
    st.success(f"✅ Feature analysis complete for {token_symbol}")
    
    # Architecture note
    st.info("""
    🧠 **Clean Architecture Notes:**
    - **Rolling Features**: Pre-computed and stored in data/features/ (ML-safe)
    - **Global Features**: Computed on-demand using price_analysis module (no redundancy)  
    - **Technical Indicators**: Computed during analysis (rolling by nature)
    """)

def run_correlation_analysis(data_loader, correlation_analyzer, available_tokens):
    """Run enhanced correlation analysis with lifecycle sync only"""
    st.header("🔗 Enhanced Token Correlation Analysis")
    
    if len(available_tokens) < 2:
        st.error("❌ Need at least 2 tokens for correlation analysis")
        return

    # Simple explanation
    st.info("""
    **🕐 Lifecycle-Based Correlation Analysis**
    
    Compares tokens based on their **position in their lifecycle** (minute 0, 1, 2... from launch)
    rather than calendar time. This allows correlation analysis between tokens launched at
    different dates by comparing their relative behavior patterns.
    
    **Example**: Compare how Token A behaved in its first 4 hours vs how Token B behaved in its first 4 hours
    """)

    # Configuration
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selection_mode = st.selectbox(
            "Token Selection:",
            ["All Tokens", "Multiple Tokens", "Random Selection", "Single Token Pair"]
        )
    
    with col2:
        data_type = st.selectbox(
            "Data to correlate:",
            ["Log Returns", "Normalized Prices", "Prices"],
            help="Log Returns: ln(price_t/price_t-1) | Normalized Prices: price/first_price"
        )
    
    with col3:
        correlation_method = st.selectbox(
            "Method:",
            ["pearson", "spearman", "kendall"],
            help="Pearson: linear relationships | Spearman: monotonic relationships"
        )
    
    # Add scaling explanation
    st.info("""
    **📊 Data Types & Scaling:**
    
    • **Log Returns**: `ln(price_t/price_t-1)` - Scale-independent, pure correlation ✅
    • **Normalized Prices**: `price/first_price` - Trajectory comparison from launch ✅  
    • **Prices**: Raw values - Uses RobustScaler for extreme price differences ⚠️
    
    **💡 Recommendation**: Use "Log Returns" for pure correlation, "Normalized Prices" for trajectory comparison
    """)

    # Simplified advanced options - lifecycle only
    with st.expander("⚙️ Advanced Options"):
        st.markdown("""
        **🔧 Lifecycle Analysis Options:**
        
        • **Lifecycle minutes**: How much data from each token's launch to analyze
        • **Minimum data per token**: Each token must have at least this much lifecycle data
        • **Rolling correlations**: See how relationships change over time (computationally intensive)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            lifecycle_minutes = st.number_input(
                "Lifecycle minutes to analyze:", 
                min_value=60, 
                max_value=1440, 
                value=240,
                help="240 = first 4 hours from launch"
            )
            
            min_data_per_token = st.number_input(
                "Minimum data per token:", 
                min_value=20, 
                value=30,
                help="Each token must have at least this many minutes of data"
            )
            
        with col2:
            use_rolling = st.checkbox(
                "Calculate rolling correlations", 
                value=False,
                help="Show how correlations change over time"
            )
            
            if use_rolling:
                rolling_window = st.selectbox(
                    "Rolling window:", 
                    [60, 240, 720],
                    format_func=lambda x: f"{x} minutes ({x//60}h)" if x >= 60 else f"{x} minutes"
                )
            else:
                rolling_window = 240

    # PCA Configuration
    with st.expander("📊 PCA Configuration"):
        col1, col2 = st.columns(2)
        with col1:
            enable_pca = st.checkbox("Enable PCA Analysis", value=True, 
                                   help="Principal Component Analysis to identify patterns and redundancy")
            max_components = st.slider(
                "Maximum PCA components:", 
                min_value=2, 
                max_value=min(10, len(available_tokens)), 
                value=min(5, len(available_tokens)),
                help="Number of principal components to calculate"
            )
        with col2:
            show_pca_plot = st.checkbox("Show PCA visualization", value=True)
            pca_plot_components = st.slider(
                "Components in plots:", 
                min_value=2, 
                max_value=6, 
                value=4,
                help="Number of components to show in visualizations"
            )

    # Token selection logic
    selected_tokens = []
    
    if selection_mode == "All Tokens":
        st.info("💡 **All Tokens mode**: Analyzes correlations between ALL available tokens")
        
        if len(available_tokens) > 50:
            st.warning(f"⚠️ You have {len(available_tokens)} tokens. Consider limiting for performance.")
            limit_tokens = st.checkbox("Limit for performance", value=True)
            if limit_tokens:
                max_tokens = st.slider("Maximum tokens:", 10, 100, 50)
                selected_tokens = available_tokens[:max_tokens]
            else:
                selected_tokens = available_tokens
        else:
            selected_tokens = available_tokens
            
        st.success(f"✅ Will analyze {len(selected_tokens)} tokens")
    
    elif selection_mode == "Multiple Tokens":
        selected_symbols = st.multiselect(
            "Select tokens:",
            options=[t['symbol'] for t in available_tokens],
            default=[t['symbol'] for t in available_tokens[:10]],
            help="Choose specific tokens for correlation analysis"
        )
        selected_tokens = [t for t in available_tokens if t['symbol'] in selected_symbols]
    
    elif selection_mode == "Random Selection":
        num_random = st.number_input(
            "Number of tokens:", 
            min_value=2, 
            max_value=min(100, len(available_tokens)), 
            value=15,
            help="Enter the number of random tokens to select for FFT (max 100)."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎲 Select Random", key="fft_random_btn"):
                # Store in session state to persist across reruns
                st.session_state.fft_random_selected_tokens = random.sample(
                    available_tokens, 
                    min(num_random, len(available_tokens))
                )
                st.rerun()  # Refresh to show the selection
        
        with col2:
            if 'fft_random_selected_tokens' in st.session_state:
                selected_tokens = st.session_state.fft_random_selected_tokens
                st.success(f"✅ Selected {len(selected_tokens)} random tokens")
                
                # Show selected token names
                token_names = [t['symbol'] for t in selected_tokens]
                st.write(f"**Selected tokens**: {', '.join(token_names[:10])}" + 
                        (f" and {len(token_names)-10} more..." if len(token_names) > 10 else ""))
                
                # Option to clear selection
                if st.button("🗑️ Clear Selection", key="fft_clear_btn"):
                    del st.session_state.fft_random_selected_tokens
                    st.rerun()
            else:
                selected_tokens = []
                st.info("Click 'Select Random' to choose tokens")
    
    else:  # Single Token Pair
        st.info("💡 **Token Pair mode**: Analyzes correlation between exactly 2 tokens")
        col1, col2 = st.columns(2)
        with col1:
            token1 = st.selectbox("First token:", [t['symbol'] for t in available_tokens])
        with col2:
            token2 = st.selectbox("Second token:", [t['symbol'] for t in available_tokens if t['symbol'] != token1])
        
        selected_tokens = [t for t in available_tokens if t['symbol'] in [token1, token2]]

    # Run analysis
    if len(selected_tokens) >= 2 and st.button("🔗 Run Lifecycle Correlation Analysis", type="primary"):
        try:
            st.write("🚀 **Starting lifecycle correlation analysis...**")
            
            with st.spinner(f"Analyzing {len(selected_tokens)} tokens..."):
                # Load token data
                token_data = {}
                progress_bar = st.progress(0)
                
                for i, token_info in enumerate(selected_tokens):
                    progress_bar.progress((i + 1) / len(selected_tokens))
                    try:
                        df = data_loader.load_token_data(token_info['file'])
                        if df is not None and len(df) >= min_data_per_token:
                            token_data[token_info['symbol']] = df
                            st.write(f"✅ {token_info['symbol']}: {len(df)} data points")
                        else:
                            st.write(f"❌ {token_info['symbol']}: Insufficient data ({len(df) if df is not None else 0} points)")
                    except Exception as e:
                        st.warning(f"Failed to load {token_info['symbol']}: {e}")
                        
                progress_bar.empty()
            
            if len(token_data) < 2:
                st.error("❌ Need at least 2 tokens with sufficient lifecycle data")
                st.write(f"📊 Successfully loaded: {list(token_data.keys())}")
                return
            
            st.success(f"📊 Loaded {len(token_data)} tokens: {list(token_data.keys())}")
            
            # Determine analysis parameters
            use_log_returns = (data_type == "Log Returns")
            use_robust_scaling = (data_type == "Prices")  # Only for raw prices
            
            st.write("🔗 **Running lifecycle correlation analysis...**")
            
            # Always use lifecycle sync
            results = correlation_analyzer.analyze_token_correlations(
                token_data=token_data,
                method=correlation_method,
                min_overlap=min_data_per_token,  # Reinterpret as minimum data per token
                use_log_returns=use_log_returns,
                use_robust_scaling=use_robust_scaling,
                use_rolling=use_rolling,
                rolling_window=rolling_window if use_rolling else None,
                use_lifecycle_sync=True,  # Always True
                lifecycle_minutes=lifecycle_minutes,
                n_components=max_components if enable_pca else None  # Pass PCA components setting
            )
            
            st.write("✅ **Analysis completed!**")
            
            # Display results
            if 'error' in results:
                st.error(f"❌ Analysis failed: {results['error']}")
                if 'suggestion' in results:
                    st.info(f"💡 Suggestion: {results['suggestion']}")
            else:
                display_correlation_results(results, correlation_analyzer)
            
            # Display PCA results if enabled
            if enable_pca and results.get('pca_analysis', {}).get('pca_available', False):
                st.divider()
                display_pca_results(results['pca_analysis'], correlation_analyzer, show_pca_plot, pca_plot_components)
                
        except Exception as e:
            st.error(f"🚨 **Unexpected error:**")
            st.write(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    elif len(selected_tokens) < 2:
        st.warning("⚠️ Please select at least 2 tokens")
    elif len(selected_tokens) == 0:
        st.info("📋 Please select tokens above")

def display_correlation_results(results, correlation_analyzer):
    """Display correlation analysis results"""
    st.header("📊 Correlation Analysis Results")
    
    # Summary statistics - use the correct key names
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tokens Analyzed", len(results.get('tokens_analyzed', [])))
    with col2:
        st.metric("Data Points", f"{results.get('data_points', 0):,}")
    with col3:
        st.metric("Time Period", f"{results.get('lifecycle_minutes', 0)} min")
    with col4:
        st.metric("Sync Method", results.get('sync_method', 'Unknown').title())
    
    # Correlation matrix visualization - use correct key name
    st.subheader("Correlation Visualization")
    viz_type = st.selectbox(
        "Visualization Type:",
        ["Heatmap", "Network Graph"],
        help="Network graph is better for many tokens."
    )

    if 'correlation_matrices' in results and 'main' in results['correlation_matrices']:
        correlation_matrix = results['correlation_matrices']['main']
        
        if viz_type == "Heatmap":
            fig = correlation_analyzer.create_correlation_heatmap(correlation_matrix)
    else:
        threshold = st.slider("Correlation Threshold", 0.1, 0.9, 0.5, 0.05)
        fig = correlation_analyzer.create_correlation_network(correlation_matrix, threshold)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Rolling correlations if available
    if results.get('rolling_analysis', False) and 'correlation_matrices' in results:
        if 'rolling' in results['correlation_matrices']:
            st.subheader("📈 Rolling Correlations Over Time")
            rolling_fig = correlation_analyzer.create_rolling_correlation_plot(results['correlation_matrices']['rolling'])
            if rolling_fig:
                st.plotly_chart(rolling_fig, use_container_width=True)
    
    # Statistical summary - use correct key name
    if 'summary_stats' in results:
        st.subheader("📋 Statistical Summary")
        stats = results['summary_stats']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Correlation Statistics:**")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    st.write(f"- {key.replace('_', ' ').title()}: {value:.3f}")
    
    with col2:
        st.write("**Analysis Details:**")
        st.write(f"- Method: {results.get('method', 'Unknown').title()}")
        st.write(f"- Scaling: {results.get('scaling_method', 'Unknown')}")
        st.write(f"- Synchronization: {results.get('sync_method', 'Unknown')}")
        if results.get('lifecycle_minutes'):
            st.write(f"- Lifecycle window: {results['lifecycle_minutes']} minutes")
    
    # Significant pairs
    if 'significant_pairs' in results and results['significant_pairs']:
        st.subheader("🔗 Significant Correlations")
        pairs_data = []
        for pair in results['significant_pairs'][:10]:  # Show top 10
            pairs_data.append({
                'Token 1': pair.get('token1', ''),
                'Token 2': pair.get('token2', ''),
                'Correlation': f"{pair.get('correlation', 0):.3f}",
                'Strength': pair.get('interpretation', '')
            })
        
        if pairs_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(pairs_data), use_container_width=True)

def display_pca_results(pca_analysis, correlation_analyzer, show_plot=True, max_plot_components=4):
    """Display PCA analysis results"""
    st.header("🔍 Principal Component Analysis (PCA)")
    
    if not pca_analysis.get('pca_available', False):
        st.warning("⚠️ PCA analysis not available")
        if 'reason' in pca_analysis:
            st.info(f"Reason: {pca_analysis['reason']}")
        if 'error' in pca_analysis:
            st.error(f"Error: {pca_analysis['error']}")
        return
    
    # PCA summary - use correct key names
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Components", pca_analysis.get('n_components', 0))
    with col2:
        st.metric("Total Variance Explained", f"{pca_analysis.get('total_variance_captured', 0):.1%}")
    with col3:
        st.metric("Tokens in Analysis", pca_analysis.get('n_tokens', 0))
    
    # Explained variance
    if 'explained_variance_ratio' in pca_analysis:
        st.subheader("📊 Explained Variance by Component")
        variance_ratios = pca_analysis['explained_variance_ratio']
        
        # Create bar chart of explained variance
        fig = go.Figure(data=go.Bar(
            x=[f'PC{i+1}' for i in range(len(variance_ratios))],
            y=variance_ratios,
            text=[f'{v:.1%}' for v in variance_ratios],
            textposition='auto',
        ))
        fig.update_layout(
            title='Explained Variance Ratio by Principal Component',
            xaxis_title='Principal Component',
            yaxis_title='Variance Explained',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Component loadings - use correct key name and structure
    if 'loadings' in pca_analysis and 'feature_names' in pca_analysis:
        st.subheader("🔗 Component Loadings")
        st.write("Shows how much each token contributes to each principal component:")
        
        loadings = pca_analysis['loadings']
        feature_names = pca_analysis['feature_names']
        
        # Create DataFrame for display
        import pandas as pd
        loadings_data = {}
        for i, component_loadings in enumerate(loadings):
            loadings_data[f'PC{i+1}'] = component_loadings
        
        loadings_df = pd.DataFrame(loadings_data, index=feature_names)
        st.dataframe(loadings_df.round(3), use_container_width=True)
    
    # PCA visualization if enabled
    if show_plot:
        st.subheader("📈 PCA Visualization")
        try:
            # Use the existing PCA visualization method
            actual_plot_components = min(max_plot_components, pca_analysis.get('n_components', 2))
            fig = correlation_analyzer.create_pca_visualization(pca_analysis, actual_plot_components)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate PCA visualization")
                
        except Exception as e:
            st.error(f"Error creating PCA visualization: {e}")
    
    # Interpretation
    st.subheader("💡 PCA Insights")
    
    # Calculate insights based on actual data
    n_components = pca_analysis.get('n_components', 0)
    total_variance = pca_analysis.get('total_variance_captured', 0)
    
    insight_text = f"""
    **How to interpret these PCA results:**
    
    • **{n_components} components** capture **{total_variance:.1%}** of total variance
    • **High variance components** = capture major patterns in token behavior
    • **Similar loadings** = tokens behave similarly in this component  
    • **First 2-3 components** usually capture most meaningful patterns
    • **Component loadings** show which tokens drive each pattern
    
    **Use case**: Identify groups of tokens with similar lifecycle patterns
    """
    
    if total_variance >= 0.8:
        insight_text += "\n\n✅ **Good dimensionality reduction** - Components capture most variance"
    elif total_variance >= 0.6:
        insight_text += "\n\n⚠️ **Moderate reduction** - Consider more components for better coverage"
    else:
        insight_text += "\n\n❌ **Poor reduction** - Data may be too diverse for effective PCA"
    
    st.info(insight_text)

def display_global_features(global_features: Dict, token_name: str):
    """Display global features computed on-demand"""
    
    if 'error' in global_features:
        st.error(f"❌ {global_features['error']}")
        return
    
    if not global_features.get('computed_on_demand', False):
        st.warning("⚠️ Global features not computed")
        return
    
    analysis = global_features['global_features']
    
    st.success("✅ Global features computed on-demand using price_analysis module")
    
    # Display global statistics
    with st.expander("📊 Global Statistics", expanded=True):
        if 'price_stats' in analysis:
            stats = analysis['price_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return %", f"{stats.get('total_return_pct', 0):.2f}%")
            with col2:
                st.metric("Max Gain %", f"{stats.get('max_gain_pct', 0):.2f}%")
            with col3:
                st.metric("Max Drawdown %", f"{stats.get('max_drawdown_pct', 0):.2f}%")
            with col4:
                st.metric("Price Range %", f"{stats.get('price_range_pct', 0):.2f}%")
    
    # Display pattern analysis
    with st.expander("🔍 Pattern Analysis"):
        if 'pattern_classification' in analysis:
            pattern = analysis['pattern_classification']
            st.info(f"**Pattern Type**: {pattern}")
        
        if 'movement_patterns' in analysis:
            movement = analysis['movement_patterns']
            st.write("**Movement Characteristics:**")
            for key, value in movement.items():
                if isinstance(value, (int, float)):
                    st.write(f"- {key.replace('_', ' ').title()}: {value:.3f}")
    
    st.info("💡 **Architecture Note**: These global features are computed on-demand to avoid redundancy with stored rolling features.")

def run_fft_analysis(data_loader, correlation_analyzer, available_tokens):
    """Run FFT analysis on multiple tokens"""
    st.header("📊 Multi-Token FFT Analysis")
    
    if not available_tokens:
        st.error("❌ No tokens found")
        return
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selection_mode = st.selectbox(
            "Token Selection:",
            ["Single Token", "Multiple Tokens", "Random Selection", "Compare Tokens"]
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analyze:",
            ["Log Returns", "Prices", "Detrended Prices", "Volume"]
        )
    
    # Token selection logic (reusing the same pattern as correlation analysis)
    selected_tokens = []
    
    if selection_mode == "Single Token":
        selected_symbol = st.selectbox(
            "Select token:",
            options=[t['symbol'] for t in available_tokens]
        )
        selected_tokens = [t for t in available_tokens if t['symbol'] == selected_symbol]
    
    elif selection_mode == "Multiple Tokens":
        selected_symbols = st.multiselect(
            "Select tokens:",
            options=[t['symbol'] for t in available_tokens],
            default=[t['symbol'] for t in available_tokens[:5]],
            help="Choose specific tokens for FFT analysis"
        )
        selected_tokens = [t for t in available_tokens if t['symbol'] in selected_symbols]
    
    elif selection_mode == "Random Selection":
        num_random = st.number_input(
            "Number of tokens:", 
            min_value=2, 
            max_value=min(100, len(available_tokens)), 
            value=15,
            help="Enter the number of random tokens to select for FFT (max 100)."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎲 Select Random", key="fft_random_btn"):
                # Store in session state to persist across reruns
                st.session_state.fft_random_selected_tokens = random.sample(
                    available_tokens, 
                    min(num_random, len(available_tokens))
                )
                st.rerun()  # Refresh to show the selection
        
        with col2:
            if 'fft_random_selected_tokens' in st.session_state:
                selected_tokens = st.session_state.fft_random_selected_tokens
                st.success(f"✅ Selected {len(selected_tokens)} random tokens")
                
                # Show selected token names
                token_names = [t['symbol'] for t in selected_tokens]
                st.write(f"**Selected tokens**: {', '.join(token_names[:10])}" + 
                        (f" and {len(token_names)-10} more..." if len(token_names) > 10 else ""))
                
                # Option to clear selection
                if st.button("🗑️ Clear Selection", key="fft_clear_btn"):
                    del st.session_state.fft_random_selected_tokens
                    st.rerun()
            else:
                selected_tokens = []
                st.info("Click 'Select Random' to choose tokens")
    
    else:  # Compare Tokens
        st.info("💡 **Compare mode**: Select 2-5 tokens for side-by-side FFT comparison")
        selected_symbols = st.multiselect(
            "Select tokens to compare:",
            options=[t['symbol'] for t in available_tokens],
            default=[t['symbol'] for t in available_tokens[:3]],
            help="Choose 2-5 tokens for comparison"
        )
        selected_tokens = [t for t in available_tokens if t['symbol'] in selected_symbols]
    
    # FFT Configuration
    with st.expander("⚙️ FFT Configuration"):
        col1, col2 = st.columns(2)
        with col1:
            window_length = st.slider("Analysis window (minutes):", 60, 1440, 240)
            overlap_pct = st.slider("Window overlap %:", 0, 75, 50)
        with col2:
            max_frequencies = st.slider("Max frequencies to show:", 5, 20, 10)
            detrend_method = st.selectbox("Detrending:", ["linear", "constant", "none"])
    
    # Run FFT Analysis
    if len(selected_tokens) >= 1 and st.button("🔍 Run FFT Analysis", type="primary"):
        try:
            st.write("🚀 **Starting FFT analysis...**")
            
            with st.spinner(f"Analyzing frequency patterns for {len(selected_tokens)} tokens..."):
                # Load token data
                token_data = {}
                progress_bar = st.progress(0)
                
                for i, token_info in enumerate(selected_tokens):
                    progress_bar.progress((i + 1) / len(selected_tokens))
                    try:
                        df = data_loader.load_token_data(token_info['file'])
                        if df is not None and len(df) >= window_length:
                            token_data[token_info['symbol']] = df
                            st.write(f"✅ {token_info['symbol']}: {len(df)} data points")
                        else:
                            st.write(f"❌ {token_info['symbol']}: Insufficient data ({len(df) if df is not None else 0} points)")
                    except Exception as e:
                        st.warning(f"Failed to load {token_info['symbol']}: {e}")
                
            progress_bar.empty()
            
            if len(token_data) == 0:
                st.error("❌ No tokens with sufficient data for FFT analysis")
                return
            
            st.success(f"📊 Loaded {len(token_data)} tokens for FFT analysis")

            # Perform and display FFT analysis for each token
            for token_name, df in token_data.items():
                with st.expander(f"📈 {token_name} - FFT Results", expanded=True):
                    fft_results = correlation_analyzer.perform_fft_analysis(
                        df, analysis_type, window_length, overlap_pct
                    )
                    
                    if fft_results['status'] == 'success':
                        fig = correlation_analyzer.create_fft_visualization(fft_results)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"FFT analysis failed for {token_name}: {fft_results['reason']}")
                
        except Exception as e:
            st.error(f"🚨 **FFT Analysis Error:**")
            st.write(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    elif len(selected_tokens) == 0:
        st.info("📋 Please select tokens above")

def run_batch_processing(data_loader, feature_engineer, available_tokens):
    """Run feature engineering on a batch of tokens and save the results"""
    st.header("⚙️ Batch Feature Engineering")
    
    st.info("""
    This section allows you to run feature engineering on a large batch of tokens
    and save the output for other processes, like ML model training.
    """)
    
    # Selection UI
    all_symbols = [t['symbol'] for t in available_tokens]

    selection_mode = st.radio(
        "Token Selection Mode:",
        ["Multiple Tokens", "Random Selection", "All Tokens"],
        horizontal=True
    )

    selected_symbols = []

    if selection_mode == "Multiple Tokens":
        selected_symbols = st.multiselect(
            "Select tokens for batch processing:",
            options=all_symbols,
            default=all_symbols[:10],
            help="Choose the specific tokens you want to process."
        )

    elif selection_mode == "Random Selection":
        num_random = st.number_input(
            "Number of random tokens:",
            min_value=1,
            max_value=len(all_symbols),
            value=min(50, len(all_symbols)),
            help="Randomly select up to the specified number of tokens for batch processing."
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎲 Select Random", key="batch_random_btn"):
                st.session_state.batch_random_tokens = random.sample(all_symbols, num_random)
        with col2:
            if 'batch_random_tokens' in st.session_state:
                selected_symbols = st.session_state.batch_random_tokens
                st.success(f"✅ Selected {len(selected_symbols)} random tokens")
                if st.button("🗑️ Clear Selection", key="batch_random_clear"):
                    del st.session_state.batch_random_tokens
                    selected_symbols = []
            else:
                st.info("Click 'Select Random' to choose tokens")

    else:  # All Tokens
        selected_symbols = all_symbols
        st.info(f"All {len(all_symbols)} tokens will be processed.")

    # Output configuration
    output_folder = st.text_input(
        "Output Folder Name:",
        value="features/batch_run_1",
        help="A new folder inside `data/` will be created to store the feature files."
    )
    
    if st.button("🚀 Run Batch Processing", type="primary"):
        if not selected_symbols:
            st.error("❌ Please select at least one token.")
            return
        
        # Get full token info for selected symbols
        selected_tokens_info = [t for t in available_tokens if t['symbol'] in selected_symbols]
        
        # Create output directory
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "data" / output_folder
        output_path.mkdir(parents=True, exist_ok=True)
        
        st.write(f"📂 Saving features to: `{output_path}`")
        
        # Run processing
        progress_bar = st.progress(0)
        total_tokens = len(selected_tokens_info)
        success_count = 0
        
        with st.spinner(f"Processing {total_tokens} tokens..."):
            for i, token_info in enumerate(selected_tokens_info):
                symbol = token_info['symbol']
                progress_bar.progress((i + 1) / total_tokens, text=f"Processing: {symbol}")
                
                try:
                    df = data_loader.load_token_data(token_info['file'])
                    if df is None:
                        st.warning(f"⚠️ Could not load data for {symbol}")
                        continue
                    
                    # Use the feature engineer to create features
                    features_result = feature_engineer.create_comprehensive_features(df, symbol)
                    
                    if features_result['status'] == 'success':
                        # Convert to ML-safe rolling feature DataFrame
                        features_df = create_rolling_features_safe(df, symbol)
                        if features_df is not None and len(features_df) > 0:
                            output_file = output_path / f"{symbol}_features.parquet"
                            features_df.write_parquet(output_file)
                        success_count += 1
                    else:
                        st.warning(f"⚠️ Feature engineering failed for {symbol}: {features_result.get('reason')}")
                
                except Exception as e:
                    st.error(f"❌ An error occurred with {symbol}: {e}")
        
        progress_bar.empty()
        st.success(f"✅ Batch processing complete! Successfully processed and saved {success_count}/{total_tokens} tokens.")

if __name__ == "__main__":
    main() 