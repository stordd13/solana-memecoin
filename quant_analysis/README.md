# Quantitative Analysis Module

## Overview

The `quant_analysis/` module provides a comprehensive quantitative analysis system for Solana memecoin trading data. This professional-grade financial analysis toolkit is specifically designed for high-frequency cryptocurrency data, focusing on price action analysis and risk-adjusted performance metrics. The system leverages Polars for efficient data processing and Plotly for interactive visualizations.

## 🏗️ Module Architecture

### Core Files

#### 1. **quant_analysis.py** - Analytical Engine
The main computational engine containing all quantitative analysis algorithms and calculations.

**Core Capabilities:**
- **Risk-Adjusted Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio calculations optimized for minute-level crypto data
- **Entry/Exit Analysis**: Statistical analysis of optimal trading windows with momentum-based signals
- **Market Regime Detection**: Algorithmic classification of market states (trending, ranging, volatile)
- **Temporal Analysis**: Risk/reward analysis across multiple time horizons (5 minutes to 24 hours)
- **Statistical Analysis**: Advanced statistical measures including Hurst exponent, entropy analysis, and efficiency ratios

**Key Methods:**
```python
# Risk metrics (annualized for minute data)
calculate_sharpe_ratio(returns, periods_per_year=525600)
calculate_sortino_ratio(returns, periods_per_year=525600) 
calculate_calmar_ratio(df, periods_per_year=525600)

# Trading analysis
optimal_entry_exit_matrix(df, entry_windows, exit_windows)
temporal_risk_reward_analysis(df, time_horizons)
enhanced_entry_exit_analysis(df, entry_method='momentum')

# Market analysis
market_regime_detection(df, lookback=60)
calculate_hurst_exponent(price_series, lags=20)
```

#### 2. **quant_viz.py** - Visualization Engine
Professional financial visualizations using Plotly for interactive, publication-ready charts.

**Visualization Categories:**

**Entry/Exit Analysis:**
- `plot_entry_exit_matrix()`: Heatmaps showing optimal trading windows with statistical significance
- `plot_entry_exit_moment_matrix()`: Minute-by-minute entry/exit timing analysis
- `plot_entry_exit_moment_matrix_optimized()`: Polars-optimized version for large datasets

**Risk & Performance Analysis:**
- `plot_multi_token_risk_metrics()`: 4-panel dashboard showing win rates, Sharpe ratios, and risk/reward across time horizons
- `plot_temporal_risk_reward()`: Risk-adjusted returns analysis across different time periods
- `plot_volatility_surface()`: 3D volatility visualization across time and percentile levels

**Lifecycle Analysis:**
- `plot_lifecycle_summary_charts()`: Multi-panel analysis of token performance across lifecycle segments
- `plot_lifecycle_aggregated_analysis()`: Statistical distribution analysis for large token datasets
- `plot_lifecycle_token_ranking()`: Performance ranking and risk/return scatter plots

**Advanced Analysis:**
- `plot_regime_analysis()`: Market regime visualization with price coloring
- `plot_correlation_dynamics()`: Rolling correlation analysis between tokens
- `plot_microstructure_analysis()`: Professional 6-panel microstructure dashboard
- `plot_microstructure_summary_dashboard()`: Key metrics displayed as interactive gauges

**Distribution Analysis:**
- `plot_price_distribution_evolution()`: Comprehensive 3-row analysis showing histograms with distribution overlays, Q-Q plots, and box plots
- `plot_distribution_evolution_summary()`: 6-panel dashboard tracking distribution metrics evolution (mean/volatility, skewness/kurtosis, normality tests, return ranges, statistical significance, quality scores)

#### 3. **trading_analysis.py** - Advanced Trading Analytics
Sophisticated trading strategy components and signal generation algorithms.

**Advanced Features:**
- **Order Flow Analysis**: Buy/sell pressure estimation using price movement patterns
- **VWAP Analysis**: Volume-weighted average price calculations with multiple time anchors
- **Market Profile (TPO)**: Time-Price-Opportunity analysis for value area identification
- **Elliott Wave Detection**: Algorithmic pattern recognition for wave structure analysis
- **Market Efficiency Metrics**: Kaufman's Efficiency Ratio and Fractal Dimension calculations
- **Entropy Analysis**: Price predictability and randomness measurement
- **Liquidity Analysis**: Market depth estimation and price impact modeling

**Key Methods:**
```python
# Strategy optimization
calculate_optimal_stop_loss_take_profit(df, lookback=20)
vwap_analysis(df, anchors=['session', 'day', 'week'])
market_profile_tpo(df, tpo_size=30, value_area_pct=0.7)

# Pattern recognition
elliott_wave_detection(df, min_wave_size=5)
calculate_market_efficiency(df, window=20)
entropy_analysis(df, window=60)
liquidity_analysis(df)
```

#### 4. **quant_app.py** - Streamlit Web Application
Professional web interface providing intuitive access to all quantitative analysis features.

**Application Features:**

**Data Management:**
- **Flexible Data Source Selection**: Browse and select from multiple data directories
- **Token Selection Modes**: 
  - Single token analysis
  - Multiple specific tokens
  - Random sample (configurable size)
  - All available tokens
- **Session State Management**: Persistent data loading and analysis results

**User Interface:**
- **Sidebar Controls**: Intuitive parameter selection and configuration
- **Progress Tracking**: Real-time analysis progress with detailed status updates
- **Interactive Results**: Plotly-based charts with hover details and zoom capabilities
- **Export Options**: Download analysis results and visualizations

## 📊 Available Analysis Types

### 🎯 Multi-Token Analysis
**1. Multi-Token Risk Metrics**
- Comparative analysis of risk-adjusted returns across multiple tokens
- Time horizon analysis (5 minutes to ~24 hours)
- Statistical metrics: Win rate, Sharpe ratio, Risk/reward ratio
- 4-panel dashboard with trend analysis and key insights

**2. 24-Hour Lifecycle Analysis**
- Token performance analysis across lifecycle segments (4, 6, 8, 12, or 24 segments)
- Metrics: Returns, Volatility, Price Momentum, Volume Proxy, Trend Strength
- Early vs Late lifecycle comparison
- Professional visualizations with aggregated statistics and token rankings

**3. Multi-Token Correlation**
- Dynamic correlation analysis between selected tokens
- Rolling correlation windows with statistical significance testing
- Correlation heatmaps and time series plots

### 📈 Entry/Exit Optimization
**4. Entry/Exit Matrix Analysis**
- Optimal trading window identification using statistical analysis
- Support for both single and multi-token analysis
- Confidence intervals and statistical significance testing
- Heatmap visualization with performance metrics

**5. Entry/Exit Moment Matrix**
- Minute-by-minute entry and exit timing optimization
- Two implementations: standard and Polars-optimized for large datasets
- Interactive heatmaps showing average returns for each entry/exit combination

### 📊 Market Structure Analysis
**6. Volatility Surface**
- 3D volatility analysis across time windows and percentile levels
- Interactive surface plots with customizable parameters
- Volatility clustering and regime identification

**7. Microstructure Analysis**
- **Professional market microstructure analysis** using price-only data
- **Roll's bid-ask spread estimator** from return serial covariance
- **Kyle's lambda (price impact coefficient)** with statistical significance
- **Amihud illiquidity measure** adapted for crypto markets without volume
- **Realized volatility analysis** across multiple time windows (1h, 4h)
- **Market quality indicators**: price efficiency, return autocorrelation, volatility clustering
- **Price dynamics**: velocity analysis and market impact estimation
- **6-panel professional dashboard** with volatility regime coloring
- **Summary gauge dashboard** for key metrics visualization
- **Comprehensive interpretation guide** with trading implications

**8. Price Distribution Evolution** ⭐ **ENHANCED**
- **Comprehensive statistical analysis** of return distribution evolution over time
- **Pure Polars implementation** for maximum performance on large datasets
- **Advanced distribution testing**: Shapiro-Wilk normality tests, Jarque-Bera statistics
- **Distribution shape analysis**: Skewness and kurtosis evolution tracking
- **Multiple distribution fitting**: Normal, t-distribution for heavy-tail detection
- **Q-Q plots** for visual normality assessment and distribution comparison
- **Outlier detection** with box plots and statistical significance testing
- **Evolution summary dashboard**: 6-panel analysis showing distribution metrics over time
- **Trading implications**: Risk management guidance based on distribution characteristics
- **Statistical significance testing** with confidence intervals and critical values
- **Distribution quality scoring** system for risk assessment

**9. Optimal Holding Period**
- Statistical analysis to determine best holding durations
- Risk-adjusted return optimization across time horizons
- Sharpe ratio maximization analysis

**10. Market Regime Analysis**
- Algorithmic detection of market regimes (trending, ranging, volatile)
- Regime transition analysis and persistence measurement
- Color-coded price charts showing regime classifications

### 🔗 Advanced Analytics
**11. Trade Timing Heatmap**
- Entry minute vs exit lag optimization analysis
- Statistical significance testing for timing strategies
- Interactive heatmaps with performance metrics

**12. Comprehensive Report**
- Complete analytical report generation combining multiple analysis types
- Professional formatting with charts, tables, and statistical summaries
- Export-ready documentation

## 🎯 Key Technical Features

### 1. **Polars-First Architecture**
- All core computations use Polars for maximum performance
- Efficient handling of large datasets (1000+ tokens)
- Memory-optimized operations for high-frequency data

### 2. **Memecoin-Specific Design**
- Optimized for 24-hour token lifecycles
- Price-only analysis (volume data often unreliable for memecoins)
- Volatility-based volume proxies and market structure analysis

### 3. **Professional Risk Management**
- Multiple risk-adjusted return metrics
- Statistical significance testing for all strategies
- Confidence intervals and uncertainty quantification

### 4. **Interactive Visualizations**
- Plotly-based professional charts
- Hover details, zoom, and pan capabilities
- Export-ready publication quality graphics

### 5. **Scalable Performance**
- Optimized algorithms for large multi-token analysis
- Parallel processing capabilities where applicable
- Memory-efficient data structures

## 🔧 Technical Specifications

### Dependencies
```python
# Core data processing
import polars as pl
import numpy as np
from scipy import stats, signal
from scipy.stats import entropy

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Utilities
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
```

### Data Requirements
- **Input Format**: Parquet files with `datetime` and `price` columns
- **Frequency**: Minute-level data (1440 minutes per token)
- **Structure**: Compatible with data_analysis module DataLoader format
- **Quality**: Cleaned data from data_cleaning pipeline recommended

### Performance Characteristics
- **Single Token Analysis**: 1-3 seconds for complete analysis
- **Multi-Token (100 tokens)**: 15-45 seconds depending on analysis type
- **Multi-Token (1000+ tokens)**: 2-10 minutes with optimized algorithms
- **Memory Usage**: 100-500MB for typical datasets

## ⚙️ Configuration Parameters

### Time Windows
```python
# Entry/Exit analysis windows (minutes)
DEFAULT_WINDOWS = [5, 10, 15, 30, 60, 120, 240]
COMMON_WINDOWS = [5, 15, 30, 60, 240]
EXTENDED_WINDOWS = [5, 10, 15, 30, 60, 120, 240, 480, 720, 1430]

# Time horizons for temporal analysis (minutes)
DEFAULT_TIME_HORIZONS = [5, 15, 30, 60, 120, 240, 480, 720, 1430]
```

### Analysis Parameters
```python
# Risk calculations
RISK_FREE_RATE = 0  # Crypto markets assumption
PERIODS_PER_YEAR = 525600  # Minutes per year for annualization
MOMENTUM_THRESHOLD = 0.01  # 1% minimum momentum for signals

# Statistical parameters
CONFIDENCE_LEVELS = [0.95, 0.99]  # For confidence intervals
MIN_OBSERVATIONS = 30  # Minimum data points for statistical validity
```

### Visualization Settings
```python
# Color schemes
COLORSCALE_RETURNS = 'RdBu'  # Red-Blue for returns (red=negative, blue=positive)
COLORSCALE_CORRELATION = 'RdYlBu'  # Red-Yellow-Blue for correlations
COLORSCALE_VOLATILITY = 'Viridis'  # Viridis for volatility surfaces

# Chart dimensions
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
HEATMAP_WIDTH = 1000
HEATMAP_HEIGHT = 700
```

## 📝 Usage Examples

### Basic Single Token Analysis
```python
from quant_analysis import QuantAnalysis
from quant_viz import QuantVisualizations

# Initialize analysis engines
qa = QuantAnalysis()
qv = QuantVisualizations()

# Load token data (Polars DataFrame)
df = pl.read_parquet("token_data.parquet")

# Calculate risk metrics
returns = df.select(pl.col('price').pct_change().alias('returns'))['returns']
sharpe = qa.calculate_sharpe_ratio(returns.drop_nulls())

# Generate entry/exit matrix
matrix_result = qa.optimal_entry_exit_matrix(
    df, 
    entry_windows=[5, 15, 30, 60], 
    exit_windows=[5, 15, 30, 60]
)

# Create visualization
fig = qv.plot_entry_exit_matrix(
    matrix_result['mean_returns'],
    matrix_result['entry_windows'],
    matrix_result['exit_windows'],
    title="Optimal Entry/Exit Analysis"
)
```

### Multi-Token Lifecycle Analysis
```python
# Load multiple tokens
token_dfs = [(name, df) for name, df in load_multiple_tokens()]

# Perform 24-hour lifecycle analysis
lifecycle_results = qa.lifecycle_analysis(
    token_dfs, 
    num_segments=6,  # 4-hour segments
    metrics=['mean_return_pct', 'volatility', 'momentum']
)

# Create comprehensive visualizations
summary_fig = qv.plot_lifecycle_summary_charts(
    lifecycle_results, 
    selected_metrics=['mean_return_pct', 'volatility']
)

ranking_fig = qv.plot_lifecycle_token_ranking(
    lifecycle_results,
    top_n=20,
    metric='mean_return_pct'
)
```

### Market Microstructure Analysis
```python
# Perform comprehensive microstructure analysis
microstructure_results = qa.microstructure_analysis(df)

# Key metrics extracted
bid_ask_spread = microstructure_results['bid_ask_spread_estimate'] * 10000  # in bps
kyle_lambda = microstructure_results['kyle_lambda']
amihud_illiquidity = microstructure_results['avg_amihud_illiquidity']
price_efficiency = microstructure_results['avg_price_efficiency']
volatility_clustering = microstructure_results['volatility_clustering']

# Create professional visualizations
microstructure_dashboard = qv.plot_microstructure_analysis(df, microstructure_results)
summary_gauges = qv.plot_microstructure_summary_dashboard(microstructure_results)

# Interpretation for trading
if bid_ask_spread < 50:  # < 50 bps
    print("Low transaction costs - suitable for high-frequency trading")
if abs(kyle_lambda) < 1e-6:
    print("Low market impact - can trade larger sizes")
if price_efficiency > 0.5:
    print("Efficient price discovery - trends likely to persist")
```

### Risk Metrics Dashboard
```python
# Multi-token risk analysis
risk_results = qa.temporal_risk_reward_analysis(
    token_dfs,
    time_horizons=[60, 240, 720, 1430]
)

# Generate 4-panel dashboard
dashboard_fig = qv.plot_multi_token_risk_metrics(
    risk_results,
    title="Multi-Token Risk Analysis Dashboard"
)
```

## 🚀 Running the Application

### Command Line
```bash
# Navigate to project directory
cd /path/to/memecoin2

# Launch Streamlit application
streamlit run quant_analysis/quant_app.py --server.port 8503
```

### Application URL
- **Local Access**: http://localhost:8503
- **Network Access**: http://[your-ip]:8503 (if configured)

### Usage Workflow
1. **Select Data Source**: Choose from available data directories
2. **Configure Token Selection**: Single, multiple, random, or all tokens
3. **Choose Analysis Type**: Select from 12 available analysis modes
4. **Set Parameters**: Configure time windows, segments, or other analysis-specific settings
5. **Run Analysis**: Execute with real-time progress tracking
6. **Explore Results**: Interactive visualizations with hover details and zoom
7. **Export Results**: Download charts and data as needed

## 🎯 Professional Applications

### Trading Strategy Development
- **Entry/Exit Optimization**: Statistical identification of optimal trading windows
- **Risk Management**: Comprehensive risk-adjusted performance metrics
- **Market Timing**: Minute-level timing analysis for high-frequency strategies

### Research & Analysis
- **Memecoin Behavior**: Understanding 24-hour lifecycle patterns
- **Market Structure**: High-frequency market behavior without traditional volume data
- **Comparative Analysis**: Multi-token performance and correlation studies

### Portfolio Management
- **Token Selection**: Data-driven token selection based on risk-adjusted metrics
- **Diversification**: Correlation analysis for portfolio construction
- **Performance Attribution**: Understanding sources of returns and risks

This quantitative analysis module provides institutional-grade analytical capabilities specifically designed for the unique characteristics of Solana memecoin markets, combining advanced statistical methods with professional visualization tools in an intuitive web interface. 