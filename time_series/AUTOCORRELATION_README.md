# 🔄 Time Series Autocorrelation & Clustering Analysis

A fresh approach to analyzing memecoin time series data using autocorrelation, clustering, and dimensionality reduction techniques.

## 🎯 Overview

This module provides a comprehensive analysis framework that works directly with raw price data (or log prices) without requiring data cleaning or feature engineering. The analysis focuses on three main components:

1. **Autocorrelation Analysis (ACF/PACF)**: Understand temporal dependencies within each token
2. **Time Series Clustering**: Group tokens with similar temporal patterns
3. **t-SNE Visualization**: Visualize high-dimensional relationships in 2D/3D space

## 🚀 Quick Start

### 1. Run Analysis Script

```bash
# Basic analysis with default parameters
python time_series/run_autocorrelation_analysis.py

# Custom analysis
python time_series/run_autocorrelation_analysis.py \
    --data_dir data/raw/dataset_2024-11-27_04-07-35 \
    --max_tokens 200 \
    --n_clusters 7 \
    --use_log_price \
    --save_plots
```

### 2. Launch Interactive App

```bash
streamlit run time_series/autocorrelation_app.py
```

## 📊 Analysis Components

### 1. Autocorrelation Analysis

For each token, we compute:
- **ACF (Autocorrelation Function)**: Measures correlation between observations at different lags
- **PACF (Partial Autocorrelation Function)**: Measures direct correlation after removing intermediate effects
- **Key Metrics**:
  - Significant lags (outside 95% confidence interval)
  - Decay rate (how quickly autocorrelation diminishes)
  - First zero crossing (where ACF first crosses zero)

### 2. Time Series Clustering

We extract features from each time series and cluster tokens based on:
- **Statistical Features**: Mean, std, percentiles of prices and returns
- **ACF Features**: ACF values at specific lags, number of significant lags
- **Trend Features**: Linear trend slope
- **Volatility Features**: Return volatility measures

Clustering methods available:
- **K-means**: Fast, requires specifying number of clusters
- **Hierarchical**: Creates a dendrogram of relationships
- **DBSCAN**: Density-based, can find outliers

### 3. t-SNE Visualization

Reduces high-dimensional feature space to 2D/3D for visualization:
- Shows natural groupings of tokens
- Reveals outliers and anomalies
- Interactive plots with token names on hover

## 📁 Output Files

After running the analysis, you'll find:

```
time_series/results/
├── cluster_assignments.json      # Token → Cluster mapping
├── cluster_statistics.json       # Detailed stats for each cluster
├── acf_summary.json             # ACF metrics for each token
├── tsne_2d.html                # Interactive 2D visualization
├── tsne_3d.html                # Interactive 3D visualization
├── acf_clusters.html            # ACF patterns by cluster
├── tsne_2d.png                 # Static 2D plot
└── acf_by_cluster.png          # Static ACF comparison
```

## 🔍 Key Features

### Features Used for Clustering

1. **Price Statistics**
   - Mean, standard deviation, median
   - 25th and 75th percentiles

2. **Return Statistics**
   - Mean return, return volatility
   - Min/max returns

3. **Autocorrelation Features**
   - ACF at lags 1, 5, and 10
   - Number of significant lags
   - Decay rate and first zero crossing

4. **Trend Features**
   - Linear regression slope

### Cluster Interpretation

Each cluster represents tokens with similar temporal patterns:
- **High autocorrelation clusters**: Tokens with strong momentum/trending behavior
- **Low autocorrelation clusters**: More random/efficient price movements
- **High volatility clusters**: Tokens with large price swings
- **Stable clusters**: Tokens with consistent, predictable patterns

## 🎨 Streamlit App Features

The interactive app provides:

1. **Overview Tab**
   - Summary statistics
   - Cluster distribution
   - Key characteristics of each cluster

2. **Autocorrelation Analysis Tab**
   - Average ACF by cluster
   - Individual token ACF/PACF plots
   - Distribution of ACF characteristics

3. **Clustering Results Tab**
   - Feature importance for clustering
   - Cluster comparison heatmap
   - Detailed cluster statistics

4. **t-SNE Visualization Tab**
   - 2D and 3D interactive plots
   - Token similarity matrix
   - Cluster boundaries

5. **Token Explorer Tab**
   - Individual token analysis
   - Price charts (raw or log)
   - Similar tokens in same cluster

## 🛠️ Advanced Usage

### Custom Feature Extraction

```python
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer

analyzer = AutocorrelationClusteringAnalyzer()

# Load your data
token_data = analyzer.load_raw_prices(Path("your_data_dir"))

# Compute autocorrelations
acf_results = analyzer.compute_all_autocorrelations(token_data, use_log_price=True)

# Extract custom features
for token, df in token_data.items():
    prices = df['log_price'].to_numpy()
    acf = acf_results[token]
    features = analyzer.extract_time_series_features(prices, acf)
```

### Custom Clustering

```python
# Prepare feature matrix
feature_matrix, token_names = analyzer.prepare_clustering_data(token_data, acf_results)

# Try different clustering methods
kmeans_results = analyzer.perform_clustering(feature_matrix, method='kmeans', n_clusters=5)
dbscan_results = analyzer.perform_clustering(feature_matrix, method='dbscan')
hier_results = analyzer.perform_clustering(feature_matrix, method='hierarchical', n_clusters=7)
```

### Custom Visualization

```python
# Generate t-SNE with different perplexity
tsne_embedding = analyzer.compute_tsne(feature_matrix, n_components=2, perplexity=50)

# Create custom plots
figures = analyzer.create_visualization_plots(results)
```

## 📈 Interpreting Results

### ACF Patterns

1. **Slow Decay**: Strong trending behavior, momentum trading
2. **Fast Decay**: More efficient pricing, less predictable
3. **Oscillating**: Cyclical patterns, mean reversion
4. **Near Zero**: Random walk, unpredictable

### Cluster Characteristics

Look for clusters with:
- **Similar volatility levels**: Risk-based grouping
- **Similar ACF patterns**: Temporal behavior grouping
- **Similar trend slopes**: Growth/decline patterns

### t-SNE Interpretation

- **Tight clusters**: Very similar tokens
- **Scattered points**: Unique/outlier tokens
- **Cluster bridges**: Tokens with mixed characteristics

## 🔧 Configuration Options

### Analysis Parameters

- `max_lag`: Maximum lag for ACF computation (default: 100)
- `n_clusters`: Number of clusters for k-means (default: 5)
- `use_log_price`: Whether to use log-transformed prices (recommended)
- `perplexity`: t-SNE perplexity parameter (default: 30)

### Performance Tips

- Start with 100-200 tokens for initial exploration
- Use log prices for better statistical properties
- Try different numbers of clusters (5-10 typically works well)
- Save plots for offline analysis

## 📚 References

- **Autocorrelation**: Box, Jenkins & Reinsel (2015) - Time Series Analysis
- **t-SNE**: van der Maaten & Hinton (2008) - Visualizing Data using t-SNE
- **Time Series Clustering**: Aghabozorgi et al. (2015) - Time-series clustering survey

## 🤝 Contributing

To add new features:
1. Extend `extract_time_series_features()` for new feature types
2. Add new clustering methods in `perform_clustering()`
3. Create new visualization functions in `create_visualization_plots()`

## 📝 Next Steps

After running this analysis:
1. Identify clusters of interest based on your trading strategy
2. Analyze tokens within promising clusters more deeply
3. Use cluster membership as a feature for ML models
4. Monitor how tokens move between clusters over time
5. Combine with other analyses (volume, social sentiment, etc.) 