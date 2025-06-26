# Feature Engineering for Memecoin Analysis

This directory contains advanced feature engineering implementations for the Solana memecoin prediction system. These modules should be used **after** data cleaning to extract meaningful features from cleaned price data.

## 🔄 **Workflow Position**

```
Raw Data → Data Cleaning → **Feature Engineering** → Model Training → Prediction
                ↗️              ↗️ YOU ARE HERE
```

## 📁 **Module Overview**

### **🧬 `advanced_feature_engineering.py`**
**Purpose**: Comprehensive feature extraction from cleaned price data

**Key Features**:
- ✅ **Log-returns calculation** (Critical roadmap requirement)
- ✅ **FFT analysis** for cyclical pattern detection
- ✅ **Advanced technical indicators**: MACD, Bollinger Bands, ATR, Enhanced RSI
- ✅ **Statistical moments**: Skewness, kurtosis, VaR, expected shortfall
- ✅ **Multi-granularity analysis**: 2-min, 5-min candlestick patterns
- ✅ **Formal outlier detection**: Winsorization, Z-score, IQR methods

**Usage**:
```python
from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineer

engineer = AdvancedFeatureEngineer()
features = engineer.create_comprehensive_features(cleaned_df, 'token_name')
```

### **📊 `correlation_analysis.py`**
**Purpose**: Analyze relationships and correlations between multiple tokens

**Key Features**:
- ✅ **Multi-method correlation**: Pearson, Spearman, Kendall
- ✅ **PCA redundancy analysis** with explained variance ratios
- ✅ **Interactive heatmaps** for correlation visualization
- ✅ **Token clustering** based on price movement patterns
- ✅ **Rolling correlations** for time-varying relationships

**Usage**:
```python
from feature_engineering.correlation_analysis import TokenCorrelationAnalyzer

analyzer = TokenCorrelationAnalyzer()
results = analyzer.analyze_token_correlations(token_data_dict)
heatmap = analyzer.create_correlation_heatmap(results['correlation_matrices']['main'])
```

### **🚀 `roadmap_dashboard.py`**
**Purpose**: Interactive Streamlit dashboard showcasing all feature engineering capabilities

**Features**:
- 📈 **Live feature engineering demos**
- 🔍 **FFT pattern visualization**
- 📊 **Correlation matrix exploration**
- 📋 **Implementation status reports**

**Usage**:
```bash
streamlit run feature_engineering/roadmap_dashboard.py
```

## 🛠️ **Prerequisites**

Before using these feature engineering modules:

1. **Data must be cleaned** using `data_cleaning/` modules
2. **Required data format**: Polars DataFrame with columns:
   - `datetime`: Timestamp column
   - `price` or `close`: Price data column

## 🎯 **Roadmap Implementation Status**

### **✅ Section 1: Data Quality & Cleaning** (Completed in `data_cleaning/`)
- [x] Outlier detection methods implemented
- [x] Incomplete token handling strategies

### **✅ Section 2: Preprocessing** (Implemented here)
- [x] **Log-returns calculation** (Critical requirement)
- [x] Rolling window normalization
- [x] Robust scaling per token
- [x] Temporal data splitting

### **✅ Section 3: Exploration & Feature Engineering** (Implemented here)
- [x] **FFT analysis** for cyclical patterns
- [x] **Correlation matrix & heatmap** analysis
- [x] **PCA explained variance ratio** calculation
- [x] **Advanced technical indicators** (MACD, Bollinger Bands, ATR)
- [x] **Multi-granularity downsampling** (2-min, 5-min)
- [x] **Statistical moments** (skewness, kurtosis)

## 🚀 **Quick Start Examples**

### **1. Single Token Feature Engineering**
```python
import polars as pl
from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineer

# Load cleaned data
df = pl.read_parquet('data/cleaned/normal_behavior_tokens/token123.parquet')

# Engineer features
engineer = AdvancedFeatureEngineer()
features = engineer.create_comprehensive_features(df, 'token123')

# Access different feature sets
log_returns = features['price_features']['log_returns']
technical_indicators = features['technical_features']
statistical_moments = features['moment_features']
```

### **2. Multi-Token Correlation Analysis**
```python
from pathlib import Path
from feature_engineering.correlation_analysis import load_tokens_for_correlation, TokenCorrelationAnalyzer

# Load multiple tokens
token_paths = list(Path('data/cleaned/normal_behavior_tokens').glob('*.parquet'))
token_data = load_tokens_for_correlation(token_paths[:50])  # Analyze 50 tokens

# Analyze correlations
analyzer = TokenCorrelationAnalyzer()
results = analyzer.analyze_token_correlations(token_data, use_log_returns=True)

# Create visualization
heatmap = analyzer.create_correlation_heatmap(results['correlation_matrices']['main'])
heatmap.show()
```

### **3. Batch Feature Engineering**
```python
from feature_engineering.advanced_feature_engineering import batch_feature_engineering
from pathlib import Path

# Process multiple tokens
token_paths = list(Path('data/cleaned/normal_behavior_tokens').glob('*.parquet'))
features_dict = batch_feature_engineering(token_paths, limit=100)

# Access features for each token
for token_name, features in features_dict.items():
    if features['status'] == 'success':
        print(f"{token_name}: {len(features['price_features']['log_returns'])} features")
```

## 📈 **Integration with ML Pipeline**

These feature engineering modules integrate seamlessly with the ML training pipeline:

```python
# 1. Feature Engineering Phase
features = engineer.create_comprehensive_features(cleaned_df, token_name)

# 2. ML Training Integration
from ML.directional_models.train_lightgbm_model import CONFIG

# Enhanced features can be used directly in ML models
enhanced_df = cleaned_df.with_columns([
    pl.Series('log_returns', features['price_features']['log_returns']),
    pl.Series('macd', features['technical_features']['macd']['macd_line']),
    pl.Series('rsi', features['technical_features']['enhanced_rsi']['values'])
])
```

## 🔧 **Advanced Configuration**

### **Outlier Detection Methods**
```python
# Choose outlier detection method
features = engineer.create_comprehensive_features(
    df, 
    token_name='token123',
    outlier_method='winsorization',  # 'z_score', 'iqr'
    winsor_limits=(0.01, 0.01)      # Lower/upper percentile limits
)
```

### **Correlation Analysis Options**
```python
# Advanced correlation analysis
results = analyzer.analyze_token_correlations(
    token_data,
    method='spearman',        # 'pearson', 'kendall'
    use_log_returns=True,     # Use log-returns vs normalized prices
    min_overlap=100           # Minimum data overlap required
)
```

## 📊 **Output Formats**

### **Feature Engineering Output**
```python
{
    'token': 'token_name',
    'status': 'success',
    'price_features': {
        'log_returns': np.array([...]),
        'cumulative_log_returns': np.array([...]),
        'price_stats': {...}
    },
    'technical_features': {
        'macd': {...},
        'bollinger_bands': {...},
        'enhanced_rsi': {...}
    },
    'moment_features': {
        'skewness': float,
        'kurtosis': float,
        'value_at_risk_95': float
    },
    'fft_features': {
        'dominant_periods_minutes': [...],
        'spectral_entropy': float
    }
}
```

### **Correlation Analysis Output**
```python
{
    'status': 'success',
    'correlation_matrices': {
        'main': pd.DataFrame,
        'rolling_240': pd.DataFrame
    },
    'significant_pairs': [...],
    'pca_analysis': {
        'explained_variance_ratio': [...],
        'n_components_95_variance': int
    }
}
```

## 🧪 **Testing & Validation**

Test the feature engineering implementations:

```bash
# Test advanced feature engineering
python -c "
from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineer
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Create test data
dates = [datetime(2024,1,1) + timedelta(minutes=i) for i in range(100)]
prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
df = pl.DataFrame({'datetime': dates, 'price': prices})

engineer = AdvancedFeatureEngineer()
features = engineer.create_comprehensive_features(df, 'test_token')
print('✅ Advanced Feature Engineering:', features['status'])
"

# Test correlation analysis
python -c "
from feature_engineering.correlation_analysis import TokenCorrelationAnalyzer
import polars as pl
import numpy as np
from datetime import datetime, timedelta

dates = [datetime(2024,1,1) + timedelta(minutes=i) for i in range(100)]
token_data = {
    'token1': pl.DataFrame({'datetime': dates, 'price': 100 + np.cumsum(np.random.normal(0, 1, 100))}),
    'token2': pl.DataFrame({'datetime': dates, 'price': 100 + np.cumsum(np.random.normal(0, 1, 100))})
}

analyzer = TokenCorrelationAnalyzer()
results = analyzer.analyze_token_correlations(token_data)
print('✅ Correlation Analysis:', results['status'])
"
```

## 🔄 **Next Steps**

After feature engineering:

1. **🤖 Model Training**: Use features in `ML/directional_models/` or `ML/forecasting_models/`
2. **📊 Quantitative Analysis**: Apply features in `quant_analysis/`
3. **⏱️ Time Series Analysis**: Use in `time_series/` modules

## 🎛️ **Dashboard Access**

Launch the interactive dashboard to explore all features:

```bash
cd feature_engineering/
streamlit run roadmap_dashboard.py
```

---

**Note**: Always ensure data is properly cleaned using `data_cleaning/` modules before applying feature engineering techniques. 