# 🧮 Feature Engineering Module

**Enhanced roadmap implementation with CLEAN ARCHITECTURE**

## 🚀 Quick Start

### **Unified Streamlit Dashboard**
```bash
# Run the comprehensive feature engineering dashboard
cd feature_engineering/
streamlit run app.py
```

**Single app handles everything:**
- 📁 **Flexible folder browsing** (any data/ subfolder)
- 🧮 **Rolling feature engineering** (ML-safe only)
- 🔗 **Multi-token correlation analysis** with PCA
- 📊 **FFT cyclical pattern detection** 
- ⚙️ **Batch processing** for large-scale feature engineering
- 📋 **Implementation report** with usage guide
- 🧠 **On-demand global features** (no redundancy)

## 🧠 **CLEAN ARCHITECTURE**

### **Core Principle: Separation of Concerns**
```
Rolling Features (ML-Safe)          Global Features (Analysis)
data/features/                  ←→  Computed on-demand in Streamlit
├── normal_behavior/                 ├── Uses price_analysis.py
├── tokens_with_extremes/            ├── No storage redundancy  
└── dead_tokens/                     └── Analysis-only features
```

### **Benefits:**
- ✅ **No redundancy** with data_analysis modules
- ✅ **Impossible to accidentally use global features in ML**
- ✅ **Cleaner separation** of rolling vs global features
- ✅ **Reduced storage** requirements
- ✅ **Leverages existing** price_analysis functionality

## 📊 **Analysis Types Available**

### 1. **🧮 Feature Engineering**
- **Rolling Features**: MACD, Bollinger position, RSI, log-returns
- **Expanding Windows**: Volatility, Sharpe ratio, statistical moments
- **✅ ML-Safe**: Only historical data, no future peeking
- **❌ Global Features**: Computed on-demand, not stored

### 2. **🔗 Correlation Analysis** 
- **Multi-method**: Pearson, Spearman, Kendall correlations
- **PCA Analysis**: Explained variance ratios for redundancy detection
- **Log-returns**: Option to use log-returns vs prices
- **Robust Scaling**: Per-token normalization option
- **Rolling Correlations**: Time-varying correlation analysis

### 3. **📊 FFT Analysis**
- **Multi-token**: Sequential analysis of multiple tokens
- **Window Functions**: Hamming, Hann, Blackman for spectral analysis
- **Pattern Detection**: Short-term, medium-term, long-term cycles
- **Advanced Options**: Detrending, phase analysis, normalization
- **Comparison Mode**: Compare FFT patterns between token groups

### 4. **⚙️ Batch Processing**
- **Rolling Features Only**: No global features computed
- **Configurable**: Select technical indicators and window sizes
- **Progress Tracking**: Real-time progress bars
- **Category-aware**: Maintains folder structure

### 5. **🧠 On-Demand Global Features**
- **Uses price_analysis.py**: Leverages existing analysis functionality
- **No Storage**: Computed when needed, not pre-stored
- **Complete Analysis**: Total returns, drawdowns, pattern classification
- **No Redundancy**: Eliminates duplicate computation

## 💾 **Data Architecture**

### **Clean Separation:**
```python
# ✅ ROLLING FEATURES (data/features/)
- log_returns, rolling_volatility, rolling_sharpe
- macd_line, macd_signal, bb_position
- rsi_values, atr_values
- rolling_skewness, rolling_kurtosis

# 🧠 GLOBAL FEATURES (computed on-demand)
- total_return_pct, max_drawdown_pct  
- global_volatility, pattern_classification
- FFT spectral_entropy, dominant_periods
- Multi-granularity candlestick patterns
```

### **ML Training Pipeline:**
```bash
# 1. Feature Engineering (rolling only)
python feature_engineering/advanced_feature_engineering.py

# 2. ML Training (uses only rolling features)
python ML/directional_models/train_lightgbm_model.py
```

### **Analysis Pipeline:**
```bash
# 1. Streamlit Analysis (computes global on-demand)
streamlit run feature_engineering/app.py

# 2. Select token → global features computed automatically
# 3. No storage waste, no redundancy
```

## 🔬 **Technical Implementation**

### **Rolling Feature Engineering**
```python
from feature_engineering import AdvancedFeatureEngineer, create_rolling_features_safe

# Only rolling features saved
engineer = AdvancedFeatureEngineer()
features = engineer.create_comprehensive_features(df, token_name)

# Clean architecture: saves only ML-safe features
save_features_to_files(features_dict, token_paths)  # Rolling only
```

### **On-Demand Global Features**
```python
from feature_engineering.app import compute_global_features_on_demand

# Computed when needed using existing price_analysis
global_features = compute_global_features_on_demand(df, token_name)
# Uses: data_analysis.price_analysis.PriceAnalyzer
```

### **Multi-Token Correlation**
```python
from feature_engineering import TokenCorrelationAnalyzer

analyzer = TokenCorrelationAnalyzer()
results = analyzer.analyze_token_correlations(
    token_data, 
    use_log_returns=True,
    use_robust_scaling=True,
    min_overlap=100
)
```

## 🎯 **Usage Examples**

### **1. Single Token Analysis**
```bash
streamlit run app.py
# → Select token
# → Check "Show Global Features" for on-demand computation
# → Rolling features always available
```

### **2. Batch Feature Engineering**
```python
python advanced_feature_engineering.py
# → Processes all cleaned tokens
# → Saves ONLY rolling features
# → No global feature storage
```

### **3. Multi-Token Correlation** 
```bash
streamlit run app.py
# → Select "Correlation Analysis"
# → Choose multiple tokens
# → PCA redundancy detection included
```

## 🛡️ **Data Leakage Prevention**

### **Safe Features (Stored):**
- ✅ Rolling windows (expanding/fixed)
- ✅ Technical indicators (MACD, RSI, BB)
- ✅ Lag features and momentum
- ✅ Expanding statistical moments

### **Unsafe Features (On-Demand Only):**
- 🧠 Total returns (uses final price)
- 🧠 Max drawdowns (uses min/max across series)  
- 🧠 FFT spectral entropy (uses entire series)
- 🧠 Pattern classification (uses full dataset)

## 📁 **File Structure**

```
feature_engineering/
├── app.py                           # 🎯 Main Streamlit dashboard
├── advanced_feature_engineering.py  # 🔄 Rolling feature extraction
├── correlation_analysis.py          # 🔗 Multi-token relationships
├── __init__.py                      # 📦 Package initialization
└── README.md                        # 📚 This documentation

data/
├── features/                        # 🔄 Rolling features (ML-safe)
│   ├── normal_behavior_tokens/
│   ├── tokens_with_extremes/
│   └── dead_tokens/
└── [no global_features directory]   # 🧠 Computed on-demand
```

## 🏆 **Architecture Benefits**

### **Before (Redundant):**
- ❌ Global features stored in feature_engineering 
- ❌ Same features computed in data_analysis
- ❌ Storage waste and confusion
- ❌ Risk of using global features in ML

### **After (Clean):**
- ✅ Rolling features only in feature_engineering
- ✅ Global features on-demand from price_analysis
- ✅ No redundancy or storage waste
- ✅ Impossible to use global features in ML by accident

## 🚀 **Migration Guide**

If you have existing global feature files, they can be safely deleted:
```bash
# Remove old global features (now computed on-demand)
rm -rf data/global_features_analysis_only/

# Keep rolling features (ML-safe)
# data/features/ directory remains unchanged
```

## 🧪 **Testing the Clean Architecture**

```python
# Test rolling features
from feature_engineering import create_rolling_features_safe
rolling_df = create_rolling_features_safe(df, token_name)
print(f"Rolling features: {rolling_df.columns}")

# Test on-demand global features  
from feature_engineering.app import compute_global_features_on_demand
global_features = compute_global_features_on_demand(df, token_name)
print(f"Global computed: {global_features['computed_on_demand']}")
```

---

## 📞 **Support**

The clean architecture ensures:
- 🎯 **Single responsibility**: Each module has a clear purpose
- 🔒 **Data safety**: Impossible to leak global features into ML
- 📊 **No redundancy**: Global features computed once when needed
- 🧠 **Leverages existing code**: Uses proven price_analysis functionality 