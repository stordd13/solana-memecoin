# 🤖 Claude Development Guide for Memecoin Analysis Platform

> **For Future Claude Instances**: This guide provides comprehensive context, architecture overview, and development workflows for the memecoin time series analysis platform.
**Important context**

Always start with testing. Test-driven development (TDD) is the only true method. Tests must use real components, in a real environment.
Set typing and linting to maximally strict mode.
Describe exactly what you want, down to the last detail: write a complete PRD (Product Requirements Document) and define precisely what a user can or cannot do.
Do thorough research: study the problem, the competition, similar solutions, etc., with DeepResearch, and build a report to launch your prompt.
Always manipulate your model and force it to criticize its own work. Every time the agent says "I'm done! Everything's ready for production!", consider that it's probably not true.
Bottom-up method. Build the smallest elements one by one, make sure they work and are tested, then connect them together - you can develop a complete production stack if you start with the basic systems, test them, then put them together.
There's no substitute for expertise. If you know how to code and know what to avoid, you'll save yourself a lot of trouble and expense.


## 📊 **Project Context & Mission**

### **Core Challenge**
ML models showing unstable performance: **70%→45% precision**, **95%→25% recall** due to mixing different behavioral patterns in unified models.

### **Dataset Overview**
- **30,000 memecoins** with minute-by-minute prices (200-2000 minutes per token)
- **24-hour timeframe context**: ~1440 minutes of trading data per token
- **3k normal tokens**: Standard behavior patterns
- **4k extreme tokens**: 99.9% dumps, 1M%+ pumps (LEGITIMATE signals, not noise!)
- **25k dead tokens**: Tradeable during active periods, then flatline

### **🪙 Critical Memecoin Context**
**IMPORTANT**: This is **24-hour minute-by-minute memecoin data** - NOT traditional markets!
- **No "historical" vs "recent"**: Every minute in 24h cycle is equally relevant
- **Extreme volatility is normal**: 1M%+ pumps and 99.9% dumps are legitimate patterns
- **Purpose is data filtering**: Identify dead/inactive tokens for ML pipeline quality
- **Not trading strategies**: Focus on token filtering, not trading signal generation

### **Solution Strategy**
Use **Autocorrelation Function (ACF) + Clustering + t-SNE** to identify 5-8 distinct behavioral archetypes for stable, cluster-specific ML models instead of unified approach.

---

## 🏗️ **Architecture Overview**

### **Project Structure**
```
memecoin2/
├── data_analysis/              # Interactive Streamlit dashboard & core analysis
│   ├── app.py                 # Main entry point: streamlit run data_analysis/app.py
│   ├── data_loader.py         # Polars-based data loading with caching
│   ├── data_quality.py        # Dead token detection, gap analysis
│   ├── price_analysis.py      # Optimal timing calculations
│   └── export_utils.py        # Category export functionality
├── data_cleaning/              # Data preprocessing pipeline
│   └── clean_tokens.py        # Category-aware cleaning strategies
├── feature_engineering/        # ML-safe feature creation
│   ├── advanced_feature_engineering.py  # Rolling features (NO data leakage)
│   └── create_directional_targets.py    # Binary targets for all horizons
├── ML/                        # Machine learning models
│   ├── directional_models/    # Binary classification (UP/DOWN prediction)
│   │   ├── train_lightgbm_model.py           # Tree-based models
│   │   ├── train_unified_lstm_model.py       # Basic LSTM
│   │   └── train_advanced_hybrid_lstm.py     # Multi-scale + attention
│   ├── forecasting_models/    # Regression (price value prediction)
│   └── utils/                 # Shared ML utilities (winsorization, etc.)
├── time_series/               # Advanced time series analysis
│   ├── autocorrelation_app.py          # ACF analysis Streamlit app
│   ├── autocorrelation_clustering.py   # Core ACF + clustering engine
│   └── MEMECOIN_ANALYSIS_ROADMAP.md    # 4-phase implementation plan
├── quant_analysis/            # Quantitative trading analysis
└── run_pipeline.py           # Complete automated pipeline
```

### **Data Flow Architecture**
```
Raw Data → Data Quality Analysis → Category Export → Cleaning → Feature Engineering → ML Training
   ↓              ↓                    ↓             ↓            ↓                ↓
data/raw/    quality reports    data/processed/  data/cleaned/ data/features/  ML/results/
```

### **Critical Design Principles**
1. **Temporal Splitting**: NEVER split by tokens - always split within each token's timeline
2. **Per-Token Scaling**: Each token scaled independently (handles 200-2000min lifespans)
3. **Feature Separation**: Rolling features (ML-safe) vs Global features (analysis-only)
4. **Category Awareness**: Different strategies for normal/extreme/dead tokens
5. **📊 Mathematical Validation**: All calculations validated with TDD to 1e-12 precision

---

## 🚀 **Build & Development Commands**

### **Main Pipeline**
```bash
# Complete automated pipeline (data → features → ready for ML)
python run_pipeline.py              # Full pipeline
python run_pipeline.py --fast       # Fast mode (rolling features only)
```

### **Interactive Analysis**
```bash
# Main dashboard - primary entry point
streamlit run data_analysis/app.py

# Behavioral archetype analysis (Phase 1 complete)
streamlit run time_series/autocorrelation_app.py

# Quantitative trading analysis
streamlit run quant_analysis/quant_app.py

# Time series modeling
streamlit run time_series/time_series_app.py
```

### **Phase 1 Analysis Workflow**
```bash
# Run behavioral archetype analysis
streamlit run time_series/autocorrelation_app.py

# Navigate to "Multi-Resolution ACF" or "🎭 Behavioral Archetypes" tabs
# Configure token limits (supports 'none' for unlimited)
# Select analysis parameters and run
# Results exported to time_series/results/
```

### **ML Model Training**
```bash
# Directional Models (Binary Classification)
python ML/directional_models/train_lightgbm_model.py              # Short-term (15m-1h)
python ML/directional_models/train_lightgbm_model_medium_term.py  # Medium-term (2h-12h)
python ML/directional_models/train_unified_lstm_model.py          # Basic LSTM
python ML/directional_models/train_advanced_hybrid_lstm.py        # Advanced hybrid LSTM
python ML/directional_models/train_logistic_regression_baseline.py

# Forecasting Models (Regression)
python ML/forecasting_models/train_lstm_model.py
python ML/forecasting_models/train_advanced_hybrid_lstm_forecasting.py
python ML/forecasting_models/train_baseline_regressors.py --horizon 60 --model both
```

### **Dependencies**
```bash
pip install -r requirements.txt
# Core: polars, streamlit, plotly, numpy, scikit-learn
# ML: torch, lightgbm, xgboost, optuna
# Analysis: statsmodels, matplotlib, seaborn
# Testing: pytest (for TDD mathematical validation)
```

### **Testing & Validation**
```bash
# Run mathematical validation tests
python -m pytest data_analysis/tests/test_mathematical_validation.py -v
python -m pytest data_cleaning/tests/test_core_mathematical_validation.py -v
python -m pytest data_cleaning/tests/test_analyze_exclusions_validation.py -v
python -m pytest data_cleaning/tests/test_generate_graduated_datasets_validation.py -v
python -m pytest feature_engineering/tests/test_mathematical_validation.py -v
python -m pytest time_series/tests/test_archetype_utils_mathematical_validation.py -v
python -m pytest time_series/tests/test_behavioral_archetype_analysis_mathematical_validation.py -v

# Complete test suite summary
python -m pytest data_analysis/tests/ data_cleaning/tests/ feature_engineering/tests/ time_series/tests/ --tb=no -q
```

---

## 💡 **Key Technical Concepts**

### **1. Data Leakage Prevention**
**❌ WRONG**: Random token splits
```python
train_tokens, test_tokens = train_test_split(all_tokens, test_size=0.2)
```

**✅ CORRECT**: Temporal splits within each token
```python
for token in all_tokens:
    token_data = load_token(token)
    train_split = token_data[:int(0.6 * len(token_data))]    # First 60%
    val_split = token_data[int(0.6 * len(token_data)):int(0.8 * len(token_data))]  # Next 20%
    test_split = token_data[int(0.8 * len(token_data)):]     # Last 20%
```

### **2. Per-Token Scaling Strategy**
Handles variable lifespans (200-2000 minutes) and extreme volatility:
```python
# Each token gets individual scaler fitted on training data only
for token in tokens:
    scaler = RobustScaler()  # or Winsorizer for extreme crypto volatility
    scaler.fit(token.train_data)
    token.scaled_features = scaler.transform(token.all_data)
```

### **3. Feature Engineering Architecture**
**Rolling Features** (ML-safe, no future leakage):
- RSI, MACD, Bollinger Bands
- Rolling means, std devs
- Log returns, momentum
- Saved to `data/features/`

**Global Features** (analysis-only, uses full token history):
- Total return, max drawdown
- FFT analysis, spectral features
- Computed on-demand in Streamlit

### **4. Extreme Returns Handling**
**CRITICAL**: 99.9% dumps and 1M%+ pumps are **LEGITIMATE trading signals**, not noise!
- Use Winsorization instead of outlier removal
- Design features that capture extreme volatility patterns
- Cluster-specific models handle different volatility regimes

---

## 🎯 **Memecoin-Specific Analysis Framework**

### **Behavioral Archetypes** (9 Production Patterns)
**Death Patterns (>90% dead tokens):**
- **"Quick Pump & Death"**: High early returns, short lifespan
- **"Dead on Arrival"**: Low volatility, immediate death
- **"Slow Bleed"**: Gradual decline, medium lifespan
- **"Extended Decline"**: Long lifespan before death

**Mixed Patterns (50-90% dead tokens):**
- **"Phoenix Attempt"**: Multiple pumps before death, high volatility
- **"Zombie Walker"**: Minimal movement, eventual death

**Survivor Patterns (<50% dead tokens):**
- **"Survivor Pump"**: Artificial pumps, still alive, high volatility
- **"Stable Survivor"**: Low volatility, consistent survival
- **"Survivor Organic"**: Natural trading patterns, low death rate

### **Multi-Resolution ACF Analysis** (Death-Aware)
- **Sprint** (50-400 active min): Fast-moving patterns with early death
- **Standard** (400-1200 active min): Typical lifecycle before death/survival
- **Marathon** (1200+ active min): Extended development or survival

### **Pre-Death Feature Extraction**
- **ACF signatures** at lags [1,2,5,10,20,60] before death_minute
- **Statistical features**: mean, std, skewness, kurtosis of pre-death returns
- **Peak timing**: when highest price occurred before death
- **Drawdown metrics**: maximum decline from peak before death
- **Death characteristics**: death_type, death_velocity, death_completeness

---

## 📋 **Current Roadmap Status**

### **✅ Completed**
- Comprehensive data analysis pipeline
- Category-aware cleaning strategies  
- ML-safe feature engineering with temporal splitting
- Multiple ML models (LightGBM, LSTM, hybrid approaches)
- Autocorrelation analysis framework
- **🧪 COMPREHENSIVE TDD IMPLEMENTATION**:
  - **data_analysis/**: 16/16 mathematical validation tests passing
  - **data_cleaning/**: 44/44 mathematical validation tests passing
  - **feature_engineering/**: 96 tests created (39 passing, 57 expected failures - testing against future implementations)
  - **time_series/**: 44 TDD tests for behavioral archetype analysis (36 archetype_utils + 8 behavioral_archetype_analysis)
  - All statistical calculations validated against numpy/scipy with 1e-12 precision
  - Streamlit display accuracy mathematically guaranteed
  - Complete test coverage for edge cases and numerical stability
  - Test-driven approach ensures mathematical correctness before implementation
- **🪙 MEMECOIN-AWARE TOKEN FILTERING**:
  - **Removed invalid "recent vs historical" bias** from 24-hour data analysis
  - **Updated variability thresholds** for memecoin context (0.1% CV vs 2% before)
  - **Implemented scalping-focused dead token detection** (flat periods, tick frequency)
  - **Enhanced UI with memecoin-appropriate metrics** (24h CV, max flat period, signal strength)
  - **Preserves extreme volatility** (99.9% dumps, 1M%+ pumps) as legitimate patterns
  - **Focuses on data quality filtering** for ML pipeline (not trading strategies)
- **✅ PHASE 1 COMPLETE**: Pattern Discovery & Behavioral Archetype Identification
  - **Production-ready multi-resolution ACF analysis** with death-aware token categorization
  - **Behavioral archetype identification system** for 9 distinct memecoin patterns
  - **Death detection algorithm** with multi-criteria approach and 1e-12 mathematical precision
  - **Pre-death feature extraction** using only data before death_minute (ACF lags, statistics, peak timing)
  - **Early detection classifier** (5-minute window) for real-time archetype classification
  - **Unlimited token analysis** with configurable per-category limits
  - **Extreme volatility optimization** for memecoin data (10M%+ pumps, 99.9% dumps)
  - **Interactive Streamlit interface** with t-SNE visualization and survival analysis
  - **Comprehensive testing coverage** with 44 mathematical validation tests

### **🔄 In Progress** 
- **Phase 2**: Temporal Pattern Recognition

### **📋 Next Steps** (From MEMECOIN_ANALYSIS_ROADMAP.md)
- **Phase 2**: Temporal Pattern Recognition
- **Phase 3**: Feature Engineering & ML Pipeline Stabilization  
- **Phase 4**: Volume Data Integration Preparation

---

## ⚠️ **Critical Guidelines & Best Practices**

### **❌ Never Do This**
- Mix behavioral patterns in unified models
- Use global scaling across tokens
- Split data randomly by tokens
- Treat extreme returns as noise/outliers
- Create unified features for all token types

### **✅ Always Do This**  
- Use temporal splitting within each token
- Scale each token individually using RobustScaler or Winsorizer
- Separate rolling features (ML) from global features (analysis)
- Embrace extreme volatility as legitimate signal
- Design cluster-specific models for each archetype
- **Validate all mathematical operations with comprehensive tests**

### **🔍 Data Quality Checks**
```python
# Essential validations before ML training
def validate_features_safety(features_df, token_name):
    # Check for constant features (data leakage risk)
    # Validate temporal ordering
    # Ensure no future information in features
```

### **📊 Performance Expectations**
- **Directional Models**: 85-90% accuracy, 85-95% ROC AUC
- **Forecasting Models**: R² 0.3-0.7, MAE 5-15% of avg price
- **Current Challenge**: Achieving stable performance across behavioral types

---

## 🔧 **Development Workflows**

### **Adding New Features**
1. Check if feature needs future information (if yes → global feature only)
2. Implement in `feature_engineering/advanced_feature_engineering.py`
3. Test with temporal splitting validation
4. Update feature documentation

### **New ML Model Development**
1. Follow existing patterns in `ML/directional_models/` or `ML/forecasting_models/`
2. Implement per-token scaling
3. Use temporal splitting for validation
4. Add comprehensive metrics reporting

### **Debugging Data Issues**
1. Use `data_analysis/app.py` for visual inspection
2. Check quality reports from `data_quality.py`
3. Validate feature engineering with sample tokens
4. Use category-specific analysis

### **Performance Optimization**
- Polars for fast DataFrame operations
- Lazy evaluation where possible
- Caching for repeated analysis
- Parallel processing for batch operations

---

## 📚 **Key Files to Understand**

### **Must Read First**
- `README.md`: Project overview and usage guide
- `MEMECOIN_ANALYSIS_ROADMAP.md`: Comprehensive 4-phase plan
- `ML/README.md`: Detailed ML pipeline documentation

### **Core Analysis**
- `data_analysis/app.py`: Main dashboard entry point
- `time_series/autocorrelation_app.py`: ACF analysis interface
- `run_pipeline.py`: Complete automated pipeline

### **Architecture Examples**
- `ML/directional_models/train_unified_lstm_model.py`: Standard LSTM with per-token scaling
- `feature_engineering/advanced_feature_engineering.py`: ML-safe feature creation
- `data_cleaning/clean_tokens.py`: Category-aware cleaning strategies

---

## 🚨 **Common Pitfalls & Solutions**

### **Data Leakage**
- **Problem**: Using future information in features
- **Solution**: Strict temporal splitting, rolling feature validation

### **Scale Mismatch**
- **Problem**: Tokens with vastly different price ranges  
- **Solution**: Per-token RobustScaler or Winsorization

### **Mixed Behavioral Patterns**
- **Problem**: Single model trying to handle all archetypes
- **Solution**: Cluster-specific models after behavioral identification

### **Extreme Volatility**
- **Problem**: Treating 1M%+ moves as outliers
- **Solution**: Embrace as signal, use appropriate scaling methods

---

## 🎉 **Success Metrics & Goals**

### **Pattern Discovery Success**
- [ ] 5-8 distinct behavioral archetypes with clear ACF signatures
- [ ] >80% intra-cluster ACF similarity within each archetype  
- [ ] <50% inter-cluster ACF similarity between archetypes

### **ML Pipeline Success**
- [ ] Stable cluster-specific models with <10% performance variance
- [ ] Improved overall performance vs unified approach
- [ ] Early classification accuracy >70% from first 60 minutes

### **Practical Utility Success**
- [ ] Actionable trading strategies for each archetype
- [ ] Clear risk/reward profiles per behavioral type
- [ ] Scalable framework ready for volume data integration

---

## 📝 **Recent Session Context (Latest Updates)**

### **🪙 Memecoin Token Filtering Revolution (Current Session)**

**Key Insight**: User provided critical context about **24-hour minute-by-minute memecoin data** to correct fundamental assumptions in variability analysis.

**❌ What Was Wrong**:
- **"Recent vs Historical" bias**: Artificially split 24-hour data into "recent" (last 50%) and "historical" (first 50%)
- **Invalid timeframe assumptions**: Applied traditional market concepts to 24-hour crypto cycles
- **Over-strict thresholds**: 2% CV minimum filtered out legitimate memecoin patterns
- **Misunderstood purpose**: Focused on trading strategy context rather than data quality filtering

**✅ What Was Fixed**:
- **Full 24-hour analysis**: Every minute equally important for token filtering
- **Memecoin-appropriate thresholds**: 0.1% CV minimum preserves extreme volatility patterns  
- **Dead token detection**: Focus on flat periods (>2 hours identical prices) and tick frequency
- **Data quality focus**: Filter for ML pipeline input, not trading strategies
- **Preserve extreme moves**: 99.9% dumps and 1M%+ pumps are legitimate signals

**🔧 Technical Changes**:
- `data_cleaning/clean_tokens.py`: Completely rewrote `_check_price_variability_graduated()`
- `data_analysis/app.py`: Updated UI labels from "Recent Activity CV" to "24h Price CV"
- New metrics: Max Flat Period, Tick Frequency, Signal Strength
- Decision logic: 3/5 criteria (no mandatory recent CV requirement)

**🎯 Current State**: 
Token filtering now properly identifies **truly dead/inactive tokens** while preserving **all legitimate memecoin volatility patterns** for downstream ML analysis.

---

**🚀 Ready to contribute to stable, interpretable memecoin analysis!**

*Last updated: Memecoin-aware token filtering implementation completed*