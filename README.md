# Solana Memecoin Analysis Platform

A comprehensive data analysis platform for analyzing Solana memecoin price patterns and trading behavior during the critical first 24 hours after token launch.

## 🎯 Project Overview

This platform analyzes a dataset of 10,000+ memecoins launched between 2022-2025, focusing exclusively on minute-by-minute price action during the first 24 hours after launch. The analysis pipeline includes data quality assessment, pattern recognition, optimal trading timing, and predictive modeling.

### Key Features
- **Real-time Interactive Dashboard** with Streamlit
- **Advanced Data Quality Analysis** with dead token detection
- **Optimal Trading Timing** calculations across multiple tokens
- **Statistical Analysis** with median/average metrics for outlier handling
- **Time Series Modeling** and pattern recognition
- **Quantitative Trading Analysis** with risk metrics
- **Data Export** functionality for processed datasets

## 📊 Dataset Specifications

- **Format**: Individual Parquet files per token
- **Timeframe**: First 24 hours only (1,440 minutes)
- **Columns**: `price` and `datetime` (pure price action)
- **Volume**: 10,000+ tokens
- **Period**: 2022-2025 launches
- **Market Context**: Different market eras (bear, recovery, bull)

## 🏗️ Project Structure

```
memecoin2/
├── data_analysis/          # Main interactive dashboard and core analysis
├── data_cleaning/          # Data preprocessing and quality assurance
├── quant_analysis/         # Quantitative trading analysis
├── time_series/           # Advanced time series modeling
├── notebooks/             # Jupyter notebooks for exploration
├── data/                  # Raw and processed datasets (gitignored)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📁 Module Descriptions

### 🎛️ `data_analysis/` - Interactive Dashboard Hub

**Main Entry Point**: `streamlit run data_analysis/app.py`

The core interactive dashboard providing real-time analysis capabilities:

- **`app.py`**: Main Streamlit application with multi-page navigation
  - Data Quality Analysis page
  - Price Analysis with optimal entry/exit timing
  - Pattern Detection and classification
  - Price Distribution visualization

- **`data_loader.py`**: Robust data loading with Polars integration
  - Subfolder selection from data directory
  - Token discovery and caching
  - DataFrame validation and type conversion

- **`data_quality.py`**: Comprehensive data quality assessment
  - Dead token detection (constant price periods)
  - Gap analysis and temporal continuity checks
  - Price anomaly detection (extreme jumps)
  - Launch context extraction (market era classification)

- **`price_analysis.py`**: Advanced price metrics and optimal timing
  - Optimal entry/exit calculation (single best trade)
  - Universal timing analysis across multiple tokens
  - Volatility and return calculations
  - Pattern classification and risk metrics

- **`export_utils.py`**: Data export functionality to processed folders

### 🧹 `data_cleaning/` - Data Preprocessing Pipeline

- **`clean_tokens.py`**: Data cleaning and quality assurance
  - Missing data imputation strategies
  - Outlier detection and treatment
  - Price continuity validation
  - Quality scoring for each token

### 📈 `quant_analysis/` - Quantitative Trading Analysis

Advanced quantitative analysis for trading strategy development:

- **`quant_app.py`**: Specialized Streamlit app for quantitative analysis
  - Portfolio-level analysis
  - Risk-adjusted returns
  - Drawdown analysis
  - Performance attribution

- **`quant_analysis.py`**: Core quantitative metrics
  - Sharpe ratio calculations
  - Maximum drawdown analysis
  - Risk-adjusted performance metrics

- **`quant_viz.py`**: Advanced visualization for quant analysis
  - Performance charts
  - Risk-return scatter plots
  - Drawdown visualizations

- **`trading_analysis.py`**: Trading strategy backtesting
  - Entry/exit signal generation
  - Position sizing algorithms
  - Transaction cost modeling

### 📊 `time_series/` - Advanced Time Series Modeling

Sophisticated time series analysis and predictive modeling:

- **`time_series_app.py`**: Streamlit interface for time series analysis
  - Model selection and training
  - Prediction visualization
  - Model performance metrics

- **`time_series_analyzer.py`**: Core time series analysis engine
  - Feature engineering from price data
  - Pattern recognition algorithms
  - Temporal alignment and normalization

- **`time_series_models.py`**: Machine learning models
  - LSTM networks for sequence prediction
  - Transformer models for pattern recognition
  - Ensemble methods for robust predictions

- **`run_analysis.py`**: Batch processing for large-scale analysis

### 📓 `notebooks/` - Exploratory Analysis

- **`exploration.ipynb`**: Jupyter notebooks for data exploration and prototyping

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/stordd13/solana-memecoin.git
cd memecoin2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
   - Place Parquet files in `data/raw/` or subdirectories
   - Each file should contain `price` and `datetime` columns
   - Files should represent 24-hour periods starting from token launch

### Quick Start

**Launch the main dashboard**:
```bash
streamlit run data_analysis/app.py
```

**Access specialized analysis**:
```bash
# Quantitative analysis
streamlit run quant_analysis/quant_app.py

# Time series modeling
streamlit run time_series/time_series_app.py
```

## 📋 Usage Guide

### 1. Data Loading
- Select data subfolder from dropdown
- View file count before loading
- Use "Load Data" button to initialize analysis

### 2. Token Selection
- **Single Token**: Detailed individual analysis
- **Multiple Tokens**: Comparative analysis with aggregated metrics
- **Random Tokens**: Statistical sampling for pattern discovery
- **All Tokens**: Comprehensive dataset analysis

### 3. Analysis Features

#### Data Quality Analysis
- Dead token identification and timing
- Gap analysis and data completeness
- Price anomaly detection
- Launch context classification by market era

#### Price Analysis
- **Individual metrics**: Return %, volatility %, optimal trade timing
- **Aggregated metrics**: Average and median statistics (outlier-robust)
- **Universal timing**: Best entry/exit times across all selected tokens
- **Detailed breakdown**: Per-token metrics in expandable DataFrame

#### Pattern Detection
- Pump & dump identification
- Trend change detection
- Momentum shift analysis
- Pattern classification

## 🔧 Technical Features

### Performance Optimizations
- **Polars integration** for fast DataFrame operations
- **Lazy evaluation** for memory efficiency
- **Caching mechanisms** for repeated analysis
- **Parallel processing** for batch operations

### Data Quality Assurance
- **Robust outlier handling** using IQR methods
- **Missing data imputation** with multiple strategies
- **Price continuity validation**
- **Quality scoring** for dataset reliability

### Statistical Robustness
- **Median statistics** alongside averages for outlier resistance
- **Bootstrap sampling** for confidence intervals
- **Cross-validation** for model validation
- **Temporal validation** respecting time series nature

## 📈 Key Metrics and Insights

### Trading Metrics
- **Optimal Return %**: Best possible single trade return
- **Entry/Exit Timing**: Minutes after launch for optimal trades
- **Universal Timing**: Best timing strategy across multiple tokens
- **Risk-Adjusted Returns**: Sharpe ratios and drawdown analysis

### Market Analysis
- **Dead Token Rate**: Percentage of tokens that "die" (constant price)
- **Market Era Effects**: Performance differences across 2022-2025
- **Launch Timing Impact**: Weekend vs weekday, hour-of-day effects
- **Volatility Patterns**: Intraday volatility evolution

## 🛠️ Development

### Branch Structure
- `main`: Production-ready code
- `dev`: Development branch
- `fix_quant_views`: Quantitative analysis improvements

### Contributing
1. Create feature branch from `dev`
2. Implement changes with tests
3. Submit pull request to `dev`
4. Merge to `main` after review

## 📊 Data Export

The platform supports exporting analyzed tokens to processed folders:
- Dead tokens → `data/processed/cleaned/dead_tokens/`
- High-quality tokens → `data/processed/cleaned/high_quality_tokens/`
- Tokens with gaps → `data/processed/cleaned/tokens_with_gaps/`
- Tokens with issues → `data/processed/cleaned/tokens_with_issues/`

## ⚠️ Important Notes

### Data Considerations
- **Pure Price Action**: Analysis based solely on price movements
- **24-Hour Window**: Focus on critical first day after launch
- **Market Context**: Results vary significantly by launch era
- **Survivorship Bias**: Consider that successful tokens may be overrepresented

### Limitations
- No volume or liquidity data available
- Market conditions vary significantly across time periods
- High volatility in memecoin space requires careful risk management
- Results are for educational/research purposes

## 📄 License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations when using for trading decisions.

## 🤝 Support

For questions or issues:
1. Check existing GitHub issues
2. Create new issue with detailed description
3. Include relevant error messages and system information

---

**⚡ Built with**: Python, Streamlit, Polars, Plotly, Scikit-learn
**🎯 Focus**: Memecoin price pattern analysis and optimal trading timing
**📊 Scale**: 10,000+ tokens across 3+ years of market data 