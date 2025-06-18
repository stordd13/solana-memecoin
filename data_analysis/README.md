# Memecoin Data Analysis Dashboard

A comprehensive Streamlit-based dashboard for analyzing memecoin price data quality and patterns.

## Overview

This dashboard provides powerful tools for analyzing memecoin data with focus on:
- Data quality assessment
- Price analysis and technical indicators
- Pattern detection (pumps, dumps, manipulation)
- Comparative analysis across multiple tokens

## Key Improvements Made

### 1. **Enhanced Data Quality Analysis** (`data_quality.py`)
- Added quality scoring system (0-100) with ratings
- Comprehensive error handling and validation
- Detection of specific issues: gaps, duplicates, extreme changes
- Recommendations for tokens suitable for analysis
- Progress tracking for large datasets

### 2. **Advanced Price Analysis** (`price_analysis.py`)
- Extended technical indicators (RSI, Bollinger Bands, etc.)
- Memecoin-specific patterns (rug pulls, honeypots, whale activity)
- Liquidity analysis and stagnation detection
- Trading signal generation
- Enhanced pump detection with classification

### 3. **Streamlined App Structure** (`app.py`)
- Consolidated functionality into single, well-organized app
- Intuitive navigation with clear sections
- Enhanced visualizations using Plotly
- Comprehensive export options
- Session state management for better performance

## Usage

### Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Data Structure Required

The app expects data in the following structure:
```
data/
└── raw/
    └── dataset_name/
        └── subdirectory/
            ├── TOKEN1_data.parquet
            ├── TOKEN2_data.parquet
            └── ...
```

Each parquet file should contain:
- `datetime`: Timestamp column
- `price`: Token price
- `volume`: (optional) Trading volume

### Features by Section

#### 🔍 Data Quality
- Analyze completeness and integrity of price data
- Identify gaps, duplicates, and anomalies
- Quality scoring and recommendations
- Export quality reports

#### 📈 Price Analysis
- Individual token deep-dive analysis
- Technical indicators and charts
- First-hour performance metrics
- Pattern detection
- Trading signals

#### 🎯 Pattern Detection
- Batch analysis of multiple tokens
- Identify common patterns (pumps, dumps, etc.)
- Risk-return profiling
- Pattern correlation analysis

#### 📊 Comparative Analysis
- Compare 2-20 tokens side-by-side
- Normalized price charts
- Performance rankings
- Pattern comparison matrix

## Key Metrics Explained

### Quality Metrics
- **Quality Score**: 0-100 score based on completeness, gaps, and data issues
- **Completeness %**: Percentage of expected data points present
- **Time Gaps**: Missing data periods > 1 minute
- **Extreme Changes**: Price changes > 1000% in one minute

### Price Metrics
- **Total Return**: Price change from first to last data point
- **Max Gain**: Maximum price increase from initial price
- **Volatility**: Standard deviation of returns
- **RSI**: Relative Strength Index (momentum indicator)

### Pattern Definitions
- **Pump & Dump**: High max gain but negative final return
- **Steady Growth**: Positive return with low volatility
- **Potential Rug Pull**: 5x+ gain followed by 90%+ drop
- **Whale Activity**: Frequent large price movements (>50%)

## Best Practices

1. **Data Quality First**: Always run quality analysis before price analysis
2. **Filter by Quality**: Use tokens with quality score > 80 for best results
3. **Compare Similar Tokens**: When comparing, select tokens from same time period
4. **Export Results**: Save analysis results for future reference

## Troubleshooting

### Common Issues

1. **"No datasets found"**: Check that data path exists and contains parquet files
2. **Import errors**: Ensure all dependencies are installed
3. **Memory issues**: Reduce number of tokens analyzed simultaneously
4. **Slow performance**: Use the max files limit in quality analysis

### Performance Tips
- Analyze in batches of 50 tokens or less
- Use quality filters to focus on good data
- Close other applications when analyzing large datasets

## Additional Tools

### Token Overlap Analyzer (`token_overlap_analyzer.py`)
A comprehensive tool for analyzing overlaps between token categories with both quick and detailed analysis modes.

**Features:**
- **Quick Mode**: Fast overlap checking between specific folders
- **Comprehensive Mode**: Detailed analysis of all folder overlaps
- **Command-line interface** for easy automation
- **Multi-category token detection**
- **Detailed reporting** with statistics
- **JSON export** for further analysis
- **Support for exclusive category system**

**Usage:**
```bash
# Quick comparison between two folders
python token_overlap_analyzer.py --mode quick --folder1 normal_behavior_tokens --folder2 dead_tokens

# Compare one folder with all others
python token_overlap_analyzer.py --mode quick --folder1 normal_behavior_tokens

# Comprehensive analysis (default)
python token_overlap_analyzer.py --mode comprehensive

# Use custom processed data path
python token_overlap_analyzer.py --processed-path /path/to/processed/data
``` 