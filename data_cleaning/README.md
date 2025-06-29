# 🧹 Token Data Cleaning System

A sophisticated **category-aware cleaning system** designed specifically for memecoin data that intelligently distinguishes between data artifacts and legitimate market behavior.

## 🚨 **CRITICAL DATA LEAKAGE FIX IMPLEMENTED**

**Problem Identified**: Dead tokens with constant price periods at the end would cause **severe data leakage** in forecasting models:
- Models would learn: "If price constant for X minutes → predict same price"  
- Result: **Artificially inflated accuracy** (90%+ on dead tokens)
- **Misleading performance metrics** that don't reflect real predictive ability

**Solution Implemented**: 
- **`_remove_death_period()`** method in minimal cleaning for dead tokens
- **Removes constant periods ≥60 minutes** at end of token timelines
- **Keeps only 2 minutes** of constant price for minimal context
- **Prevents models from learning trivial constant-price patterns**

**Impact**: Ensures **realistic model performance** and **honest accuracy metrics**

## 🎯 **Overview**

This system revolutionizes token data cleaning by applying different strategies based on token behavior patterns, ensuring that:

- **Real extreme movements** (5,000% pumps, 95% dumps) are preserved
- **Data artifacts** (listing glitches, data corruption) are removed
- **Market reality** is maintained while improving data quality

## 🚀 **Key Features**

### **Category-Aware Cleaning**
Different cleaning strategies for different token types:

| Category | Strategy | Focus |
|----------|----------|-------|
| **Normal Behavior Tokens** | `gentle` | Preserve natural volatility |
| **Dead Tokens** | `minimal` | Basic cleaning only |
| **Tokens with Extremes** | `preserve` | Keep ALL extreme movements |
| **Tokens with Gaps** | `aggressive` | Comprehensive gap filling |

### **Smart Anomaly Detection**
Distinguishes between:
- **Data Artifacts**: Listing spikes (>20x + 99% drop), data errors (>100,000% isolated moves)
- **Legitimate Behavior**: Real pumps (up to 5,000%), sustained movements, market crashes

### **Intelligent Processing**
- Context-aware decision making
- Volume/activity confirmation
- Sustained vs isolated movement analysis
- Reversible operations with comprehensive logging

## 📁 **Files**

- **`clean_tokens.py`** - Main cleaning engine with category-aware strategies
- **`CATEGORY_AWARE_CLEANING_REVIEW.md`** - Detailed technical documentation and design rationale

## 🛠️ **Usage**

### **Clean All Categories**
```python
from clean_tokens import clean_all_categories

# Clean all token categories with appropriate strategies
summary = clean_all_categories()
print(f"Successfully cleaned {summary['total_successfully_cleaned']} tokens")
```

### **Clean Specific Category**
```python
from clean_tokens import clean_category

# Clean only extreme tokens (preserve their extremes!)
result = clean_category('tokens_with_extremes', limit=100)
print(f"Cleaned {result['successfully_cleaned']} extreme tokens")
```

### **Clean Single Token**
```python
from clean_tokens import CategoryAwareTokenCleaner

cleaner = CategoryAwareTokenCleaner()
log = cleaner.clean_token_file(token_path, category='normal_behavior_tokens')
```

## 📊 **Input/Output Structure**

### **Input** (Expected structure)
```
data/processed/
├── normal_behavior_tokens/     # Regular memecoin behavior
├── dead_tokens/               # Inactive/abandoned tokens
├── tokens_with_extremes/      # Tokens with extreme price movements
└── tokens_with_gaps/          # Tokens with data gaps
```

### **Output** (Cleaned data)
```
data/cleaned/
├── normal_behavior_tokens/     # Gently cleaned, volatility preserved
├── dead_tokens/               # Minimally cleaned
├── tokens_with_extremes/      # Extremes preserved, artifacts removed
├── tokens_with_gaps/          # Gaps filled comprehensively
├── normal_behavior_tokens_cleaning_log.json
├── dead_tokens_cleaning_log.json
├── tokens_with_extremes_cleaning_log.json
├── tokens_with_gaps_cleaning_log.json
└── overall_cleaning_summary.json
```

## 🎯 **Cleaning Strategies Explained**

### **1. GENTLE CLEANING** (`normal_behavior_tokens`)
**Philosophy**: Preserve natural market volatility, remove only obvious artifacts

**What it does**:
- ✅ **Removes listing artifacts (20x spike + 99% immediate drop)**
  - *How*: Analyzes first 3 minutes, compares price to median of next 10 minutes
  - *Method*: `_remove_listing_artifacts()` - removes if >20x median + >99% immediate drop
  - *Detection*: Only removes obvious launch glitches, preserves real explosive launches
- ✅ **Fixes severe data errors (>100,000% isolated moves)**
  - *How*: Identifies single-point extreme moves that revert immediately
  - *Method*: `_fix_data_errors()` - replaces with average of surrounding prices
  - *Logic*: Preserves sustained movements, only fixes isolated data corruption
- ✅ **Fills small gaps (1-2 minutes) with linear interpolation**
  - *How*: Creates complete minute-by-minute timeline, identifies missing periods
  - *Method*: `_fill_small_gaps_only()` - linear interpolation for gaps ≤2 minutes
  - *Process*: Handles duplicates first (averages), then fills gaps progressively
- ✅ **Handles zero/negative prices conservatively**
  - *How*: Identifies impossible price values (≤0)
  - *Method*: `_handle_invalid_prices_conservative()` - interpolates between valid prices
  - *Safety*: Excludes token if >5% of data is invalid (likely corrupted dataset)
- ❌ **Preserves**: Natural volatility, legitimate pumps/dumps, market behavior

### **2. MINIMAL CLEANING** (`dead_tokens`)
**Philosophy**: Don't over-clean inactive tokens, fix only critical issues

**What it does**:
- ✅ **🛡️ REMOVES CONSTANT PRICE PERIODS** (**CRITICAL ANTI-LEAK FIX**)
  - *How*: Detects constant price periods ≥60 minutes at the end of token timeline
  - *Method*: `_remove_death_period()` - removes bulk of constant period, keeps only 2 minutes for context
  - *Why*: **PREVENTS DATA LEAKAGE** - without this, models learn "constant price → predict constant price"
  - *Impact*: Prevents artificially inflated accuracy metrics on dead tokens
- ✅ **Fixes severe data errors only**
  - *How*: Detects isolated >100,000% price moves that revert immediately
  - *Method*: `_fix_data_errors()` - replaces with interpolated value between prev/next prices
- ✅ **Handles invalid prices (basic fixes)**
  - *How*: Identifies prices ≤0 (negative or zero)
  - *Method*: `_handle_invalid_prices_conservative()` - interpolates using surrounding valid prices
  - *Threshold*: Excludes token if >5% of data is invalid
- ❌ **Skips**: Gap filling, volatility cleaning (token is inactive anyway)

### **3. PRESERVE EXTREMES** (`tokens_with_extremes`)
**Philosophy**: Keep ALL legitimate extreme movements - they define these tokens!

**What it does**:
- ✅ **Handles only impossible values (negative prices, exact zeros)**
  - *How*: Only fixes mathematically impossible values (price ≤0)
  - *Method*: `_handle_impossible_values_only()` - higher tolerance (>10% invalid before exclusion)
  - *Conservative*: Does NOT fix low prices, only truly impossible ones
- ✅ **Fixes extreme corruption (>1,000,000% obvious errors)**
  - *How*: Only detects moves >10,000x (1,000,000%) - clearly data corruption
  - *Method*: `_fix_extreme_data_corruption()` - replaces with local median of 5 surrounding points
  - *Threshold*: Much higher than normal tokens to preserve legitimate extremes
- ✅ **Fills only critical gaps that break continuity**
  - *How*: Only fills 1-minute gaps that break data flow
  - *Method*: `_fill_critical_gaps_only()` - minimal gap filling (≤1 minute only)
  - *Purpose*: Maintains data continuity without altering price patterns
- ❌ **Preserves**: ALL extreme movements (1,000%+ pumps, 95%+ dumps)

### **4. AGGRESSIVE CLEANING** (`tokens_with_gaps`)
**Philosophy**: Fix data quality issues comprehensively, gaps are main problem

**What it does**:
- ✅ **Removes listing artifacts**
  - *How*: Same as gentle cleaning - analyzes first 3 minutes for launch glitches
  - *Method*: `_remove_listing_artifacts()` - standard artifact detection
- ✅ **Fills gaps comprehensively (up to 10-minute gaps)**
  - *How*: Multi-method approach based on gap size
  - *Method*: `_fill_gaps_comprehensive()` with intelligent interpolation:
    - 1-minute gaps: Linear interpolation
    - 2-3 minute gaps: Linear interpolation
    - 4-6 minute gaps: Polynomial interpolation (order 2)
    - 7-10 minute gaps: Forward fill + linear combination
    - >10 minute gaps: Left unfilled (too large to reliably interpolate)
- ✅ **Fixes data errors thoroughly**
  - *How*: Standard data error detection for isolated extreme moves
  - *Method*: `_fix_data_errors()` - same logic as gentle cleaning
- ✅ **Handles invalid prices**
  - *How*: Conservative approach to zero/negative prices
  - *Method*: `_handle_invalid_prices_conservative()` - standard interpolation
- 🎯 **Focus**: Comprehensive gap filling while preserving movements

## 🔍 **What Gets Cleaned vs Preserved**

### **❌ REMOVED (Data Artifacts)**
- **Listing spikes**: >20x median price + >99% immediate drop
- **Data corruption**: >100,000% isolated single-minute moves that revert
- **Impossible values**: Negative prices, exact zeros from data errors
- **Technical glitches**: Clear data feed errors

### **✅ PRESERVED (Legitimate Market Behavior)**
- **Real pumps**: Up to 5,000% can be legitimate in memecoins
- **Real dumps**: Up to 95% drops are realistic market crashes
- **Sustained movements**: Multi-minute price actions with context
- **Natural volatility**: Normal memecoin price swings

## 📈 **Benefits**

- ✅ **Realistic extreme movements preserved** - 5,000% pumps and 95% dumps are kept when legitimate
- ✅ **Only true artifacts removed** - Smart detection distinguishes real movements from data errors
- ✅ **Data reflects actual market behavior** - Cleaned data maintains memecoin market reality
- ✅ **Category-appropriate processing** - Different strategies for different token types

## 🚨 **Important Notes**

### **For Extreme Tokens**
- **NEVER** removes movements >1,000% unless >1,000,000% (obvious corruption)
- Uses **higher thresholds** for "impossible" values
- **Preserves defining characteristics** of extreme tokens

### **For All Categories**
- **Comprehensive logging** of all changes made
- **Reversible operations** where possible
- **Clear reasoning** documented for each modification

## 📋 **Logging & Monitoring**

Each cleaning operation generates detailed logs:

```json
{
  "token": "PEPE_TOKEN",
  "category": "tokens_with_extremes", 
  "original_rows": 1440,
  "final_rows": 1438,
  "modifications": [
    {
      "type": "listing_artifact_removed",
      "position": 2,
      "price_ratio": 25.3,
      "reason": "Obvious listing artifact: extreme spike followed by immediate crash"
    }
  ],
  "strategy_used": "preserve",
  "status": "cleaned_successfully"
}
```

## 🔧 **Configuration**

### **Artifact Detection Thresholds**
```python
artifact_thresholds = {
    'listing_spike_multiplier': 20,     # 20x median for listing artifacts
    'listing_drop_threshold': 0.99,     # 99% drop after spike
    'data_error_threshold': 1000,       # 100,000% (obvious data errors)
}
```

### **Market Behavior Thresholds** (Preserved!)
```python
market_thresholds = {
    'max_realistic_pump': 50,           # 5,000% pumps can be real
    'max_realistic_dump': 0.95,         # 95% dumps can be real
    'sustained_movement_minutes': 3,    # Real movements last >3 minutes
}
```

## 🎉 **Success Metrics**

The system achieves:
- **High data quality** while preserving market reality
- **Category-appropriate processing** for different token types
- **Transparent operations** with comprehensive logging
- **Reversible changes** for quality assurance

## 🚀 **Quick Start**

1. **Ensure your data is categorized** in the expected folder structure
2. **Run the cleaning system**:
   ```python
   from clean_tokens import clean_all_categories
   summary = clean_all_categories()
   ```
3. **Check the results** in `data/cleaned/` and review logs
4. **Verify the cleaning** worked as expected for your use case

---

This **intelligent, context-aware** data processing system ensures your cleaned data is both high-quality AND representative of actual memecoin market dynamics! 🚀 