# Category-Aware Token Cleaning System Review

## 🚀 **NEW SYSTEM OVERVIEW**

The cleaning system has been completely redesigned to implement **category-aware cleaning** that preserves realistic market behavior while removing data artifacts. This addresses the critical issue of distinguishing between:

- **Data Artifacts**: Technical errors, listing glitches, data corruption
- **Legitimate Market Behavior**: Real pumps, dumps, and extreme movements

## 🎯 **KEY IMPROVEMENTS**

### 1. **Category-Specific Strategies**
Different cleaning approaches based on token behavior:

```python
CATEGORIES = {
    'normal_behavior_tokens': 'gentle',     # Preserve natural volatility
    'dead_tokens': 'minimal',               # Basic cleaning only  
    'tokens_with_extremes': 'preserve',     # Keep extreme movements
    'tokens_with_gaps': 'aggressive'        # Fill gaps aggressively
}
```

### 2. **Smart Anomaly Detection**
Two sets of thresholds to distinguish artifacts from legitimate behavior:

#### **Artifact Thresholds** (Remove these)
- **Listing Spike**: >20x median + >99% immediate drop
- **Data Errors**: >100,000% isolated single-minute moves
- **Data Corruption**: >1,000,000% obvious errors

#### **Market Thresholds** (Preserve these!)
- **Realistic Pumps**: Up to 5,000% can be legitimate
- **Realistic Dumps**: Up to 95% drops can be real
- **Sustained Movements**: Real moves last >3 minutes

## 🛠️ **CLEANING STRATEGIES**

### **1. GENTLE CLEANING** (Normal Behavior Tokens)
**Philosophy**: Preserve natural market volatility, remove only obvious artifacts

**Process**:
1. **Remove Listing Artifacts**: Only obvious launch glitches (20x spike + 99% drop)
2. **Fix Data Errors**: Only >100,000% isolated single-point errors
3. **Fill Small Gaps**: Only 1-2 minute gaps with linear interpolation
4. **Handle Invalid Prices**: Conservative approach to zero/negative prices

**Preserves**: Natural volatility, legitimate pumps/dumps, market behavior

### **2. MINIMAL CLEANING** (Dead Tokens)
**Philosophy**: Don't over-clean inactive tokens, fix only critical issues

**Process**:
1. **🛡️ Remove Death Period (CRITICAL)**: Remove constant price periods ≥60 minutes at end
2. **Fix Severe Data Errors**: Only obvious corruption
3. **Handle Invalid Prices**: Basic negative/zero price fixes
4. **Skip Gap Filling**: Don't bother with gaps since token is inactive
5. **Skip Volatility Cleaning**: Dead tokens don't need volatility management

**CRITICAL ANTI-LEAKAGE FIX**: 
- **Problem**: Models learn "constant price → predict constant price" 
- **Solution**: Remove bulk of constant periods, keep only 2 minutes for minimal context
- **Impact**: Prevents artificially inflated forecasting accuracy

**Rationale**: Dead tokens need minimal cleaning but MUST prevent data leakage

### **3. PRESERVE EXTREMES** (Tokens with Extremes)
**Philosophy**: Keep ALL legitimate extreme movements - they define these tokens!

**Process**:
1. **Handle Impossible Values**: Only fix negative prices, exact zeros
2. **Fix Extreme Corruption**: Only >1,000,000% obvious data errors
3. **Fill Critical Gaps**: Only gaps that break data continuity
4. **Preserve Everything Else**: All extreme movements are kept

**Critical**: Uses higher thresholds to avoid removing legitimate extreme behavior

### **4. AGGRESSIVE CLEANING** (Tokens with Gaps)
**Philosophy**: Fix data quality issues comprehensively, gaps are main problem

**Process**:
1. **Remove Listing Artifacts**: Standard artifact detection
2. **Fill Gaps Comprehensively**: Up to 10-minute gaps with appropriate methods
3. **Fix Data Errors**: Standard error correction
4. **Handle Invalid Prices**: Standard price cleaning

**Focus**: Comprehensive gap filling while preserving market movements

## 🔍 **SMART ANOMALY DETECTION METHODS**

### **Listing Artifact Detection**
```python
def _remove_listing_artifacts(self, df):
    # Check first 3 minutes only
    # Criteria: Price >20x median of next 10 minutes
    # AND >99% drop immediately after
    # This preserves real explosive launches
```

### **Data Error vs Legitimate Movement**
```python
def _fix_data_errors(self, df):
    # Data Error: Isolated >100,000% move that reverts immediately
    # Legitimate: Sustained movement with context
    # Only fix clear isolated errors
```

### **Gap Filling Intelligence**
```python
def _fill_gaps_with_size_limit(self, df, max_gap_minutes):
    # 1-minute gaps: Linear interpolation
    # 2-3 minute gaps: Linear interpolation  
    # 4-6 minute gaps: Polynomial interpolation
    # 7-10 minute gaps: Forward fill + linear
    # >10 minute gaps: Leave unfilled (too large)
```

## 📊 **CLEANING LOGIC COMPARISON**

| Issue | Old System | New System |
|-------|------------|------------|
| **5,000% Pump** | ❌ Removed as "extreme" | ✅ Preserved (realistic) |
| **Listing Spike** | ❌ Sometimes kept | ✅ Removed (artifact) |
| **Data Error** | ❌ Fixed with same logic | ✅ Smart detection |
| **61-min Gap** | ❌ Always filled | ✅ Category-dependent |
| **Dead Token** | ❌ Over-cleaned | ✅ Minimal cleaning |
| **Extreme Token** | ❌ Movements removed | ✅ Movements preserved |

## 🎯 **BENEFITS OF NEW SYSTEM**

### **1. Preserves Market Reality**
- Real 1000%+ pumps are kept (they happen in memecoins!)
- Legitimate extreme movements preserved
- Natural volatility maintained

### **2. Removes True Artifacts**
- Listing glitches detected and removed
- Data corruption fixed intelligently
- Technical errors cleaned appropriately

### **3. Category-Appropriate Processing**
- Normal tokens: Gentle, preserve behavior
- Dead tokens: Minimal, don't over-process
- Extreme tokens: Preserve extremes at all costs
- Gappy tokens: Focus on gap filling

### **4. Intelligent Decision Making**
- Context-aware anomaly detection
- Sustained vs isolated movement analysis
- Volume/activity confirmation where available

## 🚨 **CRITICAL SAFEGUARDS**

### **For Extreme Tokens**
- **Never remove movements >1,000% unless >1,000,000%**
- **Higher thresholds for "impossible" values**
- **Preserve defining characteristics**

### **For Normal Tokens**  
- **Preserve natural volatility patterns**
- **Only remove obvious technical artifacts**
- **Maintain realistic market behavior**

### **For All Categories**
- **Comprehensive logging of all changes**
- **Reversible operations where possible**
- **Clear reasoning for each modification**

## 📈 **EXPECTED OUTCOMES**

### **Before (Old System)**
- Legitimate 2000% pumps removed as "anomalies"
- Real market behavior cleaned away
- Over-processed data that doesn't reflect reality
- Same cleaning for all token types

### **After (New System)**
- Realistic extreme movements preserved
- Only true artifacts removed
- Data reflects actual market behavior
- Appropriate cleaning per token category

## 🔧 **USAGE**

### **Clean All Categories**
```python
from clean_tokens import clean_all_categories
summary = clean_all_categories()
```

### **Clean Specific Category**
```python
from clean_tokens import clean_category
result = clean_category('tokens_with_extremes', limit=100)
```

### **Output Structure**
```
data/processed/cleaned/
├── normal_behavior_tokens/     # Gently cleaned
├── dead_tokens/               # Minimally cleaned  
├── tokens_with_extremes/      # Extremes preserved
├── tokens_with_gaps/          # Gaps filled
└── *_cleaning_log.json        # Detailed logs
```

## 🎉 **CONCLUSION**

This new category-aware cleaning system represents a fundamental shift from **one-size-fits-all** to **intelligent, context-aware** data processing. It preserves the reality of memecoin markets while removing true data artifacts, resulting in cleaner data that still reflects legitimate market behavior.

The system is designed to be:
- **Conservative** with extreme movements (preserve them!)
- **Aggressive** with obvious artifacts (remove them!)
- **Intelligent** about context and category
- **Transparent** with comprehensive logging

This ensures that our cleaned data is both high-quality AND representative of actual market dynamics. 