# Corruption Detection Testing Tools

Tools for testing and validating the improved corruption detection algorithm that distinguishes between legitimate massive pumps and staircase artifacts.

## Scripts

### `test_improved_corruption_detection.py`
**Comprehensive testing of the improved corruption detection logic**

- Tests multiple tokens from different categories
- Compares behavior on legitimate pumps vs staircase artifacts
- Provides summary statistics of filtering decisions
- Validates the temporal pattern analysis approach

**Usage:**
```bash
python utils/corruption_detection/test_improved_corruption_detection.py
```

**Output:**
- Console analysis for each tested token
- Summary table showing price ratios, max returns, and modification status
- Statistics on preserved pumps vs removed artifacts

### `test_specific_extreme_token.py`
**Detailed analysis of extreme corruption cases**

- Focuses on tokens with massive (>1000%) single-minute returns
- Analyzes post-move volatility patterns
- Demonstrates staircase artifact detection in action
- Shows detailed decision-making process

**Usage:**
```bash
python utils/corruption_detection/test_specific_extreme_token.py
```

**Key Features:**
- Targets tokens with known extreme corruption
- Detailed pattern analysis (10-minute post-move window)
- Step-by-step decision explanation
- Modification tracking and reporting

### `examine_specific_token.py`
**General-purpose token examination for pattern analysis**

- Analyzes any specified token's price patterns
- Generates comprehensive plots (price, returns, rolling volatility)
- Classifies extreme moves as legitimate vs staircase
- Provides visual and statistical analysis

**Usage:**
```bash
python utils/corruption_detection/examine_specific_token.py
```

**Output:**
- `utils/results/pattern_analysis_[token_name].png` - High-resolution analysis plots
- Pattern classification results
- Detailed console statistics

## Algorithm Logic

### Improved Corruption Detection
The enhanced algorithm uses temporal pattern analysis instead of simple magnitude thresholds:

#### Previous Approach (Problematic)
- ❌ Only looked at single-minute return magnitude (>500,000%)
- ❌ Would remove legitimate multi-minute pumps
- ❌ Missed staircase artifacts with smaller single-minute jumps

#### New Approach (Improved)
- ✅ Analyzes extreme moves (>1000%) with temporal context
- ✅ Examines 10-minute post-move volatility patterns
- ✅ Distinguishes patterns:
  - **Legitimate Pumps**: High volatility continues (>0.5% avg)
  - **Staircase Artifacts**: Extreme flatness follows (<0.5% avg, <1% CV, <2% max)

### Pattern Classification Criteria

**🚀 LEGITIMATE PUMP** (preserved):
- Large single-minute return (>1000%)
- **AND** continued high volatility in next 10 minutes
- Indicates real trading activity

**🪜 STAIRCASE ARTIFACT** (removed):
- Large single-minute return (>1000%)
- **AND** extremely low post-move volatility
- Indicates instant data corruption

## Real-World Results

The improved algorithm successfully:
- ✅ **Preserved** 6,371% pump with 66% continued volatility
- ✅ **Removed** 48 billion% artifact with 0% continued volatility  
- ✅ **Kept** 3,070x gains over several minutes (eL5fUxj2J4CiQsmW85k5FG9DvuQjjUoBHoQBi2Kpump)

## Integration

This logic is implemented in `data_cleaning/clean_tokens.py` in the `_fix_extreme_data_corruption()` method and is automatically applied during the data cleaning pipeline.