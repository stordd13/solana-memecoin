# Utils Directory

This directory contains utility scripts and tools for analyzing and testing the memecoin data processing pipeline.

## Quick Start

**Run all tests to validate the improved pipeline:**
```bash
python utils/run_all_tests.py
```

**View generated analysis results:**
```bash
python utils/show_results.py
```

**Generate comprehensive variability analysis:**
```bash
python utils/variability_analysis/analyze_token_variability.py
```

## Directory Structure

### `variability_analysis/`
Tools for analyzing token price variability to distinguish real market movements from "straight line" tokens.

- `analyze_token_variability.py` - Comprehensive analysis across token categories
- `examine_individual_tokens.py` - Detailed individual token examination with plots

### `corruption_detection/`
Tools for testing and analyzing the improved corruption detection that distinguishes legitimate massive pumps from staircase artifacts.

- `test_specific_extreme_token.py` - Test extreme corruption cases
- `test_improved_corruption_detection.py` - Test multiple tokens
- `examine_specific_token.py` - General pattern analysis tool

### Root Utils

- `run_all_tests.py` - **Main test runner** - validates all improvements
- `show_results.py` - **Results viewer** - shows summary of generated analysis files
- `results/` - **Output directory** - all plots and analysis results saved here

## Key Improvements Validated

✅ **Legitimate massive pumps are preserved** (e.g., 6,371% with continued volatility)  
✅ **Staircase artifacts are detected and removed** (e.g., 48 billion% with 0% volatility)  
✅ **Temporal pattern analysis** distinguishes real vs fake movements  
✅ **Multi-granularity approach** considers post-move volatility patterns  

## Usage

Each subdirectory contains specialized tools with their own documentation. These utilities are designed to help with:

- **Data Quality Analysis**: Understanding token behavior patterns
- **Algorithm Testing**: Validating cleaning and filtering logic
- **Visual Analysis**: Generating plots and charts for inspection
- **Performance Validation**: Testing pipeline improvements

## Integration

These tools are designed to work with the main memecoin data processing pipeline and can be run independently for analysis and debugging purposes. The variability analysis tools are also integrated into the main Streamlit app under the "Variability Analysis" section.