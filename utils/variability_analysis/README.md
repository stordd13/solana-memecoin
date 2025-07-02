# Token Variability Analysis Tools

Tools for analyzing price patterns to distinguish real market variations from "straight line" tokens.

## Scripts

### `analyze_token_variability.py`
**Comprehensive variability analysis across token categories**

- Analyzes sample tokens from each category (normal, dead, extremes, gaps)
- Generates detailed plots showing variability distributions
- Compares filtering effectiveness across categories
- Outputs comprehensive analysis plots and statistics

**Usage:**
```bash
python utils/variability_analysis/analyze_token_variability.py
```

**Output:**
- `utils/results/token_variability_analysis.png` - Comprehensive analysis plots
- Console output with filtering statistics by category

### `examine_individual_tokens.py`
**Detailed analysis of specific tokens with plots and metrics**

- Examines individual tokens with detailed price and variability analysis
- Generates 4-panel plots: raw price, log price, returns distribution, rolling CV
- Provides decision explanation (filtered vs passed)
- Interactive tool for understanding why tokens are filtered

**Usage:**
```bash
python utils/variability_analysis/examine_individual_tokens.py
```

**Key Features:**
- Automatic token discovery from processed categories
- Interactive examination with pause between tokens
- Detailed metrics display with threshold comparisons
- High-resolution plot generation

## Variability Metrics

Both tools use the same comprehensive variability analysis:

### Core Metrics
- **Price CV**: Coefficient of variation for price data
- **Log Price CV**: CV for log-transformed prices (relative movements)
- **Flat Periods**: Fraction of time with minimal price change (<0.1%)
- **Range Efficiency**: Proportion of meaningful price moves (>1%)
- **Normalized Entropy**: Information content in price movements

### Filtering Thresholds
A token is filtered (low variability) if ALL conditions are met:
- Price CV < 0.05
- Log Price CV < 0.1  
- Flat periods > 0.8 (80%)
- Range efficiency < 0.1
- Normalized entropy < 0.3

## Integration with Streamlit

These tools are integrated into the main Streamlit app (`data_analysis/app.py`) under the "Variability Analysis" section, providing an interactive web interface for the same functionality.