# LSTM Architecture Analysis for Memecoin Prediction

## Overview

This document analyzes different LSTM architectures and their suitability for cryptocurrency price prediction, specifically comparing fixed lookback windows vs expanding windows approaches.

## Current Architecture Summary

### 1. **Directional Models** (Classification Task)
- **Unified LSTM**: Fixed 60-minute lookback window
- **Expanding Window LSTM**: Variable window from token launch (3-240 minutes)
- **Output**: Binary classification (UP/DOWN) for multiple horizons

### 2. **Forecasting Models** (Regression Task)
- **Forecasting LSTM**: Fixed 60-minute lookback window
- **Output**: Continuous price predictions for next N minutes

## Fixed Lookback vs Expanding Windows

### Fixed Lookback Window (Current Default)

**Pros:**
- ✅ **Consistent input size**: Easier to batch and train
- ✅ **Memory efficient**: Fixed memory requirements
- ✅ **Stable training**: Uniform sequence lengths
- ✅ **Fast inference**: Predictable computation time
- ✅ **Good for short-term patterns**: Captures recent momentum

**Cons:**
- ❌ **Limited context**: Misses long-term patterns
- ❌ **No lifecycle awareness**: Can't see token launch behavior
- ❌ **Fixed receptive field**: May be too short or too long

**Best for:**
- High-frequency trading (minutes to hours)
- Stable tokens with consistent patterns
- Real-time prediction systems

### Expanding Windows (Innovation)

**Pros:**
- ✅ **Full lifecycle context**: Sees entire token history
- ✅ **Adaptive to token age**: More data as token matures
- ✅ **Captures launch patterns**: Critical for memecoins
- ✅ **Better for regime changes**: Adapts to market shifts

**Cons:**
- ❌ **Variable sequence lengths**: Complex batching
- ❌ **Memory intensive**: Can grow very large
- ❌ **Slower training**: Variable computation per sample
- ❌ **Risk of overfitting**: Too much historical noise

**Best for:**
- Memecoin lifecycle prediction
- Pump detection from launch
- Long-term trend analysis

## Recommendations for Memecoin Data

### 1. **Hybrid Approach** (Recommended)

```python
class HybridLSTM(nn.Module):
    def __init__(self):
        # Short-term LSTM (fixed 60-min window)
        self.short_term_lstm = nn.LSTM(...)
        
        # Long-term LSTM (expanding window, capped at 6h)
        self.long_term_lstm = nn.LSTM(...)
        
        # Attention mechanism to combine
        self.attention = nn.MultiheadAttention(...)
```

**Benefits:**
- Captures both immediate momentum and lifecycle patterns
- Computationally manageable
- Best of both worlds

### 2. **Multi-Scale Fixed Windows**

Instead of one expanding window, use multiple fixed windows:

```python
windows = [15, 60, 240, 720]  # 15m, 1h, 4h, 12h
features = []
for window in windows:
    features.append(extract_features(data[-window:]))
```

**Benefits:**
- Fixed computation cost
- Multi-scale pattern recognition
- Easier to implement and debug

### 3. **Attention-Based Architecture**

Use Transformer-style attention instead of expanding windows:

```python
class AttentionLSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(...)
        self.self_attention = nn.TransformerEncoder(...)
```

**Benefits:**
- Can "attend" to any part of history
- More parameter efficient
- State-of-the-art performance

## Specific Recommendations

### For Directional Models (Classification)
1. **Keep current unified LSTM** with fixed window for baseline
2. **Enhance expanding window model** with:
   - Maximum cap at 360 minutes (6 hours)
   - Hierarchical pooling for very long sequences
   - Add positional encoding for time awareness

### For Forecasting Models (Regression)
1. **Update to use Winsorization** ✅ (Already implemented)
2. **Add multi-horizon architecture** like directional models
3. **Consider probabilistic outputs** (quantile regression)
4. **Add attention mechanism** for important price levels

### Implementation Priority

1. **High Priority**:
   - Add learning curves to all models ✅
   - Switch to Winsorization for crypto data ✅
   - Add attention mechanisms

2. **Medium Priority**:
   - Implement hybrid fixed/expanding approach
   - Add probabilistic forecasting
   - Multi-scale feature extraction

3. **Low Priority**:
   - Full transformer architecture
   - Advanced position encodings
   - Ensemble methods

## Code Example: Enhanced Forecasting LSTM

```python
class EnhancedForecastingLSTM(nn.Module):
    def __init__(self, 
                 input_size=1,
                 hidden_size=128,
                 num_layers=3,
                 dropout=0.3,
                 forecast_horizons=[15, 30, 60, 120]):
        super().__init__()
        
        # Multi-scale feature extraction
        self.scale_lstms = nn.ModuleList([
            nn.LSTM(input_size, hidden_size//4, batch_first=True)
            for _ in [15, 60, 240]  # Different time scales
        ])
        
        # Main LSTM with larger hidden size
        self.main_lstm = nn.LSTM(
            hidden_size//4 * 3,  # Concatenated multi-scale features
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention for focusing on important timepoints
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads=8,
            dropout=dropout
        )
        
        # Multi-horizon prediction heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size//2, 1)
            ) for _ in forecast_horizons
        ])
        
    def forward(self, x, lengths=None):
        # Extract multi-scale features
        scale_features = []
        for scale_lstm in self.scale_lstms:
            out, _ = scale_lstm(x)
            scale_features.append(out)
        
        # Concatenate scale features
        combined = torch.cat(scale_features, dim=-1)
        
        # Main LSTM processing
        lstm_out, _ = self.main_lstm(combined)
        
        # Apply attention
        attended, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Generate multi-horizon predictions
        final_hidden = attended[:, -1, :]
        predictions = [head(final_hidden) for head in self.horizon_heads]
        
        return torch.cat(predictions, dim=-1)
```

## Conclusion

For memecoin prediction:
1. **Fixed windows** are good for production systems and short-term trading
2. **Expanding windows** capture unique lifecycle patterns but are complex
3. **Hybrid approaches** offer the best balance
4. **Attention mechanisms** are the future direction

The current implementation provides a solid foundation. The next steps should focus on:
- Adding attention mechanisms
- Implementing multi-scale features
- Creating ensemble methods that combine both approaches 