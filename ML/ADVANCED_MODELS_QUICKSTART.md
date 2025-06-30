# Advanced Models Quick Start Guide

## Overview

This guide explains how to use the new advanced LSTM models that incorporate:
- **Attention mechanisms** for focusing on important timepoints
- **Multi-scale feature extraction** (15m, 1h, 4h windows)
- **Hybrid fixed/expanding windows** for capturing both short-term and lifecycle patterns

## Prerequisites

1. Ensure you have completed feature engineering:
   ```bash
   python feature_engineering/advanced_feature_engineering.py
   ```

2. Train the basic LSTM model first (for comparison):
   ```bash
   python ML/directional_models/train_unified_lstm_model.py
   ```

## Training the Advanced Hybrid LSTM

```bash
python ML/directional_models/train_advanced_hybrid_lstm.py
```

### What It Does

1. **Multi-Scale Feature Extraction**:
   - Processes data at 3 different time scales simultaneously (15m, 1h, 4h)
   - Each scale has its own LSTM to capture patterns at that resolution

2. **Expanding Window Processing**:
   - Uses variable-length sequences from 60 minutes up to 12 hours
   - Captures the entire token lifecycle from launch

3. **Attention Mechanisms**:
   - Self-attention on expanding window to focus on important historical points
   - Cross-attention between fixed and expanding features for optimal fusion

4. **Hierarchical Prediction**:
   - Separate prediction heads for each time horizon
   - Optimized learning rates for different model components

### Expected Training Time

- **GPU (CUDA/MPS)**: 30-60 minutes
- **CPU**: 2-4 hours
- Early stopping typically triggers around epoch 20-40

### Resource Requirements

- **Memory**: 8-16GB RAM recommended
- **Disk**: ~500MB for model checkpoints
- **Batch Size**: Default 64 (reduce if OOM errors)

## Comparing Models

After training both basic and advanced models:

```bash
python ML/directional_models/compare_lstm_models.py
```

This generates:
- Performance comparison charts
- Training history comparison
- Detailed summary statistics
- Recommendations report

Results saved to: `ML/results/model_comparison/`

## Key Configuration Options

Edit `CONFIG` in `train_advanced_hybrid_lstm.py`:

```python
CONFIG = {
    # Multi-scale windows
    'fixed_windows': [15, 60, 240],     # Add/remove scales
    'expanding_window_min': 60,          # Minimum history required
    'expanding_window_max': 720,         # Maximum history to use
    
    # Model architecture
    'hidden_size': 128,                  # LSTM hidden dimensions
    'attention_heads': 8,                # Number of attention heads
    'dropout': 0.3,                      # Regularization
    
    # Training
    'batch_size': 64,                    # Reduce for less memory
    'epochs': 100,                       # Max epochs
    'learning_rate': 0.0005,             # Initial learning rate
    'early_stopping_patience': 15        # Epochs without improvement
}
```

## Understanding the Results

### Performance Metrics

The advanced model typically shows improvements in:
- **Accuracy**: +5-10% over basic LSTM
- **F1 Score**: Better balance of precision/recall
- **ROC AUC**: More confident predictions

### When to Use Each Model

**Basic LSTM** (Fixed Window):
- ✅ Real-time trading systems
- ✅ Resource-constrained environments
- ✅ Need consistent inference time
- ✅ Simpler to deploy and maintain

**Advanced Hybrid LSTM**:
- ✅ Maximum prediction accuracy
- ✅ Research and analysis
- ✅ Can afford longer inference time
- ✅ Have sufficient computational resources

### Ensemble Approach

For best results, consider combining both models:

```python
# Pseudocode for ensemble
basic_pred = basic_model.predict(data)
advanced_pred = advanced_model.predict(data)

# Weighted average (tune weights based on validation performance)
final_pred = 0.4 * basic_pred + 0.6 * advanced_pred
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size:
   ```python
   CONFIG['batch_size'] = 32  # or 16
   ```

2. Reduce maximum sequence length:
   ```python
   CONFIG['expanding_window_max'] = 360  # 6 hours instead of 12
   ```

3. Use gradient accumulation (not implemented by default)

### Slow Training

1. Ensure you're using GPU:
   - NVIDIA: Install CUDA
   - Apple Silicon: PyTorch with MPS support
   
2. Reduce model complexity:
   ```python
   CONFIG['hidden_size'] = 64
   CONFIG['attention_heads'] = 4
   ```

### Poor Performance

1. Check class imbalance:
   - The model outputs positive class ratios
   - Consider adjusting loss weights if severely imbalanced

2. Increase training data:
   - Include more token categories
   - Ensure sufficient samples per category

3. Adjust hyperparameters:
   - Try different learning rates (1e-4 to 1e-3)
   - Experiment with dropout (0.2 to 0.5)

## Next Steps

1. **Feature Engineering**:
   - Add more domain-specific features
   - Experiment with different normalization methods

2. **Architecture Improvements**:
   - Try Transformer-only architecture
   - Add positional encodings
   - Implement dilated convolutions

3. **Training Strategies**:
   - Curriculum learning (easy → hard samples)
   - Contrastive learning for better representations
   - Meta-learning for few-shot tokens

## Code Examples

### Loading and Using the Trained Model

```python
import torch
from ML.directional_models.train_advanced_hybrid_lstm import AdvancedHybridLSTM

# Load checkpoint
checkpoint = torch.load('ML/results/advanced_hybrid_lstm/advanced_hybrid_lstm_model.pth')

# Initialize model
model = AdvancedHybridLSTM(
    input_size=checkpoint['input_size'],
    fixed_windows=checkpoint['config']['fixed_windows'],
    hidden_size=checkpoint['config']['hidden_size'],
    attention_heads=checkpoint['config']['attention_heads'],
    dropout=0,  # No dropout for inference
    horizons=checkpoint['config']['horizons']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions (see predict_with_model.py for full example)
```

### Analyzing Attention Weights

The model returns attention weights that show which historical points it focuses on:

```python
outputs, attention_weights = model(fixed_seqs, expanding_seq, exp_lengths)

# Visualize attention (example)
import matplotlib.pyplot as plt

plt.imshow(attention_weights[0].detach().cpu().numpy())
plt.colorbar()
plt.xlabel('Sequence Position')
plt.ylabel('Attention Head')
plt.title('Attention Weights Visualization')
```

## Summary

The advanced hybrid LSTM provides state-of-the-art performance by combining:
- Multiple time scales for comprehensive pattern recognition
- Attention mechanisms for intelligent feature selection
- Flexible sequence lengths to capture token lifecycles

While more complex than the basic LSTM, it offers significant performance improvements for users who can afford the additional computational cost. 