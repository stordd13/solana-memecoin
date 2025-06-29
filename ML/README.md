# 🤖 Machine Learning Models for Memecoin Price Prediction

This comprehensive guide documents all machine learning models in the ML pipeline, covering architectures, data handling, feature engineering, and training strategies in complete detail.

## 📁 **Directory Structure**

```
ML/
├── directional_models/          # Binary classification (UP/DOWN prediction)
├── forecasting_models/         # Regression (price value prediction)  
├── utils/                      # Shared utilities
└── results/                    # Model outputs and metrics
```

---

## 🎯 **DIRECTIONAL MODELS** (Binary Classification)

These models predict **direction** of price movement (UP/DOWN) over time horizons.

### **🚀 Data Processing Pipeline**

All directional models follow this pipeline:
1. **Data Analysis** → **Categorization** → **Cleaning** → **Feature Engineering** → **ML Training**
2. **Mutually Exclusive Categories**: Each token in exactly ONE category
3. **Category Hierarchy**: `gaps > normal > extremes > dead`

---

## 📊 **1. LightGBM Models**

### **Files**
- `train_lightgbm_model.py` - Short-term (15m, 30m, 1h)
- `train_lightgbm_model_medium_term.py` - Medium-term (2h, 4h, 6h, 12h)

### **Architecture**
- **Algorithm**: Gradient Boosting Decision Trees
- **Input**: Pre-engineered feature vectors (flat, not sequences)
- **Output**: Binary probabilities per horizon
- **Strategy**: Separate model per horizon

### **Data Pipeline**

#### **Feature Loading**
```python
CONFIG = {
    'features_dir': Path("data/features"),
    'categories': [
        "normal_behavior_tokens",    # Highest quality
        "tokens_with_extremes",      # Volatile patterns  
        "dead_tokens",               # Complete lifecycles
        # "tokens_with_gaps"         # EXCLUDED
    ]
}
```

Features are **pre-engineered** by `feature_engineering/advanced_feature_engineering.py`:

**Technical Indicators**:
- RSI (14-period)
- MACD (Moving Average Convergence Divergence)  
- Bollinger Bands (upper, lower, width)
- ATR (Average True Range)

**Price Features**:
- Log returns (1, 5, 15, 30 minute lags)
- Rolling means (5, 10, 15, 30, 60 minute windows)
- Rolling standard deviations
- Price momentum indicators

**Statistical Features**:
- Skewness, kurtosis of price distributions
- Value at Risk (VaR) calculations
- Higher-order statistical moments

**FFT Analysis**:
- Frequency domain features
- Cyclical pattern detection
- Spectral density measures

#### **Temporal Data Splitting** 🔥
**CRITICAL**: Prevents data leakage by splitting WITHIN each token:

```python
def prepare_data_fixed(data_paths, horizons, split_type):
    for token_path in data_paths:
        features_df = load_features_from_file(token_path)
        n_rows = features_df.height
        
        if split_type == 'train':
            start_idx, end_idx = 0, int(n_rows * 0.6)          # First 60%
        elif split_type == 'val':
            start_idx, end_idx = int(n_rows * 0.6), int(n_rows * 0.8)  # Next 20%
        elif split_type == 'test':
            start_idx, end_idx = int(n_rows * 0.8), n_rows     # Last 20%
        
        token_split = features_df.slice(start_idx, end_idx - start_idx)
```

**Why This Matters**:
- ✅ No future leakage: Training always from EARLIER periods
- ✅ Realistic evaluation: Mimics real-world trading
- ✅ Per-token integrity: Each token contributes based on TIME

#### **Label Creation**
```python
def create_labels_for_horizons(df, horizons):
    for h in horizons:
        df = df.with_columns([
            (pl.col('price').shift(-h) > pl.col('price')).alias(f'label_{h}m')
        ])
    return df
```

#### **Data Leakage Validation** 🛡️
```python
def validate_features_safety(features_df, token_name):
    unsafe_patterns = [
        'total_return', 'max_gain', 'max_drawdown', 'price_range',
        'global_', 'spectral_entropy', 'max_periodicity'
    ]
    
    for col in features_df.columns:
        unique_count = features_df[col].n_unique()
        if unique_count == 1:
            unique_val = features_df[col].drop_nulls().unique().to_list()[0]
            if abs(unique_val) < 1e-10:
                continue  # Low variability from dead token - acceptable
            else:
                return False  # Truly constant - data leakage risk
```

### **Training Configuration**
```python
CONFIG = {
    'lgb_params': {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'n_estimators': 500
    }
}
```

---

## 🧠 **2. LSTM Models**

### **Files**
- `train_unified_lstm_model.py` - Fixed-window LSTM
- `train_unified_lstm_expanding_windows.py` - Variable expanding windows

### **Architecture**
- **Algorithm**: Long Short-Term Memory Neural Networks
- **Input**: Sequential feature data (time series)
- **Output**: Binary probabilities for all horizons simultaneously
- **Strategy**: Unified model predicting all horizons

### **Data Pipeline**

#### **Per-Token Scaling** 🔑
```python
class UnifiedDirectionalDataset(Dataset):
    def _load_data(self, data_paths):
        for path in data_paths:
            df = pl.read_parquet(path)
            
            # PER-TOKEN ROBUST SCALING
            feature_cols = [c for c in df.columns if c not in ['datetime', 'price']]
            feature_matrix = df[feature_cols].to_numpy()
            
            # Individual scaler per token
            scaler = RobustScaler()
            scaler.fit(feature_matrix)
            
            # Critical fix for zero IQR
            zero_iqr_mask = np.isclose(scaler.scale_, 0)
            if np.any(zero_iqr_mask):
                scaler.scale_[zero_iqr_mask] = 1.0
            
            scaled_features = scaler.transform(feature_matrix)
```

**Why Per-Token Scaling?**
- ✅ Handles price variations across tokens
- ✅ Preserves relative movements within tokens
- ✅ Robust to outliers (uses median/IQR)
- ✅ Prevents zero division errors

#### **Sequence Creation**

**Standard LSTM (Fixed Windows)**:
```python
def _create_sequences(self, feature_matrix, prices, token_id, scaler):
    for i in range(self.sequence_length, len(feature_matrix)):
        # Fixed 60-minute lookback
        feature_seq = scaled_features[i-self.sequence_length:i]  # (60, n_features)
        
        # Labels for all horizons
        labels = {}
        for h in self.horizons:
            if i + h < len(prices):
                current_price = prices[i]
                future_price = prices[i + h]
                labels[h] = 1 if future_price > current_price else 0
        
        sequences.append((feature_seq, labels, token_id))
```

**Expanding Windows LSTM**:
```python
def _create_expanding_sequences(self, feature_matrix, prices, token_id, scaler):
    for i in range(self.min_sequence_length, len(feature_matrix)):
        # Variable-length sequence (ALL history from launch)
        start_idx = max(0, i - self.max_sequence_length)
        feature_seq = scaled_features[start_idx:i]  # (variable, n_features)
        actual_length = len(feature_seq)
        
        sequences.append((feature_seq, labels, token_id, actual_length))
```

### **Model Architecture**

**Unified LSTM**:
```python
class UnifiedLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2):
        super().__init__()
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # Separate prediction heads for each horizon
        self.prediction_heads = nn.ModuleDict({
            f'{h}m': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ) for h in horizons
        })
```

**Expanding Windows LSTM (with Attention)**:
```python
class ExpandingWindowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True
        )
        
        # Multi-head attention for long sequences
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, x, lengths):
        # Pack sequences for efficiency
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, _ = self.lstm(packed_x)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Apply attention
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Use last valid output for each sequence
        final_outputs = []
        for i, length in enumerate(lengths):
            final_outputs.append(attn_output[i, length-1, :])
        
        final_output = torch.stack(final_outputs)
        
        # Multi-horizon prediction
        predictions = {}
        for horizon, head in self.prediction_heads.items():
            predictions[horizon] = head(final_output)
        
        return predictions
```

### **Loss Function: Focal Loss**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()
```

**Why Focal Loss?**
- ✅ Handles class imbalance
- ✅ Focuses on hard examples
- ✅ Better than BCE for memecoin data

---

## 📊 **3. Logistic Regression Baseline**

### **Architecture**
- **Algorithm**: Logistic Regression with L2 regularization
- **Input**: Pre-engineered features (same as LightGBM)
- **Purpose**: Simple baseline for comparison

### **Per-Token Scaling**
```python
def prepare_and_split_data():
    token_scalers = {}
    
    for path in token_paths:
        df = pl.read_parquet(path)
        train_split_df = df.slice(0, int(0.6 * df.height))
        
        X_train_token = train_split_df[feature_cols].to_numpy()
        
        scaler = RobustScaler()
        scaler.fit(X_train_token[valid_rows])
        
        # Fix zero IQR
        zero_iqr_mask = np.isclose(scaler.scale_, 0)
        if np.any(zero_iqr_mask):
            scaler.scale_[zero_iqr_mask] = 1.0
        
        token_scalers[path.stem] = scaler
```

---

## 📈 **FORECASTING MODELS** (Regression)

These models predict **actual price values** at future time points.

## 🧠 **4. LSTM Forecasting Model**

### **Data Pipeline**

#### **Global Scaling**
```python
class MemecoinDataset(Dataset):
    def __init__(self, data_paths, lookback=60, forecast_horizon=15):
        # Global scaling across all tokens
        all_prices = []
        for path in data_paths:
            df = pl.read_parquet(path)
            all_prices.extend(df['price'].to_list())
        
        self.scaler = StandardScaler()
        self.scaler.fit(np.array(all_prices).reshape(-1, 1))
```

#### **Sequence Creation**
```python
def _create_sequences(self, prices, token_name, category):
    scaled_prices = self.scaler.transform(prices.reshape(-1, 1)).flatten()
    
    for i in range(self.lookback, len(scaled_prices) - self.forecast_horizon):
        input_seq = scaled_prices[i-self.lookback:i]  # 60-minute window
        target_price = scaled_prices[i + self.forecast_horizon]  # Future price
        
        self.sequences.append((input_seq, target_price, token_name, category))
```

### **Model Architecture**
```python
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True
        )
        
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)  # Single price output
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
        
        lstm_out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        
        prediction = self.forecast_head(last_hidden)
        return prediction.squeeze(-1)
```

## 📊 **5. Baseline Regressors**

### **Linear Regression**
```python
def build_dataset(horizon):
    for path in token_paths:
        df = pl.read_parquet(path)
        
        # Create regression target
        df = df.with_columns([
            (pl.col('price').shift(-horizon)).alias('target_price')
        ])
        
        # Per-token scaling
        scaler = RobustScaler()
        scaler.fit(X_train_token)
        X_scaled = scaler.transform(X_full)
```

### **XGBoost Regression**
```python
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)
```

---

## 🛡️ **DATA LEAKAGE PREVENTION**

### **Critical Fixes Implemented**

#### **1. Temporal Splitting Within Tokens**
```python
# WRONG: Random token splits
train_tokens, test_tokens = train_test_split(all_tokens, test_size=0.2)

# CORRECT: Temporal splits within each token
for token in all_tokens:
    token_data = load_token(token)
    n_rows = len(token_data)
    
    train_split = token_data[:int(0.6 * n_rows)]      # First 60%
    val_split = token_data[int(0.6 * n_rows):int(0.8 * n_rows)]  # Next 20%
    test_split = token_data[int(0.8 * n_rows):]       # Last 20%
```

#### **2. Per-Token Scaling**
```python
# WRONG: Global scaling
all_prices = concatenate([token.prices for token in tokens])
global_scaler.fit(all_prices)

# CORRECT: Individual scaling per token
for token in tokens:
    token_scaler = RobustScaler()
    token_scaler.fit(token.train_prices)  # Only training data
```

#### **3. Dead Token Constant Period Removal**
```python
def _remove_death_period(df, token_name):
    """Remove constant price periods to prevent data leakage"""
    constant_count = count_constant_prices_at_end(df)
    
    if constant_count >= 60:  # 1+ hour of constant prices
        remove_count = constant_count - 2  # Keep 2 minutes for context
        df_cleaned = df.head(df.height - remove_count)
        return df_cleaned
```

---

## 📊 **PERFORMANCE METRICS**

### **Classification Metrics (Directional)**
```python
def classification_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),      # PRIMARY metric
        'precision': precision_score(y_true, y_pred),    # When predict UP, % correct
        'recall': recall_score(y_true, y_pred),          # % of actual UPs caught
        'f1_score': f1_score(y_true, y_pred),           # Harmonic mean
        'roc_auc': roc_auc_score(y_true, y_prob)        # Overall discrimination
    }
```

### **Regression Metrics (Forecasting)**
```python
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred)
    }
```

### **Expected Performance**

**Directional Models**:
- **Accuracy**: 85-90% (excellent for balanced data)
- **ROC AUC**: 85-95% (strong discrimination)
- **F1 Score**: 60-80% (balanced precision/recall)

**Forecasting Models**:
- **R²**: 0.3-0.7 (captures 30-70% of variance)
- **MAE**: 5-15% of average price
- **RMSE**: 10-25% of average price

---

## 🚀 **RUNNING THE MODELS**

### **Prerequisites**
```bash
# 1. Data analysis and categorization
streamlit run data_analysis/app.py
# → Use "Export All Categories (Mutually Exclusive)" button

# 2. Data cleaning
python data_cleaning/clean_tokens.py

# 3. Feature engineering (CRITICAL for LightGBM/LogReg)
python feature_engineering/advanced_feature_engineering.py

# 4. Install dependencies
pip install polars lightgbm torch scikit-learn plotly tqdm xgboost
```

### **Training Commands**

**Directional Models**:
```bash
python ML/directional_models/train_lightgbm_model.py
python ML/directional_models/train_lightgbm_model_medium_term.py
python ML/directional_models/train_unified_lstm_model.py
python ML/directional_models/train_unified_lstm_expanding_windows.py
python ML/directional_models/train_logistic_regression_baseline.py
```

**Forecasting Models**:
```bash
python ML/forecasting_models/train_lstm_model.py
python ML/forecasting_models/train_baseline_regressors.py --horizon 60 --model both
```

---

## 🔧 **CONFIGURATION**

### **LightGBM**
```python
CONFIG = {
    'lgb_params': {
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'n_estimators': 500,
    }
}
```

### **LSTM**
```python
CONFIG = {
    'sequence_length': 60,          # Lookback window
    'hidden_size': 32,              # LSTM capacity
    'num_layers': 2,                # LSTM depth
    'dropout': 0.2,                 # Regularization
    'batch_size': 32,
    'learning_rate': 0.001,
}
```

---

## 🎉 **MODEL COMPARISON**

| Model | Type | Input | Strengths | Weaknesses |
|-------|------|-------|-----------|------------|
| **LightGBM** | Tree-based | Pre-engineered features | Fast, interpretable | Requires feature engineering |
| **LSTM Standard** | Neural | Feature sequences | Learns temporal patterns | Needs more data |
| **LSTM Expanding** | Neural | Variable sequences | Uses all history | Complex, memory intensive |
| **Logistic Regression** | Linear | Pre-engineered features | Simple, interpretable | Limited capacity |
| **LSTM Forecasting** | Neural | Price sequences | Direct price prediction | Harder to evaluate |

This ML pipeline provides a comprehensive approach to memecoin prediction with rigorous data leakage prevention, category-aware processing, and multiple model architectures for robust evaluation. 