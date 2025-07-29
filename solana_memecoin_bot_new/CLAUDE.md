# 🤖 Claude Development Guide for Solana Memecoin Trading Bot

> **For Future Claude Instances**: This guide provides comprehensive context for the automated ML/RL-powered trading bot focused on Solana memecoins in their first 24 hours post-launch.

**GitHub Repository**: https://github.com/stordd13/solana-memecoin  
**Working Directory**: `solana_memecoin_bot_new/`

---

## 🎯 **Core Goal & Mission**

Build an **automated, ML/RL-powered trading bot** for Solana memecoins with emphasis on:
- **First 24 hours post-launch** (highest volatility period)
- **High-frequency, low-stake trades** ($5-50 each)
- **Low-liquidity environments** to exploit volatility while minimizing slippage/risk
- **Beat sniper bots** by predicting pumps via patterns, volume, and on-chain data

### **Target Performance Metrics**
- **Win Rate**: >60%
- **Sharpe Ratio**: >1.5
- **Max Drawdown**: <5% per session
- **Pipeline Runtime**: <1 minute for 30k+ tokens
- **Pump Prediction F1**: >30% (baseline achieved: 22.5% with 10% threshold)

---

## 🏗️ **Project Architecture**

### **Current Structure (Updated)**
```
solana_memecoin_bot_new/
├── config.py                   # Configuration and parameters (PUMP_RETURN_THRESHOLD = 0.10)
├── scripts/                    # Core pipeline scripts
│   ├── run_pipeline.py        # Original pipeline with archetypes
│   ├── run_pipeline2.py       # Alternative pipeline (SQLite)
│   ├── run_pipeline3.py       # 🆕 NEW: Unified pipeline (NO archetypes)
│   ├── death_detection.py     # Token death identification
│   ├── feature_engineering.py # Feature creation (ACF, RSI, MACD)
│   ├── real_time_update.py    # Real-time data updates
│   ├── setup_db.py           # Database initialization
│   └── utils.py              # Helper functions
├── ml/                        # Machine learning models
│   ├── run_baseline.py        # 🆕 NEW: Enhanced baseline XGBoost training
│   ├── archetype_clustering.py # K-means behavioral clustering (DEPRECATED)
│   ├── baseline_models.py     # Original XGBoost pump prediction
│   ├── transformer_forecast.py # 🆕 UPDATED: Unified transformer with rolling windows
│   └── rl_agent.py           # PPO reinforcement learning (TO BE UPDATED)
├── analysis/                  # Analysis and visualization
│   ├── eda.py                # Exploratory data analysis
│   ├── death_distribution.py  # Death pattern analysis
│   ├── trading_sim.py        # Trading simulation framework (TO BE UPDATED)
│   ├── 5m/                   # 🆕 NEW: Results organized by interval
│   │   └── baseline_results.json
│   └── 1m/                   # 🆕 NEW: Results organized by interval
├── models/                   # 🆕 NEW: Trained model storage
├── data/                     # Processed data storage
└── tests/                    # Test suite (200+ tests goal)
```

### **Data Flow (Updated)**

**🔴 OLD (Archetype-Based):**
```
Raw Data → Death Detection → Resample (1m/5m) → Feature Engineering → 
→ Archetype Clustering → ML Training (per archetype) → RL Agent → Trading Sim
```

**🟢 NEW (Unified Approach):**
```
Raw Data → Death Detection → Resample (1m/5m) → Feature Engineering → 
→ Token Metadata → Rolling Window Sequences → Unified ML Training → RL Agent → Trading Sim
```

### **Key Design Principles (Updated)**
1. **Polars-Only**: NO Pandas - Polars is faster and memory-efficient
2. **Per-Token Processing**: Handle variable lifespans (200-2000 minutes)
3. **Temporal Splitting**: Prevent data leakage (80/20 train/test by tokens)
4. **🆕 Unified Cross-Token Models**: Single models trained on all tokens (better generalization)
5. **🆕 Rolling Window Training**: 5-60 minute sequences for realistic trading simulation
6. **Death-Aware**: Filter dead tokens early to improve ML quality
7. **🆕 10% Pump Threshold**: Realistic pump detection (reduced from 50%)

---

## 📋 **Key Constraints & Rules**

### **Data Handling**
- **Use Polars exclusively** - no Pandas (already in pipeline)
- **Handle 30k+ tokens** efficiently (<1 min full pipeline)
- **Resample to 2-5 min intervals** due to API limits
- **Incorporate upcoming features**: volume, buy/sell counts, liquidity
- **Always prevent data leakage** with temporal splitting

### **ML/RL Focus (Updated)**
- **🚫 Dropped archetype clustering** - added complexity without clear benefits
- **🆕 Unified cross-token models** - better generalization and performance
- **🆕 Advanced Transformers** for sequence forecasting (128-dim, 8 heads, 3 layers)
- **🆕 Rolling window training** - 5,10,15,20,30 minute sequences
- **RL (PPO/A2C)** for trading decisions using transformer predictions
- **Proven baseline**: XGBoost achieved F1=0.225, AUC=0.770

### **Tech Stack Requirements**
- **Languages**: Python 3.12+ only
- **Core Libraries**: 
  - Polars (data processing)
  - PyTorch (DL/Transformers/RL)
  - Stable-Baselines3 or RLlib (RL)
  - XGBoost/LightGBM (baselines)
- **NO new installs** - use available tools (torch, networkx from environment)
- **Testing**: Maintain 200+ tests; aim for 100% coverage

### **Solana-Specific Constraints**
- **Simulate low liquidity**: 1-5% slippage
- **Fast transactions**: Sub-second execution
- **Rug detection**: Identify and avoid scams
- **RPC APIs**: Use QuickNode or similar for real-time data

### **Code Quality Standards**
- **Review and refactor** poorly written code from previous AI
- **Modular, readable code** with comprehensive docstrings
- **Fix existing issues**:
  - Class imbalance in pump prediction
  - Low F1 score (22%)
  - Marathon misclassification (74%)

### **Risk Management & Ethics**
- **Kelly criterion** for position sizing
- **Stop losses** on all positions
- **No market manipulation** - only exploit existing inefficiencies
- **Log all simulations/trades** for audit trail
- **Simulate fees**: 0.1-1% transaction costs

---

## 🔧 **Current Pipeline Overview**

### **1. Data Loading & Preprocessing**
```python
# config.py key parameters
ZERO_THRESHOLD = 120  # 2-hour death detection
DUMP_RETURN_THRESHOLD = -0.1  # 10% drop for dumps
EARLY_MINUTES = 10  # Early feature window
```

### **2. Death Detection** (`death_detection.py`)
- **Backward check**: Find constant price periods from end
- **120-minute threshold**: Mark tokens with 2+ hours of no movement
- **Export**: `data/processed/death_summary.parquet`

### **3. Feature Engineering** (`feature_engineering.py`)
- **Standard features**: ~20 including:
  - ACF lags [1, 2, 5, 10, 20, 60]
  - RSI (14-period)
  - MACD (12, 26, 9)
  - Rolling statistics (mean, std, skew, kurt)
  - Momentum indicators
- **Early features**: Optimized for 10-minute windows
- **Dump detection**: Flag tokens with <-10% in first 2 minutes

### **4. Archetype Clustering** (`archetype_clustering.py`)
- **Automatic K selection**: Silhouette score + elbow method
- **Two-stage approach**:
  1. Early clustering (first 10 minutes)
  2. Full clustering (all features)
- **Dual-track scaling**: Only scale volatility, preserve raw returns

### **5. ML Models**

#### **Baseline** (`baseline_models.py`)
- XGBoost classifier for pump prediction (>50% return)
- Per-archetype training with class balancing
- Current performance: ~22% F1 (needs improvement)

#### **Transformer** (`transformer_forecast.py`)
- Custom architecture for price sequence forecasting
- Walk-forward validation (5 steps)
- 10-minute input sequences

#### **RL Agent** (`rl_agent.py`)
- PPO-based trading agent
- Actions: Hold, Buy, Sell
- Custom gym environment with transformer forecasts
- Per-token episodes with realistic fees/slippage

### **6. Trading Simulation** (`trading_sim.py`)
- Hybrid strategy: RL actions + transformer forecasts
- Tracks P&L, win rate, Sharpe ratio, drawdown
- Focuses on high-volatility archetypes (e.g., "Sprint")

---

## 🚀 **Development Workflow**

### **Running the Pipeline**
```bash
# Main pipeline (recommended)
python scripts/run_pipeline.py

# Alternative pipeline (from SQLite)
python scripts/run_pipeline2.py

# Real-time updates
python scripts/real_time_update.py
```

### **Analysis & Visualization**
```bash
# Generate EDA plots
python analysis/eda.py

# Analyze death patterns
python analysis/death_distribution.py

# Run trading simulation
python analysis/trading_sim.py
```

### **ML Training**
```bash
# Train baseline XGBoost
python ml/baseline_models.py

# Train transformer forecaster
python ml/transformer_forecast.py

# Train RL agent
python ml/rl_agent.py
```

---

## 🔍 **Current Issues & Fixes Needed**

### **1. Low Pump Prediction Performance**
- **Current**: 22% F1 score
- **Target**: >35% F1 score
- **Solutions**:
  - Better class balancing (SMOTE, focal loss)
  - More sophisticated features (volume, on-chain data)
  - Ensemble methods combining XGBoost + Transformer

### **2. Marathon Misclassification**
- **Current**: 74% of marathons misclassified as standard
- **Solutions**:
  - Improve archetype clustering with more features
  - Add temporal pattern recognition
  - Use attention mechanisms to identify long-term patterns

### **3. Memory Optimization**
- **Current**: Aggressive GC but still memory-intensive
- **Solutions**:
  - Implement chunked processing
  - Use Polars lazy evaluation more effectively
  - Consider distributed processing for 100k+ tokens

### **4. Real-Time Integration**
- **Current**: Batch processing only
- **Needed**:
  - WebSocket integration for live price feeds
  - Queue-based architecture for new tokens
  - Sub-second decision making

---

## 📊 **RL Integration Roadmap**

### **Phase 1: Enhanced Feature Engineering** (Week 1)
- [ ] Integrate volume data when available
- [ ] Add on-chain metrics (holder count, liquidity)
- [ ] Implement advanced technical indicators
- [ ] Create archetype-specific feature sets

### **Phase 2: Transformer Development** (Week 2)
- [ ] Design attention-based architecture for sequences
- [ ] Implement multi-scale temporal features
- [ ] Add positional encoding for time awareness
- [ ] Create ensemble with XGBoost predictions

### **Phase 3: RL Agent Training** (Week 3-4)
- [ ] Implement PPO with custom reward function
- [ ] Add A2C as alternative algorithm
- [ ] Create realistic market simulator
- [ ] Train separate agents per archetype

### **Phase 4: Production Integration** (Week 5)
- [ ] Real-time data pipeline
- [ ] Risk management layer
- [ ] Performance monitoring dashboard
- [ ] A/B testing framework

### **Phase 5: Optimization & Scaling** (Week 6+)
- [ ] Hyperparameter optimization with Optuna
- [ ] Distributed training for large datasets
- [ ] Edge case handling (rugs, low liquidity)
- [ ] Portfolio-level optimization

---

## 💡 **Best Practices & Guidelines**

### **✅ Always Do (Updated)**
- Use Polars for all data operations
- Implement temporal splitting (80/20 by tokens, not sequences)
- Scale features per token
- **🆕 Train unified models** across all tokens (better generalization)
- **🆕 Use rolling windows** for realistic trading simulation
- Log all experiments with metrics
- Write tests for new features
- Document architectural decisions

### **❌ Never Do (Updated)**
- Use Pandas (Polars only!)
- **🚫 Don't use archetype clustering** (adds complexity without clear benefits)
- Use future data in features
- Ignore extreme volatility (it's signal!)
- Deploy without backtesting
- Trade without risk limits
- **🚫 Don't use 50% pump thresholds** (use 10% for realistic detection)

### **Code Style**
```python
# Good: Clear, modular, tested
def calculate_acf_features(prices: pl.Series, lags: List[int]) -> Dict[str, float]:
    """Calculate autocorrelation features at specified lags.
    
    Args:
        prices: Price series (Polars)
        lags: List of lag values
        
    Returns:
        Dictionary of ACF values keyed by lag
    """
    # Implementation with proper error handling
    
# Bad: Unclear, monolithic, untested
def proc_data(df):
    # 500 lines of unmaintainable code
```

---

## 🎯 **Success Criteria**

### **Technical Metrics (Updated)**
- [x] **Pipeline runtime <1 minute for 30k tokens** ✅ ACHIEVED
- [x] **Pump prediction F1 >20%** ✅ ACHIEVED (22.5% with XGBoost baseline)
- [ ] **Advanced transformer F1 >30%** (target with new architecture)
- [x] **Memory usage <16GB for full pipeline** ✅ ACHIEVED
- [ ] Test coverage >95%
- [x] **Results organization by interval** ✅ ACHIEVED

### **Trading Performance (Updated)**
- [ ] Win rate >55% in backtesting (revised realistic target)
- [ ] Sharpe ratio >1.2 (revised realistic target)
- [ ] Maximum drawdown <10% (revised realistic target)
- [ ] Profitable after fees/slippage
- [ ] **Consistent performance across 1m vs 5m intervals** (NEW REQUIREMENT)

### **Production Readiness**
- [ ] Real-time data integration working
- [ ] Risk management implemented
- [ ] Monitoring/alerting in place
- [ ] Deployment documentation complete
- [ ] Disaster recovery plan tested

---

## 🚨 **Common Pitfalls & Solutions**

### **Data Leakage**
- **Problem**: Using future data in features
- **Solution**: Strict temporal splits, validate feature timing

### **Overfitting to Backtests**
- **Problem**: Perfect backtest, poor live performance
- **Solution**: Walk-forward validation, out-of-sample testing

### **Liquidity Assumptions**
- **Problem**: Assuming infinite liquidity
- **Solution**: Model slippage, limit position sizes

### **Class Imbalance**
- **Problem**: 95% tokens don't pump
- **Solution**: Focal loss, SMOTE, per-archetype thresholds

---

## 🚀 **Current Status & Recent Achievements**

### **✅ Major Breakthrough: Unified Approach (Latest Session)**

**Key Decision**: Dropped archetype clustering in favor of unified cross-token models after baseline achieved decent performance (F1=0.225, AUC=0.770) but archetype complexity wasn't providing clear benefits.

#### **What We've Accomplished:**

1. **🔧 Fixed Critical Data Issues**:
   - **Reduced pump threshold** from 50% to 10% (increased pump rate from 0.5% to ~2.1%)
   - **Fixed JSON serialization** errors in results saving
   - **Enhanced baseline model** with better class weighting and metrics

2. **🆕 Created Unified Pipeline (`run_pipeline3.py`)**:
   - **Removed archetype clustering** completely
   - **Added token metadata** (lifetime, volatility regime, max returns)
   - **Generates unified datasets**: `processed_features_{interval}_unified.parquet`
   - **Maintains all core functionality** (death detection, feature engineering, per-token scaling)

3. **🤖 Advanced Transformer Architecture (`transformer_forecast.py`)**:
   - **`UnifiedPumpTransformer`** class: 128-dim embeddings, 8 attention heads, 3 layers
   - **Rolling window sequences**: 5, 10, 15, 20, 30 minute windows for realistic trading
   - **Proper data splits**: By tokens (not sequences) to prevent leakage
   - **Enhanced training**: Learning rate scheduling, early stopping, best model saving
   - **Command line interface**: Multiple window sizes, interval selection

4. **📊 Enhanced Results Organization**:
   - **Results by interval**: `analysis/5m/` and `analysis/1m/` folders
   - **Model versioning**: Saved with metadata and timestamps
   - **Comprehensive metrics**: F1, AUC, precision, recall, class balance info

#### **Current Performance:**
- **Baseline XGBoost**: F1=0.225, AUC=0.770, Precision=0.140 (with 10% threshold)
- **Pipeline runtime**: <1 minute for 30k+ tokens
- **Data quality**: ~2.1% pump rate (much better than 0.5% with 50% threshold)

---

## 📋 **Current Roadmap & Next Steps**

### **🔥 Immediate Next Steps (1-2 Hours)**

#### **Phase 1: Complete Unified ML Pipeline**
1. **✅ DONE: Fixed baseline models and data issues**
2. **✅ DONE: Created unified pipeline without archetypes**
3. **✅ DONE: Advanced transformer with rolling windows**
4. **🔄 IN PROGRESS: Update `rl_agent.py`** for unified approach
5. **📋 TODO: Update `trading_sim.py`** for unified models

#### **Phase 2: Training & Testing** 
1. **Generate unified data**:
   ```bash
   python scripts/run_pipeline3.py
   ```

2. **Train transformer models**:
   ```bash
   python ml/transformer_forecast.py --interval 5m --window-sizes 10 20 30
   python ml/transformer_forecast.py --interval 1m --window-sizes 10 20 30
   ```

3. **Train RL agents** (after updating):
   ```bash
   python ml/rl_agent.py --interval 5m --transformer-window 20
   ```

4. **Run trading simulation**:
   ```bash
   python analysis/trading_sim.py --strategy unified --interval 5m
   ```

### **🎯 Success Criteria for Next Phase**
- **Transformer F1 >30%** with rolling windows
- **RL agent profit >5%** average per token
- **1m vs 5m comparison** shows clear performance difference
- **Trading simulation** shows realistic P&L with fees/slippage

---

## 🔗 **Related Documentation**

- **Main Project**: `../CLAUDE.md` (comprehensive memecoin2 documentation)
- **GitHub**: https://github.com/stordd13/solana-memecoin
- **Data Sources**: Parquet files in `../data/processed/`
- **Results**: Clustering outputs in `processed_features_*.parquet`

---

**Last Updated**: Latest session - Unified approach implementation with advanced transformer architecture

## 🏁 **Quick Start Guide**

### **For New Claude Instances**:

1. **Current State**: We've moved from archetype-based to unified cross-token models
2. **What Works**: Baseline XGBoost (F1=0.225), unified pipeline, advanced transformer architecture
3. **What's Next**: Update RL agent, complete trading simulation, test 1m vs 5m performance
4. **Key Files**: `run_pipeline3.py`, `transformer_forecast.py`, `run_baseline.py`

### **Ready-to-Run Commands**:
```bash
# Generate unified data (no archetypes)
python scripts/run_pipeline3.py

# Train advanced transformer
python ml/transformer_forecast.py --interval 5m --window-sizes 10 20 30

# Enhanced baseline for comparison
python ml/run_baseline.py --interval 5m
```

*Remember: This is a high-risk, experimental trading system. Always test thoroughly and never risk more than you can afford to lose.*