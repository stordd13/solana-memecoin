# Project Structure Overview

## 📁 solana_memecoin_bot_new/
```
solana_memecoin_bot_new/
├── __pycache__/
├── analysis/
│   ├── eda.py                    # Exploratory data analysis
│   └── trading_sim.py            # Trading simulation
├── logs/
├── ml/
│   ├── __pycache__/
│   ├── archetype_clustering.py   # Behavioral archetype clustering
│   ├── baseline_models.py        # Baseline ML models
│   ├── rl_agent.py              # Reinforcement learning agent
│   └── transformer_forecast.py   # Transformer-based forecasting
├── scripts/
│   ├── __pycache__/
│   ├── death_detection.py        # Token death detection
│   ├── feature_engineering.py    # Feature creation
│   └── real_time_update.py      # Real-time data updates
├── tests/
└── config.py                     # Configuration settings
```

## 📊 data/ (Current Structure)
```
data/
├── raw/
│   └── dataset/                  # Raw token data (parquet files)
│       ├── CudisfkgWvMKnZ3TWf6iCuHm8pN2ikXhDcWytwz6f6RN.parquet
│       ├── [... 19,045 more parquet files ...]
│       └── zzzVPNFidF4YN4BYEk3tAr45brUkPnmQVeer1mwpump.parquet
```

## 📊 Expected Data Structure (from documentation)
Based on the CLAUDE.md and project documentation, the full data pipeline should create:
```
data/
├── raw/                          # Original scraped data
├── processed/                    # Categorized tokens (normal, extreme, dead, gap)
├── cleaned/                      # Quality-checked, cleaned data
├── features/                     # ML-ready engineered features
└── with_archetypes_fixed/        # Tokens with behavioral archetype labels
```

## 🔍 Key Observations:
1. The `solana_memecoin_bot_new` folder appears to be a newer, more focused implementation
2. Currently only `data/raw/dataset/` exists with ~19,047 parquet files
3. The processed/cleaned/features folders haven't been created yet or are in a different location
4. Each parquet file represents one memecoin token with minute-by-minute price data

## 📝 Notes:
- The `solana_memecoin_bot_new` structure suggests a shift towards real-time trading bot implementation
- The ML components include advanced approaches (RL agent, transformers)
- The scripts folder contains core functionality for real-time operation
- The data pipeline stages (processed, cleaned, features) may need to be run to generate the expected folders