"""
Test and demonstrate the walk-forward validation splitter
"""

import numpy as np
import polars as pl
from walk_forward_splitter import (
    WalkForwardSplitter, 
    WalkForwardConfig,
    create_walk_forward_splits
)
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def create_sample_data(n_tokens: int = 5, min_length: int = 400, max_length: int = 2000):
    """Create sample token data with varying lengths"""
    data_frames = []
    
    for i in range(n_tokens):
        token_length = np.random.randint(min_length, max_length)
        token_address = f"token_{i:03d}"
        
        # Create time series data
        minutes = np.arange(token_length)
        prices = 100 * np.exp(np.cumsum(np.random.randn(token_length) * 0.01))
        volume = np.random.lognormal(10, 2, token_length)
        
        token_df = pl.DataFrame({
            'token_address': [token_address] * token_length,
            'minutes_since_launch': minutes,
            'price': prices,
            'volume': volume,
            'timestamp': [datetime.now() + timedelta(minutes=int(m)) for m in minutes]
        })
        
        data_frames.append(token_df)
    
    return pl.concat(data_frames)


def test_adaptive_config():
    """Test adaptive configuration based on data length"""
    print("Testing Adaptive Configuration")
    print("-" * 50)
    
    splitter = WalkForwardSplitter()
    
    test_lengths = [400, 800, 1200, 1600, 2000, 3000]
    
    for length in test_lengths:
        config = splitter.get_adaptive_config(length)
        dummy_data = np.arange(length)
        fold_info = splitter.get_fold_info(length)
        
        print(f"\nData length: {length} minutes ({length/60:.1f} hours)")
        print(f"Config: min_train={config.min_train_size}, "
              f"step={config.step_size}, test={config.test_size}")
        print(f"Number of folds: {len(fold_info)}")
        
        if fold_info:
            print(f"First fold: train[{fold_info[0]['train_start']}:{fold_info[0]['train_end']}], "
                  f"test[{fold_info[0]['test_start']}:{fold_info[0]['test_end']}]")
            print(f"Last fold: train[{fold_info[-1]['train_start']}:{fold_info[-1]['train_end']}], "
                  f"test[{fold_info[-1]['test_start']}:{fold_info[-1]['test_end']}]")


def test_expanding_vs_sliding():
    """Compare expanding vs sliding window strategies"""
    print("\n\nComparing Expanding vs Sliding Window")
    print("-" * 50)
    
    data_length = 1000
    dummy_data = np.arange(data_length)
    
    # Expanding window
    config_expanding = WalkForwardConfig(
        min_train_size=200,
        step_size=100,
        test_size=100,
        strategy='expanding'
    )
    splitter_expanding = WalkForwardSplitter(config=config_expanding)
    
    # Sliding window
    config_sliding = WalkForwardConfig(
        min_train_size=200,
        step_size=100,
        test_size=100,
        strategy='sliding',
        max_train_size=400
    )
    splitter_sliding = WalkForwardSplitter(config=config_sliding)
    
    print("\nExpanding Window:")
    for i, (train_idx, test_idx) in enumerate(splitter_expanding.split(dummy_data)):
        print(f"Fold {i}: train size={len(train_idx)}, "
              f"range=[{train_idx[0]},{train_idx[-1]}], "
              f"test range=[{test_idx[0]},{test_idx[-1]}]")
    
    print("\nSliding Window:")
    for i, (train_idx, test_idx) in enumerate(splitter_sliding.split(dummy_data)):
        print(f"Fold {i}: train size={len(train_idx)}, "
              f"range=[{train_idx[0]},{train_idx[-1]}], "
              f"test range=[{test_idx[0]},{test_idx[-1]}]")


def test_token_splitting():
    """Test per-token splitting"""
    print("\n\nTesting Per-Token Splitting")
    print("-" * 50)
    
    # Create sample data
    data = create_sample_data(n_tokens=3, min_length=400, max_length=1000)
    
    splitter = WalkForwardSplitter(config='short')
    token_splits = splitter.split_by_token(data, min_token_length=300)
    
    for token, splits in token_splits.items():
        token_data = data.filter(pl.col('token_address') == token)
        print(f"\n{token}: {len(token_data)} minutes")
        print(f"Number of folds: {len(splits)}")
        
        for i, (train_df, test_df) in enumerate(splits[:3]):  # Show first 3 folds
            print(f"  Fold {i}: train={len(train_df)}, test={len(test_df)}")


def test_global_splitting():
    """Test global splitting across all tokens"""
    print("\n\nTesting Global Splitting")
    print("-" * 50)
    
    # Create sample data
    data = create_sample_data(n_tokens=5, min_length=400, max_length=800)
    
    splitter = WalkForwardSplitter(config='medium')
    global_splits = splitter.get_global_splits(data)
    
    print(f"Total data points: {len(data)}")
    print(f"Number of tokens: {data['token_address'].n_unique()}")
    print(f"Number of folds: {len(global_splits)}")
    
    for i, (train_df, test_df) in enumerate(global_splits):
        train_tokens = train_df['token_address'].n_unique()
        test_tokens = test_df['token_address'].n_unique()
        print(f"\nFold {i}:")
        print(f"  Train: {len(train_df)} points, {train_tokens} tokens")
        print(f"  Test: {len(test_df)} points, {test_tokens} tokens")


def visualize_walk_forward():
    """Create visualization of walk-forward validation"""
    print("\n\nVisualizing Walk-Forward Validation")
    print("-" * 50)
    
    data_length = 1440  # 24 hours
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Expanding window
    splitter1 = WalkForwardSplitter(config='medium')
    fold_info1 = splitter1.get_fold_info(data_length)
    
    for i, fold in enumerate(fold_info1):
        # Train window
        ax1.barh(i, fold['train_size'], left=fold['train_start'], 
                height=0.8, color='blue', alpha=0.7, label='Train' if i == 0 else '')
        # Test window
        ax1.barh(i, fold['test_size'], left=fold['test_start'], 
                height=0.8, color='red', alpha=0.7, label='Test' if i == 0 else '')
    
    ax1.set_xlabel('Minutes since launch')
    ax1.set_ylabel('Fold')
    ax1.set_title('Expanding Window Walk-Forward Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sliding window
    config2 = WalkForwardConfig(
        min_train_size=360,
        step_size=120,
        test_size=120,
        strategy='sliding',
        max_train_size=480
    )
    splitter2 = WalkForwardSplitter(config=config2)
    fold_info2 = splitter2.get_fold_info(data_length)
    
    for i, fold in enumerate(fold_info2):
        # Train window
        ax2.barh(i, fold['train_size'], left=fold['train_start'], 
                height=0.8, color='green', alpha=0.7, label='Train' if i == 0 else '')
        # Test window
        ax2.barh(i, fold['test_size'], left=fold['test_start'], 
                height=0.8, color='orange', alpha=0.7, label='Test' if i == 0 else '')
    
    ax2.set_xlabel('Minutes since launch')
    ax2.set_ylabel('Fold')
    ax2.set_title('Sliding Window Walk-Forward Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ML/utils/walk_forward_visualization.png', dpi=150)
    print("Saved visualization to ML/utils/walk_forward_visualization.png")


def test_integration_example():
    """Example of integrating with existing model training"""
    print("\n\nIntegration Example")
    print("-" * 50)
    
    # Simulate feature data
    data = create_sample_data(n_tokens=2, min_length=600, max_length=800)
    
    # Add some features
    data = data.with_columns([
        (pl.col('price').shift(1) / pl.col('price') - 1).alias('return_1m'),
        pl.col('volume').log().alias('log_volume'),
        (pl.col('price').rolling_mean(5) / pl.col('price')).alias('ma_ratio_5')
    ])
    
    # Example: Training with walk-forward validation
    splitter = WalkForwardSplitter()
    token = data['token_address'].unique()[0]
    token_data = data.filter(pl.col('token_address') == token).drop_nulls()
    
    print(f"\nTraining example for {token}:")
    print(f"Total samples: {len(token_data)}")
    
    metrics_per_fold = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(token_data)):
        train_data = token_data[train_idx]
        test_data = token_data[test_idx]
        
        # Simulate model training and evaluation
        # In real code, you'd fit your model here
        fake_mae = np.random.uniform(0.01, 0.05)
        metrics_per_fold.append(fake_mae)
        
        print(f"Fold {fold_idx}: train={len(train_data)}, test={len(test_data)}, MAE={fake_mae:.4f}")
    
    print(f"\nAverage MAE across {len(metrics_per_fold)} folds: {np.mean(metrics_per_fold):.4f}")


if __name__ == "__main__":
    # Run all tests
    test_adaptive_config()
    test_expanding_vs_sliding()
    test_token_splitting()
    test_global_splitting()
    visualize_walk_forward()
    test_integration_example()
    
    print("\n\nAll tests completed!") 