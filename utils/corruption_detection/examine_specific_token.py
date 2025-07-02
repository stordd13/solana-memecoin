#!/usr/bin/env python3
"""
Examine the specific token eL5fUxj2J4CiQsmW85k5FG9DvuQjjUoBHoQBi2Kpump 
to understand the pattern of legitimate massive gains vs staircase artifacts
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_token_pattern(file_path):
    """Analyze the pattern of a specific token"""
    
    df = pl.read_parquet(file_path)
    token_name = Path(file_path).stem
    
    print(f"Analyzing token: {token_name}")
    print(f"Data points: {len(df)}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Calculate returns
    if 'returns' not in df.columns:
        df = df.sort('datetime').with_columns([
            (pl.col('price').pct_change()).alias('returns')
        ]).with_columns([
            pl.col('returns').fill_null(0.0)
        ])
    
    prices = df['price'].to_numpy()
    returns = df['returns'].to_numpy()
    datetime_col = df['datetime'].to_numpy()
    
    # Basic statistics
    print(f"\nPrice Statistics:")
    print(f"  Min price: {np.min(prices):.2e}")
    print(f"  Max price: {np.max(prices):.2e}")
    print(f"  Price ratio: {np.max(prices)/np.min(prices):.1f}x")
    
    print(f"\nReturns Statistics:")
    finite_returns = returns[np.isfinite(returns)]
    print(f"  Max positive return: {np.max(finite_returns):.6f} ({np.max(finite_returns)*100:.1f}%)")
    print(f"  Max negative return: {np.min(finite_returns):.6f} ({np.min(finite_returns)*100:.1f}%)")
    print(f"  Returns > 10.0 (1000%): {np.sum(finite_returns > 10.0)}")
    print(f"  Returns > 50.0 (5000%): {np.sum(finite_returns > 50.0)}")
    
    # Find extreme moves
    extreme_indices = np.where(np.abs(finite_returns) > 10.0)[0]
    print(f"\nExtreme moves (>1000%):")
    for i, idx in enumerate(extreme_indices[:10]):  # Show first 10
        print(f"  Minute {idx}: {finite_returns[idx]*100:.1f}% return")
        
        # Check what happens after this extreme move
        if idx < len(prices) - 10:
            next_5_returns = finite_returns[idx+1:idx+6]
            next_5_avg = np.mean(np.abs(next_5_returns)) * 100
            print(f"    -> Next 5 minutes avg |return|: {next_5_avg:.1f}%")
    
    # Analyze time patterns
    print(f"\nTime Pattern Analysis:")
    
    # Find periods of sustained high volatility vs single spikes
    window_size = 5  # 5-minute windows
    volatility_windows = []
    
    for i in range(0, len(finite_returns) - window_size):
        window_returns = finite_returns[i:i+window_size]
        avg_abs_return = np.mean(np.abs(window_returns))
        max_abs_return = np.max(np.abs(window_returns))
        volatility_windows.append({
            'start_idx': i,
            'avg_abs_return': avg_abs_return,
            'max_abs_return': max_abs_return,
            'sustained_high': avg_abs_return > 0.5  # 50% average return in window
        })
    
    high_vol_windows = [w for w in volatility_windows if w['sustained_high']]
    print(f"  High volatility 5-min windows: {len(high_vol_windows)}")
    
    # Plot the price and returns
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Price over time
    axes[0].plot(prices, linewidth=1.5, color='blue')
    axes[0].set_title(f'Price Over Time: {token_name}')
    axes[0].set_ylabel('Price')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Returns over time
    axes[1].plot(finite_returns * 100, linewidth=1, color='orange')
    axes[1].set_title('Returns Over Time (%)')
    axes[1].set_ylabel('Return (%)')
    axes[1].axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='1000% threshold')
    axes[1].axhline(y=-1000, color='red', linestyle='--', alpha=0.7)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Rolling 5-minute average absolute returns
    rolling_returns = []
    x_positions = []
    for i in range(5, len(finite_returns)):
        window = finite_returns[i-5:i]
        rolling_returns.append(np.mean(np.abs(window)) * 100)
        x_positions.append(i)
    
    axes[2].plot(x_positions, rolling_returns, linewidth=2, color='purple')
    axes[2].set_title('Rolling 5-min Average |Return| (%)')
    axes[2].set_xlabel('Time (minutes)')
    axes[2].set_ylabel('Avg |Return| (%)')
    axes[2].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to utils/results directory
    output_path = Path(__file__).parent.parent / 'results' / f'pattern_analysis_{token_name}.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Analysis plot saved to: {output_path}")
    
    # Pattern classification
    print(f"\nPattern Classification:")
    
    # Check for staircase pattern: extreme jump followed by flat period
    staircase_artifacts = 0
    legitimate_pumps = 0
    
    for idx in extreme_indices:
        if idx < len(finite_returns) - 10:
            # Check volatility in next 10 minutes
            post_jump_returns = finite_returns[idx+1:idx+11]
            post_jump_volatility = np.mean(np.abs(post_jump_returns))
            
            if post_jump_volatility < 0.01:  # <1% average volatility after extreme move
                staircase_artifacts += 1
                print(f"  STAIRCASE: Minute {idx}, {finite_returns[idx]*100:.1f}% return, then flat")
            else:
                legitimate_pumps += 1
                print(f"  LEGITIMATE: Minute {idx}, {finite_returns[idx]*100:.1f}% return, continued volatility")
    
    print(f"\nSummary:")
    print(f"  Staircase artifacts: {staircase_artifacts}")
    print(f"  Legitimate pumps: {legitimate_pumps}")
    
    return {
        'staircase_artifacts': staircase_artifacts,
        'legitimate_pumps': legitimate_pumps,
        'max_price_ratio': np.max(prices)/np.min(prices),
        'max_single_return': np.max(finite_returns),
        'high_vol_windows': len(high_vol_windows)
    }

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    token_file = project_root / "data/processed/tokens_with_extremes/eL5fUxj2J4CiQsmW85k5FG9DvuQjjUoBHoQBi2Kpump.parquet"
    
    result = analyze_token_pattern(token_file)
    print(f"\nAnalysis complete. Check 'utils/results/pattern_analysis_eL5fUxj2J4CiQsmW85k5FG9DvuQjjUoBHoQBi2Kpump.png' for plots.")