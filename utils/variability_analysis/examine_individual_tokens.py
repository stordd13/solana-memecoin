#!/usr/bin/env python3
"""
Individual Token Examination Tool
Detailed analysis of specific tokens with price plots and variability metrics
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from data_cleaning.clean_tokens import CategoryAwareTokenCleaner

def examine_token(file_path: str, show_log_scale: bool = True):
    """Examine a single token in detail with plots and metrics"""
    
    cleaner = CategoryAwareTokenCleaner()
    
    try:
        # Load the token data
        df = pl.read_parquet(file_path)
        token_name = Path(file_path).stem
        
        # Add returns if missing
        if 'returns' not in df.columns:
            df = cleaner._calculate_returns(df)
        
        # Get variability analysis
        result_df, mods = cleaner._check_price_variability(df, token_name)
        
        # Extract metrics from modifications
        metrics = None
        for mod in mods:
            if 'metrics' in mod:
                metrics = mod['metrics']
                break
        
        if not metrics:
            print(f"Could not analyze {token_name}")
            return
        
        prices = df['price'].to_numpy()
        returns = df['returns'].to_numpy()
        datetime_col = df['datetime'].to_numpy()
        
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Token Analysis: {token_name}', fontsize=16, fontweight='bold')
        
        # 1. Raw price plot
        ax1 = axes[0, 0]
        ax1.plot(prices, linewidth=1.5, color='blue')
        ax1.set_title('Raw Price Over Time')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # 2. Log price plot (if requested and prices > 0)
        ax2 = axes[0, 1]
        if show_log_scale and np.all(prices > 0):
            log_prices = np.log(prices)
            ax2.plot(log_prices, linewidth=1.5, color='green')
            ax2.set_title('Log Price Over Time')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Log(Price)')
        else:
            ax2.plot(prices, linewidth=1.5, color='green')
            ax2.set_title('Price Over Time (Linear Scale)')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Price')
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns distribution
        ax3 = axes[1, 0]
        ax3.hist(returns[np.isfinite(returns)], bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Returns')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling variability
        ax4 = axes[1, 1]
        window_size = min(60, len(prices) // 4)
        rolling_cv = []
        x_positions = []
        
        for i in range(0, len(prices) - window_size + 1, window_size // 2):
            window_prices = prices[i:i + window_size]
            cv = np.std(window_prices) / np.mean(window_prices) if np.mean(window_prices) > 0 else 0
            rolling_cv.append(cv)
            x_positions.append(i + window_size // 2)
        
        ax4.plot(x_positions, rolling_cv, linewidth=2, color='purple', marker='o', markersize=4)
        ax4.set_title('Rolling Coefficient of Variation')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('CV')
        ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Filter threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add metrics text box
        metrics_text = f"""
VARIABILITY METRICS:
• Price CV: {metrics['price_cv']:.4f} {'❌' if metrics['price_cv'] < 0.05 else '✅'}
• Log Price CV: {metrics['log_price_cv']:.4f} {'❌' if metrics['log_price_cv'] < 0.1 else '✅'}
• Flat Periods: {metrics['flat_periods_fraction']:.3f} {'❌' if metrics['flat_periods_fraction'] > 0.8 else '✅'}
• Range Efficiency: {metrics['range_efficiency']:.3f} {'❌' if metrics['range_efficiency'] < 0.1 else '✅'}
• Entropy: {metrics['normalized_entropy']:.3f} {'❌' if metrics['normalized_entropy'] < 0.3 else '✅'}

DECISION: {'🔴 FILTERED (Low Variability)' if metrics['is_low_variability'] else '🟢 PASSED'}

BASIC STATS:
• Length: {len(prices)} minutes
• Price Range: {np.min(prices):.2e} to {np.max(prices):.2e}
• Price Ratio: {np.max(prices)/np.min(prices):.1f}x
• Max |Return|: {np.max(np.abs(returns[np.isfinite(returns)])):.2f}
        """
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                   verticalalignment='bottom')
        
        # Save the plot to utils/results directory
        output_path = Path(__file__).parent.parent / 'results' / f'token_analysis_{token_name}.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Analysis saved to: {output_path}")
        
        # Print detailed console output
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS: {token_name}")
        print(f"{'='*60}")
        print(f"File: {file_path}")
        print(f"Length: {len(prices)} data points")
        print(f"")
        print(f"🎯 VARIABILITY METRICS:")
        print(f"   Price CV: {metrics['price_cv']:.6f} (threshold: < 0.05)")
        print(f"   Log Price CV: {metrics['log_price_cv']:.6f} (threshold: < 0.1)")
        print(f"   Flat Periods: {metrics['flat_periods_fraction']:.3f} (threshold: > 0.8)")
        print(f"   Range Efficiency: {metrics['range_efficiency']:.3f} (threshold: < 0.1)")
        print(f"   Normalized Entropy: {metrics['normalized_entropy']:.3f} (threshold: < 0.3)")
        print(f"")
        print(f"📊 PRICE STATISTICS:")
        print(f"   Min Price: {np.min(prices):.2e}")
        print(f"   Max Price: {np.max(prices):.2e}")
        print(f"   Price Ratio: {np.max(prices)/np.min(prices):.1f}x")
        print(f"   Mean Price: {np.mean(prices):.2e}")
        print(f"   Std Price: {np.std(prices):.2e}")
        print(f"")
        print(f"📈 RETURNS STATISTICS:")
        finite_returns = returns[np.isfinite(returns)]
        print(f"   Max |Return|: {np.max(np.abs(finite_returns)):.6f}")
        print(f"   Mean |Return|: {np.mean(np.abs(finite_returns)):.6f}")
        print(f"   Std Returns: {np.std(finite_returns):.6f}")
        print(f"")
        print(f"🎯 FINAL DECISION: {'🔴 FILTERED (Low Variability)' if metrics['is_low_variability'] else '🟢 PASSED'}")
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def find_example_tokens():
    """Find example tokens of different types"""
    categories = ['normal_behavior_tokens', 'dead_tokens', 'tokens_with_extremes', 'tokens_with_gaps']
    examples = {}
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    for category in categories:
        processed_dir = project_root / 'data' / 'processed' / category
        if processed_dir.exists():
            files = list(processed_dir.glob('*.parquet'))[:5]  # Get first 5 files
            examples[category] = files
    
    return examples

if __name__ == "__main__":
    print("🔍 Individual Token Examination Tool")
    print("="*50)
    
    # Find some example tokens
    examples = find_example_tokens()
    
    print("Available examples:")
    for category, files in examples.items():
        print(f"\n{category}:")
        for i, file in enumerate(files):
            print(f"  {i+1}. {file.stem}")
    
    print(f"\nTo examine a specific token, modify the script or call:")
    print(f"examine_token('path/to/token.parquet')")
    
    # Example: Examine one token from each category
    print(f"\n🔬 Examining sample tokens...")
    
    for category, files in examples.items():
        if files:
            print(f"\n" + "="*80)
            print(f"EXAMINING SAMPLE FROM: {category.upper()}")
            print(f"="*80)
            examine_token(str(files[0]))
            
            # Ask user if they want to continue
            response = input(f"\nPress Enter to continue to next category, or 'q' to quit: ")
            if response.lower() == 'q':
                break