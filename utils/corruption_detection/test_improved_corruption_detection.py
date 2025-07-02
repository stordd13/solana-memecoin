#!/usr/bin/env python3
"""
Test the improved corruption detection logic on various tokens
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_cleaning.clean_tokens import CategoryAwareTokenCleaner

def test_token(file_path):
    """Test the improved corruption detection on a specific token"""
    
    cleaner = CategoryAwareTokenCleaner()
    
    df = pl.read_parquet(file_path)
    token_name = Path(file_path).stem
    
    print(f"\n{'='*80}")
    print(f"TESTING: {token_name}")
    print(f"{'='*80}")
    
    # Add returns if missing
    if 'returns' not in df.columns:
        df = cleaner._calculate_returns(df)
    
    # Show basic stats
    prices = df['price'].to_numpy()
    returns = df['returns'].to_numpy()
    finite_returns = returns[np.isfinite(returns)]
    
    print(f"Original data points: {len(df)}")
    print(f"Price range: {np.min(prices):.2e} to {np.max(prices):.2e} ({np.max(prices)/np.min(prices):.1f}x)")
    print(f"Max single-minute return: {np.max(finite_returns)*100:.1f}%")
    print(f"Returns > 1000%: {np.sum(finite_returns > 10.0)}")
    
    # Test the improved corruption detection
    df_clean, modifications = cleaner._fix_extreme_data_corruption(df)
    
    print(f"\nResults:")
    print(f"Final data points: {len(df_clean)}")
    print(f"Modifications: {len(modifications)}")
    
    for mod in modifications:
        print(f"  - {mod['type']}")
        if 'legitimate_pumps_preserved' in mod:
            print(f"    Legitimate pumps preserved: {mod['legitimate_pumps_preserved']}")
        if 'count' in mod:
            print(f"    Staircase artifacts removed: {mod['count']}")
    
    return {
        'token': token_name,
        'original_points': len(df),
        'final_points': len(df_clean),
        'modifications': modifications,
        'max_return': np.max(finite_returns),
        'price_ratio': np.max(prices)/np.min(prices)
    }

def main():
    """Test on various tokens from different categories"""
    
    # Test on the specific token mentioned
    test_tokens = [
        project_root / "data/processed/tokens_with_extremes/eL5fUxj2J4CiQsmW85k5FG9DvuQjjUoBHoQBi2Kpump.parquet"
    ]
    
    # Add a few more extreme tokens for testing
    extreme_dir = project_root / "data/processed/tokens_with_extremes"
    if extreme_dir.exists():
        extreme_files = list(extreme_dir.glob("*.parquet"))[:5]  # Test first 5
        test_tokens.extend(extreme_files)
    
    results = []
    
    for token_file in test_tokens[:6]:  # Test up to 6 tokens
        if Path(token_file).exists():
            try:
                result = test_token(token_file)
                results.append(result)
            except Exception as e:
                print(f"Error testing {token_file}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    total_tested = len(results)
    preserved_pumps = sum(mod.get('legitimate_pumps_preserved', 0) 
                         for r in results for mod in r['modifications'])
    removed_artifacts = sum(mod.get('count', 0) 
                           for r in results for mod in r['modifications'] 
                           if mod['type'] == 'staircase_artifacts_removed')
    
    print(f"Tokens tested: {total_tested}")
    print(f"Total legitimate pumps preserved: {preserved_pumps}")
    print(f"Total staircase artifacts removed: {removed_artifacts}")
    
    print(f"\nToken details:")
    for r in results:
        status = "MODIFIED" if r['final_points'] < r['original_points'] else "UNCHANGED"
        print(f"  {r['token'][:50]:<50} | {r['price_ratio']:>8.1f}x | {r['max_return']*100:>6.1f}% | {status}")

if __name__ == "__main__":
    main()