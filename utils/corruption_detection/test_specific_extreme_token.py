#!/usr/bin/env python3
"""
Test the improved corruption detection on a specific extreme token
"""

import sys
import polars as pl
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_cleaning.clean_tokens import CategoryAwareTokenCleaner

def test_specific_extreme_token():
    """Test the improved corruption detection on the token that had massive corrupted returns"""
    
    cleaner = CategoryAwareTokenCleaner()
    
    # This token had 4,899,223,971.3% return and was correctly identified as staircase
    token_file = project_root / "data/processed/tokens_with_extremes/HRdv28P1stNkcac6v2VFnhbMbkwmd1VnLQdLxRSE7Q1e.parquet"
    
    if not Path(token_file).exists():
        print(f"Token file not found: {token_file}")
        return
    
    df = pl.read_parquet(token_file)
    token_name = Path(token_file).stem
    
    print(f"Testing extreme token: {token_name}")
    print(f"Original data points: {len(df)}")
    
    # Add returns if missing
    if 'returns' not in df.columns:
        df = cleaner._calculate_returns(df)
    
    # Show extreme statistics
    prices = df['price'].to_numpy()
    returns = df['returns'].to_numpy()
    finite_returns = returns[np.isfinite(returns)]
    
    print(f"Price range: {np.min(prices):.2e} to {np.max(prices):.2e}")
    print(f"Price ratio: {np.max(prices)/np.min(prices):.1f}x")
    print(f"Max return: {np.max(finite_returns)*100:.1f}%")
    print(f"Min return: {np.min(finite_returns)*100:.1f}%")
    
    # Find extreme moves
    extreme_threshold = 10.0  # 1000%
    extreme_indices = np.where(np.abs(finite_returns) > extreme_threshold)[0]
    print(f"Returns > 1000%: {len(extreme_indices)}")
    
    for i, idx in enumerate(extreme_indices):
        print(f"  Extreme move {i+1}: Minute {idx}, {finite_returns[idx]*100:.1f}% return")
        
        # Check what happens in next 10 minutes
        if idx < len(finite_returns) - 10:
            next_10 = finite_returns[idx+1:idx+11]
            avg_volatility = np.mean(np.abs(next_10)) * 100
            print(f"    -> Next 10 minutes avg volatility: {avg_volatility:.3f}%")
    
    print(f"\n{'='*50}")
    print("TESTING IMPROVED CORRUPTION DETECTION")
    print(f"{'='*50}")
    
    # Test the improved corruption detection
    df_clean, modifications = cleaner._fix_extreme_data_corruption(df)
    
    print(f"\nResults:")
    print(f"Original points: {len(df)}")
    print(f"Final points: {len(df_clean)}")
    print(f"Points removed: {len(df) - len(df_clean)}")
    print(f"Modifications: {len(modifications)}")
    
    for mod in modifications:
        print(f"\nModification: {mod['type']}")
        if 'count' in mod:
            print(f"  Staircase artifacts removed: {mod['count']}")
        if 'legitimate_pumps_preserved' in mod:
            print(f"  Legitimate pumps preserved: {mod['legitimate_pumps_preserved']}")
        if 'artifacts_data' in mod:
            print("  Artifact details:")
            for artifact in mod['artifacts_data']:
                print(f"    - Minute {artifact['idx']}: {artifact['return']*100:.1f}% return, "
                      f"post-volatility: {artifact['post_volatility']*100:.3f}%")

if __name__ == "__main__":
    test_specific_extreme_token()