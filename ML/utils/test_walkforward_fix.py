"""
Test script to verify walk-forward validation fixes
"""

import polars as pl
import numpy as np
from pathlib import Path
from walk_forward_splitter import WalkForwardSplitter

def test_horizon_buffer():
    """Test that horizon buffer correctly limits test folds"""
    
    # Create sample data with 1000 minutes
    n_samples = 1000
    
    # Test without buffer
    print("Testing WITHOUT horizon buffer:")
    splitter_no_buffer = WalkForwardSplitter(config='short', horizon_buffer=0)
    folds_no_buffer = list(splitter_no_buffer.split(np.arange(n_samples)))
    
    print(f"  Number of folds: {len(folds_no_buffer)}")
    if folds_no_buffer:
        last_fold = folds_no_buffer[-1]
        print(f"  Last fold test end: {last_fold[1][-1]}")
        print(f"  Distance from data end: {n_samples - last_fold[1][-1] - 1} minutes")
    
    # Test with 720 minute buffer (12 hours)
    print("\nTesting WITH 720-minute horizon buffer:")
    splitter_with_buffer = WalkForwardSplitter(config='short', horizon_buffer=720)
    folds_with_buffer = list(splitter_with_buffer.split(np.arange(n_samples)))
    
    print(f"  Number of folds: {len(folds_with_buffer)}")
    if folds_with_buffer:
        last_fold = folds_with_buffer[-1]
        print(f"  Last fold test end: {last_fold[1][-1]}")
        print(f"  Distance from data end: {n_samples - last_fold[1][-1] - 1} minutes")
        print(f"  ✅ Buffer ensures at least 720 minutes for label creation!")
    
    # Verify the buffer is working
    if folds_with_buffer:
        buffer_distance = n_samples - folds_with_buffer[-1][1][-1] - 1
        assert buffer_distance >= 720, f"Buffer distance {buffer_distance} is less than 720!"
        print(f"\n✅ TEST PASSED: Buffer distance is {buffer_distance} minutes (>= 720)")


def test_label_creation():
    """Test that labels can be created for all horizons with buffer"""
    
    # Load a sample feature file
    features_dir = Path("data/features")
    sample_files = []
    
    for category in ["normal_behavior_tokens", "tokens_with_extremes", "dead_tokens"]:
        cat_dir = features_dir / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*.parquet"))[:1]  # Get one file
            sample_files.extend(files)
            if sample_files:
                break
    
    if not sample_files:
        print("No sample files found for testing")
        return
    
    # Load sample data
    df = pl.read_parquet(sample_files[0])
    print(f"\nTesting label creation with {sample_files[0].name}")
    print(f"Data length: {len(df)} minutes")
    
    # Create labels for different horizons
    horizons = [15, 30, 60, 120, 240, 360, 720]
    
    # Without buffer - check last possible label
    print("\nLabel creation WITHOUT buffer:")
    for h in horizons:
        last_valid_idx = len(df) - h - 1
        if last_valid_idx >= 0:
            print(f"  {h}min horizon: Can create labels up to index {last_valid_idx}")
        else:
            print(f"  {h}min horizon: ❌ Cannot create any labels!")
    
    # With 720 buffer - check last possible label in test fold
    print("\nLabel creation WITH 720-minute buffer:")
    effective_length = len(df) - 720
    for h in horizons:
        last_valid_idx = effective_length - h - 1
        if last_valid_idx >= 0:
            print(f"  {h}min horizon: Can create labels up to index {last_valid_idx}")
            print(f"    (Still have {720 - h} minutes of buffer remaining)")
        else:
            print(f"  {h}min horizon: ❌ Cannot create labels even with buffer!")


if __name__ == "__main__":
    print("="*60)
    print("Testing Walk-Forward Validation Fixes")
    print("="*60)
    
    test_horizon_buffer()
    print("\n" + "="*60 + "\n")
    test_label_creation()
    
    print("\n✅ All tests completed!") 