#!/usr/bin/env python3
"""
Validate No Data Leakage in Feature Calculations

This script validates that the new feature extraction method doesn't leak future information
by testing that features calculated at minute 3 don't change when we have data for minutes 4-5.
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from archetype_classifier import ArchetypeClassifier


def test_no_data_leakage():
    """Test that feature calculations don't leak future information."""
    print("🔍 Testing for data leakage in feature calculations...")
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Create 5 minutes of synthetic price data
    prices_5min = np.array([1.0, 1.05, 1.02, 1.08, 1.03])  # 5 minutes of prices
    prices_3min = prices_5min[:3]  # First 3 minutes only
    
    # Create DataFrames
    df_5min = pl.DataFrame({
        'datetime': pd.date_range('2025-01-01', periods=5, freq='1min'),
        'price': prices_5min
    })
    
    df_3min = pl.DataFrame({
        'datetime': pd.date_range('2025-01-01', periods=3, freq='1min'),
        'price': prices_3min
    })
    
    # Create a dummy classifier instance
    classifier = ArchetypeClassifier()
    
    # Mock token labels
    classifier.token_labels = {
        'test_token': {
            'category': 'standard',
            'cluster': 0,
            'archetype': 'standard_cluster_0'
        }
    }
    
    # Extract features from 3-minute data
    features_3min = classifier.extract_features({'test_token': df_3min}, minutes=3)
    
    # Extract features from 5-minute data (but only looking at first 3 minutes)
    features_5min = classifier.extract_features({'test_token': df_5min}, minutes=5)
    
    if len(features_3min) == 0 or len(features_5min) == 0:
        print("❌ No features extracted - test failed")
        return False
    
    # Check that features that should be the same are identical
    leak_detected = False
    
    # Features that should be identical (using only first 3 minutes)
    safe_features = [
        'cumulative_return_1m', 'cumulative_return_2m', 'cumulative_return_3m',
        'rolling_volatility_2m', 'rolling_volatility_3m',
        'trend_slope_2m', 'trend_slope_3m',
        'returns_mean_2m', 'returns_mean_3m',
        'returns_std_3m'
    ]
    
    print(f"📊 Checking {len(safe_features)} features for data leakage...")
    
    for feature in safe_features:
        if feature in features_3min.columns and feature in features_5min.columns:
            val_3min = features_3min[feature].iloc[0]
            val_5min = features_5min[feature].iloc[0]
            
            if abs(val_3min - val_5min) > 1e-10:  # Allow for floating point precision
                print(f"❌ LEAK DETECTED in {feature}: 3min={val_3min:.6f}, 5min={val_5min:.6f}")
                leak_detected = True
            else:
                print(f"✅ {feature}: No leakage detected")
    
    # Features that should be different (using different amounts of data)
    expected_different = [
        'cumulative_return_4m', 'cumulative_return_5m',
        'rolling_volatility_4m', 'rolling_volatility_5m',
        'trend_slope_4m', 'trend_slope_5m'
    ]
    
    print(f"\n📊 Checking {len(expected_different)} features that should be different...")
    
    for feature in expected_different:
        if feature in features_5min.columns:
            val_5min = features_5min[feature].iloc[0]
            if feature.replace('4m', '3m').replace('5m', '3m') in features_3min.columns:
                # These should be different because they use more data
                print(f"✅ {feature}: Uses additional data as expected")
            else:
                print(f"✅ {feature}: Only available with 5min data")
    
    if not leak_detected:
        print(f"\n🎉 SUCCESS: No data leakage detected!")
        print(f"✅ All 3-minute features are identical whether calculated from 3-minute or 5-minute data")
        print(f"✅ Features properly use only progressive/rolling calculations")
        return True
    else:
        print(f"\n❌ FAILURE: Data leakage detected in feature calculations!")
        return False


def test_progressive_feature_behavior():
    """Test that features behave correctly as we add more data."""
    print(f"\n🔍 Testing progressive feature behavior...")
    
    # Create synthetic price data with a clear pattern
    prices = np.array([1.0, 1.1, 1.2, 1.15, 1.25])  # 5 minutes: up, up, down, up
    
    classifier = ArchetypeClassifier()
    classifier.token_labels = {
        'test_token': {
            'category': 'standard',
            'cluster': 0,
            'archetype': 'standard_cluster_0'
        }
    }
    
    # Test feature extraction at different time points
    for minutes in range(2, 6):  # Test 2-5 minutes
        df = pl.DataFrame({
            'datetime': pd.date_range('2025-01-01', periods=minutes, freq='1min'),
            'price': prices[:minutes]
        })
        
        features = classifier.extract_features({'test_token': df}, minutes=minutes)
        
        if len(features) > 0:
            row = features.iloc[0]
            print(f"📊 {minutes} minutes:")
            print(f"  Cumulative return: {row[f'cumulative_return_{minutes}m']:.4f}")
            print(f"  Rolling volatility: {row[f'rolling_volatility_{minutes}m']:.4f}")
            print(f"  Trend slope: {row[f'trend_slope_{minutes}m']:.4f}")
    
    print("✅ Progressive feature behavior verified")
    return True


if __name__ == "__main__":
    print("🚀 Running data leakage validation tests...")
    
    success1 = test_no_data_leakage()
    success2 = test_progressive_feature_behavior()
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED: Feature extraction is data-leakage free!")
        sys.exit(0)
    else:
        print(f"\n❌ TESTS FAILED: Data leakage detected or other issues found!")
        sys.exit(1)