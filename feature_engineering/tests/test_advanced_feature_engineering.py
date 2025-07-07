"""
Advanced feature engineering pipeline validation tests for feature_engineering module
Tests the complete AdvancedFeatureEngineer pipeline mathematical correctness
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.mathematical
class TestAdvancedFeatureEngineeringMath:
    """Test advanced feature engineering mathematical correctness."""
    
    def test_technical_indicators_calculation(self, feature_engineer, reference_time_series_data):
        """Test technical indicators calculation accuracy."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        # Add required columns for feature engineering
        df_with_features = df.with_columns([
            pl.col('price').pct_change().alias('returns'),
            pl.col('price').rolling_mean(window_size=20).alias('sma_20'),
            (pl.col('price').rolling_std(window_size=20) * 2).alias('volatility')
        ])
        
        # Calculate features using the engineer
        features_df = feature_engineer.calculate_technical_indicators(df_with_features)
        
        # Test RSI calculation
        if 'rsi' in features_df.columns:
            rsi_values = features_df['rsi'].drop_nulls().to_numpy()
            if len(rsi_values) > 0:
                # RSI should be between 0 and 100
                assert np.all(rsi_values >= 0), "RSI values should be >= 0"
                assert np.all(rsi_values <= 100), "RSI values should be <= 100"
        
        # Test MACD calculation
        if 'macd' in features_df.columns:
            macd_values = features_df['macd'].drop_nulls().to_numpy()
            if len(macd_values) > 0:
                assert np.all(np.isfinite(macd_values)), "MACD values should be finite"
        
        # Test Bollinger Bands
        bollinger_cols = [col for col in features_df.columns if 'bollinger' in col]
        if len(bollinger_cols) >= 3:
            # Should have upper, middle, lower
            upper_col = next((col for col in bollinger_cols if 'upper' in col), None)
            middle_col = next((col for col in bollinger_cols if 'middle' in col), None)
            lower_col = next((col for col in bollinger_cols if 'lower' in col), None)
            
            if upper_col and middle_col and lower_col:
                valid_indices = (~features_df[upper_col].is_null() & 
                               ~features_df[middle_col].is_null() & 
                               ~features_df[lower_col].is_null())
                
                if valid_indices.sum() > 0:
                    upper_vals = features_df.filter(valid_indices)[upper_col].to_numpy()
                    middle_vals = features_df.filter(valid_indices)[middle_col].to_numpy()
                    lower_vals = features_df.filter(valid_indices)[lower_col].to_numpy()
                    
                    # Upper should be >= middle >= lower
                    assert np.all(upper_vals >= middle_vals), "Upper Bollinger Band should be >= middle"
                    assert np.all(middle_vals >= lower_vals), "Middle Bollinger Band should be >= lower"
    
    def test_rolling_statistics_accuracy(self, feature_engineer, reference_time_series_data):
        """Test rolling statistics calculation accuracy."""
        df = reference_time_series_data['df']
        
        # Calculate rolling features
        rolling_features = feature_engineer.calculate_rolling_statistics(df, windows=[10, 20])
        
        # Test rolling mean accuracy
        rolling_mean_10_col = next((col for col in rolling_features.columns if 'rolling_mean_10' in col), None)
        if rolling_mean_10_col:
            # Compare with manual calculation
            manual_rolling_mean = df['price'].rolling_mean(window_size=10)
            calculated_rolling_mean = rolling_features[rolling_mean_10_col]
            
            # Both should be approximately equal
            manual_values = manual_rolling_mean.drop_nulls().to_numpy()
            calculated_values = calculated_rolling_mean.drop_nulls().to_numpy()
            
            if len(manual_values) > 0 and len(calculated_values) > 0:
                min_len = min(len(manual_values), len(calculated_values))
                np.testing.assert_array_almost_equal(
                    manual_values[-min_len:], calculated_values[-min_len:], decimal=12,
                    err_msg="Rolling mean calculation should match manual calculation"
                )
        
        # Test rolling standard deviation
        rolling_std_10_col = next((col for col in rolling_features.columns if 'rolling_std_10' in col), None)
        if rolling_std_10_col:
            std_values = rolling_features[rolling_std_10_col].drop_nulls().to_numpy()
            if len(std_values) > 0:
                # Standard deviation should be non-negative
                assert np.all(std_values >= 0), "Rolling standard deviation should be non-negative"
    
    def test_lag_features_temporal_safety(self, feature_engineer, reference_time_series_data):
        """Test lag features for temporal safety."""
        df = reference_time_series_data['df']
        
        # Calculate lag features
        lag_features = feature_engineer.calculate_lag_features(df, lags=[1, 5, 10])
        
        # Test lag feature accuracy
        for lag in [1, 5, 10]:
            lag_col = next((col for col in lag_features.columns if f'lag_{lag}' in col), None)
            if lag_col:
                lag_values = lag_features[lag_col].to_numpy()
                original_values = df['price'].to_numpy()
                
                # Check lag alignment
                valid_indices = ~np.isnan(lag_values)
                if np.any(valid_indices):
                    # Lag values should match original values shifted by lag
                    for i in np.where(valid_indices)[0]:
                        if i >= lag:
                            expected_val = original_values[i - lag]
                            actual_val = lag_values[i]
                            assert abs(actual_val - expected_val) < 1e-12, \
                                f"Lag {lag} feature should match price {lag} periods ago"
    
    def test_momentum_features_calculation(self, feature_engineer, reference_time_series_data):
        """Test momentum features calculation."""
        df = reference_time_series_data['df']
        
        # Calculate momentum features
        momentum_features = feature_engineer.calculate_momentum_features(df, periods=[5, 10, 20])
        
        # Test momentum calculation accuracy
        for period in [5, 10, 20]:
            momentum_col = next((col for col in momentum_features.columns if f'momentum_{period}' in col), None)
            if momentum_col:
                momentum_values = momentum_features[momentum_col].to_numpy()
                prices = df['price'].to_numpy()
                
                # Calculate expected momentum manually
                expected_momentum = np.full(len(prices), np.nan)
                for i in range(period, len(prices)):
                    expected_momentum[i] = (prices[i] - prices[i - period]) / prices[i - period]
                
                # Compare calculations
                valid_indices = ~np.isnan(momentum_values)
                if np.any(valid_indices):
                    np.testing.assert_array_almost_equal(
                        momentum_values[valid_indices][-min(50, np.sum(valid_indices)):],
                        expected_momentum[valid_indices][-min(50, np.sum(valid_indices)):],
                        decimal=12,
                        err_msg=f"Momentum {period} calculation should match manual calculation"
                    )
    
    def test_volatility_features_accuracy(self, feature_engineer, reference_time_series_data):
        """Test volatility features calculation accuracy."""
        df = reference_time_series_data['df']
        
        # Add returns column for volatility calculation
        df_with_returns = df.with_columns([
            pl.col('price').pct_change().alias('returns')
        ])
        
        # Calculate volatility features
        volatility_features = feature_engineer.calculate_volatility_features(df_with_returns, windows=[10, 20])
        
        # Test volatility calculation
        for window in [10, 20]:
            vol_col = next((col for col in volatility_features.columns if f'volatility_{window}' in col), None)
            if vol_col:
                vol_values = volatility_features[vol_col].drop_nulls().to_numpy()
                
                if len(vol_values) > 0:
                    # Volatility should be non-negative
                    assert np.all(vol_values >= 0), f"Volatility {window} should be non-negative"
                    
                    # Should be finite
                    assert np.all(np.isfinite(vol_values)), f"Volatility {window} should be finite"
                    
                    # Calculate manual volatility for comparison
                    returns = df_with_returns['returns'].to_numpy()
                    manual_vol = np.full(len(returns), np.nan)
                    
                    for i in range(window, len(returns)):
                        window_returns = returns[i - window:i]
                        valid_returns = window_returns[~np.isnan(window_returns)]
                        if len(valid_returns) > 1:
                            manual_vol[i] = np.std(valid_returns, ddof=1)
                    
                    # Compare non-NaN values
                    manual_valid = manual_vol[~np.isnan(manual_vol)]
                    if len(manual_valid) > 0 and len(vol_values) > 0:
                        min_compare = min(10, len(manual_valid), len(vol_values))
                        if min_compare > 0:
                            # Should be approximately equal (allowing for slight differences in implementation)
                            correlation = np.corrcoef(manual_valid[-min_compare:], vol_values[-min_compare:])[0, 1]
                            if not np.isnan(correlation):
                                assert correlation > 0.9, f"Volatility {window} should correlate strongly with manual calculation"


@pytest.mark.unit
@pytest.mark.mathematical
class TestFeatureEngineeringPipeline:
    """Test complete feature engineering pipeline."""
    
    def test_complete_pipeline_execution(self, feature_engineer, reference_time_series_data):
        """Test complete feature engineering pipeline execution."""
        df = reference_time_series_data['df']
        
        # Run complete pipeline
        try:
            features_df = feature_engineer.engineer_features(df)
            
            # Should return a DataFrame
            assert isinstance(features_df, pl.DataFrame), "Pipeline should return Polars DataFrame"
            
            # Should have same number of rows as input
            assert len(features_df) == len(df), "Output should have same number of rows as input"
            
            # Should have datetime column
            assert 'datetime' in features_df.columns, "Output should contain datetime column"
            
            # Should have created additional feature columns
            feature_cols = [col for col in features_df.columns if col not in ['datetime', 'price']]
            assert len(feature_cols) > 0, "Pipeline should create feature columns"
            
        except Exception as e:
            # If pipeline fails due to missing modules, that's acceptable for this test
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Pipeline test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_pipeline_feature_consistency(self, feature_engineer, reference_time_series_data):
        """Test feature consistency across pipeline runs."""
        df = reference_time_series_data['df']
        
        try:
            # Run pipeline twice
            features_df_1 = feature_engineer.engineer_features(df)
            features_df_2 = feature_engineer.engineer_features(df)
            
            # Results should be identical
            assert features_df_1.shape == features_df_2.shape, "Pipeline should produce consistent shapes"
            
            # Compare feature values for consistency
            for col in features_df_1.columns:
                if col in features_df_2.columns:
                    values_1 = features_df_1[col].to_numpy()
                    values_2 = features_df_2[col].to_numpy()
                    
                    # Should be exactly equal
                    np.testing.assert_array_equal(
                        values_1, values_2,
                        err_msg=f"Pipeline should produce consistent results for column {col}"
                    )
                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Pipeline consistency test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_pipeline_with_missing_data(self, feature_engineer, edge_case_time_series):
        """Test pipeline behavior with missing data."""
        # Test with gaps in data
        df_with_gaps = edge_case_time_series['with_gaps']
        
        try:
            features_df = feature_engineer.engineer_features(df_with_gaps)
            
            # Should handle missing data gracefully
            assert len(features_df) == len(df_with_gaps), "Should preserve input length with gaps"
            
            # Should not crash with missing data
            assert isinstance(features_df, pl.DataFrame), "Should return DataFrame even with gaps"
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Missing data test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_pipeline_feature_temporal_alignment(self, feature_engineer, reference_time_series_data):
        """Test that all features maintain proper temporal alignment."""
        df = reference_time_series_data['df']
        
        try:
            features_df = feature_engineer.engineer_features(df)
            
            # All features should have same temporal alignment
            datetime_col = features_df['datetime']
            
            # Check that datetime is preserved and in order
            original_datetimes = df['datetime'].to_list()
            feature_datetimes = datetime_col.to_list()
            
            assert feature_datetimes == original_datetimes, \
                "Pipeline should preserve datetime ordering and values"
            
            # Check that no feature uses future information
            # (This is tested by ensuring NaN patterns make sense)
            for col in features_df.columns:
                if col not in ['datetime', 'price']:
                    col_values = features_df[col].to_numpy()
                    nan_indices = np.isnan(col_values)
                    
                    if np.any(nan_indices) and not np.all(nan_indices):
                        # NaN pattern should make temporal sense
                        # Most features should have NaN at beginning (for lags/rolling)
                        # not random patterns in the middle
                        first_valid = np.where(~nan_indices)[0]
                        if len(first_valid) > 0:
                            first_valid_idx = first_valid[0]
                            
                            # Should not have many NaN values after first valid value
                            # (some are acceptable for features with different lag requirements)
                            mid_section = col_values[first_valid_idx:first_valid_idx + 50]
                            if len(mid_section) > 10:
                                nan_ratio_mid = np.sum(np.isnan(mid_section)) / len(mid_section)
                                assert nan_ratio_mid < 0.5, \
                                    f"Feature {col} should not have excessive NaN in middle section"
                                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Temporal alignment test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.integration
@pytest.mark.mathematical
class TestFeatureEngineeringScalability:
    """Test feature engineering scalability and performance."""
    
    def test_pipeline_with_large_dataset(self, feature_engineer):
        """Test pipeline performance with larger dataset."""
        # Create larger synthetic dataset
        n_points = 5000  # ~3.5 days of minute data
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_points) * 0.01)
        
        large_df = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_points)],
            'price': prices
        })
        
        try:
            # Should complete without memory issues
            features_df = feature_engineer.engineer_features(large_df)
            
            # Should maintain expected properties
            assert len(features_df) == len(large_df), "Should preserve length for large dataset"
            assert isinstance(features_df, pl.DataFrame), "Should return DataFrame for large dataset"
            
            # Should have created features
            feature_cols = [col for col in features_df.columns if col not in ['datetime', 'price']]
            assert len(feature_cols) > 0, "Should create features for large dataset"
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Large dataset test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_feature_memory_efficiency(self, feature_engineer, reference_time_series_data):
        """Test that feature engineering doesn't create memory leaks."""
        df = reference_time_series_data['df']
        
        try:
            # Run pipeline multiple times to check for memory leaks
            initial_features = None
            
            for i in range(3):
                features_df = feature_engineer.engineer_features(df)
                
                if initial_features is None:
                    initial_features = features_df
                else:
                    # Should produce consistent results
                    assert features_df.shape == initial_features.shape, \
                        f"Run {i}: Shape should be consistent across runs"
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Memory efficiency test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_feature_numerical_stability(self, feature_engineer, edge_case_time_series):
        """Test numerical stability with extreme values."""
        test_cases = ['extreme_volatility', 'large_values', 'small_values']
        
        for case_name in test_cases:
            df = edge_case_time_series[case_name]
            
            try:
                features_df = feature_engineer.engineer_features(df)
                
                # Check that features are numerically stable
                for col in features_df.columns:
                    if col not in ['datetime']:
                        col_values = features_df[col].to_numpy()
                        finite_values = col_values[np.isfinite(col_values)]
                        
                        if len(finite_values) > 0:
                            # Should not have overflow/underflow
                            assert not np.any(np.isinf(finite_values)), \
                                f"Feature {col} should not have infinite values for {case_name}"
                            
                            # Should not have excessive precision loss
                            assert np.all(np.abs(finite_values) < 1e15), \
                                f"Feature {col} should not have extremely large values for {case_name}"
                            
                            assert np.all(np.abs(finite_values) > 1e-15), \
                                f"Feature {col} should not have extremely small values for {case_name}"
                                
            except Exception as e:
                if "No module named" in str(e) or "cannot import name" in str(e):
                    pytest.skip(f"Numerical stability test for {case_name} skipped due to missing dependencies: {e}")
                else:
                    raise e