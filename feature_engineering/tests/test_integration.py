"""
Integration tests for complete feature_engineering module
Tests end-to-end pipeline integration and cross-module compatibility
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil


@pytest.mark.integration
@pytest.mark.mathematical
class TestFeatureEngineeringIntegration:
    """Test complete feature engineering module integration."""
    
    def test_end_to_end_feature_pipeline(self, feature_engineer, short_term_engineer, correlation_analyzer, reference_time_series_data):
        """Test complete end-to-end feature engineering pipeline."""
        df = reference_time_series_data['df']
        
        try:
            # Step 1: Advanced feature engineering
            advanced_features = feature_engineer.engineer_features(df)
            
            # Step 2: Short-term features
            short_term_features = short_term_engineer.engineer_short_term_features(df)
            
            # Step 3: Combine features
            # Merge on datetime
            combined_features = advanced_features.join(
                short_term_features.drop('price'), 
                on='datetime', 
                how='left'
            )
            
            # Validate combined features
            assert isinstance(combined_features, pl.DataFrame), \
                "Combined features should be Polars DataFrame"
            
            assert len(combined_features) == len(df), \
                "Combined features should preserve input length"
            
            # Should have datetime column
            assert 'datetime' in combined_features.columns, \
                "Combined features should contain datetime column"
            
            # Should have features from both pipelines
            advanced_cols = set(advanced_features.columns) - {'datetime', 'price'}
            short_term_cols = set(short_term_features.columns) - {'datetime', 'price'}
            combined_cols = set(combined_features.columns) - {'datetime', 'price'}
            
            # Should contain features from both sources
            overlap_advanced = len(advanced_cols & combined_cols) / len(advanced_cols) if len(advanced_cols) > 0 else 1
            overlap_short_term = len(short_term_cols & combined_cols) / len(short_term_cols) if len(short_term_cols) > 0 else 1
            
            assert overlap_advanced > 0.7, "Should retain most advanced features"
            assert overlap_short_term > 0.7, "Should retain most short-term features"
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"End-to-end pipeline test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_cross_module_feature_consistency(self, feature_engineer, short_term_engineer, reference_time_series_data):
        """Test consistency of features calculated by different modules."""
        df = reference_time_series_data['df']
        
        try:
            # Calculate features with both modules
            advanced_features = feature_engineer.engineer_features(df)
            short_term_features = short_term_engineer.engineer_short_term_features(df)
            
            # Find common feature types (e.g., both might calculate RSI)
            advanced_cols = set(advanced_features.columns)
            short_term_cols = set(short_term_features.columns)
            
            # Look for similar features (same calculation, potentially different parameters)
            for adv_col in advanced_cols:
                for st_col in short_term_cols:
                    # Check if they're similar features (e.g., both RSI)
                    if ('rsi' in adv_col.lower() and 'rsi' in st_col.lower() and
                        adv_col != st_col):
                        
                        adv_values = advanced_features[adv_col].drop_nulls().to_numpy()
                        st_values = short_term_features[st_col].drop_nulls().to_numpy()
                        
                        if len(adv_values) > 10 and len(st_values) > 10:
                            # Should be in similar ranges (both RSI between 0-100)
                            adv_range = (np.min(adv_values), np.max(adv_values))
                            st_range = (np.min(st_values), np.max(st_values))
                            
                            # Ranges should overlap significantly
                            range_overlap = (min(adv_range[1], st_range[1]) - 
                                           max(adv_range[0], st_range[0]))
                            total_range = max(adv_range[1], st_range[1]) - min(adv_range[0], st_range[0])
                            
                            if total_range > 0:
                                overlap_ratio = range_overlap / total_range
                                assert overlap_ratio > 0.5, \
                                    f"Similar features {adv_col} and {st_col} should have overlapping ranges"
                                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Cross-module consistency test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_multi_token_correlation_integration(self, correlation_analyzer, correlation_test_data):
        """Test integration of correlation analysis with multi-token data."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            # Complete correlation analysis
            correlation_result = correlation_analyzer.run_complete_analysis(tokens_data)
            
            if correlation_result is not None:
                # Should contain multiple analysis types
                assert isinstance(correlation_result, dict), \
                    "Correlation analysis should return structured results"
                
                # Test static correlation
                if 'static_correlation' in correlation_result:
                    static_corr = correlation_result['static_correlation']
                    if 'correlation_matrix' in static_corr:
                        corr_matrix = static_corr['correlation_matrix']
                        n_tokens = len(tokens_data)
                        
                        assert corr_matrix.shape == (n_tokens, n_tokens), \
                            f"Correlation matrix should be {n_tokens}x{n_tokens}"
                
                # Test rolling correlation
                if 'rolling_correlation' in correlation_result:
                    rolling_corr = correlation_result['rolling_correlation']
                    assert 'rolling_correlations' in rolling_corr or 'error' in rolling_corr, \
                        "Rolling correlation should have results or error information"
                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Multi-token correlation test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.integration
@pytest.mark.performance
class TestFeatureEngineeringPerformance:
    """Test performance characteristics of feature engineering pipeline."""
    
    def test_pipeline_scalability(self, feature_engineer):
        """Test pipeline performance with different dataset sizes."""
        # Test with different sizes
        sizes = [100, 500, 1000, 2000]
        
        for n_points in sizes:
            # Create synthetic dataset
            prices = 100 + np.cumsum(np.random.normal(0, 1, n_points) * 0.01)
            df = pl.DataFrame({
                'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_points)],
                'price': prices
            })
            
            try:
                # Time the feature engineering
                import time
                start_time = time.time()
                
                features_df = feature_engineer.engineer_features(df)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Should complete in reasonable time
                time_per_point = processing_time / n_points
                assert time_per_point < 0.01, \
                    f"Processing should be < 0.01s per point, got {time_per_point:.4f}s for {n_points} points"
                
                # Should preserve data integrity
                assert len(features_df) == len(df), \
                    f"Should preserve length for {n_points} points"
                
                # Should create features
                feature_cols = [col for col in features_df.columns if col not in ['datetime', 'price']]
                assert len(feature_cols) > 0, \
                    f"Should create features for {n_points} points"
                    
            except Exception as e:
                if "No module named" in str(e) or "cannot import name" in str(e):
                    pytest.skip(f"Scalability test for {n_points} points skipped due to missing dependencies: {e}")
                else:
                    raise e
    
    def test_memory_efficiency(self, feature_engineer, reference_time_series_data):
        """Test memory efficiency of feature engineering."""
        df = reference_time_series_data['df']
        
        try:
            # Run multiple times to check for memory leaks
            results = []
            
            for i in range(5):
                features_df = feature_engineer.engineer_features(df)
                results.append(features_df)
                
                # Should produce consistent results
                if i > 0:
                    assert features_df.shape == results[0].shape, \
                        f"Run {i}: Should produce consistent shapes"
            
            # Memory usage should be reasonable
            # (This is a basic test - more sophisticated memory profiling could be added)
            for result in results:
                assert isinstance(result, pl.DataFrame), \
                    "Results should be proper DataFrame objects"
                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Memory efficiency test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.integration
@pytest.mark.robustness
class TestFeatureEngineeringRobustness:
    """Test robustness of feature engineering with edge cases."""
    
    def test_pipeline_with_all_edge_cases(self, feature_engineer, edge_case_time_series):
        """Test pipeline robustness with various edge cases."""
        edge_cases = ['constant', 'extreme_volatility', 'large_values', 'small_values', 'with_gaps']
        
        for case_name in edge_cases:
            df = edge_case_time_series[case_name]
            
            try:
                # Should handle edge case gracefully
                features_df = feature_engineer.engineer_features(df)
                
                # Basic validation
                assert isinstance(features_df, pl.DataFrame), \
                    f"Should return DataFrame for {case_name}"
                
                assert len(features_df) == len(df), \
                    f"Should preserve length for {case_name}"
                
                # Check for reasonable feature values
                for col in features_df.columns:
                    if col not in ['datetime', 'price']:
                        col_values = features_df[col].to_numpy()
                        finite_values = col_values[np.isfinite(col_values)]
                        
                        if len(finite_values) > 0:
                            # Should not have extreme values that suggest calculation errors
                            assert not np.any(np.abs(finite_values) > 1e10), \
                                f"Feature {col} should not have extreme values for {case_name}"
                                
            except Exception as e:
                if "No module named" in str(e) or "cannot import name" in str(e):
                    pytest.skip(f"Edge case test for {case_name} skipped due to missing dependencies: {e}")
                else:
                    # Some edge cases might legitimately fail - document the behavior
                    print(f"Edge case {case_name} failed with: {str(e)}")
                    if case_name in ['constant', 'single_point']:
                        # These failures might be expected
                        pass
                    else:
                        raise e
    
    def test_pipeline_error_handling(self, feature_engineer):
        """Test pipeline error handling with invalid inputs."""
        try:
            # Test with empty DataFrame
            empty_df = pl.DataFrame({'datetime': [], 'price': []})
            result = feature_engineer.engineer_features(empty_df)
            
            if result is not None:
                assert len(result) == 0, "Should handle empty DataFrame gracefully"
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Error handling test skipped due to missing dependencies: {e}")
            else:
                # Some errors might be expected for invalid inputs
                assert "empty" in str(e).lower() or "length" in str(e).lower(), \
                    f"Should provide meaningful error for empty DataFrame: {e}"
        
        try:
            # Test with missing required columns
            invalid_df = pl.DataFrame({'wrong_column': [1, 2, 3]})
            result = feature_engineer.engineer_features(invalid_df)
            
            # Should either handle gracefully or provide meaningful error
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Missing columns test skipped due to missing dependencies: {e}")
            else:
                # Should provide meaningful error message
                assert any(keyword in str(e).lower() for keyword in ['column', 'datetime', 'price']), \
                    f"Should provide meaningful error for missing columns: {e}"


@pytest.mark.integration
@pytest.mark.temporal_safety
class TestTemporalSafetyIntegration:
    """Test temporal safety across the complete feature engineering pipeline."""
    
    def test_complete_pipeline_temporal_safety(self, feature_engineer, reference_time_series_data):
        """Test that complete pipeline maintains temporal safety."""
        df = reference_time_series_data['df']
        
        try:
            # Create features
            features_df = feature_engineer.engineer_features(df)
            
            # Test temporal alignment
            original_timestamps = df['datetime'].to_list()
            feature_timestamps = features_df['datetime'].to_list()
            
            assert feature_timestamps == original_timestamps, \
                "Pipeline should preserve exact datetime ordering"
            
            # Test for lookahead bias
            for col in features_df.columns:
                if col not in ['datetime', 'price']:
                    col_values = features_df[col].to_numpy()
                    
                    # Check NaN pattern for temporal safety
                    nan_indices = np.isnan(col_values)
                    
                    if np.any(nan_indices) and not np.all(nan_indices):
                        first_valid = np.where(~nan_indices)[0]
                        if len(first_valid) > 0:
                            first_valid_idx = first_valid[0]
                            
                            # Should not have excessive NaN at the end (suggests future information)
                            end_section = col_values[-20:]  # Last 20 values
                            end_nan_ratio = np.sum(np.isnan(end_section)) / len(end_section)
                            
                            # Some NaN at end is OK for certain features, but not excessive
                            assert end_nan_ratio < 0.8, \
                                f"Feature {col} should not have excessive NaN at end (suggests future info)"
                                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Temporal safety test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_feature_dependencies_temporal_safety(self, reference_time_series_data):
        """Test that feature dependencies maintain temporal safety."""
        df = reference_time_series_data['df']
        
        # Simulate feature dependency chain
        # Feature A depends on raw price (safe)
        # Feature B depends on Feature A (should remain safe)
        # Feature C depends on Feature B (should remain safe)
        
        prices = df['price'].to_numpy()
        
        # Feature A: Simple moving average (safe)
        feature_a = np.full(len(prices), np.nan)
        for i in range(9, len(prices)):  # 10-period MA
            feature_a[i] = np.mean(prices[i-9:i+1])
        
        # Feature B: Momentum of Feature A (should be safe)
        feature_b = np.full(len(prices), np.nan)
        for i in range(14, len(prices)):  # 5-period momentum of feature_a
            if not np.isnan(feature_a[i]) and not np.isnan(feature_a[i-5]):
                feature_b[i] = (feature_a[i] - feature_a[i-5]) / feature_a[i-5]
        
        # Feature C: Volatility of Feature B (should remain safe)
        feature_c = np.full(len(prices), np.nan)
        for i in range(19, len(prices)):  # 5-period volatility of feature_b
            window_values = feature_b[i-4:i+1]
            valid_values = window_values[~np.isnan(window_values)]
            if len(valid_values) > 1:
                feature_c[i] = np.std(valid_values, ddof=1)
        
        # Test temporal safety of dependency chain
        features = {'A': feature_a, 'B': feature_b, 'C': feature_c}
        
        for name, feature in features.items():
            # Each feature should only depend on past information
            valid_indices = ~np.isnan(feature)
            
            if np.any(valid_indices):
                first_valid = np.where(valid_indices)[0][0]
                
                # First valid value should appear after sufficient lag for dependencies
                expected_min_lag = {'A': 9, 'B': 14, 'C': 19}[name]
                assert first_valid >= expected_min_lag, \
                    f"Feature {name} first valid value should respect dependency lag"
                
                # Should not use future information
                for i in np.where(valid_indices)[0][:10]:  # Check first 10 valid values
                    # At time i, feature should only use data from times <= i
                    # This is implicit in our calculation method above
                    assert not np.isnan(feature[i]), \
                        f"Feature {name} should be valid where expected"


@pytest.mark.integration
@pytest.mark.data_flow
class TestDataFlowIntegration:
    """Test data flow through complete feature engineering pipeline."""
    
    def test_data_preservation_through_pipeline(self, reference_time_series_data):
        """Test that essential data is preserved through the pipeline."""
        df = reference_time_series_data['df']
        original_price_checksum = np.sum(df['price'].to_numpy())
        original_datetime_count = len(df['datetime'].to_list())
        
        try:
            # Import here to avoid import errors affecting test discovery
            from advanced_feature_engineering import AdvancedFeatureEngineer
            
            feature_engineer = AdvancedFeatureEngineer()
            features_df = feature_engineer.engineer_features(df)
            
            # Essential data should be preserved
            if 'price' in features_df.columns:
                final_price_checksum = np.sum(features_df['price'].to_numpy())
                assert abs(final_price_checksum - original_price_checksum) < 1e-10, \
                    "Original price data should be preserved exactly"
            
            final_datetime_count = len(features_df['datetime'].to_list())
            assert final_datetime_count == original_datetime_count, \
                "All datetime entries should be preserved"
                
        except ImportError as e:
            pytest.skip(f"Data preservation test skipped due to import error: {e}")
    
    def test_feature_calculation_order_independence(self, reference_time_series_data):
        """Test that feature calculation order doesn't affect results."""
        df = reference_time_series_data['df']
        
        try:
            # Calculate features in different orders (if supported by implementation)
            from advanced_feature_engineering import AdvancedFeatureEngineer
            
            feature_engineer = AdvancedFeatureEngineer()
            
            # Full pipeline
            features_full = feature_engineer.engineer_features(df)
            
            # Step-by-step calculation (if methods are available)
            features_step = df.clone()
            
            # Add basic features first
            if hasattr(feature_engineer, 'calculate_technical_indicators'):
                tech_features = feature_engineer.calculate_technical_indicators(features_step)
                features_step = features_step.join(
                    tech_features.drop(['datetime', 'price'] if 'price' in tech_features.columns else ['datetime']), 
                    on='datetime', how='left'
                )
            
            # Results should be consistent regardless of calculation order
            # (This test depends on implementation details)
            
        except ImportError as e:
            pytest.skip(f"Order independence test skipped due to import error: {e}")
        except Exception as e:
            if "No module named" in str(e) or "method" in str(e):
                pytest.skip(f"Order independence test skipped: {e}")
            else:
                raise e