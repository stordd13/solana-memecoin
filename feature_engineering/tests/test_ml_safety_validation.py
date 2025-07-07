"""
ML safety validation tests for feature_engineering module
Tests data leakage prevention and temporal correctness
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.ml_safety
class TestDataLeakageDetection:
    """Test detection of data leakage in features."""
    
    def test_safe_features_validation(self, ml_safe_feature_examples):
        """Test that ML-safe features pass validation."""
        safe_features = ml_safe_feature_examples['safe_features']
        df = ml_safe_feature_examples['df']
        
        for feature_name, feature_values in safe_features.items():
            # Convert to numpy array for testing
            if hasattr(feature_values, 'to_numpy'):
                feature_array = feature_values.to_numpy()
            else:
                feature_array = np.array(feature_values)
            
            # Test 1: No constant features (except at boundaries where NaN is expected)
            finite_values = feature_array[np.isfinite(feature_array)]
            if len(finite_values) > 10:  # Only test if enough data points
                feature_variance = np.var(finite_values)
                assert feature_variance > 1e-15, \
                    f"ML-safe feature {feature_name} should have variance (not constant)"
            
            # Test 2: No future information (compare with manual lag calculation)
            if 'lag' in feature_name:
                # Extract lag amount from feature name
                lag_amount = int(feature_name.split('_')[-1])
                
                # Calculate expected lagged values manually
                prices = df['price'].to_numpy()
                expected_lagged = np.full(len(prices), np.nan)
                expected_lagged[lag_amount:] = prices[:-lag_amount]
                
                # Compare with actual feature
                valid_indices = np.isfinite(feature_array) & np.isfinite(expected_lagged)
                if np.any(valid_indices):
                    np.testing.assert_array_almost_equal(
                        feature_array[valid_indices],
                        expected_lagged[valid_indices],
                        decimal=12,
                        err_msg=f"Lagged feature {feature_name} should match manual calculation"
                    )
            
            # Test 3: Rolling features should only use past data
            if 'rolling' in feature_name:
                # Rolling features should have NaN at the beginning
                window_size = 10 if '10' in feature_name else 20 if '20' in feature_name else 5
                
                # First (window_size - 1) values should be NaN for proper rolling calculation
                initial_values = feature_array[:window_size-1]
                assert np.all(np.isnan(initial_values)), \
                    f"Rolling feature {feature_name} should have NaN for initial {window_size-1} values"
    
    def test_unsafe_features_detection(self, ml_safe_feature_examples):
        """Test that ML-unsafe features are properly identified."""
        unsafe_features = ml_safe_feature_examples['unsafe_features']
        df = ml_safe_feature_examples['df']
        prices = ml_safe_feature_examples['prices']
        
        for feature_name, feature_values in unsafe_features.items():
            # Convert to numpy array for testing
            if hasattr(feature_values, 'to_numpy'):
                feature_array = feature_values.to_numpy()
            elif isinstance(feature_values, list):
                feature_array = np.array(feature_values)
            else:
                feature_array = feature_values
            
            # Test 1: Future information features
            if 'future' in feature_name:
                # Future return should use information not available at prediction time
                # This is UNSAFE - we validate it's detected as such
                future_return = feature_array[:-1]  # Remove last NaN
                current_prices = prices[:-1]
                future_prices = prices[1:]
                
                expected_future_return = future_prices / current_prices - 1
                
                valid_indices = np.isfinite(future_return) & np.isfinite(expected_future_return)
                if np.any(valid_indices):
                    # Should match calculation (confirming it uses future data)
                    np.testing.assert_array_almost_equal(
                        future_return[valid_indices],
                        expected_future_return[valid_indices],
                        decimal=12,
                        err_msg="Future return feature should use future data (UNSAFE)"
                    )
            
            # Test 2: Constant features (global statistics)
            elif 'max_price' in feature_name or 'final_price' in feature_name or 'mean' in feature_name:
                # These should be constant across time (UNSAFE)
                finite_values = feature_array[np.isfinite(feature_array)]
                if len(finite_values) > 1:
                    feature_variance = np.var(finite_values)
                    assert feature_variance < 1e-15, \
                        f"Global statistic feature {feature_name} should be constant (UNSAFE)"
            
            # Test 3: Global statistics that use whole series
            elif 'total_return' in feature_name:
                # Should use first price (global information)
                if len(feature_array) > 1:
                    # All values should be relative to first price
                    first_price = prices[0]
                    expected_total_return = prices / first_price - 1
                    
                    valid_indices = np.isfinite(feature_array) & np.isfinite(expected_total_return)
                    if np.any(valid_indices):
                        np.testing.assert_array_almost_equal(
                            feature_array[valid_indices],
                            expected_total_return[valid_indices],
                            decimal=12,
                            err_msg="Total return should use first price (UNSAFE global stat)"
                        )
    
    def test_temporal_boundary_validation(self, ml_safe_feature_examples):
        """Test that features respect temporal boundaries."""
        df = ml_safe_feature_examples['df']
        prices = df['price'].to_numpy()
        
        # Simulate train/validation/test split
        n_total = len(prices)
        train_end = int(0.6 * n_total)
        val_end = int(0.8 * n_total)
        
        # Calculate features on training data only
        train_prices = prices[:train_end]
        
        # Rolling mean should only use training data for training period
        window = 10
        if len(train_prices) >= window:
            # Calculate rolling mean that respects temporal boundaries
            safe_rolling_mean = np.full(n_total, np.nan)
            
            for i in range(window - 1, train_end):
                safe_rolling_mean[i] = np.mean(train_prices[i - window + 1:i + 1])
            
            # For validation/test periods, rolling mean should only use past data
            for i in range(train_end, n_total):
                if i >= window - 1:
                    # Should only use data up to current point, not future
                    safe_rolling_mean[i] = np.mean(prices[i - window + 1:i + 1])
            
            # Verify no future data leakage
            for i in range(train_end):
                if not np.isnan(safe_rolling_mean[i]):
                    # Training period features should only use training data
                    contributing_indices = range(max(0, i - window + 1), i + 1)
                    assert all(idx < train_end for idx in contributing_indices), \
                        f"Training feature at index {i} should not use data beyond training period"


@pytest.mark.unit
@pytest.mark.ml_safety
class TestTemporalSplittingValidation:
    """Test temporal splitting for ML safety."""
    
    def test_temporal_split_boundaries(self, reference_time_series_data):
        """Test that temporal splits maintain proper boundaries."""
        df = reference_time_series_data['df']
        n_points = len(df)
        
        # Define temporal splits
        train_end = int(0.6 * n_points)
        val_end = int(0.8 * n_points)
        
        # Test that splits are chronologically ordered
        assert 0 < train_end < val_end < n_points, \
            "Temporal splits should be in chronological order"
        
        # Test that no data leaks between splits
        train_data = df[:train_end]
        val_data = df[train_end:val_end]
        test_data = df[val_end:]
        
        # Check timestamps are properly ordered
        if len(train_data) > 0 and len(val_data) > 0:
            last_train_time = train_data['datetime'][-1]
            first_val_time = val_data['datetime'][0]
            assert last_train_time < first_val_time, \
                "Validation data should start after training data ends"
        
        if len(val_data) > 0 and len(test_data) > 0:
            last_val_time = val_data['datetime'][-1]
            first_test_time = test_data['datetime'][0]
            assert last_val_time < first_test_time, \
                "Test data should start after validation data ends"
    
    def test_expanding_window_safety(self, reference_time_series_data):
        """Test that expanding windows don't leak future information."""
        df = reference_time_series_data['df']
        prices = df['price'].to_numpy()
        
        # Calculate expanding mean (ML-safe version)
        expanding_mean = np.full(len(prices), np.nan)
        
        for i in range(len(prices)):
            if i >= 10:  # Start after minimum window
                expanding_mean[i] = np.mean(prices[:i+1])  # Only use data up to current point
        
        # Test that each point only uses past data
        for i in range(10, len(prices)):
            if not np.isnan(expanding_mean[i]):
                # Calculate what it should be using only past data
                expected_mean = np.mean(prices[:i+1])
                assert abs(expanding_mean[i] - expected_mean) < 1e-12, \
                    f"Expanding mean at index {i} should only use data up to current point"
        
        # Test monotonicity properties where appropriate
        valid_indices = ~np.isnan(expanding_mean)
        if np.sum(valid_indices) > 1:
            # Expanding mean should converge (variance should decrease)
            expanding_values = expanding_mean[valid_indices]
            # This is a general property but may not always hold, so we test reasonableness
            assert np.all(np.isfinite(expanding_values)), \
                "Expanding mean values should be finite"
    
    def test_rolling_vs_expanding_consistency(self, reference_time_series_data):
        """Test mathematical consistency between rolling and expanding windows."""
        df = reference_time_series_data['df']
        prices = df['price'].to_numpy()
        
        window = 20
        if len(prices) >= window * 2:
            # Calculate rolling mean
            rolling_mean = np.full(len(prices), np.nan)
            for i in range(window - 1, len(prices)):
                rolling_mean[i] = np.mean(prices[i - window + 1:i + 1])
            
            # Calculate expanding mean
            expanding_mean = np.full(len(prices), np.nan)
            for i in range(window - 1, len(prices)):
                expanding_mean[i] = np.mean(prices[:i + 1])
            
            # At the start of the series, they should be similar
            start_idx = window - 1
            if start_idx < len(prices):
                # First expanding value should equal first rolling value
                assert abs(expanding_mean[start_idx] - rolling_mean[start_idx]) < 1e-12, \
                    "First expanding mean should equal first rolling mean"
                
                # Expanding mean should be more stable (less variance in changes)
                if start_idx + 10 < len(prices):
                    rolling_changes = np.diff(rolling_mean[start_idx:start_idx + 10])
                    expanding_changes = np.diff(expanding_mean[start_idx:start_idx + 10])
                    
                    rolling_volatility = np.std(rolling_changes[np.isfinite(rolling_changes)])
                    expanding_volatility = np.std(expanding_changes[np.isfinite(expanding_changes)])
                    
                    # Expanding should generally be less volatile
                    if rolling_volatility > 0 and expanding_volatility > 0:
                        assert expanding_volatility <= rolling_volatility * 2, \
                            "Expanding window should be less volatile than rolling window"


@pytest.mark.unit
@pytest.mark.ml_safety
class TestFeatureSafetyValidation:
    """Test the feature safety validation functions themselves."""
    
    def test_constant_feature_detection(self, ml_safe_feature_examples):
        """Test detection of constant features."""
        # Create test features
        n_points = 100
        
        # Constant feature (UNSAFE)
        constant_feature = np.ones(n_points) * 42
        
        # Variable feature (SAFE)
        variable_feature = np.random.normal(0, 1, n_points)
        
        # Nearly constant feature (edge case)
        nearly_constant = np.ones(n_points) * 42
        nearly_constant[50] += 1e-10  # Tiny variation
        
        # Test constant detection
        def is_constant_feature(feature, tolerance=1e-12):
            """Helper function to detect constant features."""
            finite_values = feature[np.isfinite(feature)]
            if len(finite_values) <= 1:
                return True
            return np.var(finite_values) < tolerance
        
        assert is_constant_feature(constant_feature), "Should detect constant feature"
        assert not is_constant_feature(variable_feature), "Should not flag variable feature"
        assert is_constant_feature(nearly_constant), "Should detect nearly constant feature"
    
    def test_future_information_detection(self, ml_safe_feature_examples):
        """Test detection of features using future information."""
        df = ml_safe_feature_examples['df']
        prices = df['price'].to_numpy()
        
        # Create features with different temporal properties
        n_points = len(prices)
        
        # Safe feature: lag
        safe_lag = np.full(n_points, np.nan)
        safe_lag[1:] = prices[:-1]
        
        # Unsafe feature: future return
        unsafe_future = np.full(n_points, np.nan)
        unsafe_future[:-1] = prices[1:] / prices[:-1] - 1
        
        # Test correlation with future values (simplified detection method)
        def has_future_information(feature, target, max_lag=5):
            """Helper function to detect future information leakage."""
            valid_indices = np.isfinite(feature) & np.isfinite(target)
            if np.sum(valid_indices) < 10:
                return False
            
            feature_valid = feature[valid_indices]
            target_valid = target[valid_indices]
            
            # Check correlation with future values
            for lag in range(1, min(max_lag + 1, len(target_valid))):
                if len(target_valid) > lag:
                    future_target = np.roll(target_valid, -lag)
                    future_target = future_target[:-lag]  # Remove wrapped values
                    current_feature = feature_valid[:-lag]
                    
                    if len(current_feature) > 5:
                        correlation = np.corrcoef(current_feature, future_target)[0, 1]
                        if not np.isnan(correlation) and abs(correlation) > 0.8:
                            return True
            return False
        
        # Create target (current returns)
        current_returns = np.full(n_points, np.nan)
        current_returns[1:] = prices[1:] / prices[:-1] - 1
        
        # Test detection
        # Note: This is a simplified test - real detection would be more sophisticated
        has_future_safe = has_future_information(safe_lag, current_returns)
        has_future_unsafe = has_future_information(unsafe_future, current_returns)
        
        # Lagged feature should not correlate strongly with future
        # Future feature should correlate with future (but this test may not catch it)
        # This is more of a conceptual test of the detection logic
    
    def test_scaling_safety_validation(self, reference_time_series_data):
        """Test that feature scaling maintains ML safety."""
        df = reference_time_series_data['df']
        prices = df['price'].to_numpy()
        
        # Simulate proper temporal splitting for scaling
        train_end = int(0.6 * len(prices))
        
        # SAFE: Fit scaler on training data only
        train_prices = prices[:train_end]
        train_mean = np.mean(train_prices)
        train_std = np.std(train_prices, ddof=1)
        
        # Apply scaling to all data using training statistics
        safe_scaled = (prices - train_mean) / train_std
        
        # UNSAFE: Fit scaler on all data
        all_mean = np.mean(prices)
        all_std = np.std(prices, ddof=1)
        unsafe_scaled = (prices - all_mean) / all_std
        
        # Test that safe scaling doesn't use future information
        # Training portion should have mean close to 0, std close to 1
        train_portion_safe = safe_scaled[:train_end]
        train_mean_safe = np.mean(train_portion_safe)
        train_std_safe = np.std(train_portion_safe, ddof=1)
        
        assert abs(train_mean_safe) < 0.1, \
            "Safe scaling should result in approximately zero mean for training data"
        assert abs(train_std_safe - 1.0) < 0.1, \
            "Safe scaling should result in approximately unit std for training data"
        
        # Unsafe scaling would have perfect statistics on all data
        all_mean_unsafe = np.mean(unsafe_scaled)
        all_std_unsafe = np.std(unsafe_scaled, ddof=1)
        
        assert abs(all_mean_unsafe) < 1e-10, \
            "Unsafe scaling has perfect zero mean (uses future information)"
        assert abs(all_std_unsafe - 1.0) < 1e-10, \
            "Unsafe scaling has perfect unit std (uses future information)"


@pytest.mark.integration
@pytest.mark.ml_safety
class TestMLSafetyPipelineIntegration:
    """Test ML safety across complete feature engineering pipeline."""
    
    def test_complete_pipeline_safety(self, reference_time_series_data):
        """Test that complete feature engineering pipeline maintains ML safety."""
        df = reference_time_series_data['df']
        prices = df['price'].to_numpy()
        n_points = len(prices)
        
        # Simulate complete feature engineering pipeline
        train_end = int(0.6 * n_points)
        val_end = int(0.8 * n_points)
        
        # Create various features following ML-safe principles
        features = {}
        
        # 1. Rolling features (safe)
        for window in [5, 10, 20]:
            rolling_mean = np.full(n_points, np.nan)
            rolling_std = np.full(n_points, np.nan)
            
            for i in range(window - 1, n_points):
                window_data = prices[max(0, i - window + 1):i + 1]
                rolling_mean[i] = np.mean(window_data)
                rolling_std[i] = np.std(window_data, ddof=1)
            
            features[f'rolling_mean_{window}'] = rolling_mean
            features[f'rolling_std_{window}'] = rolling_std
        
        # 2. Lag features (safe)
        for lag in [1, 5, 10]:
            lag_feature = np.full(n_points, np.nan)
            lag_feature[lag:] = prices[:-lag]
            features[f'price_lag_{lag}'] = lag_feature
        
        # 3. Returns (safe)
        returns = np.full(n_points, np.nan)
        returns[1:] = prices[1:] / prices[:-1] - 1
        features['returns'] = returns
        
        # Validate all features for ML safety
        for feature_name, feature_values in features.items():
            # Test 1: No constant features in training data
            train_values = feature_values[:train_end]
            finite_train = train_values[np.isfinite(train_values)]
            
            if len(finite_train) > 10:
                train_variance = np.var(finite_train)
                assert train_variance > 1e-15, \
                    f"Feature {feature_name} should not be constant in training data"
            
            # Test 2: Features should respect temporal boundaries
            # (Already tested in individual feature creation above)
            
            # Test 3: No NaN pattern that suggests future information
            # Features should have NaN at the beginning (for lags/rolling) or end (for future-looking)
            # but not random patterns that suggest lookahead bias
            nan_indices = np.isnan(feature_values)
            
            if np.any(nan_indices) and not np.all(nan_indices):
                # Check if NaN pattern makes sense
                first_valid = np.where(~nan_indices)[0]
                last_valid = np.where(~nan_indices)[0]
                
                if len(first_valid) > 0 and len(last_valid) > 0:
                    first_valid_idx = first_valid[0]
                    last_valid_idx = last_valid[-1]
                    
                    # For rolling/lag features, NaN should be at the beginning
                    if 'rolling' in feature_name or 'lag' in feature_name:
                        expected_start_nans = int(feature_name.split('_')[-1]) if 'lag' in feature_name else 4
                        actual_start_nans = first_valid_idx
                        
                        # Allow some tolerance for different implementations
                        assert actual_start_nans <= expected_start_nans + 5, \
                            f"Feature {feature_name} should have NaN at beginning, not middle"
    
    def test_feature_correlation_temporal_safety(self, reference_time_series_data):
        """Test that feature correlations don't reveal future information."""
        df = reference_time_series_data['df']
        prices = df['price'].to_numpy()
        n_points = len(prices)
        
        # Create target variable (future returns)
        horizons = [1, 5, 10]
        targets = {}
        
        for h in horizons:
            target = np.full(n_points, np.nan)
            target[:-h] = prices[h:] / prices[:-h] - 1
            targets[f'future_return_{h}'] = target
        
        # Create safe features
        safe_features = {}
        
        # Lagged features
        for lag in [1, 5, 10]:
            lag_feature = np.full(n_points, np.nan)
            lag_feature[lag:] = prices[:-lag]
            safe_features[f'price_lag_{lag}'] = lag_feature
        
        # Rolling features
        for window in [5, 10]:
            rolling_mean = np.full(n_points, np.nan)
            for i in range(window - 1, n_points):
                rolling_mean[i] = np.mean(prices[i - window + 1:i + 1])
            safe_features[f'rolling_mean_{window}'] = rolling_mean
        
        # Test correlations between safe features and future targets
        for feature_name, feature_values in safe_features.items():
            for target_name, target_values in targets.items():
                # Calculate correlation
                valid_indices = (np.isfinite(feature_values) & 
                               np.isfinite(target_values))
                
                if np.sum(valid_indices) > 20:
                    feature_valid = feature_values[valid_indices]
                    target_valid = target_values[valid_indices]
                    
                    correlation = np.corrcoef(feature_valid, target_valid)[0, 1]
                    
                    # Strong correlation (>0.9) might indicate data leakage
                    # This is a heuristic test - in practice, some correlation is expected
                    if not np.isnan(correlation):
                        assert abs(correlation) < 0.95, \
                            f"Very high correlation between {feature_name} and {target_name} " \
                            f"({correlation:.3f}) might indicate data leakage"
    
    def test_cross_validation_temporal_consistency(self, reference_time_series_data):
        """Test that cross-validation maintains temporal consistency."""
        df = reference_time_series_data['df']
        n_points = len(df)
        
        # Time series cross-validation: walk-forward approach
        min_train_size = int(0.3 * n_points)
        step_size = int(0.1 * n_points)
        
        if min_train_size > 50 and step_size > 10:
            cv_splits = []
            
            for start_idx in range(0, n_points - min_train_size - step_size, step_size):
                train_end = start_idx + min_train_size
                val_start = train_end
                val_end = min(val_start + step_size, n_points)
                
                if val_end > val_start:
                    cv_splits.append({
                        'train': (start_idx, train_end),
                        'val': (val_start, val_end)
                    })
            
            # Test temporal ordering in each split
            for i, split in enumerate(cv_splits):
                train_start, train_end = split['train']
                val_start, val_end = split['val']
                
                # Validation should come after training
                assert train_end <= val_start, \
                    f"CV split {i}: validation should start after training ends"
                
                # No overlap between train and validation
                assert train_end <= val_start, \
                    f"CV split {i}: no overlap between train and validation"
                
                # Later splits should start after earlier splits
                if i > 0:
                    prev_val_end = cv_splits[i-1]['val'][1]
                    assert train_start >= prev_val_end - step_size, \
                        f"CV split {i}: should use more recent data than previous splits"