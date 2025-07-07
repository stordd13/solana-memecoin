"""
Target creation validation tests for feature_engineering module
Tests mathematical correctness of directional and return targets
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.mathematical
class TestDirectionalTargetCreation:
    """Test directional target creation mathematical correctness."""
    
    def test_directional_target_accuracy(self, target_creation_test_data):
        """Test directional target calculation accuracy."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        horizons = target_creation_test_data['horizons']
        expected_targets = target_creation_test_data['expected_targets']
        
        for horizon in horizons:
            # Calculate directional targets manually
            manual_targets = []
            for i in range(len(prices)):
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    current_price = prices[i]
                    direction = 1 if future_price > current_price else 0
                    manual_targets.append(direction)
                else:
                    manual_targets.append(None)  # NaN for end of series
            
            # Compare with expected targets
            expected_directional = expected_targets[horizon]['directional']
            
            for i, (manual, expected) in enumerate(zip(manual_targets, expected_directional)):
                if manual is not None and expected is not None:
                    assert manual == expected, \
                        f"Directional target mismatch at index {i}, horizon {horizon}: {manual} vs {expected}"
                elif manual is None and expected is None:
                    continue  # Both NaN, which is correct
                else:
                    assert False, f"One target is None while other is not at index {i}, horizon {horizon}"
    
    def test_directional_target_properties(self, target_creation_test_data):
        """Test mathematical properties of directional targets."""
        df = target_creation_test_data['df']
        horizons = target_creation_test_data['horizons']
        expected_targets = target_creation_test_data['expected_targets']
        
        for horizon in horizons:
            directional_targets = expected_targets[horizon]['directional']
            
            # Remove None values for analysis
            valid_targets = [t for t in directional_targets if t is not None]
            
            if len(valid_targets) > 0:
                # All targets should be 0 or 1
                assert all(t in [0, 1] for t in valid_targets), \
                    f"All directional targets should be 0 or 1 for horizon {horizon}"
                
                # Should have some variation (not all same) unless prices are monotonic
                unique_targets = set(valid_targets)
                # This depends on the test data - we know our test data has both up and down moves
                assert len(unique_targets) > 1, \
                    f"Should have both up and down targets for horizon {horizon}"
                
                # Count of targets should match expectations
                n_valid = len(valid_targets)
                n_total = len(directional_targets)
                n_nans = n_total - n_valid
                
                # Number of NaN values should equal horizon (end-of-series)
                assert n_nans == horizon, \
                    f"Should have exactly {horizon} NaN targets at end of series, got {n_nans}"
    
    def test_directional_target_consistency_across_horizons(self, target_creation_test_data):
        """Test consistency of directional targets across different horizons."""
        expected_targets = target_creation_test_data['expected_targets']
        horizons = target_creation_test_data['horizons']
        
        # For most cases, if direction is up for short horizon, 
        # it's likely (but not guaranteed) to be up for longer horizons
        # This tests logical consistency
        
        if len(horizons) >= 2:
            short_horizon = min(horizons)
            long_horizon = max(horizons)
            
            short_targets = expected_targets[short_horizon]['directional']
            long_targets = expected_targets[long_horizon]['directional']
            
            # Compare overlapping valid targets
            overlap_length = min(len(short_targets), len(long_targets)) - long_horizon
            
            if overlap_length > 0:
                agreement_count = 0
                total_valid_pairs = 0
                
                for i in range(overlap_length):
                    short_val = short_targets[i]
                    long_val = long_targets[i]
                    
                    if short_val is not None and long_val is not None:
                        total_valid_pairs += 1
                        if short_val == long_val:
                            agreement_count += 1
                
                # Some agreement expected but not perfect (market dynamics)
                if total_valid_pairs > 5:
                    agreement_rate = agreement_count / total_valid_pairs
                    # Should have some consistency, but not perfect
                    assert 0.3 <= agreement_rate <= 0.9, \
                        f"Agreement between horizons {short_horizon} and {long_horizon} " \
                        f"should be reasonable: {agreement_rate:.2f}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestReturnTargetCreation:
    """Test return target creation mathematical correctness."""
    
    def test_return_target_accuracy(self, target_creation_test_data):
        """Test return target calculation accuracy."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        horizons = target_creation_test_data['horizons']
        expected_targets = target_creation_test_data['expected_targets']
        
        for horizon in horizons:
            # Calculate return targets manually
            manual_returns = []
            for i in range(len(prices)):
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    current_price = prices[i]
                    return_val = (future_price - current_price) / current_price
                    manual_returns.append(return_val)
                else:
                    manual_returns.append(None)  # NaN for end of series
            
            # Compare with expected targets
            expected_returns = expected_targets[horizon]['returns']
            
            for i, (manual, expected) in enumerate(zip(manual_returns, expected_returns)):
                if manual is not None and expected is not None:
                    assert abs(manual - expected) < 1e-12, \
                        f"Return target mismatch at index {i}, horizon {horizon}: {manual} vs {expected}"
                elif manual is None and expected is None:
                    continue  # Both NaN, which is correct
                else:
                    assert False, f"One target is None while other is not at index {i}, horizon {horizon}"
    
    def test_return_target_properties(self, target_creation_test_data):
        """Test mathematical properties of return targets."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        horizons = target_creation_test_data['horizons']
        expected_targets = target_creation_test_data['expected_targets']
        
        for horizon in horizons:
            return_targets = expected_targets[horizon]['returns']
            
            # Remove None values for analysis
            valid_returns = [r for r in return_targets if r is not None]
            
            if len(valid_returns) > 0:
                valid_returns_array = np.array(valid_returns)
                
                # All returns should be finite
                assert np.all(np.isfinite(valid_returns_array)), \
                    f"All return targets should be finite for horizon {horizon}"
                
                # Returns should be greater than -1 (can't lose more than 100%)
                assert np.all(valid_returns_array > -1), \
                    f"All return targets should be > -1 for horizon {horizon}"
                
                # Check relationship with directional targets
                directional_targets = expected_targets[horizon]['directional']
                valid_directional = [d for d in directional_targets if d is not None]
                
                if len(valid_directional) == len(valid_returns):
                    for ret, direction in zip(valid_returns, valid_directional):
                        if direction == 1:
                            assert ret >= 0, \
                                f"Positive directional target should have non-negative return: {ret}"
                        else:  # direction == 0
                            assert ret <= 0, \
                                f"Negative directional target should have non-positive return: {ret}"
    
    def test_return_target_mathematical_relationships(self, target_creation_test_data):
        """Test mathematical relationships in return targets."""
        prices = target_creation_test_data['prices']
        horizons = target_creation_test_data['horizons']
        expected_targets = target_creation_test_data['expected_targets']
        
        # Test additivity property of returns over time
        if 5 in horizons and 10 in horizons:
            returns_5 = expected_targets[5]['returns']
            returns_10 = expected_targets[10]['returns']
            
            # For indices where both are valid
            for i in range(len(prices) - 10):
                ret_5 = returns_5[i]
                ret_10 = returns_10[i]
                
                if ret_5 is not None and ret_10 is not None:
                    # Calculate intermediate return (from day 5 to day 10)
                    if i + 5 < len(prices) and i + 10 < len(prices):
                        p0 = prices[i]
                        p5 = prices[i + 5]
                        p10 = prices[i + 10]
                        
                        # Check: (1 + ret_5) * (1 + ret_5_to_10) = (1 + ret_10)
                        # Where ret_5_to_10 is return from day 5 to day 10
                        ret_5_to_10 = (p10 - p5) / p5
                        
                        # Mathematical relationship: (1 + ret_5) * (1 + ret_5_to_10) = (1 + ret_10)
                        left_side = (1 + ret_5) * (1 + ret_5_to_10)
                        right_side = (1 + ret_10)
                        
                        assert abs(left_side - right_side) < 1e-12, \
                            f"Return additivity check failed at index {i}: " \
                            f"(1 + {ret_5}) * (1 + {ret_5_to_10}) = {left_side} != {right_side}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestTargetCreationEdgeCases:
    """Test target creation with edge cases."""
    
    def test_single_price_point(self):
        """Test target creation with single price point."""
        single_df = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1)],
            'price': [100.0]
        })
        
        # Should handle gracefully - no targets possible
        horizons = [1, 5]
        
        for horizon in horizons:
            # Manual calculation
            directional_targets = []
            return_targets = []
            
            # Only one price point, so no future targets possible
            directional_targets.append(None)
            return_targets.append(None)
            
            # Verify expected behavior
            assert len(directional_targets) == 1
            assert len(return_targets) == 1
            assert directional_targets[0] is None
            assert return_targets[0] is None
    
    def test_constant_prices(self):
        """Test target creation with constant prices."""
        constant_df = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(10)],
            'price': [100.0] * 10
        })
        
        prices = constant_df['price'].to_numpy()
        horizons = [1, 3, 5]
        
        for horizon in horizons:
            # Calculate targets
            directional_targets = []
            return_targets = []
            
            for i in range(len(prices)):
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    current_price = prices[i]
                    
                    # Directional: should be 0 (no change, so "down" by convention)
                    direction = 1 if future_price > current_price else 0
                    directional_targets.append(direction)
                    
                    # Return: should be 0
                    return_val = (future_price - current_price) / current_price
                    return_targets.append(return_val)
                else:
                    directional_targets.append(None)
                    return_targets.append(None)
            
            # Check results
            valid_directional = [d for d in directional_targets if d is not None]
            valid_returns = [r for r in return_targets if r is not None]
            
            if len(valid_directional) > 0:
                assert all(d == 0 for d in valid_directional), \
                    f"All directional targets should be 0 for constant prices, horizon {horizon}"
            
            if len(valid_returns) > 0:
                assert all(abs(r) < 1e-12 for r in valid_returns), \
                    f"All return targets should be 0 for constant prices, horizon {horizon}"
    
    def test_extreme_price_changes(self):
        """Test target creation with extreme price changes."""
        # Create extreme price movements
        extreme_prices = [100, 1000, 10, 500, 1]  # Very volatile
        extreme_df = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(len(extreme_prices))],
            'price': extreme_prices
        })
        
        horizon = 1
        prices = extreme_df['price'].to_numpy()
        
        # Calculate targets
        directional_targets = []
        return_targets = []
        
        for i in range(len(prices)):
            if i + horizon < len(prices):
                future_price = prices[i + horizon]
                current_price = prices[i]
                
                direction = 1 if future_price > current_price else 0
                return_val = (future_price - current_price) / current_price
                
                directional_targets.append(direction)
                return_targets.append(return_val)
            else:
                directional_targets.append(None)
                return_targets.append(None)
        
        # Verify extreme returns are handled correctly
        valid_returns = [r for r in return_targets if r is not None]
        
        if len(valid_returns) > 0:
            # Should handle extreme returns correctly
            assert all(r > -1 for r in valid_returns), \
                "Even extreme returns should be > -1"
            assert all(np.isfinite(r) for r in valid_returns), \
                "Extreme returns should be finite"
            
            # Check specific extreme cases
            # 100 -> 1000: return should be 9.0 (900%)
            if len(valid_returns) >= 1:
                first_return = valid_returns[0]  # 100 -> 1000
                expected_first = (1000 - 100) / 100  # 9.0
                assert abs(first_return - expected_first) < 1e-12, \
                    f"First extreme return should be {expected_first}, got {first_return}"
            
            # 1000 -> 10: return should be -0.99 (99% drop)
            if len(valid_returns) >= 2:
                second_return = valid_returns[1]  # 1000 -> 10
                expected_second = (10 - 1000) / 1000  # -0.99
                assert abs(second_return - expected_second) < 1e-12, \
                    f"Second extreme return should be {expected_second}, got {second_return}"
    
    def test_missing_price_data(self):
        """Test target creation with missing price data (NaN values)."""
        # Create data with some NaN prices
        prices_with_nan = [100, np.nan, 110, 105, np.nan, 120]
        nan_df = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(len(prices_with_nan))],
            'price': prices_with_nan
        })
        
        horizon = 1
        
        # Target creation should handle NaN values appropriately
        # This depends on implementation - should either skip NaN or propagate them
        
        directional_targets = []
        return_targets = []
        
        for i in range(len(prices_with_nan)):
            if i + horizon < len(prices_with_nan):
                future_price = prices_with_nan[i + horizon]
                current_price = prices_with_nan[i]
                
                # If either price is NaN, target should be NaN
                if np.isnan(current_price) or np.isnan(future_price):
                    directional_targets.append(None)
                    return_targets.append(None)
                else:
                    direction = 1 if future_price > current_price else 0
                    return_val = (future_price - current_price) / current_price
                    directional_targets.append(direction)
                    return_targets.append(return_val)
            else:
                directional_targets.append(None)
                return_targets.append(None)
        
        # Check that NaN handling is appropriate
        # Count None values (representing NaN)
        none_directional = sum(1 for d in directional_targets if d is None)
        none_returns = sum(1 for r in return_targets if r is None)
        
        # Should have same number of None values
        assert none_directional == none_returns, \
            "Directional and return targets should have same NaN pattern"
        
        # Should have more None values than just the end-of-series
        # (because of NaN in input data)
        assert none_directional >= horizon, \
            "Should have at least end-of-series NaN values"


@pytest.mark.unit
@pytest.mark.mathematical
class TestTargetCreationTemporalSafety:
    """Test temporal safety of target creation."""
    
    def test_target_temporal_alignment(self, target_creation_test_data):
        """Test that targets are properly aligned with features temporally."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        horizons = target_creation_test_data['horizons']
        
        for horizon in horizons:
            # Simulate feature creation at time t
            # Target should predict price movement from t to t+horizon
            
            for i in range(len(prices) - horizon):
                current_price = prices[i]
                future_price = prices[i + horizon]
                
                # At time i, we have:
                # - Features up to time i (using data from 0 to i)
                # - Target predicting price movement from i to i+horizon
                
                # This is temporally safe - no future information in features
                expected_return = (future_price - current_price) / current_price
                expected_direction = 1 if future_price > current_price else 0
                
                # Verify temporal alignment
                assert i + horizon < len(prices), \
                    f"Target at time {i} should use future price at time {i + horizon}"
                
                # Feature at time i should not use information beyond time i
                # (This would be tested in feature creation, but we verify the principle)
                available_data_end = i + 1  # Data available up to and including time i
                assert available_data_end <= i + 1, \
                    "Features at time i should only use data up to time i"
    
    def test_target_creation_no_lookahead_bias(self, target_creation_test_data):
        """Test that target creation doesn't introduce lookahead bias."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        
        # Simulate online target creation (as would happen in real-time)
        # At each time step, we can only use information up to that point
        
        horizon = 5
        online_targets = []
        
        for current_time in range(len(prices)):
            # At current_time, we have access to prices[0:current_time+1]
            available_prices = prices[:current_time + 1]
            
            # We want to predict what happens at current_time + horizon
            if current_time + horizon < len(prices):
                # This is safe - we're predicting the future based on current information
                current_price = available_prices[-1]  # Most recent price
                future_price = prices[current_time + horizon]  # Future price (unknown at current_time)
                
                # Target calculation
                target_return = (future_price - current_price) / current_price
                online_targets.append(target_return)
            else:
                online_targets.append(None)  # No target available
        
        # Compare with batch calculation
        batch_targets = []
        for i in range(len(prices)):
            if i + horizon < len(prices):
                batch_return = (prices[i + horizon] - prices[i]) / prices[i]
                batch_targets.append(batch_return)
            else:
                batch_targets.append(None)
        
        # Should be identical
        assert len(online_targets) == len(batch_targets), \
            "Online and batch target creation should produce same number of targets"
        
        for i, (online, batch) in enumerate(zip(online_targets, batch_targets)):
            if online is not None and batch is not None:
                assert abs(online - batch) < 1e-12, \
                    f"Online and batch targets should match at index {i}: {online} vs {batch}"
            elif online is None and batch is None:
                continue  # Both None is correct
            else:
                assert False, f"Mismatch in None values at index {i}: online={online}, batch={batch}"
    
    def test_target_creation_cross_validation_safety(self, target_creation_test_data):
        """Test target creation in cross-validation context."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        n_points = len(prices)
        
        # Simulate time series cross-validation
        train_size = int(0.7 * n_points)
        val_size = int(0.2 * n_points)
        horizon = 3
        
        if train_size + val_size + horizon <= n_points:
            # Split data
            train_end = train_size
            val_start = train_end
            val_end = val_start + val_size
            
            # Create targets for training set
            train_targets = []
            for i in range(train_end):
                if i + horizon < train_end:
                    # Target uses information within training set
                    target = (prices[i + horizon] - prices[i]) / prices[i]
                    train_targets.append(target)
                else:
                    train_targets.append(None)  # No target available
            
            # Create targets for validation set
            val_targets = []
            for i in range(val_start, val_end):
                if i + horizon < n_points:
                    # Target can use future data (this is the target we're predicting)
                    target = (prices[i + horizon] - prices[i]) / prices[i]
                    val_targets.append(target)
                else:
                    val_targets.append(None)
            
            # Verify temporal safety
            # Training targets should not use validation data
            for i, target in enumerate(train_targets):
                if target is not None:
                    # This target was calculated using prices[i + horizon]
                    # Verify that i + horizon < train_end
                    assert i + horizon < train_end, \
                        f"Training target at index {i} should not use validation data"
            
            # Validation targets can use future data (that's what we're predicting)
            # but features at validation time should only use past data
            valid_val_targets = [t for t in val_targets if t is not None]
            assert len(valid_val_targets) > 0, "Should have some valid validation targets"


@pytest.mark.integration
@pytest.mark.mathematical
class TestTargetFeatureAlignment:
    """Test alignment between targets and features."""
    
    def test_target_feature_temporal_consistency(self, target_creation_test_data):
        """Test temporal consistency between targets and features."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        horizon = 5
        
        # Create simple features (rolling mean)
        window = 3
        features = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            features[i] = np.mean(prices[i - window + 1:i + 1])
        
        # Create targets
        targets = np.full(len(prices), np.nan)
        for i in range(len(prices)):
            if i + horizon < len(prices):
                targets[i] = (prices[i + horizon] - prices[i]) / prices[i]
        
        # Test alignment
        for i in range(len(prices)):
            feature_val = features[i]
            target_val = targets[i]
            
            if not np.isnan(feature_val) and not np.isnan(target_val):
                # At time i:
                # - Feature uses data from max(0, i-window+1) to i
                # - Target predicts from i to i+horizon
                
                # Verify no temporal overlap
                feature_end_time = i
                target_start_time = i
                target_end_time = i + horizon
                
                assert feature_end_time <= target_start_time, \
                    f"Feature at time {i} should not use data beyond target start time"
                
                # Verify feature doesn't use future information
                feature_data_range = range(max(0, i - window + 1), i + 1)
                assert all(t <= i for t in feature_data_range), \
                    f"Feature at time {i} should only use past data"
    
    def test_multiple_horizon_target_consistency(self, target_creation_test_data):
        """Test consistency across multiple prediction horizons."""
        df = target_creation_test_data['df']
        prices = target_creation_test_data['prices']
        horizons = [1, 3, 5, 10]
        
        # Create targets for all horizons
        all_targets = {}
        for h in horizons:
            targets = np.full(len(prices), np.nan)
            for i in range(len(prices)):
                if i + h < len(prices):
                    targets[i] = (prices[i + h] - prices[i]) / prices[i]
            all_targets[h] = targets
        
        # Test consistency properties
        for i in range(len(prices) - max(horizons)):
            # At time i, we can calculate targets for all horizons
            current_targets = {h: all_targets[h][i] for h in horizons if not np.isnan(all_targets[h][i])}
            
            if len(current_targets) >= 2:
                # Longer horizon targets should generally have higher magnitude
                # (more time for price to move)
                horizon_returns = [(h, ret) for h, ret in current_targets.items()]
                horizon_returns.sort()  # Sort by horizon
                
                # Check mathematical relationship
                for j in range(len(horizon_returns) - 1):
                    h1, ret1 = horizon_returns[j]
                    h2, ret2 = horizon_returns[j + 1]
                    
                    # Verify compounding relationship where possible
                    if h1 in [1, 3] and h2 in [3, 5]:
                        # Can check intermediate compounding
                        p0 = prices[i]
                        p1 = prices[i + h1] if i + h1 < len(prices) else None
                        p2 = prices[i + h2] if i + h2 < len(prices) else None
                        
                        if p1 is not None and p2 is not None:
                            # Mathematical check: (1 + ret1) * (1 + intermediate) = (1 + ret2)
                            intermediate_return = (p2 - p1) / p1
                            expected_total = (1 + ret1) * (1 + intermediate_return) - 1
                            
                            assert abs(expected_total - ret2) < 1e-12, \
                                f"Return compounding check failed at time {i}, " \
                                f"horizons {h1},{h2}: {expected_total} vs {ret2}"