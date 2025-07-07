"""
Unit tests for price analysis mathematical validation
Tests mathematical correctness of all price calculation functions
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import math


@pytest.mark.unit
@pytest.mark.mathematical
class TestPriceStatisticsCalculations:
    """Test basic price statistics calculations for mathematical correctness."""
    
    def test_calculate_price_stats_against_numpy(self, price_analyzer, reference_calculations):
        """Test price statistics against numpy reference implementations."""
        prices = reference_calculations['prices']
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=len(prices)-1),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        # Calculate statistics using our implementation
        stats = price_analyzer._calculate_price_stats(df)
        ref = reference_calculations
        
        # Validate basic statistics
        assert abs(stats['mean'] - ref['mean']) < 1e-10, f"Mean mismatch: {stats['mean']} vs {ref['mean']}"
        assert abs(stats['std'] - ref['std']) < 1e-10, f"Std mismatch: {stats['std']} vs {ref['std']}"
        assert abs(stats['min'] - ref['min']) < 1e-10, f"Min mismatch: {stats['min']} vs {ref['min']}"
        assert abs(stats['max'] - ref['max']) < 1e-10, f"Max mismatch: {stats['max']} vs {ref['max']}"
        
        # Validate percentiles
        for p, expected in ref['percentiles'].items():
            actual = stats.get(p, stats.get(f'price_{p}'))
            assert actual is not None, f"Percentile {p} not found in results"
            assert abs(actual - expected) < 1e-10, f"Percentile {p} mismatch: {actual} vs {expected}"
    
    def test_calculate_price_stats_edge_cases(self, price_analyzer, edge_case_data):
        """Test price statistics with edge cases."""
        # Test single point
        stats = price_analyzer._calculate_price_stats(edge_case_data['single_point'])
        assert stats['mean'] == 100.0
        assert stats['std'] == 0.0 or math.isnan(stats['std'])  # Single point has no variance
        assert stats['min'] == stats['max'] == 100.0
        
        # Test two points
        stats = price_analyzer._calculate_price_stats(edge_case_data['two_points'])
        assert stats['mean'] == 125.0  # (100 + 150) / 2
        assert stats['min'] == 100.0
        assert stats['max'] == 150.0
        
        # Test constant prices
        constant_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': [42.0] * 100
        })
        stats = price_analyzer._calculate_price_stats(constant_df)
        assert stats['mean'] == 42.0
        assert stats['std'] == 0.0
        assert stats['min'] == stats['max'] == 42.0
    
    def test_calculate_price_stats_with_invalid_data(self, price_analyzer, edge_case_data):
        """Test price statistics with invalid data (NaN, inf, negative, zero)."""
        # Test with NaN values - should handle gracefully
        try:
            stats = price_analyzer._calculate_price_stats(edge_case_data['with_nan'])
            # Should either exclude NaN or handle gracefully
            assert stats is not None
        except (ValueError, TypeError):
            # Acceptable to raise clear error for invalid data
            pass
        
        # Test with infinite values
        try:
            stats = price_analyzer._calculate_price_stats(edge_case_data['with_inf'])
            assert stats is not None
        except (ValueError, TypeError):
            pass
        
        # Test with zero prices
        stats = price_analyzer._calculate_price_stats(edge_case_data['zero_prices'])
        assert stats['mean'] == 0.0
        assert stats['min'] == stats['max'] == 0.0


@pytest.mark.unit
@pytest.mark.mathematical
class TestReturnsCalculations:
    """Test returns calculation mathematical correctness."""
    
    def test_returns_calculation_accuracy(self, price_analyzer, reference_calculations):
        """Test returns calculation against reference implementation."""
        prices = reference_calculations['prices']
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=len(prices)-1),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        # Calculate temporal features (includes returns)
        temporal_features = price_analyzer._calculate_temporal_features(df)
        
        # Validate returns calculation
        expected_returns = reference_calculations['returns']
        
        # Check that returns are calculated correctly
        # Note: We'll validate the overall return calculation
        initial_price = prices[0]
        final_price = prices[-1]
        expected_total_return = (final_price - initial_price) / initial_price
        
        assert 'total_return' in temporal_features
        actual_total_return = temporal_features['total_return']
        assert abs(actual_total_return - expected_total_return) < 1e-10, \
            f"Total return mismatch: {actual_total_return} vs {expected_total_return}"
    
    def test_log_returns_calculation(self, price_analyzer):
        """Test log returns calculation."""
        # Create simple test case with known log returns
        prices = [100, 110, 121, 133.1]  # 10% increases
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=3),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        temporal_features = price_analyzer._calculate_temporal_features(df)
        
        # For 10% increases, log return should be approximately ln(1.1) ≈ 0.0953
        expected_log_return_per_step = math.log(1.1)
        expected_total_log_return = expected_log_return_per_step * 3
        
        # The implementation should calculate some form of log returns
        # We'll check if the magnitude is reasonable
        if 'log_total_return' in temporal_features:
            actual_log_return = temporal_features['log_total_return']
            assert abs(actual_log_return - expected_total_log_return) < 0.01, \
                f"Log return mismatch: {actual_log_return} vs {expected_total_log_return}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestVolatilityCalculations:
    """Test volatility calculation mathematical correctness."""
    
    def test_volatility_calculation_accuracy(self, price_analyzer, reference_calculations):
        """Test volatility calculation against reference implementation."""
        prices = reference_calculations['prices']
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=len(prices)-1),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        volatility_metrics = price_analyzer._calculate_volatility_metrics(df)
        
        # Validate volatility calculation
        expected_volatility = reference_calculations['volatility']
        
        # Check basic volatility metric
        assert 'volatility' in volatility_metrics or 'price_volatility' in volatility_metrics
        volatility_key = 'volatility' if 'volatility' in volatility_metrics else 'price_volatility'
        actual_volatility = volatility_metrics[volatility_key]
        
        # Allow for reasonable tolerance in volatility calculation
        assert abs(actual_volatility - expected_volatility) < 0.01, \
            f"Volatility mismatch: {actual_volatility} vs {expected_volatility}"
    
    def test_volatility_edge_cases(self, price_analyzer):
        """Test volatility calculation with edge cases."""
        # Constant prices should have zero volatility
        constant_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': [100.0] * 100
        })
        
        volatility_metrics = price_analyzer._calculate_volatility_metrics(constant_df)
        volatility_key = 'volatility' if 'volatility' in volatility_metrics else 'price_volatility'
        
        if volatility_key in volatility_metrics:
            assert volatility_metrics[volatility_key] == 0.0, \
                f"Constant prices should have zero volatility, got {volatility_metrics[volatility_key]}"
    
    def test_coefficient_of_variation(self, price_analyzer):
        """Test coefficient of variation calculation."""
        # Create data with known mean and std
        prices = [90, 100, 110]  # Mean=100, Std≈10
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=2),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        volatility_metrics = price_analyzer._calculate_volatility_metrics(df)
        
        if 'coefficient_of_variation' in volatility_metrics:
            expected_cv = np.std(prices, ddof=1) / np.mean(prices)
            actual_cv = volatility_metrics['coefficient_of_variation']
            assert abs(actual_cv - expected_cv) < 1e-10, \
                f"CV mismatch: {actual_cv} vs {expected_cv}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestMovementPatternDetection:
    """Test movement pattern detection algorithms."""
    
    def test_pump_detection(self, price_analyzer, synthetic_token_data):
        """Test pump pattern detection accuracy."""
        pump_df = synthetic_token_data['extreme_pump']
        movement_patterns = price_analyzer._calculate_movement_patterns(pump_df)
        
        # Should detect significant pump
        assert 'pump_detected' in movement_patterns or 'major_pump' in movement_patterns
        pump_key = 'pump_detected' if 'pump_detected' in movement_patterns else 'major_pump'
        
        if pump_key in movement_patterns:
            assert movement_patterns[pump_key] == True, "Should detect pump in extreme pump data"
        
        # Should have high total return
        if 'total_return' in movement_patterns:
            assert movement_patterns['total_return'] > 5.0, "Pump should have >500% return"
    
    def test_dump_detection(self, price_analyzer, synthetic_token_data):
        """Test dump pattern detection accuracy."""
        dump_df = synthetic_token_data['extreme_dump']
        movement_patterns = price_analyzer._calculate_movement_patterns(dump_df)
        
        # Should detect significant dump
        assert 'dump_detected' in movement_patterns or 'major_dump' in movement_patterns
        dump_key = 'dump_detected' if 'dump_detected' in movement_patterns else 'major_dump'
        
        if dump_key in movement_patterns:
            assert movement_patterns[dump_key] == True, "Should detect dump in extreme dump data"
        
        # Should have large drawdown
        if 'max_drawdown' in movement_patterns:
            assert movement_patterns['max_drawdown'] < -0.8, "Dump should have >80% drawdown"
    
    def test_pattern_thresholds_consistency(self, price_analyzer):
        """Test that pattern detection thresholds are mathematically consistent."""
        thresholds = price_analyzer.pattern_thresholds
        
        # Pump threshold should be positive
        assert thresholds['pump_threshold'] > 0, "Pump threshold should be positive"
        
        # Dump threshold should be negative
        assert thresholds['dump_threshold'] < 0, "Dump threshold should be negative"
        
        # Volatility threshold should be positive
        assert thresholds['volatility_threshold'] > 0, "Volatility threshold should be positive"
        
        # Thresholds should be reasonable for memecoin data
        assert 0.1 <= thresholds['pump_threshold'] <= 10.0, "Pump threshold should be reasonable"
        assert -1.0 <= thresholds['dump_threshold'] <= -0.1, "Dump threshold should be reasonable"


@pytest.mark.unit
@pytest.mark.mathematical
class TestMomentumCalculations:
    """Test momentum calculation mathematical correctness."""
    
    def test_momentum_calculation_accuracy(self, price_analyzer):
        """Test momentum calculation with known data."""
        # Create upward trending data
        prices = [100 * (1.01 ** i) for i in range(50)]  # 1% growth per minute
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=49),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        momentum_metrics = price_analyzer._calculate_momentum_metrics(df)
        
        # Should detect positive momentum
        if 'momentum' in momentum_metrics:
            assert momentum_metrics['momentum'] > 0, "Should detect positive momentum in upward trend"
        
        if 'trend_strength' in momentum_metrics:
            assert momentum_metrics['trend_strength'] > 0, "Should detect positive trend strength"
    
    def test_momentum_edge_cases(self, price_analyzer, synthetic_token_data):
        """Test momentum calculation with edge cases."""
        # Constant prices should have zero momentum
        constant_df = synthetic_token_data['constant_price']
        momentum_metrics = price_analyzer._calculate_momentum_metrics(constant_df)
        
        if 'momentum' in momentum_metrics:
            assert abs(momentum_metrics['momentum']) < 1e-10, \
                f"Constant prices should have zero momentum, got {momentum_metrics['momentum']}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestOptimalTimingCalculations:
    """Test optimal timing calculation mathematical correctness."""
    
    def test_optimal_return_calculation(self, price_analyzer, synthetic_token_data):
        """Test optimal return calculation accuracy."""
        # Use pump data where optimal timing should give high returns
        pump_df = synthetic_token_data['extreme_pump']
        optimal_metrics = price_analyzer._calculate_optimal_return(pump_df)
        
        # Should find optimal entry and exit points
        assert 'optimal_entry_time' in optimal_metrics
        assert 'optimal_exit_time' in optimal_metrics
        assert 'optimal_return' in optimal_metrics
        
        # Optimal return should be positive for pump data
        assert optimal_metrics['optimal_return'] > 0, "Optimal return should be positive for pump data"
        
        # Entry time should be before exit time
        entry_time = optimal_metrics['optimal_entry_time']
        exit_time = optimal_metrics['optimal_exit_time']
        
        if isinstance(entry_time, (int, float)) and isinstance(exit_time, (int, float)):
            assert entry_time < exit_time, "Entry time should be before exit time"
    
    def test_recovery_time_calculation(self, price_analyzer, synthetic_token_data):
        """Test recovery time calculation."""
        # Use dump data to test recovery time
        dump_df = synthetic_token_data['extreme_dump']
        
        # This should either calculate recovery time or handle gracefully
        try:
            recovery_time = price_analyzer._calculate_recovery_time(dump_df)
            if recovery_time is not None:
                assert isinstance(recovery_time, (int, float)), "Recovery time should be numeric"
                assert recovery_time >= 0, "Recovery time should be non-negative"
        except (ValueError, TypeError):
            # It's acceptable to not find recovery for extreme dumps
            pass


@pytest.mark.unit
@pytest.mark.mathematical
class TestCalculationConsistency:
    """Test consistency across different calculation methods."""
    
    def test_price_analysis_complete_pipeline(self, price_analyzer, synthetic_token_data):
        """Test complete price analysis pipeline for mathematical consistency."""
        for token_type, df in synthetic_token_data.items():
            # Run complete analysis
            result = price_analyzer.analyze_prices(df, token_type)
            
            # Should return valid result
            assert isinstance(result, dict), f"Result should be dict for {token_type}"
            assert 'status' in result, f"Result should have status for {token_type}"
            
            # If successful, should have key metrics
            if result.get('status') == 'success':
                assert 'price_stats' in result or 'basic_stats' in result, \
                    f"Should have price stats for {token_type}"
    
    def test_numerical_stability(self, price_analyzer):
        """Test numerical stability with extreme values."""
        # Very large prices
        large_prices = [1e15, 1.1e15, 1.2e15, 1.05e15]
        large_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=3),
                interval="1m",
                eager=True
            ),
            'price': large_prices
        })
        
        # Should handle large values without overflow
        result = price_analyzer.analyze_prices(large_df, "large_values_test")
        assert result is not None, "Should handle large values"
        
        # Very small prices
        small_prices = [1e-10, 1.1e-10, 1.2e-10, 1.05e-10]
        small_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=3),
                interval="1m",
                eager=True
            ),
            'price': small_prices
        })
        
        # Should handle small values without underflow
        result = price_analyzer.analyze_prices(small_df, "small_values_test")
        assert result is not None, "Should handle small values"


@pytest.mark.integration
@pytest.mark.mathematical
class TestCrossModuleConsistency:
    """Test mathematical consistency across different analysis modules."""
    
    def test_basic_stats_consistency(self, price_analyzer, token_analyzer, synthetic_token_data):
        """Test that basic statistics are consistent between price_analyzer and token_analyzer."""
        test_df = synthetic_token_data['normal_behavior']
        
        # Get stats from both analyzers
        price_result = price_analyzer.analyze_prices(test_df, "test_token")
        token_result = token_analyzer.calculate_basic_stats(test_df)
        
        # Both should calculate similar basic statistics
        # This test ensures mathematical consistency across modules
        if (price_result.get('status') == 'success' and 
            'price_stats' in price_result and
            token_result is not None):
            
            price_stats = price_result['price_stats']
            
            # Compare mean values (allowing for small differences in implementation)
            if 'mean' in price_stats and 'mean_price' in token_result:
                assert abs(price_stats['mean'] - token_result['mean_price']) < 1e-6, \
                    "Mean calculations should be consistent across modules"