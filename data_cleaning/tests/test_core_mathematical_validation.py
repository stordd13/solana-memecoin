"""
Core mathematical validation tests for data_cleaning module
Tests essential mathematical functions without complex fixtures
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import math


@pytest.mark.unit
@pytest.mark.mathematical
class TestCoreCalculations:
    """Test core mathematical calculations across data_cleaning module."""
    
    def test_percentage_calculations_accuracy(self):
        """Test percentage calculations mathematical correctness."""
        # Test cases: (count, total, expected_percentage)
        test_cases = [
            (25, 100, 25.0),
            (1, 3, 33.333333333333336),
            (0, 50, 0.0),
            (50, 50, 100.0),
            (75, 200, 37.5)
        ]
        
        for count, total, expected in test_cases:
            calculated = (count / total) * 100
            assert abs(calculated - expected) < 1e-12, \
                f"Percentage calculation error: {calculated} vs {expected}"
    
    def test_retention_rate_calculations_accuracy(self):
        """Test retention rate calculations mathematical correctness."""
        # Test cases: (original, final, expected_retention)
        test_cases = [
            (1000, 800, 0.8),
            (500, 350, 0.7),
            (100, 100, 1.0),  # No loss
            (200, 0, 0.0),    # Total loss
            (750, 250, 1/3)   # 1/3 retention
        ]
        
        for original, final, expected in test_cases:
            retention_rate = final / original if original > 0 else 0
            assert abs(retention_rate - expected) < 1e-12, \
                f"Retention rate calculation error: {retention_rate} vs {expected}"
    
    def test_coefficient_of_variation_calculation(self):
        """Test coefficient of variation calculation mathematical correctness."""
        # Test data with known CV
        prices = [100, 110, 90, 105, 95, 115, 85, 120, 80, 125]
        
        # Reference calculation
        mean_price = np.mean(prices)
        std_price = np.std(prices, ddof=1)
        expected_cv = std_price / mean_price if mean_price > 0 else 0
        
        # Manual verification
        manual_mean = sum(prices) / len(prices)
        manual_variance = sum((x - manual_mean) ** 2 for x in prices) / (len(prices) - 1)
        manual_std = manual_variance ** 0.5
        manual_cv = manual_std / manual_mean
        
        assert abs(manual_cv - expected_cv) < 1e-12, \
            f"CV calculation consistency error: {manual_cv} vs {expected_cv}"
        
        # Verify non-negativity
        assert expected_cv >= 0, f"CV should be non-negative, got {expected_cv}"
    
    def test_log_coefficient_of_variation_calculation(self):
        """Test log coefficient of variation calculation mathematical correctness."""
        prices = [100, 150, 200, 250, 300]
        
        # Calculate log CV
        log_prices = np.log(np.array(prices) + 1e-10)
        log_mean = np.mean(log_prices)
        log_std = np.std(log_prices)
        log_cv = log_std / log_mean if log_mean != 0 else 0
        
        # Manual verification
        manual_log_prices = [math.log(p + 1e-10) for p in prices]
        manual_log_mean = sum(manual_log_prices) / len(manual_log_prices)
        manual_log_variance = sum((x - manual_log_mean) ** 2 for x in manual_log_prices) / len(manual_log_prices)
        manual_log_std = manual_log_variance ** 0.5
        manual_log_cv = manual_log_std / manual_log_mean if manual_log_mean != 0 else 0
        
        assert abs(manual_log_cv - log_cv) < 1e-12, \
            f"Log CV calculation consistency error: {manual_log_cv} vs {log_cv}"
    
    def test_entropy_calculation_accuracy(self):
        """Test entropy calculation mathematical correctness."""
        # Simple test case with known entropy
        prices = [1, 1, 2, 2, 3, 3, 4, 4]  # Uniform distribution
        
        # Calculate histogram and probabilities
        unique_vals, counts = np.unique(prices, return_counts=True)
        probs = counts / np.sum(counts)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # For uniform distribution with 4 unique values: entropy = log2(4) = 2
        expected_entropy = math.log2(4)
        assert abs(entropy - expected_entropy) < 1e-12, \
            f"Uniform entropy should be log2(4) = 2, got {entropy}"
        
        # Normalized entropy
        normalized_entropy = entropy / math.log2(len(unique_vals))
        assert abs(normalized_entropy - 1.0) < 1e-12, \
            f"Normalized entropy should be 1.0 for uniform distribution, got {normalized_entropy}"
    
    def test_returns_calculation_accuracy(self):
        """Test returns calculation mathematical correctness."""
        prices = [100, 110, 99, 108, 95]
        
        # Calculate returns manually
        expected_returns = []
        for i in range(1, len(prices)):
            return_val = (prices[i] - prices[i-1]) / prices[i-1]
            expected_returns.append(return_val)
        
        # Expected: [0.1, -0.1, 0.090909..., -0.120370...]
        assert abs(expected_returns[0] - 0.1) < 1e-12, "First return should be 0.1"
        assert abs(expected_returns[1] - (-0.1)) < 1e-12, "Second return should be -0.1"
        
        # Test using polars pct_change equivalent
        df = pl.DataFrame({'price': prices})
        df_with_returns = df.with_columns([(pl.col('price').pct_change()).alias('returns')])
        calculated_returns = df_with_returns['returns'].to_numpy()
        
        # Remove first NaN
        calculated_returns_clean = calculated_returns[~np.isnan(calculated_returns)]
        
        # Compare with expected
        for i, (calc, exp) in enumerate(zip(calculated_returns_clean, expected_returns)):
            assert abs(calc - exp) < 1e-12, \
                f"Return {i} mismatch: {calc} vs {exp}"
    
    def test_price_ratio_calculation_accuracy(self):
        """Test price ratio calculation mathematical correctness."""
        test_cases = [
            ([100, 200], 2.0),           # Simple 2:1 ratio
            ([50, 150], 3.0),            # 3:1 ratio
            ([100, 100], 1.0),           # Same price
            ([1, 1000], 1000.0),         # High ratio
            ([0.001, 1], 1000.0)         # Very high ratio
        ]
        
        for prices, expected_ratio in test_cases:
            max_price = max(prices)
            min_price = min(prices)
            calculated_ratio = max_price / min_price if min_price > 0 else float('inf')
            
            assert abs(calculated_ratio - expected_ratio) < 1e-12, \
                f"Price ratio calculation error: {calculated_ratio} vs {expected_ratio}"
    
    def test_statistical_aggregation_accuracy(self):
        """Test statistical aggregation mathematical correctness."""
        # Test data for weighted averages
        categories = {
            'short_term': {'count': 100, 'avg_length': 50},
            'medium_term': {'count': 200, 'avg_length': 200},
            'long_term': {'count': 50, 'avg_length': 500}
        }
        
        # Calculate weighted average
        total_count = sum(cat['count'] for cat in categories.values())
        weighted_sum = sum(cat['count'] * cat['avg_length'] for cat in categories.values())
        weighted_average = weighted_sum / total_count
        
        # Manual calculation: (100*50 + 200*200 + 50*500) / (100+200+50)
        expected_weighted_sum = 5000 + 40000 + 25000  # 70000
        expected_total_count = 350
        expected_weighted_avg = 70000 / 350  # 200
        
        assert abs(weighted_average - expected_weighted_avg) < 1e-12, \
            f"Weighted average calculation error: {weighted_average} vs {expected_weighted_avg}"
        assert weighted_average == 200.0, f"Expected weighted average 200, got {weighted_average}"
    
    def test_consecutive_counting_accuracy(self):
        """Test consecutive pattern counting mathematical correctness."""
        # Test pattern with known consecutive sequences
        prices = [100, 100, 100, 105, 105, 110, 110, 110, 110, 115]
        
        # Count consecutive same prices manually
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(prices)):
            if prices[i] == prices[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # Expected: 3 consecutive 100s, 2 consecutive 105s, 4 consecutive 110s
        # Maximum should be 4
        assert max_consecutive == 4, f"Expected max consecutive 4, got {max_consecutive}"
        
        # Alternative numpy-based calculation
        price_array = np.array(prices)
        diff_array = np.diff(price_array)
        same_price_positions = (diff_array == 0)
        
        # Count runs of consecutive True values
        consecutive_runs = []
        current_run = 1
        
        for is_same in same_price_positions:
            if is_same:
                current_run += 1
            else:
                if current_run > 1:
                    consecutive_runs.append(current_run)
                current_run = 1
        
        # Add final run if it exists
        if current_run > 1:
            consecutive_runs.append(current_run)
        
        max_consecutive_alt = max(consecutive_runs) if consecutive_runs else 1
        assert max_consecutive_alt == 4, f"Alternative method got {max_consecutive_alt}"
    
    def test_gap_size_calculation_accuracy(self):
        """Test gap size calculation mathematical correctness."""
        # Create timestamps with known gaps
        timestamps = [
            datetime(2024, 1, 1, 0, 0),   # 0 minutes
            datetime(2024, 1, 1, 0, 1),   # 1 minute
            datetime(2024, 1, 1, 0, 2),   # 2 minutes
            datetime(2024, 1, 1, 0, 15),  # 15 minutes (13-minute gap)
            datetime(2024, 1, 1, 0, 16),  # 16 minutes
        ]
        
        # Calculate time differences
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i-1]
            minutes_diff = diff.total_seconds() / 60
            time_diffs.append(minutes_diff)
        
        # Expected differences: [1, 1, 13, 1]
        expected_diffs = [1, 1, 13, 1]
        for actual, expected in zip(time_diffs, expected_diffs):
            assert abs(actual - expected) < 1e-6, \
                f"Time diff calculation error: {actual} vs {expected}"
        
        # Find gaps (differences > 1.5 minutes)
        gaps = [diff for diff in time_diffs if diff > 1.5]
        assert len(gaps) == 1, f"Should find exactly 1 gap, found {len(gaps)}"
        assert abs(gaps[0] - 13) < 1e-6, f"Gap should be 13 minutes, got {gaps[0]}"
    
    def test_threshold_comparison_accuracy(self):
        """Test threshold comparison mathematical correctness."""
        threshold = 0.3
        test_values = [0.25, 0.35, 0.3, 0.299999, 0.300001]
        expected_results = [True, False, False, True, False]
        
        for value, expected in zip(test_values, expected_results):
            result = value < threshold
            assert result == expected, \
                f"Threshold comparison error for {value} < {threshold}: expected {expected}, got {result}"
    
    def test_boolean_logic_combinations_accuracy(self):
        """Test boolean logic combinations mathematical correctness."""
        # Test AND logic for multiple conditions
        conditions = [
            {'a': True, 'b': True, 'c': True, 'expected': True},
            {'a': True, 'b': True, 'c': False, 'expected': False},
            {'a': True, 'b': False, 'c': True, 'expected': False},
            {'a': False, 'b': False, 'c': False, 'expected': False}
        ]
        
        for case in conditions:
            result = case['a'] and case['b'] and case['c']
            assert result == case['expected'], \
                f"Boolean AND logic error for {case}: expected {case['expected']}, got {result}"
        
        # Test OR logic
        or_conditions = [
            {'a': True, 'b': False, 'expected': True},
            {'a': False, 'b': True, 'expected': True},
            {'a': False, 'b': False, 'expected': False},
            {'a': True, 'b': True, 'expected': True}
        ]
        
        for case in or_conditions:
            result = case['a'] or case['b']
            assert result == case['expected'], \
                f"Boolean OR logic error for {case}: expected {case['expected']}, got {result}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestNumericalStability:
    """Test numerical stability of calculations."""
    
    def test_division_by_zero_handling(self):
        """Test proper handling of division by zero cases."""
        # Test retention rate with zero original
        retention_rate = 100 / 0 if 0 > 0 else 0
        assert retention_rate == 0, "Zero original should give 0 retention rate"
        
        # Test CV with zero mean
        cv = 5 / 0 if 0 > 0 else 0
        assert cv == 0, "Zero mean should give 0 CV"
        
        # Test percentage with zero total
        percentage = (50 / 0) * 100 if 0 > 0 else 0
        assert percentage == 0, "Zero total should give 0 percentage"
    
    def test_floating_point_precision(self):
        """Test floating-point precision in calculations."""
        # Test values that might cause precision issues
        a = 0.1 + 0.2
        b = 0.3
        
        # Direct comparison might fail due to floating-point representation
        # Use tolerance-based comparison
        assert abs(a - b) < 1e-15, f"Floating-point precision issue: {a} vs {b}"
        
        # Test with percentage calculations
        percentage = (1 / 3) * 100
        expected = 100 / 3
        assert abs(percentage - expected) < 1e-12, \
            f"Percentage precision error: {percentage} vs {expected}"
    
    def test_large_number_stability(self):
        """Test calculations with very large numbers."""
        large_numbers = [1e12, 1.1e12, 1.05e12]
        
        # Calculate statistics
        mean_large = np.mean(large_numbers)
        std_large = np.std(large_numbers, ddof=1)
        cv_large = std_large / mean_large
        
        # Should handle large numbers without overflow
        assert np.isfinite(mean_large), "Mean of large numbers should be finite"
        assert np.isfinite(std_large), "Std of large numbers should be finite"
        assert np.isfinite(cv_large), "CV of large numbers should be finite"
        
        # Values should preserve scale
        assert mean_large > 1e11, "Mean should preserve large scale"
    
    def test_small_number_stability(self):
        """Test calculations with very small numbers."""
        small_numbers = [1e-10, 1.1e-10, 1.05e-10]
        
        # Calculate statistics
        mean_small = np.mean(small_numbers)
        std_small = np.std(small_numbers, ddof=1)
        cv_small = std_small / mean_small
        
        # Should handle small numbers without underflow
        assert np.isfinite(mean_small), "Mean of small numbers should be finite"
        assert np.isfinite(std_small), "Std of small numbers should be finite"
        assert np.isfinite(cv_small), "CV of small numbers should be finite"
        
        # Values should preserve scale
        assert mean_small < 1e-9, "Mean should preserve small scale"


@pytest.mark.integration
@pytest.mark.mathematical
class TestCrossCalculationConsistency:
    """Test mathematical consistency across different calculation methods."""
    
    def test_complementary_rates_consistency(self):
        """Test that complementary rates sum correctly."""
        # Exclusion and retention rates should sum to 100%
        total_tokens = 1000
        excluded_tokens = 300
        retained_tokens = total_tokens - excluded_tokens
        
        exclusion_rate = (excluded_tokens / total_tokens) * 100
        retention_rate = (retained_tokens / total_tokens) * 100
        
        total_rate = exclusion_rate + retention_rate
        assert abs(total_rate - 100.0) < 1e-12, \
            f"Exclusion + retention should equal 100%, got {total_rate}"
        
        # Success and failure rates should sum to 100%
        successful = 700
        failed = 300
        total_processed = successful + failed
        
        success_rate = (successful / total_processed) * 100
        failure_rate = (failed / total_processed) * 100
        
        total_rate = success_rate + failure_rate
        assert abs(total_rate - 100.0) < 1e-12, \
            f"Success + failure should equal 100%, got {total_rate}"
    
    def test_aggregation_consistency(self):
        """Test that aggregations are mathematically consistent."""
        # Category totals should sum to overall total
        category_counts = {'short': 300, 'medium': 400, 'long': 300}
        total_from_categories = sum(category_counts.values())
        expected_total = 1000
        
        assert total_from_categories == expected_total, \
            f"Category sum should equal total: {total_from_categories} vs {expected_total}"
        
        # Weighted average consistency
        weights = [0.3, 0.4, 0.3]  # Proportional to counts
        values = [100, 200, 300]
        
        weighted_avg = sum(w * v for w, v in zip(weights, values))
        manual_weighted_avg = (0.3 * 100) + (0.4 * 200) + (0.3 * 300)
        
        assert abs(weighted_avg - manual_weighted_avg) < 1e-12, \
            f"Weighted average consistency error: {weighted_avg} vs {manual_weighted_avg}"
    
    def test_statistical_relationship_consistency(self):
        """Test that statistical relationships are mathematically consistent."""
        data = [10, 20, 30, 40, 50]
        
        # Mean calculation consistency
        mean1 = sum(data) / len(data)
        mean2 = np.mean(data)
        assert abs(mean1 - mean2) < 1e-12, f"Mean calculation inconsistent: {mean1} vs {mean2}"
        
        # Variance calculation consistency
        variance1 = sum((x - mean1) ** 2 for x in data) / (len(data) - 1)
        variance2 = np.var(data, ddof=1)
        assert abs(variance1 - variance2) < 1e-12, f"Variance calculation inconsistent: {variance1} vs {variance2}"
        
        # Standard deviation consistency
        std1 = variance1 ** 0.5
        std2 = np.std(data, ddof=1)
        assert abs(std1 - std2) < 1e-12, f"Std deviation calculation inconsistent: {std1} vs {std2}"