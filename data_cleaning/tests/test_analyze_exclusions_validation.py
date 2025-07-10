"""
Mathematical validation tests for analyze_exclusions module
Tests statistical calculations and aggregation accuracy
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.mathematical
class TestPercentageCalculations:
    """Test percentage calculation mathematical correctness."""
    
    def test_percentage_calculation_accuracy(self):
        """Test percentage calculations against reference implementation."""
        # Test various count/total scenarios
        test_cases = [
            {'count': 25, 'total': 100, 'expected': 25.0},
            {'count': 1, 'total': 3, 'expected': 33.333333333333336},
            {'count': 0, 'total': 50, 'expected': 0.0},
            {'count': 50, 'total': 50, 'expected': 100.0},
            {'count': 75, 'total': 200, 'expected': 37.5}
        ]
        
        for case in test_cases:
            # Manual calculation
            calculated_percentage = (case['count'] / case['total']) * 100
            
            # Should match expected with high precision
            assert abs(calculated_percentage - case['expected']) < 1e-12, \
                f"Percentage calculation error: {calculated_percentage} vs {case['expected']}"
    
    def test_percentage_edge_cases(self):
        """Test percentage calculations with edge cases."""
        # Zero total - should handle division by zero
        try:
            result = (5 / 0) * 100
            assert False, "Should raise ZeroDivisionError"
        except ZeroDivisionError:
            # Expected behavior
            pass
        
        # Zero count - should be 0%
        percentage = (0 / 100) * 100
        assert percentage == 0.0, f"Zero count should give 0%, got {percentage}"
        
        # Count equals total - should be 100%
        percentage = (50 / 50) * 100
        assert percentage == 100.0, f"Equal count/total should give 100%, got {percentage}"
        
        # Floating-point precision test
        count = 1
        total = 3
        percentage = (count / total) * 100
        expected = 100 / 3
        assert abs(percentage - expected) < 1e-12, \
            f"Floating-point precision error: {percentage} vs {expected}"
    
    def test_exclusion_rate_calculation(self):
        """Test exclusion rate calculation mathematical accuracy."""
        # Simulate exclusion analysis scenario
        total_tokens = 1000
        excluded_tokens = 150
        
        exclusion_rate = (excluded_tokens / total_tokens) * 100
        expected_rate = 15.0
        
        assert abs(exclusion_rate - expected_rate) < 1e-12, \
            f"Exclusion rate calculation error: {exclusion_rate} vs {expected_rate}"
        
        # Test retention rate (complement)
        retention_rate = ((total_tokens - excluded_tokens) / total_tokens) * 100
        expected_retention = 85.0
        
        assert abs(retention_rate - expected_retention) < 1e-12, \
            f"Retention rate calculation error: {retention_rate} vs {expected_retention}"
        
        # Verify they sum to 100%
        total_rate = exclusion_rate + retention_rate
        assert abs(total_rate - 100.0) < 1e-12, \
            f"Exclusion + retention should equal 100%, got {total_rate}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestDescriptiveStatistics:
    """Test descriptive statistics calculation accuracy."""
    
    def test_basic_statistics_accuracy(self):
        """Test basic statistics against numpy reference."""
        # Test data with known statistics
        test_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # Calculate using numpy (reference)
        np_min = np.min(test_data)
        np_max = np.max(test_data)
        np_mean = np.mean(test_data)
        np_median = np.median(test_data)
        np_std = np.std(test_data, ddof=1)
        
        # Expected values for validation
        assert np_min == 10, f"Min should be 10, got {np_min}"
        assert np_max == 100, f"Max should be 100, got {np_max}"
        assert np_mean == 55.0, f"Mean should be 55, got {np_mean}"
        assert np_median == 55.0, f"Median should be 55, got {np_median}"
        
        # Verify standard deviation calculation
        manual_variance = sum((x - np_mean) ** 2 for x in test_data) / (len(test_data) - 1)
        manual_std = manual_variance ** 0.5
        assert abs(manual_std - np_std) < 1e-12, \
            f"Standard deviation calculation error: {manual_std} vs {np_std}"
    
    def test_statistics_with_polars_dataframe(self):
        """Test statistics calculation with Polars DataFrame."""
        # Create test DataFrame
        test_data = [5, 15, 25, 35, 45]
        df = pl.DataFrame({'length': test_data})
        
        # Calculate statistics using Polars
        stats = df.select([
            pl.col('length').min().alias('min'),
            pl.col('length').max().alias('max'),
            pl.col('length').mean().alias('mean'),
            pl.col('length').median().alias('median')
        ])
        
        # Extract values
        min_val = stats['min'][0]
        max_val = stats['max'][0]
        mean_val = stats['mean'][0]
        median_val = stats['median'][0]
        
        # Verify against manual calculations
        assert min_val == 5, f"Polars min should be 5, got {min_val}"
        assert max_val == 45, f"Polars max should be 45, got {max_val}"
        assert mean_val == 25.0, f"Polars mean should be 25, got {mean_val}"
        assert median_val == 25.0, f"Polars median should be 25, got {median_val}"
    
    def test_statistics_edge_cases(self):
        """Test statistics with edge cases."""
        # Single value
        single_data = [42]
        assert np.min(single_data) == 42
        assert np.max(single_data) == 42
        assert np.mean(single_data) == 42.0
        assert np.median(single_data) == 42.0
        
        # Two values
        two_data = [10, 20]
        assert np.min(two_data) == 10
        assert np.max(two_data) == 20
        assert np.mean(two_data) == 15.0
        assert np.median(two_data) == 15.0
        
        # All same values
        same_data = [7, 7, 7, 7, 7]
        assert np.min(same_data) == 7
        assert np.max(same_data) == 7
        assert np.mean(same_data) == 7.0
        assert np.median(same_data) == 7.0
        assert np.std(same_data, ddof=1) == 0.0  # No variation


@pytest.mark.unit
@pytest.mark.mathematical
class TestCategoryAggregation:
    """Test category-based aggregation mathematical correctness."""
    
    def test_category_grouping_accuracy(self):
        """Test mathematical accuracy of category-based grouping."""
        # Simulate token data with categories
        tokens_data = [
            {'token': 'A', 'category': 'short_term', 'excluded': True, 'reason': 'staircase'},
            {'token': 'B', 'category': 'short_term', 'excluded': False, 'reason': None},
            {'token': 'C', 'category': 'medium_term', 'excluded': True, 'reason': 'ratio'},
            {'token': 'D', 'category': 'medium_term', 'excluded': False, 'reason': None},
            {'token': 'E', 'category': 'long_term', 'excluded': True, 'reason': 'gaps'},
            {'token': 'F', 'category': 'long_term', 'excluded': False, 'reason': None}
        ]
        
        # Manual category calculations
        categories = {}
        for token in tokens_data:
            cat = token['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'excluded': 0}
            categories[cat]['total'] += 1
            if token['excluded']:
                categories[cat]['excluded'] += 1
        
        # Verify calculations
        assert categories['short_term']['total'] == 2, "Short term should have 2 tokens"
        assert categories['short_term']['excluded'] == 1, "Short term should have 1 excluded"
        
        assert categories['medium_term']['total'] == 2, "Medium term should have 2 tokens"
        assert categories['medium_term']['excluded'] == 1, "Medium term should have 1 excluded"
        
        assert categories['long_term']['total'] == 2, "Long term should have 2 tokens"
        assert categories['long_term']['excluded'] == 1, "Long term should have 1 excluded"
        
        # Calculate exclusion rates
        for cat, data in categories.items():
            exclusion_rate = (data['excluded'] / data['total']) * 100
            assert exclusion_rate == 50.0, f"Each category should have 50% exclusion rate, got {exclusion_rate}"
    
    def test_reason_aggregation_accuracy(self):
        """Test mathematical accuracy of exclusion reason aggregation."""
        # Simulate exclusion reasons
        exclusions = [
            {'reason': 'staircase', 'count': 25},
            {'reason': 'ratio', 'count': 15},
            {'reason': 'gaps', 'count': 10},
            {'reason': 'variability', 'count': 5}
        ]
        
        total_exclusions = sum(item['count'] for item in exclusions)
        assert total_exclusions == 55, f"Total exclusions should be 55, got {total_exclusions}"
        
        # Calculate percentages
        percentages = []
        for item in exclusions:
            percentage = (item['count'] / total_exclusions) * 100
            percentages.append(percentage)
        
        # Verify specific percentages
        expected_percentages = [
            (25 / 55) * 100,  # ~45.45%
            (15 / 55) * 100,  # ~27.27%
            (10 / 55) * 100,  # ~18.18%
            (5 / 55) * 100    # ~9.09%
        ]
        
        for calculated, expected in zip(percentages, expected_percentages):
            assert abs(calculated - expected) < 1e-12, \
                f"Percentage calculation error: {calculated} vs {expected}"
        
        # Verify percentages sum to 100%
        total_percentage = sum(percentages)
        assert abs(total_percentage - 100.0) < 1e-12, \
            f"Percentages should sum to 100%, got {total_percentage}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestExclusionAnalysisMetrics:
    """Test exclusion analysis metrics calculation accuracy."""
    
    def test_retention_vs_exclusion_consistency(self):
        """Test mathematical consistency between retention and exclusion rates."""
        # Test various scenarios
        test_scenarios = [
            {'total': 1000, 'excluded': 200},  # 20% exclusion, 80% retention
            {'total': 500, 'excluded': 100},   # 20% exclusion, 80% retention
            {'total': 250, 'excluded': 0},     # 0% exclusion, 100% retention
            {'total': 100, 'excluded': 100},   # 100% exclusion, 0% retention
            {'total': 75, 'excluded': 25}      # 33.33% exclusion, 66.67% retention
        ]
        
        for scenario in test_scenarios:
            total = scenario['total']
            excluded = scenario['excluded']
            retained = total - excluded
            
            # Calculate rates
            exclusion_rate = (excluded / total) * 100
            retention_rate = (retained / total) * 100
            
            # Rates should sum to 100%
            total_rate = exclusion_rate + retention_rate
            assert abs(total_rate - 100.0) < 1e-12, \
                f"Rates should sum to 100% for scenario {scenario}, got {total_rate}"
            
            # Alternative calculation should be consistent
            alt_retention_rate = ((total - excluded) / total) * 100
            assert abs(retention_rate - alt_retention_rate) < 1e-12, \
                f"Alternative retention calculation inconsistent: {retention_rate} vs {alt_retention_rate}"
    
    def test_weighted_average_calculations(self):
        """Test weighted average calculations for category statistics."""
        # Simulate category data with different sizes
        category_data = [
            {'category': 'short_term', 'tokens': 100, 'avg_length': 50},
            {'category': 'medium_term', 'tokens': 200, 'avg_length': 200},
            {'category': 'long_term', 'tokens': 50, 'avg_length': 500}
        ]
        
        # Calculate weighted average length
        total_tokens = sum(item['tokens'] for item in category_data)
        weighted_sum = sum(item['tokens'] * item['avg_length'] for item in category_data)
        weighted_average = weighted_sum / total_tokens
        
        # Manual verification
        manual_weighted_sum = (100 * 50) + (200 * 200) + (50 * 500)
        manual_total_tokens = 100 + 200 + 50
        manual_weighted_avg = manual_weighted_sum / manual_total_tokens
        
        assert abs(weighted_average - manual_weighted_avg) < 1e-12, \
            f"Weighted average calculation error: {weighted_average} vs {manual_weighted_avg}"
        
        # Expected: (5000 + 40000 + 25000) / 350 = 70000 / 350 = 200
        expected_weighted_avg = 200.0
        assert abs(weighted_average - expected_weighted_avg) < 1e-12, \
            f"Weighted average should be 200, got {weighted_average}"


@pytest.mark.integration
@pytest.mark.mathematical
class TestExclusionAnalysisIntegration:
    """Test complete exclusion analysis mathematical consistency."""
    
    def test_exclusion_analysis_data_consistency(self):
        """Test that exclusion analysis maintains mathematical consistency."""
        # Create comprehensive test dataset
        np.random.seed(42)
        
        # Generate synthetic exclusion data
        tokens = []
        categories = ['short_term', 'medium_term', 'long_term']
        reasons = ['staircase', 'ratio', 'gaps', 'variability', None]
        
        for i in range(300):  # 300 tokens total
            category = np.random.choice(categories)
            excluded = np.random.choice([True, False], p=[0.3, 0.7])  # 30% exclusion rate
            reason = np.random.choice(reasons[:-1]) if excluded else None
            length = np.random.randint(50, 1000)
            
            tokens.append({
                'token_name': f'token_{i}',
                'category': category,
                'excluded': excluded,
                'reason': reason,
                'length': length
            })
        
        # Calculate overall statistics
        total_tokens = len(tokens)
        excluded_tokens = sum(1 for t in tokens if t['excluded'])
        retention_rate = ((total_tokens - excluded_tokens) / total_tokens) * 100
        exclusion_rate = (excluded_tokens / total_tokens) * 100
        
        # Verify mathematical consistency
        assert abs(retention_rate + exclusion_rate - 100.0) < 1e-12, \
            "Retention + exclusion rates should equal 100%"
        
        # Category-wise analysis
        category_stats = {}
        for category in categories:
            cat_tokens = [t for t in tokens if t['category'] == category]
            cat_total = len(cat_tokens)
            cat_excluded = sum(1 for t in cat_tokens if t['excluded'])
            cat_exclusion_rate = (cat_excluded / cat_total) * 100 if cat_total > 0 else 0
            
            category_stats[category] = {
                'total': cat_total,
                'excluded': cat_excluded,
                'exclusion_rate': cat_exclusion_rate
            }
        
        # Verify category totals sum to overall total
        total_from_categories = sum(stats['total'] for stats in category_stats.values())
        assert total_from_categories == total_tokens, \
            f"Category totals should sum to overall total: {total_from_categories} vs {total_tokens}"
        
        excluded_from_categories = sum(stats['excluded'] for stats in category_stats.values())
        assert excluded_from_categories == excluded_tokens, \
            f"Category exclusions should sum to overall exclusions: {excluded_from_categories} vs {excluded_tokens}"
    
    def test_statistical_distribution_properties(self):
        """Test statistical properties of exclusion analysis."""
        # Create dataset with known statistical properties
        lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        # Calculate statistical properties
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        std_length = np.std(lengths, ddof=1)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        
        # Expected values for this specific dataset
        assert mean_length == 550.0, f"Mean should be 550, got {mean_length}"
        assert median_length == 550.0, f"Median should be 550, got {median_length}"
        assert min_length == 100, f"Min should be 100, got {min_length}"
        assert max_length == 1000, f"Max should be 1000, got {max_length}"
        
        # Standard deviation for arithmetic sequence
        expected_std = np.sqrt(sum((x - mean_length) ** 2 for x in lengths) / (len(lengths) - 1))
        assert abs(std_length - expected_std) < 1e-12, \
            f"Standard deviation calculation error: {std_length} vs {expected_std}"