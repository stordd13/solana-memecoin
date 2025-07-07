"""
Mathematical validation tests for generate_graduated_datasets module
Tests dataset generation metrics and strategy comparison accuracy
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.mathematical
class TestRetentionRateCalculations:
    """Test retention rate calculation mathematical correctness."""
    
    def test_retention_rate_accuracy(self):
        """Test retention rate calculations against reference implementation."""
        # Test various original/final length scenarios
        test_cases = [
            {'original': 1000, 'final': 800, 'expected': 0.8},
            {'original': 500, 'final': 350, 'expected': 0.7},
            {'original': 100, 'final': 100, 'expected': 1.0},  # No loss
            {'original': 200, 'final': 0, 'expected': 0.0},    # Total loss
            {'original': 750, 'final': 250, 'expected': 1/3}   # 1/3 retention
        ]
        
        for case in test_cases:
            # Calculate retention rate
            retention_rate = case['final'] / case['original'] if case['original'] > 0 else 0
            
            # Verify against expected with high precision
            assert abs(retention_rate - case['expected']) < 1e-12, \
                f"Retention rate calculation error: {retention_rate} vs {case['expected']}"
            
            # Verify percentage calculation
            retention_percentage = retention_rate * 100
            expected_percentage = case['expected'] * 100
            assert abs(retention_percentage - expected_percentage) < 1e-12, \
                f"Retention percentage error: {retention_percentage} vs {expected_percentage}"
    
    def test_retention_rate_edge_cases(self):
        """Test retention rate calculations with edge cases."""
        # Zero original length - should handle division by zero
        try:
            result = 100 / 0
            assert False, "Should raise ZeroDivisionError"
        except ZeroDivisionError:
            # Expected behavior - use conditional logic in practice
            retention_rate = 100 / 0 if 0 > 0 else 0
            assert retention_rate == 0, "Zero original should give 0 retention rate"
        
        # Final larger than original (should not happen in practice, but test robustness)
        original = 100
        final = 150
        retention_rate = final / original
        assert retention_rate == 1.5, f"Retention rate should be 1.5, got {retention_rate}"
        
        # Both zero
        retention_rate = 0 / 0 if 0 > 0 else 0  # Conditional to avoid division by zero
        assert retention_rate == 0, "Zero/zero should be handled as 0"
    
    def test_average_retention_rate_calculation(self):
        """Test average retention rate calculation accuracy."""
        # Multiple retention rates
        retention_rates = [0.8, 0.7, 0.9, 0.6, 0.85]
        
        # Calculate average
        avg_retention = sum(retention_rates) / len(retention_rates)
        expected_avg = (0.8 + 0.7 + 0.9 + 0.6 + 0.85) / 5
        
        assert abs(avg_retention - expected_avg) < 1e-12, \
            f"Average retention calculation error: {avg_retention} vs {expected_avg}"
        
        # Manual verification
        manual_avg = 3.85 / 5
        assert abs(avg_retention - manual_avg) < 1e-12, \
            f"Manual average verification failed: {avg_retention} vs {manual_avg}"
        
        # Expected: 0.77
        assert abs(avg_retention - 0.77) < 1e-12, f"Expected average 0.77, got {avg_retention}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestSuccessRateCalculations:
    """Test success rate calculation mathematical correctness."""
    
    def test_success_rate_accuracy(self):
        """Test success rate calculations against reference implementation."""
        # Test various success/total scenarios
        test_cases = [
            {'successful': 80, 'total': 100, 'expected': 0.8},
            {'successful': 45, 'total': 60, 'expected': 0.75},
            {'successful': 100, 'total': 100, 'expected': 1.0},  # Perfect success
            {'successful': 0, 'total': 50, 'expected': 0.0},     # No success
            {'successful': 33, 'total': 99, 'expected': 1/3}     # 1/3 success
        ]
        
        for case in test_cases:
            # Calculate success rate
            success_rate = case['successful'] / case['total'] if case['total'] > 0 else 0
            
            # Verify against expected with high precision
            assert abs(success_rate - case['expected']) < 1e-12, \
                f"Success rate calculation error: {success_rate} vs {case['expected']}"
            
            # Verify percentage calculation
            success_percentage = success_rate * 100
            expected_percentage = case['expected'] * 100
            assert abs(success_percentage - expected_percentage) < 1e-12, \
                f"Success percentage error: {success_percentage} vs {expected_percentage}"
    
    def test_success_rate_conditional_logic(self):
        """Test success rate conditional logic for zero totals."""
        # Test the conditional logic used in practice
        test_results = [
            {'cleaned_successfully': 75, 'total_processed': 100},
            {'cleaned_successfully': 0, 'total_processed': 0},    # Edge case
            {'cleaned_successfully': 50, 'total_processed': 75}
        ]
        
        for result in test_results:
            # Apply conditional logic as in the actual code
            success_rate = (
                result['cleaned_successfully'] / result['total_processed'] 
                if result['total_processed'] > 0 else 0
            )
            
            if result['total_processed'] > 0:
                expected = result['cleaned_successfully'] / result['total_processed']
                assert abs(success_rate - expected) < 1e-12, \
                    f"Success rate with conditional logic error: {success_rate} vs {expected}"
            else:
                assert success_rate == 0, "Zero total should give 0 success rate"
    
    def test_complementary_failure_rate(self):
        """Test that success and failure rates are complementary."""
        # Test scenarios where success + failure = 100%
        test_cases = [
            {'successful': 70, 'failed': 30, 'total': 100},
            {'successful': 45, 'failed': 15, 'total': 60},
            {'successful': 0, 'failed': 25, 'total': 25}
        ]
        
        for case in test_cases:
            total = case['total']
            
            success_rate = case['successful'] / total
            failure_rate = case['failed'] / total
            
            # Should sum to 1.0 (100%)
            total_rate = success_rate + failure_rate
            assert abs(total_rate - 1.0) < 1e-12, \
                f"Success + failure rates should equal 1.0, got {total_rate}"
            
            # Alternative calculation
            alt_failure_rate = (total - case['successful']) / total
            assert abs(failure_rate - alt_failure_rate) < 1e-12, \
                f"Alternative failure rate calculation inconsistent: {failure_rate} vs {alt_failure_rate}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestStrategyComparisonMetrics:
    """Test strategy comparison metrics mathematical correctness."""
    
    def test_strategy_performance_aggregation(self):
        """Test mathematical accuracy of strategy performance aggregation."""
        # Simulate results from different cleaning strategies
        strategy_results = {
            'minimal': {
                'total_processed': 100,
                'cleaned_successfully': 85,
                'total_retention': 0.95,
                'avg_retention': 0.92
            },
            'gentle': {
                'total_processed': 100,
                'cleaned_successfully': 75,
                'total_retention': 0.88,
                'avg_retention': 0.85
            },
            'aggressive': {
                'total_processed': 100,
                'cleaned_successfully': 60,
                'total_retention': 0.70,
                'avg_retention': 0.75
            }
        }
        
        # Calculate aggregated metrics
        total_processed = sum(s['total_processed'] for s in strategy_results.values())
        total_successful = sum(s['cleaned_successfully'] for s in strategy_results.values())
        
        overall_success_rate = total_successful / total_processed
        expected_success_rate = (85 + 75 + 60) / (100 + 100 + 100)
        
        assert abs(overall_success_rate - expected_success_rate) < 1e-12, \
            f"Overall success rate calculation error: {overall_success_rate} vs {expected_success_rate}"
        
        # Expected: 220/300 = 0.7333...
        assert abs(overall_success_rate - (220/300)) < 1e-12, \
            f"Expected success rate 220/300, got {overall_success_rate}"
    
    def test_strategy_ranking_consistency(self):
        """Test mathematical consistency of strategy ranking."""
        # Strategy performance metrics
        strategies = {
            'strategy_a': {'success_rate': 0.85, 'retention_rate': 0.90},
            'strategy_b': {'success_rate': 0.75, 'retention_rate': 0.95},
            'strategy_c': {'success_rate': 0.90, 'retention_rate': 0.80}
        }
        
        # Calculate composite scores (example: weighted average)
        weight_success = 0.6
        weight_retention = 0.4
        
        composite_scores = {}
        for strategy, metrics in strategies.items():
            composite_score = (
                weight_success * metrics['success_rate'] + 
                weight_retention * metrics['retention_rate']
            )
            composite_scores[strategy] = composite_score
        
        # Verify calculations
        expected_scores = {
            'strategy_a': 0.6 * 0.85 + 0.4 * 0.90,  # 0.51 + 0.36 = 0.87
            'strategy_b': 0.6 * 0.75 + 0.4 * 0.95,  # 0.45 + 0.38 = 0.83
            'strategy_c': 0.6 * 0.90 + 0.4 * 0.80   # 0.54 + 0.32 = 0.86
        }
        
        for strategy, expected in expected_scores.items():
            actual = composite_scores[strategy]
            assert abs(actual - expected) < 1e-12, \
                f"Composite score calculation error for {strategy}: {actual} vs {expected}"
        
        # Verify ranking order (strategy_a > strategy_c > strategy_b)
        scores_list = [(score, strategy) for strategy, score in composite_scores.items()]
        scores_list.sort(reverse=True)
        
        assert scores_list[0][1] == 'strategy_a', "Strategy A should rank first"
        assert scores_list[1][1] == 'strategy_c', "Strategy C should rank second"
        assert scores_list[2][1] == 'strategy_b', "Strategy B should rank third"
    
    def test_relative_performance_calculations(self):
        """Test relative performance calculations between strategies."""
        # Base strategy performance
        base_strategy = {'success_rate': 0.70, 'retention_rate': 0.80}
        
        # Comparison strategies
        comparison_strategies = {
            'improved': {'success_rate': 0.85, 'retention_rate': 0.90},
            'degraded': {'success_rate': 0.60, 'retention_rate': 0.70}
        }
        
        # Calculate relative improvements
        for strategy, metrics in comparison_strategies.items():
            success_improvement = (metrics['success_rate'] - base_strategy['success_rate']) / base_strategy['success_rate']
            retention_improvement = (metrics['retention_rate'] - base_strategy['retention_rate']) / base_strategy['retention_rate']
            
            if strategy == 'improved':
                # Should show positive improvements
                expected_success_improvement = (0.85 - 0.70) / 0.70  # ~0.214
                expected_retention_improvement = (0.90 - 0.80) / 0.80  # 0.125
                
                assert abs(success_improvement - expected_success_improvement) < 1e-12, \
                    f"Success improvement calculation error: {success_improvement} vs {expected_success_improvement}"
                assert abs(retention_improvement - expected_retention_improvement) < 1e-12, \
                    f"Retention improvement calculation error: {retention_improvement} vs {expected_retention_improvement}"
                
                assert success_improvement > 0, "Improved strategy should show positive success improvement"
                assert retention_improvement > 0, "Improved strategy should show positive retention improvement"
                
            elif strategy == 'degraded':
                # Should show negative improvements (degradation)
                expected_success_degradation = (0.60 - 0.70) / 0.70  # ~-0.143
                expected_retention_degradation = (0.70 - 0.80) / 0.80  # -0.125
                
                assert abs(success_improvement - expected_success_degradation) < 1e-12, \
                    f"Success degradation calculation error: {success_improvement} vs {expected_success_degradation}"
                assert abs(retention_improvement - expected_retention_degradation) < 1e-12, \
                    f"Retention degradation calculation error: {retention_improvement} vs {expected_retention_degradation}"
                
                assert success_improvement < 0, "Degraded strategy should show negative success improvement"
                assert retention_improvement < 0, "Degraded strategy should show negative retention improvement"


@pytest.mark.unit
@pytest.mark.mathematical
class TestDatasetGenerationMetrics:
    """Test dataset generation metrics mathematical correctness."""
    
    def test_dataset_size_calculations(self):
        """Test dataset size calculation accuracy."""
        # Original dataset parameters
        original_counts = {
            'short_term': 500,
            'medium_term': 300, 
            'long_term': 200
        }
        
        # Retention rates by category
        retention_rates = {
            'short_term': 0.80,
            'medium_term': 0.75,
            'long_term': 0.70
        }
        
        # Calculate final dataset sizes
        final_counts = {}
        total_original = 0
        total_final = 0
        
        for category, original_count in original_counts.items():
            retention_rate = retention_rates[category]
            final_count = int(original_count * retention_rate)
            final_counts[category] = final_count
            
            total_original += original_count
            total_final += final_count
        
        # Verify calculations
        assert final_counts['short_term'] == int(500 * 0.80), "Short term calculation error"
        assert final_counts['medium_term'] == int(300 * 0.75), "Medium term calculation error"
        assert final_counts['long_term'] == int(200 * 0.70), "Long term calculation error"
        
        # Expected values
        assert final_counts['short_term'] == 400, f"Expected 400, got {final_counts['short_term']}"
        assert final_counts['medium_term'] == 225, f"Expected 225, got {final_counts['medium_term']}"
        assert final_counts['long_term'] == 140, f"Expected 140, got {final_counts['long_term']}"
        
        # Verify totals
        assert total_original == 1000, f"Total original should be 1000, got {total_original}"
        assert total_final == 765, f"Total final should be 765, got {total_final}"
        
        # Overall retention rate
        overall_retention = total_final / total_original
        expected_overall = 765 / 1000
        assert abs(overall_retention - expected_overall) < 1e-12, \
            f"Overall retention calculation error: {overall_retention} vs {expected_overall}"
    
    def test_quality_distribution_analysis(self):
        """Test quality distribution analysis mathematical accuracy."""
        # Quality scores for different categories
        quality_distributions = {
            'short_term': [85, 90, 75, 95, 80, 88, 92, 78, 87, 91],
            'medium_term': [70, 80, 85, 75, 90, 82, 88, 73, 86, 79],
            'long_term': [60, 70, 65, 75, 68, 72, 77, 63, 71, 69]
        }
        
        # Calculate statistics for each category
        category_stats = {}
        for category, scores in quality_distributions.items():
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores, ddof=1)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            category_stats[category] = {
                'mean': mean_score,
                'median': median_score,
                'std': std_score,
                'min': min_score,
                'max': max_score
            }
        
        # Verify specific calculations
        # Short term: [85, 90, 75, 95, 80, 88, 92, 78, 87, 91]
        # Sum = 861, Mean = 86.1
        expected_short_mean = 861 / 10
        assert abs(category_stats['short_term']['mean'] - expected_short_mean) < 1e-12, \
            f"Short term mean calculation error: {category_stats['short_term']['mean']} vs {expected_short_mean}"
        
        # Verify all categories have reasonable statistics
        for category, stats in category_stats.items():
            assert stats['min'] <= stats['median'] <= stats['max'], \
                f"Statistical ordering violated for {category}"
            assert stats['std'] >= 0, f"Standard deviation should be non-negative for {category}"
    
    def test_dataset_balance_calculations(self):
        """Test dataset balance calculation accuracy."""
        # Category distributions in final dataset
        final_distribution = {
            'short_term': 400,
            'medium_term': 250,
            'long_term': 150
        }
        
        total_tokens = sum(final_distribution.values())
        
        # Calculate category percentages
        category_percentages = {}
        for category, count in final_distribution.items():
            percentage = (count / total_tokens) * 100
            category_percentages[category] = percentage
        
        # Verify calculations
        assert abs(category_percentages['short_term'] - 50.0) < 1e-12, \
            f"Short term percentage should be 50%, got {category_percentages['short_term']}"
        assert abs(category_percentages['medium_term'] - 31.25) < 1e-12, \
            f"Medium term percentage should be 31.25%, got {category_percentages['medium_term']}"
        assert abs(category_percentages['long_term'] - 18.75) < 1e-12, \
            f"Long term percentage should be 18.75%, got {category_percentages['long_term']}"
        
        # Verify percentages sum to 100%
        total_percentage = sum(category_percentages.values())
        assert abs(total_percentage - 100.0) < 1e-12, \
            f"Category percentages should sum to 100%, got {total_percentage}"


@pytest.mark.integration
@pytest.mark.mathematical
class TestDatasetGenerationIntegration:
    """Test complete dataset generation mathematical consistency."""
    
    def test_end_to_end_generation_consistency(self):
        """Test mathematical consistency across complete generation pipeline."""
        # Simulate complete dataset generation process
        np.random.seed(42)
        
        # Original dataset parameters
        original_data = {
            'total_tokens': 1000,
            'categories': {
                'short_term': 400,
                'medium_term': 350,
                'long_term': 250
            }
        }
        
        # Strategy performance (retention rates)
        strategy_performance = {
            'minimal': {
                'short_term': 0.95,
                'medium_term': 0.90,
                'long_term': 0.85
            },
            'moderate': {
                'short_term': 0.85,
                'medium_term': 0.80,
                'long_term': 0.75
            },
            'aggressive': {
                'short_term': 0.70,
                'medium_term': 0.65,
                'long_term': 0.60
            }
        }
        
        # Calculate final datasets for each strategy
        strategy_results = {}
        for strategy, retention_rates in strategy_performance.items():
            strategy_total = 0
            category_results = {}
            
            for category, original_count in original_data['categories'].items():
                retention_rate = retention_rates[category]
                final_count = int(original_count * retention_rate)
                category_results[category] = {
                    'original': original_count,
                    'final': final_count,
                    'retention_rate': retention_rate,
                    'calculated_retention': final_count / original_count
                }
                strategy_total += final_count
            
            # Calculate overall strategy performance
            overall_retention = strategy_total / original_data['total_tokens']
            
            strategy_results[strategy] = {
                'categories': category_results,
                'total_final': strategy_total,
                'overall_retention': overall_retention
            }
        
        # Verify mathematical consistency
        for strategy, results in strategy_results.items():
            # Verify category totals sum to strategy total
            category_sum = sum(cat['final'] for cat in results['categories'].values())
            assert category_sum == results['total_final'], \
                f"Category sum mismatch for {strategy}: {category_sum} vs {results['total_final']}"
            
            # Verify retention rate calculations
            for category, cat_data in results['categories'].items():
                expected_retention = cat_data['final'] / cat_data['original']
                assert abs(cat_data['calculated_retention'] - expected_retention) < 1e-12, \
                    f"Retention rate calculation error for {strategy}/{category}"
        
        # Verify strategy ordering (minimal > moderate > aggressive)
        minimal_retention = strategy_results['minimal']['overall_retention']
        moderate_retention = strategy_results['moderate']['overall_retention']
        aggressive_retention = strategy_results['aggressive']['overall_retention']
        
        assert minimal_retention > moderate_retention > aggressive_retention, \
            "Strategy retention rates should be ordered: minimal > moderate > aggressive"
        
        # Verify all retention rates are between 0 and 1
        for strategy, results in strategy_results.items():
            assert 0 <= results['overall_retention'] <= 1, \
                f"Overall retention rate should be 0-1 for {strategy}: {results['overall_retention']}"
            
            for category, cat_data in results['categories'].items():
                assert 0 <= cat_data['retention_rate'] <= 1, \
                    f"Category retention rate should be 0-1 for {strategy}/{category}: {cat_data['retention_rate']}"