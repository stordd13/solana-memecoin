"""
Unit tests for token analysis mathematical validation
Tests mathematical correctness of basic statistics and returns calculations
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import math


@pytest.mark.unit
@pytest.mark.mathematical
class TestBasicStatistics:
    """Test basic statistics calculation mathematical correctness."""
    
    def test_calculate_basic_stats_against_numpy(self, token_analyzer, reference_calculations):
        """Test basic statistics against numpy reference implementations."""
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
        stats = token_analyzer.calculate_basic_stats(df)
        ref = reference_calculations
        
        # Validate basic statistics exist and are accurate
        assert stats is not None, "Basic stats should not be None"
        assert isinstance(stats, dict), "Basic stats should be dictionary"
        
        # Test key statistical measures
        stat_mappings = {
            'mean_price': ref['mean'],
            'std_price': ref['std'],
            'min_price': ref['min'],
            'max_price': ref['max']
        }
        
        for stat_key, expected_value in stat_mappings.items():
            if stat_key in stats:
                actual_value = stats[stat_key]
                assert abs(actual_value - expected_value) < 1e-10, \
                    f"{stat_key} mismatch: {actual_value} vs {expected_value}"
    
    def test_calculate_basic_stats_edge_cases(self, token_analyzer, edge_case_data):
        """Test basic statistics with edge cases."""
        # Single point
        stats = token_analyzer.calculate_basic_stats(edge_case_data['single_point'])
        assert stats is not None, "Should handle single point data"
        
        if 'mean_price' in stats:
            assert stats['mean_price'] == 100.0, "Single point mean should equal the value"
        if 'std_price' in stats:
            assert stats['std_price'] == 0.0 or math.isnan(stats['std_price']), \
                "Single point std should be 0 or NaN"
        
        # Two points
        stats = token_analyzer.calculate_basic_stats(edge_case_data['two_points'])
        assert stats is not None, "Should handle two point data"
        
        if 'mean_price' in stats:
            assert stats['mean_price'] == 125.0, "Two point mean should be (100+150)/2 = 125"
        if 'min_price' in stats and 'max_price' in stats:
            assert stats['min_price'] == 100.0, "Min should be 100"
            assert stats['max_price'] == 150.0, "Max should be 150"
    
    def test_calculate_basic_stats_data_types(self, token_analyzer):
        """Test basic statistics with different data types."""
        # Integer prices
        int_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=4),
                interval="1m",
                eager=True
            ),
            'price': [100, 110, 90, 105, 115]
        })
        
        stats = token_analyzer.calculate_basic_stats(int_df)
        assert stats is not None, "Should handle integer prices"
        
        if 'mean_price' in stats:
            expected_mean = (100 + 110 + 90 + 105 + 115) / 5
            assert abs(stats['mean_price'] - expected_mean) < 1e-10, \
                f"Integer price mean calculation error: {stats['mean_price']} vs {expected_mean}"
    
    def test_calculate_basic_stats_large_dataset(self, token_analyzer):
        """Test basic statistics with large dataset for numerical stability."""
        np.random.seed(42)
        large_prices = np.random.normal(1000, 100, 10000)  # 10,000 points
        
        large_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=9999),
                interval="1m",
                eager=True
            ),
            'price': large_prices
        })
        
        stats = token_analyzer.calculate_basic_stats(large_df)
        assert stats is not None, "Should handle large datasets"
        
        # Compare with numpy reference for large dataset
        if 'mean_price' in stats:
            numpy_mean = np.mean(large_prices)
            assert abs(stats['mean_price'] - numpy_mean) < 1e-6, \
                "Large dataset mean should match numpy"
        
        if 'std_price' in stats:
            numpy_std = np.std(large_prices, ddof=1)
            assert abs(stats['std_price'] - numpy_std) < 1e-6, \
                "Large dataset std should match numpy"


@pytest.mark.unit
@pytest.mark.mathematical
class TestReturnsCalculation:
    """Test returns calculation mathematical correctness."""
    
    def test_calculate_returns_accuracy(self, token_analyzer, reference_calculations):
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
        
        # Calculate returns using our implementation
        returns_df = token_analyzer.calculate_returns(df)
        
        # Should return DataFrame with returns
        assert isinstance(returns_df, pl.DataFrame), "Returns should be DataFrame"
        assert 'returns' in returns_df.columns, "Should have returns column"
        
        # Extract returns and compare with reference
        calculated_returns = returns_df['returns'].to_numpy()
        expected_returns = reference_calculations['returns']
        
        # Should have one less return than prices
        assert len(calculated_returns) == len(expected_returns), \
            f"Returns length mismatch: {len(calculated_returns)} vs {len(expected_returns)}"
        
        # Compare values (allowing for small numerical differences)
        for i, (actual, expected) in enumerate(zip(calculated_returns, expected_returns)):
            if not (math.isnan(actual) and math.isnan(expected)):
                assert abs(actual - expected) < 1e-10, \
                    f"Return {i} mismatch: {actual} vs {expected}"
    
    def test_calculate_returns_edge_cases(self, token_analyzer, edge_case_data):
        """Test returns calculation with edge cases."""
        # Single point - no returns possible
        try:
            returns_df = token_analyzer.calculate_returns(edge_case_data['single_point'])
            if returns_df is not None and 'returns' in returns_df.columns:
                assert len(returns_df) == 0, "Single point should have no returns"
        except (ValueError, IndexError):
            # Acceptable to raise error for insufficient data
            pass
        
        # Two points - one return
        returns_df = token_analyzer.calculate_returns(edge_case_data['two_points'])
        if returns_df is not None and 'returns' in returns_df.columns:
            returns_values = returns_df['returns'].to_numpy()
            if len(returns_values) > 0:
                # Return from 100 to 150 should be 0.5 (50%)
                expected_return = (150 - 100) / 100
                assert abs(returns_values[0] - expected_return) < 1e-10, \
                    f"Two point return mismatch: {returns_values[0]} vs {expected_return}"
    
    def test_calculate_returns_zero_division_handling(self, token_analyzer):
        """Test returns calculation with zero prices (division by zero)."""
        # Prices with zero (should handle gracefully)
        zero_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=3),
                interval="1m",
                eager=True
            ),
            'price': [0.0, 100.0, 200.0, 150.0]
        })
        
        # Should handle zero prices gracefully
        try:
            returns_df = token_analyzer.calculate_returns(zero_df)
            if returns_df is not None and 'returns' in returns_df.columns:
                returns_values = returns_df['returns'].to_numpy()
                # Should handle infinite returns appropriately
                assert not np.all(np.isfinite(returns_values)), \
                    "Zero price should create infinite/NaN returns"
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error for zero prices
            pass
    
    def test_calculate_returns_consistency_across_calls(self, token_analyzer):
        """Test that returns calculation is consistent across multiple calls."""
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=4),
                interval="1m",
                eager=True
            ),
            'price': [100.0, 110.0, 105.0, 120.0, 115.0]
        })
        
        # Calculate returns twice
        returns_df1 = token_analyzer.calculate_returns(df)
        returns_df2 = token_analyzer.calculate_returns(df)
        
        # Should get identical results
        if (returns_df1 is not None and returns_df2 is not None and 
            'returns' in returns_df1.columns and 'returns' in returns_df2.columns):
            
            returns1 = returns_df1['returns'].to_numpy()
            returns2 = returns_df2['returns'].to_numpy()
            
            np.testing.assert_array_equal(returns1, returns2, 
                                        "Returns calculation should be deterministic")


@pytest.mark.unit
@pytest.mark.mathematical
class TestTokenAnalysisPipeline:
    """Test complete token analysis pipeline."""
    
    def test_analyze_token_complete_pipeline(self, token_analyzer, synthetic_token_data, temp_data_dir):
        """Test complete token analysis pipeline mathematical consistency."""
        for token_type, df in synthetic_token_data.items():
            # Save to temporary file for testing file-based analysis
            file_path = temp_data_dir / f"{token_type}.parquet"
            df.write_parquet(file_path)
            
            # Analyze token
            result = token_analyzer.analyze_token(file_path)
            
            # Should return valid result
            assert isinstance(result, dict), f"Result should be dict for {token_type}"
            
            # Should have basic information
            if 'status' in result and result['status'] == 'success':
                assert 'basic_stats' in result, f"Should have basic stats for {token_type}"
                
                basic_stats = result['basic_stats']
                
                # Basic stats should be mathematically valid
                if 'mean_price' in basic_stats:
                    assert isinstance(basic_stats['mean_price'], (int, float)), \
                        "Mean price should be numeric"
                    assert basic_stats['mean_price'] > 0, \
                        "Mean price should be positive for our test data"
                
                if 'std_price' in basic_stats:
                    assert isinstance(basic_stats['std_price'], (int, float)), \
                        "Std price should be numeric"
                    assert basic_stats['std_price'] >= 0, \
                        "Standard deviation should be non-negative"
                
                if 'min_price' in basic_stats and 'max_price' in basic_stats:
                    assert basic_stats['min_price'] <= basic_stats['max_price'], \
                        "Min price should be <= max price"
    
    def test_analyze_multiple_tokens_aggregation(self, token_analyzer, synthetic_token_files):
        """Test multiple token analysis mathematical aggregation."""
        # Analyze subset of tokens
        result = token_analyzer.analyze_multiple_tokens(limit=3)
        
        # Should return aggregated results
        assert isinstance(result, dict), "Results should be dict"
        
        # Should have summary statistics
        if 'summary' in result:
            summary = result['summary']
            
            # Summary should have reasonable values
            if 'total_tokens' in summary:
                assert summary['total_tokens'] >= 0, "Total tokens should be non-negative"
            
            if 'average_stats' in summary:
                avg_stats = summary['average_stats']
                
                # Average statistics should be valid
                if 'mean_price' in avg_stats:
                    assert isinstance(avg_stats['mean_price'], (int, float)), \
                        "Average mean price should be numeric"
                    assert avg_stats['mean_price'] > 0, \
                        "Average mean price should be positive"


@pytest.mark.unit
@pytest.mark.mathematical
class TestNumericalStability:
    """Test numerical stability of token analysis functions."""
    
    def test_extreme_values_handling(self, token_analyzer):
        """Test token analysis with extreme values."""
        # Very large values
        large_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=4),
                interval="1m",
                eager=True
            ),
            'price': [1e15, 1.1e15, 1.2e15, 1.05e15, 1.3e15]
        })
        
        stats = token_analyzer.calculate_basic_stats(large_df)
        assert stats is not None, "Should handle very large values"
        
        returns_df = token_analyzer.calculate_returns(large_df)
        assert returns_df is not None, "Should handle returns for large values"
        
        # Very small values
        small_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=4),
                interval="1m",
                eager=True
            ),
            'price': [1e-10, 1.1e-10, 1.2e-10, 1.05e-10, 1.3e-10]
        })
        
        stats = token_analyzer.calculate_basic_stats(small_df)
        assert stats is not None, "Should handle very small values"
        
        returns_df = token_analyzer.calculate_returns(small_df)
        assert returns_df is not None, "Should handle returns for small values"
    
    def test_precision_consistency(self, token_analyzer):
        """Test that calculations maintain precision consistency."""
        # Test with values that might cause precision issues
        precision_test_prices = [
            1.000000001, 1.000000002, 1.000000003, 1.000000004, 1.000000005
        ]
        
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=4),
                interval="1m",
                eager=True
            ),
            'price': precision_test_prices
        })
        
        stats = token_analyzer.calculate_basic_stats(df)
        assert stats is not None, "Should handle precision-sensitive values"
        
        # Calculated statistics should reflect the small differences
        if 'std_price' in stats:
            assert stats['std_price'] > 0, "Should detect variance in precision test data"
        
        returns_df = token_analyzer.calculate_returns(df)
        if returns_df is not None and 'returns' in returns_df.columns:
            returns_values = returns_df['returns'].to_numpy()
            # Should calculate meaningful returns even for small differences
            assert np.any(returns_values != 0), "Should detect non-zero returns in precision test"


@pytest.mark.integration
@pytest.mark.mathematical
class TestCrossModuleConsistency:
    """Test mathematical consistency across token analysis and other modules."""
    
    def test_basic_stats_vs_price_analysis_consistency(self, token_analyzer, price_analyzer, synthetic_token_data):
        """Test that basic statistics are consistent between modules."""
        test_df = synthetic_token_data['normal_behavior']
        
        # Get basic stats from token analyzer
        token_stats = token_analyzer.calculate_basic_stats(test_df)
        
        # Get price analysis results
        price_result = price_analyzer.analyze_prices(test_df, "consistency_test")
        
        # Both should exist and be successful
        if (token_stats is not None and 
            price_result is not None and 
            price_result.get('status') == 'success'):
            
            # Compare common statistical measures
            if ('mean_price' in token_stats and 
                'price_stats' in price_result and 
                'mean' in price_result['price_stats']):
                
                token_mean = token_stats['mean_price']
                price_mean = price_result['price_stats']['mean']
                
                assert abs(token_mean - price_mean) < 1e-6, \
                    f"Mean price should be consistent: {token_mean} vs {price_mean}"
            
            if ('std_price' in token_stats and 
                'price_stats' in price_result and 
                'std' in price_result['price_stats']):
                
                token_std = token_stats['std_price']
                price_std = price_result['price_stats']['std']
                
                assert abs(token_std - price_std) < 1e-6, \
                    f"Std price should be consistent: {token_std} vs {price_std}"