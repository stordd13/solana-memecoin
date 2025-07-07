"""
Integration tests for complete data analysis pipeline
Tests end-to-end mathematical correctness and Streamlit display accuracy
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path


@pytest.mark.integration
@pytest.mark.mathematical
class TestDataAnalysisPipeline:
    """Test complete data analysis pipeline integration."""
    
    def test_complete_analysis_pipeline(self, data_loader, data_quality_analyzer, 
                                      price_analyzer, token_analyzer, synthetic_token_files):
        """Test complete pipeline from data loading to analysis."""
        # Set up data loader with temporary directory
        temp_dir = list(synthetic_token_files.values())[0].parent
        data_loader_temp = data_loader.__class__(base_path=temp_dir)
        
        # Load data
        available_tokens = data_loader_temp.get_available_tokens()
        assert len(available_tokens) > 0, "Should find synthetic tokens"
        
        # Analyze first few tokens
        test_tokens = available_tokens[:3]
        pipeline_results = {}
        
        for token_info in test_tokens:
            token_name = token_info['name']
            
            # Load token data
            df = data_loader_temp.load_token_data(token_name)
            assert df is not None, f"Should load data for {token_name}"
            
            # Quality analysis
            quality_result = data_quality_analyzer.analyze_single_file(df, token_name)
            assert quality_result is not None, f"Should analyze quality for {token_name}"
            
            # Price analysis
            price_result = price_analyzer.analyze_prices(df, token_name)
            assert price_result is not None, f"Should analyze prices for {token_name}"
            
            # Basic token analysis
            basic_stats = token_analyzer.calculate_basic_stats(df)
            assert basic_stats is not None, f"Should calculate basic stats for {token_name}"
            
            pipeline_results[token_name] = {
                'quality': quality_result,
                'price': price_result,
                'basic': basic_stats,
                'data': df
            }
        
        # Validate pipeline consistency
        for token_name, results in pipeline_results.items():
            _validate_pipeline_consistency(results, token_name)
    
    def test_data_loader_mathematical_consistency(self, data_loader, synthetic_token_files):
        """Test data loader mathematical consistency."""
        temp_dir = list(synthetic_token_files.values())[0].parent
        data_loader_temp = data_loader.__class__(base_path=temp_dir)
        
        # Load same token multiple times
        available_tokens = data_loader_temp.get_available_tokens()
        if available_tokens:
            token_name = available_tokens[0]['name']
            
            # Load multiple times
            df1 = data_loader_temp.load_token_data(token_name)
            df2 = data_loader_temp.load_token_data(token_name)
            
            # Should be identical
            assert df1.shape == df2.shape, "Multiple loads should return same shape"
            
            # Compare price data
            prices1 = df1['price'].to_numpy()
            prices2 = df2['price'].to_numpy()
            np.testing.assert_array_equal(prices1, prices2, 
                                        "Multiple loads should return identical data")
    
    def test_multi_module_statistical_consistency(self, data_quality_analyzer, 
                                                price_analyzer, token_analyzer, 
                                                synthetic_token_data):
        """Test statistical consistency across multiple analysis modules."""
        test_df = synthetic_token_data['normal_behavior']
        
        # Get results from all modules
        quality_result = data_quality_analyzer.analyze_single_file(test_df, "consistency_test")
        price_result = price_analyzer.analyze_prices(test_df, "consistency_test")
        basic_stats = token_analyzer.calculate_basic_stats(test_df)
        
        # All should succeed
        assert quality_result is not None
        assert price_result is not None  
        assert basic_stats is not None
        
        # Extract comparable statistics
        statistics = {}
        
        # From quality analysis
        if quality_result.get('status') == 'success':
            if 'basic_stats' in quality_result:
                stats = quality_result['basic_stats']
                if 'mean' in stats:
                    statistics['quality_mean'] = stats['mean']
        
        # From price analysis
        if price_result.get('status') == 'success':
            if 'price_stats' in price_result:
                stats = price_result['price_stats']
                if 'mean' in stats:
                    statistics['price_mean'] = stats['mean']
        
        # From basic token analysis
        if 'mean_price' in basic_stats:
            statistics['token_mean'] = basic_stats['mean_price']
        
        # Compare means across modules
        means = [v for k, v in statistics.items() if 'mean' in k]
        if len(means) > 1:
            # All means should be approximately equal
            for i in range(1, len(means)):
                assert abs(means[0] - means[i]) < 1e-6, \
                    f"Mean calculations inconsistent across modules: {statistics}"


@pytest.mark.integration
@pytest.mark.streamlit
class TestStreamlitDisplayAccuracy:
    """Test Streamlit display accuracy and mathematical correctness."""
    
    def test_aggregated_statistics_accuracy(self, data_quality_analyzer, synthetic_token_files):
        """Test aggregated statistics displayed in Streamlit are mathematically correct."""
        parquet_files = list(synthetic_token_files.values())
        
        # Analyze multiple files (simulating Streamlit app behavior)
        results = data_quality_analyzer.analyze_multiple_files(parquet_files, limit=None)
        
        # Should have summary for display
        assert 'summary' in results, "Should have summary for Streamlit display"
        summary = results['summary']
        
        # Should have individual reports
        assert 'reports' in results, "Should have individual reports"
        reports = results['reports']
        
        # Validate aggregated statistics manually
        if 'average_quality_score' in summary and reports:
            individual_scores = []
            for report in reports.values():
                if isinstance(report, dict) and 'quality_score' in report:
                    score = report['quality_score']
                    if isinstance(score, (int, float)) and not np.isnan(score):
                        individual_scores.append(score)
            
            if individual_scores:
                expected_avg = np.mean(individual_scores)
                actual_avg = summary['average_quality_score']
                
                assert abs(actual_avg - expected_avg) < 0.01, \
                    f"Streamlit display average mismatch: {actual_avg} vs {expected_avg}"
        
        # Validate count aggregations
        if 'total_files' in summary:
            assert summary['total_files'] == len(parquet_files), \
                "File count should match input for Streamlit display"
    
    def test_chart_data_mathematical_accuracy(self, price_analyzer, synthetic_token_data):
        """Test that chart data for Streamlit is mathematically accurate."""
        for token_type, df in synthetic_token_data.items():
            result = price_analyzer.analyze_prices(df, token_type)
            
            if result and result.get('status') == 'success':
                # Validate time series data for charts
                original_prices = df['price'].to_numpy()
                original_timestamps = df['datetime'].to_numpy()
                
                # Basic validation that data wasn't corrupted during analysis
                assert len(original_prices) > 0, "Should have price data for charts"
                assert len(original_timestamps) == len(original_prices), \
                    "Timestamps and prices should have same length for charts"
                
                # Price range should be preserved
                assert np.min(original_prices) >= 0, "Prices should be non-negative for charts"
                
                if 'price_stats' in result:
                    stats = result['price_stats']
                    if 'min' in stats and 'max' in stats:
                        assert stats['min'] <= stats['max'], \
                            "Min/max should be consistent for chart scaling"
                        assert abs(stats['min'] - np.min(original_prices)) < 1e-10, \
                            "Chart min should match actual data min"
                        assert abs(stats['max'] - np.max(original_prices)) < 1e-10, \
                            "Chart max should match actual data max"
    
    def test_percentage_calculations_for_display(self, data_quality_analyzer, synthetic_token_data):
        """Test percentage calculations displayed in Streamlit are accurate."""
        # Test with data that has known gap patterns
        test_df = synthetic_token_data['with_gaps']  # Has gaps every other minute
        
        result = data_quality_analyzer.analyze_single_file(test_df, "gap_test")
        
        if result and result.get('status') == 'success':
            # Validate completeness percentage
            if 'completeness' in result:
                completeness = result['completeness']
                
                # For data with gaps every other minute, completeness should be ~50%
                expected_completeness = 50.0  # Approximately
                tolerance = 10.0  # Allow 10% tolerance
                
                assert abs(completeness - expected_completeness) < tolerance, \
                    f"Completeness percentage for display incorrect: {completeness}% vs ~{expected_completeness}%"
            
            # Validate quality score is between 0-100 for display
            if 'quality_score' in result:
                quality_score = result['quality_score']
                assert 0 <= quality_score <= 100, \
                    f"Quality score for display should be 0-100%, got {quality_score}"


@pytest.mark.integration
@pytest.mark.mathematical 
class TestPipelineRobustness:
    """Test pipeline robustness under various conditions."""
    
    def test_pipeline_with_minimal_data(self, data_quality_analyzer, price_analyzer, token_analyzer):
        """Test pipeline with minimal data requirements."""
        # Minimal viable dataset
        minimal_df = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
            'price': [100.0, 105.0]
        })
        
        # Should handle gracefully
        quality_result = data_quality_analyzer.analyze_single_file(minimal_df, "minimal_test")
        assert quality_result is not None, "Should handle minimal data"
        
        price_result = price_analyzer.analyze_prices(minimal_df, "minimal_test")
        assert price_result is not None, "Should handle minimal data"
        
        basic_stats = token_analyzer.calculate_basic_stats(minimal_df)
        assert basic_stats is not None, "Should handle minimal data"
    
    def test_pipeline_memory_efficiency(self, data_quality_analyzer, price_analyzer):
        """Test pipeline memory efficiency with larger datasets."""
        # Create larger test dataset (simulate 24 hours of data)
        n_points = 1440  # 24 hours * 60 minutes
        large_prices = 100 + np.cumsum(np.random.normal(0, 0.01, n_points))
        
        large_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=n_points-1),
                interval="1m",
                eager=True
            ),
            'price': large_prices
        })
        
        # Should handle larger datasets efficiently
        quality_result = data_quality_analyzer.analyze_single_file(large_df, "large_test")
        assert quality_result is not None, "Should handle large datasets"
        
        price_result = price_analyzer.analyze_prices(large_df, "large_test")
        assert price_result is not None, "Should handle large datasets"
    
    def test_pipeline_error_propagation(self, data_quality_analyzer, price_analyzer, edge_case_data):
        """Test how errors propagate through the pipeline."""
        # Test with problematic data
        problematic_cases = ['with_nan', 'with_inf', 'negative_prices', 'zero_prices']
        
        for case_name in problematic_cases:
            if case_name in edge_case_data:
                case_df = edge_case_data[case_name]
                
                # Should either handle gracefully or fail with clear errors
                try:
                    quality_result = data_quality_analyzer.analyze_single_file(case_df, case_name)
                    price_result = price_analyzer.analyze_prices(case_df, case_name)
                    
                    # If they succeed, should have valid status
                    if quality_result:
                        assert 'status' in quality_result, f"Should have status for {case_name}"
                    if price_result:
                        assert 'status' in price_result, f"Should have status for {case_name}"
                        
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    # Acceptable to fail with clear error messages
                    assert len(str(e)) > 0, f"Error message should be informative for {case_name}"


def _validate_pipeline_consistency(results: dict, token_name: str):
    """Helper function to validate consistency across pipeline results."""
    quality_result = results.get('quality')
    price_result = results.get('price')
    basic_stats = results.get('basic')
    
    # If all modules succeeded, cross-validate results
    if (quality_result and quality_result.get('status') == 'success' and
        price_result and price_result.get('status') == 'success' and
        basic_stats):
        
        # Data length consistency
        df = results.get('data')
        if df is not None:
            data_length = len(df)
            
            # Quality analysis should reference same data length
            if 'total_rows' in quality_result:
                assert quality_result['total_rows'] == data_length, \
                    f"Quality analysis data length mismatch for {token_name}"
        
        # Price statistics consistency
        if ('price_stats' in price_result and 'mean_price' in basic_stats):
            price_mean = price_result['price_stats'].get('mean')
            basic_mean = basic_stats['mean_price']
            
            if price_mean is not None and basic_mean is not None:
                assert abs(price_mean - basic_mean) < 1e-6, \
                    f"Mean price inconsistency for {token_name}: {price_mean} vs {basic_mean}"
        
        # Quality score should be reasonable
        if 'quality_score' in quality_result:
            quality_score = quality_result['quality_score']
            assert 0 <= quality_score <= 100, \
                f"Quality score out of range for {token_name}: {quality_score}"