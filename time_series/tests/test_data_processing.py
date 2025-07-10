"""
Unit tests for data loading and processing pipeline
Tests data loading, categorization, and filtering logic
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


@pytest.mark.unit
class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_raw_prices_basic(self, analyzer, synthetic_token_data):
        """Test basic data loading functionality."""
        # Get the temp directory from fixture
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Load data
        token_data = analyzer.load_raw_prices(temp_dir)
        
        # Should load all tokens
        assert len(token_data) > 0, "Should load at least some tokens"
        
        # Check that each token has expected structure
        for token_name, df in token_data.items():
            assert isinstance(df, pl.DataFrame), f"Token {token_name} should be DataFrame"
            assert 'datetime' in df.columns, f"Token {token_name} missing datetime column"
            assert 'price' in df.columns, f"Token {token_name} missing price column"
            assert 'log_price' in df.columns, f"Token {token_name} should have log_price column"
    
    def test_load_raw_prices_with_limit(self, analyzer, synthetic_token_data):
        """Test data loading with token limit."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Load with limit
        limit = 3
        token_data = analyzer.load_raw_prices(temp_dir, limit=limit)
        
        # Should respect limit
        assert len(token_data) <= limit, f"Should load at most {limit} tokens"
    
    def test_load_empty_directory(self, analyzer, temp_data_dir):
        """Test loading from empty directory."""
        # Load from empty directory
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        # Should return empty dict
        assert len(token_data) == 0, "Empty directory should return empty dict"
    
    def test_load_invalid_files(self, analyzer, temp_data_dir):
        """Test loading with invalid files in directory."""
        # Create invalid file
        invalid_file = temp_data_dir / "invalid.parquet"
        invalid_file.write_text("not a parquet file")
        
        # Should handle gracefully
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        # Should not crash, may return empty dict
        assert isinstance(token_data, dict), "Should return dict even with invalid files"


@pytest.mark.unit
class TestLifespanCategorization:
    """Test lifespan categorization logic."""
    
    def test_lifespan_category_ranges(self, analyzer, synthetic_token_data):
        """Test that tokens are categorized into correct lifespan ranges."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Run multi-resolution analysis to test categorization
        results = analyzer.analyze_by_lifespan_category(
            temp_dir, 
            method='returns',
            max_tokens_per_category=10
        )
        
        # Should have category results
        assert 'categories' in results, "Results should contain categories"
        
        categories = results['categories']
        
        # Check each category has expected tokens
        for category_name, category_results in categories.items():
            if len(category_results.get('token_data', {})) > 0:
                # Verify lifespan ranges
                for token_name, token_df in category_results['token_data'].items():
                    token_length = len(token_df)
                    
                    if category_name == 'Sprint':
                        assert 200 <= token_length <= 400, \
                            f"Sprint token {token_name} has invalid length {token_length}"
                    elif category_name == 'Standard':
                        assert 400 < token_length <= 1200, \
                            f"Standard token {token_name} has invalid length {token_length}"
                    elif category_name == 'Marathon':
                        assert token_length > 1200, \
                            f"Marathon token {token_name} has invalid length {token_length}"
    
    def test_category_token_limits(self, analyzer, synthetic_token_data):
        """Test that category token limits are respected."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        max_per_category = 2
        results = analyzer.analyze_by_lifespan_category(
            temp_dir,
            method='returns',
            max_tokens_per_category=max_per_category
        )
        
        # Check that each category respects the limit
        for category_name, category_results in results['categories'].items():
            n_tokens = len(category_results.get('token_data', {}))
            assert n_tokens <= max_per_category, \
                f"Category {category_name} has {n_tokens} tokens, exceeds limit {max_per_category}"
    
    def test_lifespan_range_descriptions(self, analyzer):
        """Test lifespan range description helper."""
        # Test the helper method
        assert analyzer._get_lifespan_range('Sprint') == '200-400 minutes'
        assert analyzer._get_lifespan_range('Standard') == '400-1200 minutes'
        assert analyzer._get_lifespan_range('Marathon') == '1200+ minutes'
        assert analyzer._get_lifespan_range('Unknown') == 'Unknown'


@pytest.mark.unit
class TestPriceTransformations:
    """Test different price transformation methods."""
    
    def test_prepare_price_only_data_returns(self, analyzer, synthetic_token_data):
        """Test returns transformation."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=3)
        
        # Test returns method
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='returns', use_log_price=True
        )
        
        assert len(feature_matrix) > 0, "Should generate feature matrix"
        assert len(token_names) > 0, "Should have token names"
        assert len(feature_matrix) == len(token_names), "Matrix and names should match"
        
        # Check that returns are reasonable (should be centered around 0)
        if len(feature_matrix) > 0:
            all_returns = feature_matrix.flatten()
            mean_return = np.mean(all_returns)
            assert abs(mean_return) < 1.0, f"Mean return seems too large: {mean_return}"
    
    def test_prepare_price_only_data_log_returns(self, analyzer, synthetic_token_data):
        """Test log returns transformation."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=3)
        
        # Test log_returns method
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='log_returns', use_log_price=True
        )
        
        assert len(feature_matrix) > 0, "Should generate feature matrix"
        
        # Log returns should not contain inf or nan values
        assert not np.any(np.isinf(feature_matrix)), "Log returns should not contain inf"
        assert not np.any(np.isnan(feature_matrix)), "Log returns should not contain nan"
    
    def test_prepare_price_only_data_raw_prices(self, analyzer, synthetic_token_data):
        """Test raw prices transformation."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=3)
        
        # Test prices method
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='prices', use_log_price=False
        )
        
        assert len(feature_matrix) > 0, "Should generate feature matrix"
        
        # Raw prices should be positive
        assert np.all(feature_matrix > 0), "Raw prices should be positive"
    
    def test_prepare_price_only_data_dtw_features(self, analyzer, synthetic_token_data):
        """Test DTW features transformation."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=3)
        
        # Test dtw_features method
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='dtw_features', use_log_price=True
        )
        
        assert len(feature_matrix) > 0, "Should generate feature matrix"
        
        # DTW features should be a 2D matrix with statistical features
        assert feature_matrix.ndim == 2, "DTW features should be 2D matrix"
        assert feature_matrix.shape[1] > 1, "DTW features should have multiple dimensions"
        
        # Features should not contain inf or nan
        assert not np.any(np.isinf(feature_matrix)), "DTW features should not contain inf"
        assert not np.any(np.isnan(feature_matrix)), "DTW features should not contain nan"
    
    def test_price_transformation_consistency(self, analyzer, synthetic_token_data):
        """Test that price transformations produce consistent results."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=2)
        
        # Test same transformation twice
        result1 = analyzer.prepare_price_only_data(
            token_data, method='returns', use_log_price=True
        )
        result2 = analyzer.prepare_price_only_data(
            token_data, method='returns', use_log_price=True
        )
        
        # Should produce identical results
        np.testing.assert_array_equal(result1[0], result2[0], 
                                     "Same transformation should produce identical results")
        assert result1[1] == result2[1], "Token names should be identical"


@pytest.mark.unit
class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_empty_token_data(self, analyzer):
        """Test handling of empty token data."""
        empty_data = {}
        
        # Should handle empty data gracefully
        try:
            feature_matrix, token_names = analyzer.prepare_price_only_data(
                empty_data, method='returns'
            )
            # If it doesn't raise an exception, check results
            assert len(feature_matrix) == 0, "Empty data should produce empty matrix"
            assert len(token_names) == 0, "Empty data should produce empty names"
        except ValueError as e:
            # It's also acceptable to raise a clear error
            assert "No valid" in str(e), "Should provide clear error message"
    
    def test_very_short_series(self, analyzer, temp_data_dir):
        """Test handling of very short price series."""
        # Create token with very short series
        short_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=2),
                interval="1m",
                eager=True
            ),
            'price': [100.0, 101.0, 99.0]
        })
        short_df = short_df.with_columns([pl.col('price').log().alias('log_price')])
        
        file_path = temp_data_dir / "short_token.parquet"
        short_df.write_parquet(file_path)
        
        # Load and process
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        # Should handle short series appropriately
        if len(token_data) > 0:
            try:
                feature_matrix, token_names = analyzer.prepare_price_only_data(
                    token_data, method='returns'
                )
                # If successful, check reasonable constraints
                if len(feature_matrix) > 0:
                    assert feature_matrix.shape[1] > 0, "Should have some features"
            except ValueError:
                # It's acceptable to reject very short series
                pass
    
    def test_constant_price_series(self, analyzer, temp_data_dir):
        """Test handling of constant price series."""
        # Create token with constant prices
        constant_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': [100.0] * 100  # Constant price
        })
        constant_df = constant_df.with_columns([pl.col('price').log().alias('log_price')])
        
        file_path = temp_data_dir / "constant_token.parquet"
        constant_df.write_parquet(file_path)
        
        # Load and process
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        # Should handle constant series gracefully
        if len(token_data) > 0:
            try:
                feature_matrix, token_names = analyzer.prepare_price_only_data(
                    token_data, method='returns'
                )
                # If processed, returns should be zero
                if len(feature_matrix) > 0:
                    # Returns of constant series should be near zero
                    assert np.allclose(feature_matrix, 0, atol=1e-10), \
                        "Returns of constant series should be zero"
            except (ValueError, ZeroDivisionError):
                # It's acceptable to reject constant series
                pass
    
    def test_invalid_price_values(self, analyzer, temp_data_dir):
        """Test handling of invalid price values (negative, zero, NaN)."""
        # Create token with problematic prices
        problematic_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=9),
                interval="1m",
                eager=True
            ),
            'price': [100.0, -50.0, 0.0, 150.0, np.nan, 200.0, np.inf, 75.0, -np.inf, 125.0]
        })
        
        file_path = temp_data_dir / "problematic_token.parquet"
        problematic_df.write_parquet(file_path)
        
        # Load and process
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        # Should handle problematic values gracefully
        if len(token_data) > 0:
            # Loading should either succeed with cleaned data or fail gracefully
            try:
                feature_matrix, token_names = analyzer.prepare_price_only_data(
                    token_data, method='returns'
                )
                # If successful, results should be clean
                if len(feature_matrix) > 0:
                    assert not np.any(np.isnan(feature_matrix)), "Result should not contain NaN"
                    assert not np.any(np.isinf(feature_matrix)), "Result should not contain inf"
            except (ValueError, RuntimeError):
                # It's acceptable to reject problematic data
                pass


@pytest.mark.unit
class TestSequenceLengthHandling:
    """Test handling of different sequence lengths."""
    
    def test_max_length_constraint(self, analyzer, synthetic_token_data):
        """Test max_length parameter in price preparation."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=3)
        
        max_length = 50
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='returns', max_length=max_length
        )
        
        if len(feature_matrix) > 0:
            # All sequences should respect max_length constraint
            for i, sequence in enumerate(feature_matrix):
                assert len(sequence) <= max_length, \
                    f"Sequence {i} length {len(sequence)} exceeds max_length {max_length}"
    
    def test_variable_length_handling(self, analyzer, temp_data_dir):
        """Test handling of tokens with very different lengths."""
        # Create tokens with different lengths
        lengths = [50, 150, 300, 500]
        
        for i, length in enumerate(lengths):
            # Create price series
            prices = 100 + np.random.normal(0, 1, length)
            timestamps = pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=length-1),
                interval="1m",
                eager=True
            )
            
            df = pl.DataFrame({
                'datetime': timestamps,
                'price': prices
            })
            df = df.with_columns([pl.col('price').log().alias('log_price')])
            
            file_path = temp_data_dir / f"token_{length}.parquet"
            df.write_parquet(file_path)
        
        # Load and process
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='returns'
        )
        
        # Should handle variable lengths by using common length
        if len(feature_matrix) > 0:
            # All resulting sequences should have same length
            sequence_lengths = [len(seq) for seq in feature_matrix]
            assert len(set(sequence_lengths)) == 1, \
                "All sequences should have same length after processing"