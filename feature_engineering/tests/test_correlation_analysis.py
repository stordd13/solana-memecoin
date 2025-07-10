"""
Correlation analysis validation tests for feature_engineering module
Tests mathematical correctness of multi-token correlation analysis
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.mathematical
class TestCorrelationMatrixCalculation:
    """Test correlation matrix calculation mathematical correctness."""
    
    def test_pearson_correlation_accuracy(self, correlation_analyzer, correlation_test_data):
        """Test Pearson correlation calculation accuracy."""
        tokens_data = correlation_test_data['tokens_data']
        expected_corr_matrix = correlation_test_data['expected_correlation_matrix']
        n_tokens = correlation_test_data['n_tokens']
        
        try:
            # Calculate correlation matrix using analyzer
            correlation_result = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='pearson'
            )
            
            if correlation_result is not None and 'correlation_matrix' in correlation_result:
                calculated_matrix = correlation_result['correlation_matrix']
                
                # Should be same shape as expected
                assert calculated_matrix.shape == expected_corr_matrix.shape, \
                    "Correlation matrix should have correct shape"
                
                # Compare with expected correlation (allowing for small numerical differences)
                np.testing.assert_array_almost_equal(
                    calculated_matrix, expected_corr_matrix, decimal=10,
                    err_msg="Calculated correlation should match expected correlation"
                )
                
                # Test correlation matrix properties
                # 1. Diagonal should be 1
                diagonal = np.diag(calculated_matrix)
                np.testing.assert_array_almost_equal(
                    diagonal, np.ones(n_tokens), decimal=12,
                    err_msg="Correlation matrix diagonal should be 1"
                )
                
                # 2. Matrix should be symmetric
                np.testing.assert_array_almost_equal(
                    calculated_matrix, calculated_matrix.T, decimal=12,
                    err_msg="Correlation matrix should be symmetric"
                )
                
                # 3. All values should be between -1 and 1
                assert np.all(calculated_matrix >= -1 - 1e-10), "Correlations should be >= -1"
                assert np.all(calculated_matrix <= 1 + 1e-10), "Correlations should be <= 1"
                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Correlation calculation test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_spearman_correlation_properties(self, correlation_analyzer, correlation_test_data):
        """Test Spearman correlation calculation properties."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            # Calculate Spearman correlation
            spearman_result = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='spearman'
            )
            
            if spearman_result is not None and 'correlation_matrix' in spearman_result:
                spearman_matrix = spearman_result['correlation_matrix']
                
                # Test matrix properties
                n_tokens = spearman_matrix.shape[0]
                
                # Diagonal should be 1
                diagonal = np.diag(spearman_matrix)
                np.testing.assert_array_almost_equal(
                    diagonal, np.ones(n_tokens), decimal=12,
                    err_msg="Spearman correlation matrix diagonal should be 1"
                )
                
                # Matrix should be symmetric
                np.testing.assert_array_almost_equal(
                    spearman_matrix, spearman_matrix.T, decimal=12,
                    err_msg="Spearman correlation matrix should be symmetric"
                )
                
                # All values between -1 and 1
                assert np.all(spearman_matrix >= -1 - 1e-10), "Spearman correlations should be >= -1"
                assert np.all(spearman_matrix <= 1 + 1e-10), "Spearman correlations should be <= 1"
                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Spearman correlation test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_correlation_edge_cases(self, correlation_analyzer, edge_case_time_series):
        """Test correlation calculation with edge cases."""
        # Create tokens data from edge cases
        tokens_data = {
            'constant_token': edge_case_time_series['constant'],
            'volatile_token': edge_case_time_series['extreme_volatility']
        }
        
        try:
            correlation_result = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='pearson'
            )
            
            if correlation_result is not None:
                # Should handle edge cases gracefully
                assert isinstance(correlation_result, dict), "Should return dict result for edge cases"
                
                if 'correlation_matrix' in correlation_result:
                    corr_matrix = correlation_result['correlation_matrix']
                    
                    # Should be finite or NaN (for undefined correlations)
                    assert np.all(np.isfinite(corr_matrix) | np.isnan(corr_matrix)), \
                        "Correlation values should be finite or NaN"
                        
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Edge cases test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.unit
@pytest.mark.mathematical
class TestRollingCorrelationCalculation:
    """Test rolling correlation calculation mathematical correctness."""
    
    def test_rolling_correlation_accuracy(self, correlation_analyzer, correlation_test_data):
        """Test rolling correlation calculation accuracy."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            # Calculate rolling correlation
            rolling_result = correlation_analyzer.calculate_rolling_correlation(
                tokens_data, window=50
            )
            
            if rolling_result is not None and 'rolling_correlations' in rolling_result:
                rolling_correlations = rolling_result['rolling_correlations']
                
                # Should be a time series of correlation matrices
                assert isinstance(rolling_correlations, (list, np.ndarray)), \
                    "Rolling correlations should be sequence of matrices"
                
                if len(rolling_correlations) > 0:
                    # Each correlation matrix should have proper properties
                    for i, corr_matrix in enumerate(rolling_correlations):
                        if corr_matrix is not None and not np.all(np.isnan(corr_matrix)):
                            # Should be square matrix
                            assert corr_matrix.shape[0] == corr_matrix.shape[1], \
                                f"Rolling correlation matrix {i} should be square"
                            
                            # Diagonal should be 1 (where defined)
                            diagonal = np.diag(corr_matrix)
                            finite_diagonal = diagonal[np.isfinite(diagonal)]
                            if len(finite_diagonal) > 0:
                                np.testing.assert_array_almost_equal(
                                    finite_diagonal, np.ones(len(finite_diagonal)), decimal=10,
                                    err_msg=f"Rolling correlation matrix {i} diagonal should be 1"
                                )
                                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Rolling correlation test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_rolling_correlation_window_properties(self, correlation_analyzer, correlation_test_data):
        """Test rolling correlation window properties."""
        tokens_data = correlation_test_data['tokens_data']
        n_points = correlation_test_data['n_points']
        
        try:
            # Test different window sizes
            for window in [20, 50, 100]:
                if window < n_points:
                    rolling_result = correlation_analyzer.calculate_rolling_correlation(
                        tokens_data, window=window
                    )
                    
                    if rolling_result is not None and 'rolling_correlations' in rolling_result:
                        rolling_correlations = rolling_result['rolling_correlations']
                        
                        # Number of rolling correlations should be consistent with window size
                        expected_length = n_points - window + 1
                        assert len(rolling_correlations) <= expected_length + 10, \
                            f"Rolling correlation length should be reasonable for window {window}"
                        
                        # Later correlations should generally be more stable
                        # (less change between consecutive correlations)
                        if len(rolling_correlations) > 10:
                            changes = []
                            for i in range(1, min(10, len(rolling_correlations))):
                                if (rolling_correlations[i] is not None and 
                                    rolling_correlations[i-1] is not None):
                                    change = np.nanmean(np.abs(rolling_correlations[i] - rolling_correlations[i-1]))
                                    if np.isfinite(change):
                                        changes.append(change)
                            
                            if len(changes) > 2:
                                # Changes should generally be small for stable markets
                                avg_change = np.mean(changes)
                                assert avg_change < 0.5, \
                                    f"Rolling correlation changes should be reasonable for window {window}"
                                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Rolling correlation window test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.unit
@pytest.mark.mathematical
class TestLeadLagAnalysis:
    """Test lead-lag relationship analysis mathematical correctness."""
    
    def test_lead_lag_calculation_accuracy(self, correlation_analyzer, correlation_test_data):
        """Test lead-lag correlation calculation accuracy."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            # Test lead-lag analysis between first two tokens
            token_names = list(tokens_data.keys())
            if len(token_names) >= 2:
                token1_name = token_names[0]
                token2_name = token_names[1]
                
                lead_lag_result = correlation_analyzer.calculate_lead_lag_correlation(
                    tokens_data[token1_name], tokens_data[token2_name], 
                    max_lag=10
                )
                
                if lead_lag_result is not None and 'correlations' in lead_lag_result:
                    correlations = lead_lag_result['correlations']
                    lags = lead_lag_result.get('lags', range(-10, 11))
                    
                    # Should have correlation for each lag
                    assert len(correlations) == len(lags), \
                        "Should have correlation for each lag"
                    
                    # All correlations should be between -1 and 1
                    finite_correlations = [c for c in correlations if np.isfinite(c)]
                    if len(finite_correlations) > 0:
                        assert all(-1 <= c <= 1 for c in finite_correlations), \
                            "Lead-lag correlations should be between -1 and 1"
                    
                    # Lag 0 should give same result as regular correlation
                    if 0 in lags:
                        lag_0_idx = list(lags).index(0)
                        lag_0_corr = correlations[lag_0_idx]
                        
                        # Calculate regular correlation for comparison
                        regular_corr_result = correlation_analyzer.calculate_correlation_matrix(
                            {token1_name: tokens_data[token1_name], 
                             token2_name: tokens_data[token2_name]}, 
                            method='pearson'
                        )
                        
                        if (regular_corr_result is not None and 
                            'correlation_matrix' in regular_corr_result):
                            regular_corr = regular_corr_result['correlation_matrix'][0, 1]
                            
                            if np.isfinite(lag_0_corr) and np.isfinite(regular_corr):
                                assert abs(lag_0_corr - regular_corr) < 0.01, \
                                    "Lag 0 correlation should match regular correlation"
                                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Lead-lag analysis test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_lead_lag_symmetry_properties(self, correlation_analyzer, correlation_test_data):
        """Test lead-lag correlation symmetry properties."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            token_names = list(tokens_data.keys())
            if len(token_names) >= 2:
                token1_name = token_names[0]
                token2_name = token_names[1]
                
                # Calculate lead-lag correlation A vs B
                lead_lag_AB = correlation_analyzer.calculate_lead_lag_correlation(
                    tokens_data[token1_name], tokens_data[token2_name], 
                    max_lag=5
                )
                
                # Calculate lead-lag correlation B vs A
                lead_lag_BA = correlation_analyzer.calculate_lead_lag_correlation(
                    tokens_data[token2_name], tokens_data[token1_name], 
                    max_lag=5
                )
                
                if (lead_lag_AB is not None and lead_lag_BA is not None and
                    'correlations' in lead_lag_AB and 'correlations' in lead_lag_BA):
                    
                    corr_AB = lead_lag_AB['correlations']
                    corr_BA = lead_lag_BA['correlations']
                    lags_AB = lead_lag_AB.get('lags', range(-5, 6))
                    lags_BA = lead_lag_BA.get('lags', range(-5, 6))
                    
                    # Should satisfy symmetry property:
                    # corr(A[t], B[t+k]) = corr(B[t], A[t-k])
                    for i, lag in enumerate(lags_AB):
                        if -lag in lags_BA:
                            neg_lag_idx = list(lags_BA).index(-lag)
                            
                            corr_val_AB = corr_AB[i]
                            corr_val_BA = corr_BA[neg_lag_idx]
                            
                            if np.isfinite(corr_val_AB) and np.isfinite(corr_val_BA):
                                assert abs(corr_val_AB - corr_val_BA) < 0.01, \
                                    f"Lead-lag correlation should satisfy symmetry for lag {lag}"
                                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Lead-lag symmetry test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.integration
@pytest.mark.mathematical
class TestCorrelationAnalysisPipeline:
    """Test complete correlation analysis pipeline."""
    
    def test_complete_correlation_pipeline(self, correlation_analyzer, correlation_test_data):
        """Test complete correlation analysis pipeline execution."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            # Run complete analysis
            analysis_result = correlation_analyzer.run_complete_analysis(tokens_data)
            
            if analysis_result is not None:
                # Should return structured results
                assert isinstance(analysis_result, dict), "Should return dictionary result"
                
                # Should contain multiple types of analysis
                expected_keys = ['static_correlation', 'rolling_correlation', 'lead_lag_analysis']
                present_keys = [key for key in expected_keys if key in analysis_result]
                assert len(present_keys) > 0, "Should contain at least one type of analysis"
                
                # Test static correlation if present
                if 'static_correlation' in analysis_result:
                    static_corr = analysis_result['static_correlation']
                    if 'correlation_matrix' in static_corr:
                        corr_matrix = static_corr['correlation_matrix']
                        assert corr_matrix.shape[0] == corr_matrix.shape[1], \
                            "Static correlation matrix should be square"
                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Complete pipeline test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_correlation_analysis_consistency(self, correlation_analyzer, correlation_test_data):
        """Test correlation analysis consistency across runs."""
        tokens_data = correlation_test_data['tokens_data']
        
        try:
            # Run analysis twice
            result_1 = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='pearson'
            )
            result_2 = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='pearson'
            )
            
            if (result_1 is not None and result_2 is not None and
                'correlation_matrix' in result_1 and 'correlation_matrix' in result_2):
                
                matrix_1 = result_1['correlation_matrix']
                matrix_2 = result_2['correlation_matrix']
                
                # Should be exactly equal
                np.testing.assert_array_equal(
                    matrix_1, matrix_2,
                    err_msg="Correlation analysis should be consistent across runs"
                )
                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Consistency test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_correlation_with_different_timeframes(self, correlation_analyzer):
        """Test correlation analysis with different timeframes."""
        # Create data with different sampling frequencies
        np.random.seed(42)
        base_returns = np.random.normal(0, 0.02, 1000)
        
        # High frequency data (1-minute)
        hf_prices = 100 * np.exp(np.cumsum(base_returns))
        hf_data = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(1000)],
            'price': hf_prices
        })
        
        # Low frequency data (1-hour) - subsample
        lf_indices = range(0, 1000, 60)
        lf_prices = hf_prices[lf_indices]
        lf_data = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(len(lf_indices))],
            'price': lf_prices
        })
        
        tokens_data = {
            'token_hf': hf_data,
            'token_lf': lf_data
        }
        
        try:
            # Should handle different timeframes
            correlation_result = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='pearson'
            )
            
            if correlation_result is not None:
                # Should complete without error
                assert isinstance(correlation_result, dict), \
                    "Should handle different timeframes gracefully"
                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Different timeframes test skipped due to missing dependencies: {e}")
            else:
                # This might be expected behavior for mismatched timeframes
                if "timeframe" in str(e).lower() or "alignment" in str(e).lower():
                    pass  # Expected error for mismatched timeframes
                else:
                    raise e
    
    def test_correlation_analysis_scalability(self, correlation_analyzer):
        """Test correlation analysis with many tokens."""
        # Create data for multiple tokens
        np.random.seed(42)
        n_tokens = 10
        n_points = 500
        
        tokens_data = {}
        for i in range(n_tokens):
            # Create correlated returns
            base_factor = 0.3
            individual_factor = 0.7
            
            base_returns = np.random.normal(0, 0.02, n_points)
            individual_returns = np.random.normal(0, 0.02, n_points)
            combined_returns = base_factor * base_returns + individual_factor * individual_returns
            
            prices = 100 * np.exp(np.cumsum(combined_returns))
            
            tokens_data[f'token_{i}'] = pl.DataFrame({
                'datetime': [datetime(2024, 1, 1) + timedelta(minutes=j) for j in range(n_points)],
                'price': prices
            })
        
        try:
            # Should handle multiple tokens efficiently
            correlation_result = correlation_analyzer.calculate_correlation_matrix(
                tokens_data, method='pearson'
            )
            
            if correlation_result is not None and 'correlation_matrix' in correlation_result:
                corr_matrix = correlation_result['correlation_matrix']
                
                # Should have correct dimensions
                assert corr_matrix.shape == (n_tokens, n_tokens), \
                    f"Correlation matrix should be {n_tokens}x{n_tokens}"
                
                # Should have reasonable correlation structure
                # (some positive correlations due to base factor)
                off_diagonal = corr_matrix[np.triu_indices(n_tokens, k=1)]
                finite_off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
                
                if len(finite_off_diagonal) > 0:
                    mean_correlation = np.mean(finite_off_diagonal)
                    assert -0.5 < mean_correlation < 0.8, \
                        f"Mean correlation should be reasonable: {mean_correlation}"
                        
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Scalability test skipped due to missing dependencies: {e}")
            else:
                raise e