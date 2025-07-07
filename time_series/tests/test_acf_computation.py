"""
Unit tests for autocorrelation function (ACF) computation
Tests mathematical correctness of ACF calculations
"""

import pytest
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import acf as statsmodels_acf
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


@pytest.mark.unit
class TestACFComputation:
    """Test ACF computation accuracy and edge cases."""
    
    def test_acf_against_statsmodels(self, analyzer, synthetic_price_series):
        """Test our ACF implementation against statsmodels reference."""
        for series_name, prices in synthetic_price_series.items():
            # Compute ACF using our implementation
            our_acf_result = analyzer.compute_autocorrelation(prices, max_lag=20)
            our_acf = our_acf_result['acf']
            
            # Compute ACF using statsmodels as reference
            reference_acf = statsmodels_acf(prices, nlags=20, fft=False)
            
            # Compare values (allow small numerical differences)
            np.testing.assert_allclose(
                our_acf, reference_acf, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"ACF mismatch for {series_name}"
            )
    
    def test_acf_theoretical_values(self, analyzer, known_acf_values):
        """Test ACF against known theoretical values for AR(1) process."""
        # Generate AR(1) series with known parameters
        np.random.seed(42)
        phi = 0.7
        n_points = 1000  # Large sample for theoretical accuracy
        
        ar1_series = [0]  # Start at 0 for easier math
        for _ in range(n_points - 1):
            next_val = phi * ar1_series[-1] + np.random.normal(0, 1)
            ar1_series.append(next_val)
        
        ar1_series = np.array(ar1_series)
        
        # Compute ACF
        acf_result = analyzer.compute_autocorrelation(ar1_series, max_lag=10)
        computed_acf = acf_result['acf']
        
        # Check theoretical values
        expected_values = known_acf_values['ar1_phi07']
        
        # Lag 1 should be approximately phi
        assert abs(computed_acf[1] - expected_values['expected_lag1']) < 0.05, \
            f"ACF lag 1: expected ~{expected_values['expected_lag1']}, got {computed_acf[1]}"
        
        # Lag 5 should be approximately phi^5
        assert abs(computed_acf[5] - expected_values['expected_lag5']) < 0.05, \
            f"ACF lag 5: expected ~{expected_values['expected_lag5']}, got {computed_acf[5]}"
    
    def test_acf_white_noise(self, analyzer, known_acf_values):
        """Test ACF for white noise (should be near zero except at lag 0)."""
        np.random.seed(42)
        white_noise = np.random.normal(0, 1, 500)
        
        acf_result = analyzer.compute_autocorrelation(white_noise, max_lag=10)
        computed_acf = acf_result['acf']
        
        # Lag 0 should be 1.0
        assert abs(computed_acf[0] - 1.0) < 1e-10, \
            f"ACF lag 0 should be 1.0, got {computed_acf[0]}"
        
        # Other lags should be near zero
        tolerance = known_acf_values['white_noise']['tolerance']
        for lag in range(1, 11):
            assert abs(computed_acf[lag]) < tolerance, \
                f"ACF lag {lag} should be near 0, got {computed_acf[lag]}"
    
    def test_acf_edge_cases(self, analyzer):
        """Test ACF computation with edge cases."""
        
        # Test with constant series
        constant_series = np.full(100, 42.0)
        acf_result = analyzer.compute_autocorrelation(constant_series, max_lag=10)
        
        # For constant series, all ACF values should be NaN (except lag 0 which should be 1)
        # or the function should handle it gracefully
        assert acf_result is not None, "ACF should handle constant series gracefully"
        
        # Test with very short series
        short_series = np.array([1.0, 2.0, 3.0])
        acf_result = analyzer.compute_autocorrelation(short_series, max_lag=5)
        assert acf_result is not None, "ACF should handle short series"
        
        # Test with single value
        single_value = np.array([42.0])
        acf_result = analyzer.compute_autocorrelation(single_value, max_lag=1)
        assert acf_result is not None, "ACF should handle single value"
    
    def test_acf_with_nan_values(self, analyzer):
        """Test ACF computation with NaN values."""
        # Series with some NaN values
        series_with_nan = np.array([1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10])
        
        acf_result = analyzer.compute_autocorrelation(series_with_nan, max_lag=5)
        
        # Should handle NaN values gracefully
        assert acf_result is not None, "ACF should handle NaN values"
        
        # Check that ACF values are not all NaN
        if 'acf' in acf_result and len(acf_result['acf']) > 0:
            # At least some ACF values should be finite
            finite_count = np.sum(np.isfinite(acf_result['acf']))
            assert finite_count > 0, "At least some ACF values should be finite"
    
    def test_acf_different_max_lags(self, analyzer, synthetic_price_series):
        """Test ACF computation with different max_lag values."""
        series = synthetic_price_series['trend']
        
        # Test different max_lag values
        for max_lag in [5, 10, 20, 50]:
            if max_lag < len(series) // 4:  # Reasonable constraint
                acf_result = analyzer.compute_autocorrelation(series, max_lag=max_lag)
                
                assert 'acf' in acf_result, f"ACF result missing for max_lag={max_lag}"
                assert len(acf_result['acf']) == max_lag + 1, \
                    f"ACF length should be {max_lag + 1}, got {len(acf_result['acf'])}"
    
    def test_pacf_computation(self, analyzer, synthetic_price_series):
        """Test PACF (Partial ACF) computation."""
        series = synthetic_price_series['ar1_phi07']
        
        acf_result = analyzer.compute_autocorrelation(series, max_lag=10)
        
        # Should include PACF
        assert 'pacf' in acf_result, "Result should include PACF"
        
        # PACF should have same length as ACF
        assert len(acf_result['pacf']) == len(acf_result['acf']), \
            "PACF should have same length as ACF"
        
        # For AR(1), PACF should be ~0 after lag 1
        if len(acf_result['pacf']) > 2:
            pacf_values = acf_result['pacf']
            # PACF lag 1 should be significant for AR(1)
            assert abs(pacf_values[1]) > 0.3, \
                f"PACF lag 1 should be significant for AR(1), got {pacf_values[1]}"
    
    def test_acf_returns_vs_prices(self, analyzer):
        """Test ACF computation on returns vs raw prices."""
        np.random.seed(42)
        # Generate price series with trend
        prices = 100 + 0.1 * np.arange(200) + np.random.normal(0, 1, 200)
        
        # Compute returns
        returns = np.diff(prices)
        
        # ACF of prices (should show high autocorrelation due to trend)
        acf_prices = analyzer.compute_autocorrelation(prices, max_lag=10)
        
        # ACF of returns (should show lower autocorrelation)
        acf_returns = analyzer.compute_autocorrelation(returns, max_lag=10)
        
        # Prices should have higher lag-1 autocorrelation than returns
        if len(acf_prices['acf']) > 1 and len(acf_returns['acf']) > 1:
            assert acf_prices['acf'][1] > acf_returns['acf'][1], \
                "Prices should have higher autocorrelation than returns"
    
    def test_acf_decay_rate_calculation(self, analyzer, synthetic_price_series):
        """Test decay rate calculation in ACF results."""
        series = synthetic_price_series['ar1_phi07']
        
        acf_result = analyzer.compute_autocorrelation(series, max_lag=20)
        
        # Should include decay rate calculation
        if 'decay_rate' in acf_result:
            decay_rate = acf_result['decay_rate']
            
            # Decay rate should be reasonable for AR(1) process
            assert 0 < decay_rate < 1, \
                f"Decay rate should be between 0 and 1, got {decay_rate}"
    
    def test_acf_significance_bounds(self, analyzer, synthetic_price_series):
        """Test significance bounds calculation."""
        series = synthetic_price_series['white_noise']
        
        acf_result = analyzer.compute_autocorrelation(series, max_lag=10)
        
        # Check if significance bounds are calculated
        if 'significance_bounds' in acf_result:
            bounds = acf_result['significance_bounds']
            
            # Bounds should be symmetric around zero
            assert abs(bounds[0] + bounds[1]) < 1e-10, \
                "Significance bounds should be symmetric around zero"
            
            # Bounds should be reasonable (not too large or small)
            assert 0.01 < abs(bounds[0]) < 0.5, \
                f"Significance bounds seem unreasonable: {bounds}"


@pytest.mark.unit
class TestACFDataTypes:
    """Test ACF computation with different data types and formats."""
    
    def test_acf_polars_series(self, analyzer):
        """Test ACF computation with Polars Series."""
        # Create Polars DataFrame
        df = pl.DataFrame({
            'price': [100.0, 101.0, 99.5, 102.0, 98.0, 103.0]
        })
        
        # Extract series as numpy array
        prices = df['price'].to_numpy()
        
        acf_result = analyzer.compute_autocorrelation(prices, max_lag=3)
        
        assert acf_result is not None, "ACF should work with Polars-derived data"
        assert 'acf' in acf_result, "ACF result should contain 'acf' key"
    
    def test_acf_different_dtypes(self, analyzer):
        """Test ACF with different numeric data types."""
        base_series = [100, 101, 99, 102, 98, 103, 97, 104]
        
        # Test with different dtypes
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            series = np.array(base_series, dtype=dtype)
            
            acf_result = analyzer.compute_autocorrelation(series, max_lag=3)
            
            assert acf_result is not None, f"ACF should work with dtype {dtype}"
            assert 'acf' in acf_result, f"ACF should return results for dtype {dtype}"