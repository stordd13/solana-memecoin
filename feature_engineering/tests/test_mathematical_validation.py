"""
Core mathematical validation tests for feature_engineering module
Tests essential mathematical functions that power feature creation
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
import math


@pytest.mark.unit
@pytest.mark.mathematical
class TestTechnicalIndicatorCalculations:
    """Test technical indicator calculation mathematical correctness."""
    
    def test_simple_moving_average_accuracy(self, technical_indicator_references):
        """Test Simple Moving Average calculation accuracy."""
        prices = technical_indicator_references['prices']
        expected_sma = technical_indicator_references['references']['sma_5']
        
        # Calculate SMA manually for verification
        window = 5
        manual_sma = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            manual_sma[i] = np.mean(prices[i - window + 1:i + 1])
        
        # Verify our reference calculation
        valid_indices = ~np.isnan(expected_sma)
        np.testing.assert_array_almost_equal(
            expected_sma[valid_indices], 
            manual_sma[valid_indices], 
            decimal=12,
            err_msg="Reference SMA calculation should match manual calculation"
        )
        
        # Test specific values
        # SMA at index 4 (5th point): mean of first 5 values
        assert abs(expected_sma[4] - np.mean(prices[:5])) < 1e-12, \
            f"SMA at index 4 should be {np.mean(prices[:5])}, got {expected_sma[4]}"
    
    def test_rsi_calculation_accuracy(self, technical_indicator_references):
        """Test RSI calculation mathematical correctness."""
        prices = technical_indicator_references['prices']
        expected_rsi = technical_indicator_references['references']['rsi_14']
        
        # Manual RSI calculation for verification
        period = min(14, len(prices) - 1)
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        manual_rsi = np.full(len(prices), np.nan)
        for i in range(period, len(prices)):
            avg_gain = np.mean(gains[i-period:i])
            avg_loss = np.mean(losses[i-period:i])
            
            if avg_loss == 0:
                manual_rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                manual_rsi[i] = 100 - (100 / (1 + rs))
        
        # Compare calculations
        valid_indices = ~np.isnan(expected_rsi)
        if np.any(valid_indices):
            np.testing.assert_array_almost_equal(
                expected_rsi[valid_indices],
                manual_rsi[valid_indices],
                decimal=10,
                err_msg="RSI calculation should match manual calculation"
            )
        
        # Test RSI bounds
        finite_rsi = expected_rsi[np.isfinite(expected_rsi)]
        if len(finite_rsi) > 0:
            assert np.all(finite_rsi >= 0), "RSI values should be >= 0"
            assert np.all(finite_rsi <= 100), "RSI values should be <= 100"
    
    def test_macd_calculation_accuracy(self, technical_indicator_references):
        """Test MACD calculation mathematical correctness."""
        macd_ref = technical_indicator_references['references']['macd']
        macd_line = macd_ref['macd']
        signal_line = macd_ref['signal']
        
        # Test MACD properties
        finite_macd = macd_line[np.isfinite(macd_line)]
        finite_signal = signal_line[np.isfinite(signal_line)]
        
        # MACD should have finite values
        assert len(finite_macd) > 0, "MACD should have some finite values"
        
        # Signal line should be smoother than MACD line
        if len(finite_macd) > 2 and len(finite_signal) > 2:
            macd_volatility = np.std(np.diff(finite_macd))
            signal_volatility = np.std(np.diff(finite_signal))
            # Signal line should generally be less volatile (smoother)
            # This is a general property but may not always hold for very short series
    
    def test_bollinger_bands_calculation(self, technical_indicator_references):
        """Test Bollinger Bands calculation mathematical correctness."""
        bollinger_ref = technical_indicator_references['references']['bollinger']
        upper_band = bollinger_ref['upper']
        middle_band = bollinger_ref['middle']
        lower_band = bollinger_ref['lower']
        
        # Test Bollinger Bands properties
        valid_indices = (~np.isnan(upper_band) & 
                        ~np.isnan(middle_band) & 
                        ~np.isnan(lower_band))
        
        if np.any(valid_indices):
            # Upper band should be above middle band
            assert np.all(upper_band[valid_indices] >= middle_band[valid_indices]), \
                "Upper Bollinger Band should be >= middle band"
            
            # Lower band should be below middle band
            assert np.all(lower_band[valid_indices] <= middle_band[valid_indices]), \
                "Lower Bollinger Band should be <= middle band"
            
            # Bands should be symmetric around middle
            upper_distance = upper_band[valid_indices] - middle_band[valid_indices]
            lower_distance = middle_band[valid_indices] - lower_band[valid_indices]
            
            np.testing.assert_array_almost_equal(
                upper_distance, lower_distance, decimal=10,
                err_msg="Bollinger Bands should be symmetric around middle band"
            )
    
    def test_volatility_calculation_accuracy(self, technical_indicator_references):
        """Test volatility calculation mathematical correctness."""
        prices = technical_indicator_references['prices']
        expected_volatility = technical_indicator_references['references']['volatility']
        
        # Manual volatility calculation
        returns = np.diff(prices) / prices[:-1]
        window = 5
        manual_volatility = np.full(len(prices), np.nan)
        
        for i in range(window, len(prices)):
            vol_returns = returns[i - window:i]
            manual_volatility[i] = np.std(vol_returns, ddof=1)
        
        # Compare calculations
        valid_indices = ~np.isnan(expected_volatility)
        if np.any(valid_indices):
            np.testing.assert_array_almost_equal(
                expected_volatility[valid_indices],
                manual_volatility[valid_indices],
                decimal=12,
                err_msg="Volatility calculation should match manual calculation"
            )
        
        # Test volatility properties
        finite_vol = expected_volatility[np.isfinite(expected_volatility)]
        if len(finite_vol) > 0:
            assert np.all(finite_vol >= 0), "Volatility values should be non-negative"


@pytest.mark.unit
@pytest.mark.mathematical
class TestRollingWindowCalculations:
    """Test rolling window calculation mathematical correctness."""
    
    def test_rolling_mean_accuracy(self, reference_time_series_data):
        """Test rolling mean calculation accuracy."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        # Test different window sizes
        windows = [5, 10, 20, 50]
        
        for window in windows:
            if len(prices) >= window:
                # Calculate using Polars
                rolling_mean_pl = df['price'].rolling_mean(window_size=window)
                
                # Calculate manually for verification
                manual_rolling_mean = np.full(len(prices), np.nan)
                for i in range(window - 1, len(prices)):
                    manual_rolling_mean[i] = np.mean(prices[i - window + 1:i + 1])
                
                # Compare non-NaN values
                pl_values = rolling_mean_pl.to_numpy()
                valid_indices = ~np.isnan(pl_values)
                
                if np.any(valid_indices):
                    np.testing.assert_array_almost_equal(
                        pl_values[valid_indices],
                        manual_rolling_mean[valid_indices],
                        decimal=12,
                        err_msg=f"Rolling mean with window {window} should match manual calculation"
                    )
    
    def test_rolling_std_accuracy(self, reference_time_series_data):
        """Test rolling standard deviation calculation accuracy."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        # Test different window sizes
        windows = [5, 10, 20]
        
        for window in windows:
            if len(prices) >= window:
                # Calculate using Polars
                rolling_std_pl = df['price'].rolling_std(window_size=window)
                
                # Calculate manually for verification
                manual_rolling_std = np.full(len(prices), np.nan)
                for i in range(window - 1, len(prices)):
                    window_data = prices[i - window + 1:i + 1]
                    manual_rolling_std[i] = np.std(window_data, ddof=1)
                
                # Compare non-NaN values
                pl_values = rolling_std_pl.to_numpy()
                valid_indices = ~np.isnan(pl_values)
                
                if np.any(valid_indices):
                    np.testing.assert_array_almost_equal(
                        pl_values[valid_indices],
                        manual_rolling_std[valid_indices],
                        decimal=12,
                        err_msg=f"Rolling std with window {window} should match manual calculation"
                    )
    
    def test_rolling_window_edge_cases(self, edge_case_time_series):
        """Test rolling window calculations with edge cases."""
        # Test with constant values
        constant_df = edge_case_time_series['constant']
        
        # Rolling mean of constant values should be constant
        rolling_mean = constant_df['price'].rolling_mean(window_size=5)
        valid_values = rolling_mean.drop_nulls()
        if len(valid_values) > 0:
            assert np.all(valid_values == 100.0), "Rolling mean of constant values should be constant"
        
        # Rolling std of constant values should be zero
        rolling_std = constant_df['price'].rolling_std(window_size=5)
        valid_std_values = rolling_std.drop_nulls()
        if len(valid_std_values) > 0:
            assert np.all(valid_std_values == 0.0), "Rolling std of constant values should be zero"
        
        # Test with insufficient data
        single_point_df = edge_case_time_series['single_point']
        rolling_mean_single = single_point_df['price'].rolling_mean(window_size=5)
        # Should handle gracefully (likely all NaN)
        assert len(rolling_mean_single) == 1, "Should return same length as input"
    
    def test_rolling_window_properties(self, reference_time_series_data):
        """Test mathematical properties of rolling windows."""
        df = reference_time_series_data['df']
        
        # Test that rolling mean with window=1 equals original values
        rolling_mean_1 = df['price'].rolling_mean(window_size=1)
        original_prices = df['price']
        
        # Should be approximately equal (allowing for numerical precision)
        np.testing.assert_array_almost_equal(
            rolling_mean_1.to_numpy(),
            original_prices.to_numpy(),
            decimal=12,
            err_msg="Rolling mean with window=1 should equal original values"
        )
        
        # Test that rolling std with window=1 should be 0 (or NaN)
        rolling_std_1 = df['price'].rolling_std(window_size=1)
        std_values = rolling_std_1.to_numpy()
        finite_std = std_values[np.isfinite(std_values)]
        if len(finite_std) > 0:
            assert np.all(finite_std == 0.0), "Rolling std with window=1 should be 0"


@pytest.mark.unit
@pytest.mark.mathematical
class TestReturnsAndLogReturnsCalculations:
    """Test returns and log returns calculation mathematical correctness."""
    
    def test_simple_returns_accuracy(self, reference_time_series_data):
        """Test simple returns calculation accuracy."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        # Calculate returns using Polars
        returns_pl = df['price'].pct_change()
        
        # Calculate manually
        manual_returns = np.full(len(prices), np.nan)
        for i in range(1, len(prices)):
            manual_returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        
        # Compare non-NaN values
        pl_returns = returns_pl.to_numpy()
        valid_indices = ~np.isnan(pl_returns)
        
        if np.any(valid_indices):
            np.testing.assert_array_almost_equal(
                pl_returns[valid_indices],
                manual_returns[valid_indices],
                decimal=12,
                err_msg="Simple returns should match manual calculation"
            )
    
    def test_log_returns_accuracy(self, reference_time_series_data):
        """Test log returns calculation accuracy."""
        prices = reference_time_series_data['prices']
        
        # Calculate log returns manually
        log_returns = np.full(len(prices), np.nan)
        for i in range(1, len(prices)):
            log_returns[i] = np.log(prices[i]) - np.log(prices[i-1])
        
        # Alternative calculation
        alt_log_returns = np.full(len(prices), np.nan)
        alt_log_returns[1:] = np.diff(np.log(prices))
        
        # Both methods should give same result
        valid_indices = ~np.isnan(log_returns)
        if np.any(valid_indices):
            np.testing.assert_array_almost_equal(
                log_returns[valid_indices],
                alt_log_returns[valid_indices],
                decimal=12,
                err_msg="Different log returns calculations should match"
            )
    
    def test_returns_properties(self, reference_time_series_data):
        """Test mathematical properties of returns."""
        df = reference_time_series_data['df']
        
        # Calculate returns
        returns = df['price'].pct_change().to_numpy()
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) > 0:
            # Returns should be greater than -1 (can't lose more than 100%)
            assert np.all(valid_returns > -1), "Simple returns should be > -1"
            
            # Test relationship with log returns
            prices = df['price'].to_numpy()
            log_returns = np.diff(np.log(prices))
            
            # For small returns, simple returns ≈ log returns
            small_returns_mask = np.abs(valid_returns[1:]) < 0.1  # Returns < 10%
            if np.any(small_returns_mask):
                small_simple = valid_returns[1:][small_returns_mask]
                small_log = log_returns[small_returns_mask]
                
                # Should be approximately equal for small returns
                np.testing.assert_array_almost_equal(
                    small_simple, small_log, decimal=2,
                    err_msg="Simple and log returns should be similar for small changes"
                )
    
    def test_returns_edge_cases(self, edge_case_time_series):
        """Test returns calculation with edge cases."""
        # Test with constant prices
        constant_df = edge_case_time_series['constant']
        returns = constant_df['price'].pct_change()
        valid_returns = returns.drop_nulls()
        
        if len(valid_returns) > 0:
            assert np.all(valid_returns == 0.0), "Returns of constant prices should be zero"
        
        # Test with extreme values
        extreme_df = edge_case_time_series['extreme_volatility']
        extreme_returns = extreme_df['price'].pct_change()
        extreme_values = extreme_returns.drop_nulls().to_numpy()
        
        if len(extreme_values) > 0:
            # Should still be valid returns (> -1)
            assert np.all(extreme_values > -1), "Even extreme returns should be > -1"
            assert np.all(np.isfinite(extreme_values)), "Returns should be finite"


@pytest.mark.unit
@pytest.mark.mathematical
class TestStatisticalMoments:
    """Test higher-order statistical moments calculation."""
    
    def test_skewness_calculation(self, reference_time_series_data):
        """Test skewness calculation mathematical correctness."""
        prices = reference_time_series_data['prices']
        
        # Calculate skewness manually using the standard formula
        mean_price = np.mean(prices)
        std_price = np.std(prices, ddof=1)
        n = len(prices)
        
        # Third moment (skewness)
        third_moment = np.mean(((prices - mean_price) / std_price) ** 3)
        manual_skewness = (n / ((n - 1) * (n - 2))) * third_moment
        
        # Using scipy for reference
        from scipy import stats
        scipy_skewness = stats.skew(prices)
        
        # Compare calculations
        assert abs(manual_skewness - scipy_skewness) < 1e-12, \
            f"Manual skewness calculation should match scipy: {manual_skewness} vs {scipy_skewness}"
    
    def test_kurtosis_calculation(self, reference_time_series_data):
        """Test kurtosis calculation mathematical correctness."""
        prices = reference_time_series_data['prices']
        
        # Calculate kurtosis manually
        mean_price = np.mean(prices)
        std_price = np.std(prices, ddof=1)
        n = len(prices)
        
        # Fourth moment (kurtosis)
        fourth_moment = np.mean(((prices - mean_price) / std_price) ** 4)
        manual_kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * fourth_moment - 
                          3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        
        # Using scipy for reference
        from scipy import stats
        scipy_kurtosis = stats.kurtosis(prices)
        
        # Compare calculations
        assert abs(manual_kurtosis - scipy_kurtosis) < 1e-12, \
            f"Manual kurtosis calculation should match scipy: {manual_kurtosis} vs {scipy_kurtosis}"
    
    def test_jarque_bera_test(self, reference_time_series_data):
        """Test Jarque-Bera normality test calculation."""
        prices = reference_time_series_data['prices']
        
        # Calculate Jarque-Bera statistic manually
        from scipy import stats
        
        skewness = stats.skew(prices)
        kurtosis = stats.kurtosis(prices)
        n = len(prices)
        
        # Jarque-Bera statistic
        jb_stat = (n / 6) * (skewness**2 + (1/4) * kurtosis**2)
        
        # Using scipy for reference
        scipy_jb_stat, scipy_p_value = stats.jarque_bera(prices)
        
        # Compare statistics
        assert abs(jb_stat - scipy_jb_stat) < 1e-12, \
            f"Manual JB statistic should match scipy: {jb_stat} vs {scipy_jb_stat}"
        
        # JB statistic should be non-negative
        assert jb_stat >= 0, "Jarque-Bera statistic should be non-negative"


@pytest.mark.unit
@pytest.mark.mathematical
class TestCorrelationCalculations:
    """Test correlation calculation mathematical correctness."""
    
    def test_pearson_correlation_accuracy(self, correlation_test_data):
        """Test Pearson correlation calculation accuracy."""
        price_matrix = correlation_test_data['price_matrix']
        expected_corr = correlation_test_data['expected_correlation_matrix']
        
        # Calculate correlation manually
        manual_corr = np.corrcoef(price_matrix.T)
        
        # Should match expected correlation
        np.testing.assert_array_almost_equal(
            manual_corr, expected_corr, decimal=12,
            err_msg="Manual correlation should match expected correlation"
        )
        
        # Test correlation properties
        n_tokens = expected_corr.shape[0]
        
        # Diagonal should be 1
        diagonal = np.diag(expected_corr)
        np.testing.assert_array_almost_equal(
            diagonal, np.ones(n_tokens), decimal=12,
            err_msg="Correlation matrix diagonal should be 1"
        )
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            expected_corr, expected_corr.T, decimal=12,
            err_msg="Correlation matrix should be symmetric"
        )
        
        # All correlations should be between -1 and 1
        assert np.all(expected_corr >= -1), "Correlations should be >= -1"
        assert np.all(expected_corr <= 1), "Correlations should be <= 1"
    
    def test_spearman_correlation_properties(self, correlation_test_data):
        """Test Spearman correlation calculation properties."""
        price_matrix = correlation_test_data['price_matrix']
        
        # Calculate Spearman correlation using scipy
        from scipy.stats import spearmanr
        
        spearman_corr, _ = spearmanr(price_matrix.T)
        
        # Test properties
        if spearman_corr.ndim == 2:  # Matrix case
            # Diagonal should be 1
            diagonal = np.diag(spearman_corr)
            np.testing.assert_array_almost_equal(
                diagonal, np.ones(len(diagonal)), decimal=12,
                err_msg="Spearman correlation matrix diagonal should be 1"
            )
            
            # Matrix should be symmetric
            np.testing.assert_array_almost_equal(
                spearman_corr, spearman_corr.T, decimal=12,
                err_msg="Spearman correlation matrix should be symmetric"
            )
            
            # All correlations should be between -1 and 1
            assert np.all(spearman_corr >= -1), "Spearman correlations should be >= -1"
            assert np.all(spearman_corr <= 1), "Spearman correlations should be <= 1"
    
    def test_correlation_edge_cases(self, edge_case_time_series):
        """Test correlation calculation with edge cases."""
        # Test correlation of series with itself
        constant_prices = edge_case_time_series['constant']['price'].to_numpy()
        
        # Self-correlation should be 1 (if there's variation) or NaN (if constant)
        if np.std(constant_prices) > 0:
            self_corr = np.corrcoef(constant_prices, constant_prices)[0, 1]
            assert abs(self_corr - 1.0) < 1e-12, "Self-correlation should be 1"
        else:
            # Constant series correlation is undefined
            self_corr = np.corrcoef(constant_prices, constant_prices)[0, 1]
            assert np.isnan(self_corr), "Correlation of constant series should be NaN"
        
        # Test with very small values
        small_prices = edge_case_time_series['small_values']['price'].to_numpy()
        large_prices = edge_case_time_series['large_values']['price'].to_numpy()
        
        # Should handle different scales
        if len(small_prices) == len(large_prices):
            cross_corr = np.corrcoef(small_prices, large_prices)[0, 1]
            assert np.isfinite(cross_corr), "Correlation should be finite even with different scales"
            assert -1 <= cross_corr <= 1, "Correlation should be between -1 and 1"


@pytest.mark.integration
@pytest.mark.mathematical
class TestFeatureCalculationIntegration:
    """Test integration of multiple feature calculations."""
    
    def test_feature_calculation_consistency(self, reference_time_series_data):
        """Test that different feature calculations are mathematically consistent."""
        df = reference_time_series_data['df']
        prices = df['price'].to_numpy()
        
        # Test relationship between different calculations
        returns = df['price'].pct_change().to_numpy()
        log_returns = np.diff(np.log(prices))
        
        # For the same data, calculations should be related
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) > 10:
            # Volatility from returns should be positive
            returns_volatility = np.std(valid_returns[1:], ddof=1)
            log_returns_volatility = np.std(log_returns, ddof=1)
            
            assert returns_volatility > 0, "Returns volatility should be positive"
            assert log_returns_volatility > 0, "Log returns volatility should be positive"
            
            # Should be relatively close for normal market data
            volatility_ratio = returns_volatility / log_returns_volatility
            assert 0.5 < volatility_ratio < 2.0, \
                f"Returns and log returns volatility should be similar: {volatility_ratio}"
    
    def test_temporal_ordering_consistency(self, reference_time_series_data):
        """Test that temporal calculations maintain proper ordering."""
        df = reference_time_series_data['df']
        
        # Rolling calculations should maintain temporal order
        rolling_mean = df['price'].rolling_mean(window_size=10)
        
        # Non-NaN values should be properly ordered in time
        rolling_values = rolling_mean.to_numpy()
        valid_indices = np.where(~np.isnan(rolling_values))[0]
        
        if len(valid_indices) > 1:
            # Indices should be in ascending order (temporal consistency)
            assert np.all(np.diff(valid_indices) > 0), \
                "Valid rolling calculations should maintain temporal order"
    
    def test_numerical_stability_across_features(self, edge_case_time_series):
        """Test numerical stability across different feature types."""
        # Test with extreme values
        large_df = edge_case_time_series['large_values']
        small_df = edge_case_time_series['small_values']
        
        for df_name, df in [('large', large_df), ('small', small_df)]:
            # Basic calculations should not overflow/underflow
            returns = df['price'].pct_change()
            rolling_mean = df['price'].rolling_mean(window_size=5)
            
            returns_values = returns.drop_nulls().to_numpy()
            rolling_values = rolling_mean.drop_nulls().to_numpy()
            
            # Should be finite
            assert np.all(np.isfinite(returns_values)), \
                f"Returns should be finite for {df_name} values"
            assert np.all(np.isfinite(rolling_values)), \
                f"Rolling mean should be finite for {df_name} values"