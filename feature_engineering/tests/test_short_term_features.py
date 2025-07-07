"""
Short-term features validation tests for feature_engineering module
Tests mathematical correctness of short-term (15-60 minute) feature calculations
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta


@pytest.mark.unit
@pytest.mark.mathematical
class TestShortTermIndicators:
    """Test short-term technical indicators mathematical correctness."""
    
    def test_short_term_rsi_calculation(self, short_term_engineer, reference_time_series_data):
        """Test short-term RSI calculation accuracy."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        try:
            # Calculate short-term RSI (typically with smaller periods)
            short_rsi_df = short_term_engineer.calculate_short_term_rsi(df, periods=[5, 10, 14])
            
            # Test RSI calculation for each period
            for period in [5, 10, 14]:
                rsi_col = next((col for col in short_rsi_df.columns if f'rsi_{period}' in col), None)
                if rsi_col:
                    rsi_values = short_rsi_df[rsi_col].drop_nulls().to_numpy()
                    
                    if len(rsi_values) > 0:
                        # RSI should be between 0 and 100
                        assert np.all(rsi_values >= 0), f"RSI {period} values should be >= 0"
                        assert np.all(rsi_values <= 100), f"RSI {period} values should be <= 100"
                        
                        # Calculate manual RSI for verification
                        if len(prices) > period:
                            deltas = np.diff(prices)
                            gains = np.where(deltas > 0, deltas, 0)
                            losses = np.where(deltas < 0, -deltas, 0)
                            
                            # Simple moving average for gains/losses
                            if len(gains) >= period:
                                avg_gain = np.mean(gains[-period:])
                                avg_loss = np.mean(losses[-period:])
                                
                                if avg_loss > 0:
                                    rs = avg_gain / avg_loss
                                    expected_rsi = 100 - (100 / (1 + rs))
                                    
                                    # Last calculated RSI should be close to manual calculation
                                    if len(rsi_values) > 0:
                                        # Allow some tolerance for different implementations
                                        assert abs(rsi_values[-1] - expected_rsi) < 5, \
                                            f"RSI {period} should be close to manual calculation"
                                            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Short-term RSI test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_short_term_moving_averages(self, short_term_engineer, reference_time_series_data):
        """Test short-term moving averages calculation accuracy."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        try:
            # Calculate short-term moving averages
            ma_df = short_term_engineer.calculate_moving_averages(
                df, windows=[5, 10, 15, 20]
            )
            
            # Test each moving average
            for window in [5, 10, 15, 20]:
                ma_col = next((col for col in ma_df.columns if f'ma_{window}' in col or f'sma_{window}' in col), None)
                if ma_col:
                    ma_values = ma_df[ma_col].to_numpy()
                    
                    # Calculate manual moving average for verification
                    manual_ma = np.full(len(prices), np.nan)
                    for i in range(window - 1, len(prices)):
                        manual_ma[i] = np.mean(prices[i - window + 1:i + 1])
                    
                    # Compare non-NaN values
                    valid_indices = ~np.isnan(ma_values)
                    if np.any(valid_indices):
                        np.testing.assert_array_almost_equal(
                            ma_values[valid_indices],
                            manual_ma[valid_indices],
                            decimal=12,
                            err_msg=f"Moving average {window} should match manual calculation"
                        )
                        
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Short-term MA test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_short_term_volatility_measures(self, short_term_engineer, reference_time_series_data):
        """Test short-term volatility measures calculation."""
        df = reference_time_series_data['df']
        
        # Add returns column if not present
        df_with_returns = df.with_columns([
            pl.col('price').pct_change().alias('returns')
        ])
        
        try:
            # Calculate short-term volatility
            vol_df = short_term_engineer.calculate_volatility_measures(
                df_with_returns, windows=[5, 10, 15]
            )
            
            # Test volatility calculation
            for window in [5, 10, 15]:
                vol_col = next((col for col in vol_df.columns if f'vol_{window}' in col or f'volatility_{window}' in col), None)
                if vol_col:
                    vol_values = vol_df[vol_col].drop_nulls().to_numpy()
                    
                    if len(vol_values) > 0:
                        # Volatility should be non-negative
                        assert np.all(vol_values >= 0), f"Volatility {window} should be non-negative"
                        
                        # Should be finite
                        assert np.all(np.isfinite(vol_values)), f"Volatility {window} should be finite"
                        
                        # Test against manual calculation
                        returns = df_with_returns['returns'].to_numpy()
                        manual_vol = np.full(len(returns), np.nan)
                        
                        for i in range(window, len(returns)):
                            window_returns = returns[i - window:i]
                            valid_returns = window_returns[~np.isnan(window_returns)]
                            if len(valid_returns) > 1:
                                manual_vol[i] = np.std(valid_returns, ddof=1)
                        
                        # Compare last few values
                        manual_valid = manual_vol[~np.isnan(manual_vol)]
                        if len(manual_valid) > 0 and len(vol_values) > 0:
                            min_compare = min(5, len(manual_valid), len(vol_values))
                            if min_compare > 0:
                                # Should be highly correlated (allowing for implementation differences)
                                correlation = np.corrcoef(
                                    manual_valid[-min_compare:], 
                                    vol_values[-min_compare:]
                                )[0, 1]
                                if not np.isnan(correlation):
                                    assert correlation > 0.8, \
                                        f"Volatility {window} should correlate with manual calculation"
                                        
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Short-term volatility test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.unit
@pytest.mark.mathematical
class TestShortTermMomentum:
    """Test short-term momentum indicators mathematical correctness."""
    
    def test_short_term_price_momentum(self, short_term_engineer, reference_time_series_data):
        """Test short-term price momentum calculation."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        try:
            # Calculate momentum features
            momentum_df = short_term_engineer.calculate_momentum_features(
                df, periods=[5, 10, 15, 30]
            )
            
            # Test each momentum period
            for period in [5, 10, 15, 30]:
                momentum_col = next((col for col in momentum_df.columns if f'momentum_{period}' in col), None)
                if momentum_col:
                    momentum_values = momentum_df[momentum_col].to_numpy()
                    
                    # Calculate manual momentum
                    manual_momentum = np.full(len(prices), np.nan)
                    for i in range(period, len(prices)):
                        manual_momentum[i] = (prices[i] - prices[i - period]) / prices[i - period]
                    
                    # Compare calculations
                    valid_indices = ~np.isnan(momentum_values)
                    if np.any(valid_indices):
                        np.testing.assert_array_almost_equal(
                            momentum_values[valid_indices],
                            manual_momentum[valid_indices],
                            decimal=12,
                            err_msg=f"Momentum {period} should match manual calculation"
                        )
                        
                        # Momentum should be greater than -1 (can't lose more than 100%)
                        finite_momentum = momentum_values[valid_indices]
                        assert np.all(finite_momentum > -1), \
                            f"Momentum {period} should be > -1"
                            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Short-term momentum test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_rate_of_change_calculation(self, short_term_engineer, reference_time_series_data):
        """Test rate of change calculation for short-term analysis."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        try:
            # Calculate rate of change
            roc_df = short_term_engineer.calculate_rate_of_change(df, periods=[5, 10, 20])
            
            for period in [5, 10, 20]:
                roc_col = next((col for col in roc_df.columns if f'roc_{period}' in col), None)
                if roc_col:
                    roc_values = roc_df[roc_col].to_numpy()
                    
                    # Calculate manual ROC
                    manual_roc = np.full(len(prices), np.nan)
                    for i in range(period, len(prices)):
                        manual_roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
                    
                    # Compare calculations
                    valid_indices = ~np.isnan(roc_values)
                    if np.any(valid_indices):
                        np.testing.assert_array_almost_equal(
                            roc_values[valid_indices],
                            manual_roc[valid_indices],
                            decimal=10,
                            err_msg=f"ROC {period} should match manual calculation"
                        )
                        
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Rate of change test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.unit
@pytest.mark.mathematical
class TestShortTermSupportResistance:
    """Test short-term support/resistance level identification."""
    
    def test_local_extrema_detection(self, short_term_engineer, reference_time_series_data):
        """Test local extrema (peaks/troughs) detection."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        try:
            # Calculate local extrema
            extrema_df = short_term_engineer.identify_local_extrema(df, window=5)
            
            # Check for peak/trough columns
            peak_col = next((col for col in extrema_df.columns if 'peak' in col), None)
            trough_col = next((col for col in extrema_df.columns if 'trough' in col), None)
            
            if peak_col:
                peaks = extrema_df[peak_col].to_numpy()
                peak_indices = np.where(peaks == 1)[0]
                
                # Verify peaks are actually local maxima
                for peak_idx in peak_indices[:10]:  # Test first 10 peaks
                    window_start = max(0, peak_idx - 2)
                    window_end = min(len(prices), peak_idx + 3)
                    window_prices = prices[window_start:window_end]
                    local_peak_idx = peak_idx - window_start
                    
                    if 0 <= local_peak_idx < len(window_prices):
                        peak_price = window_prices[local_peak_idx]
                        # Peak should be highest in its local window
                        assert peak_price >= np.max(window_prices), \
                            f"Peak at index {peak_idx} should be local maximum"
            
            if trough_col:
                troughs = extrema_df[trough_col].to_numpy()
                trough_indices = np.where(troughs == 1)[0]
                
                # Verify troughs are actually local minima
                for trough_idx in trough_indices[:10]:  # Test first 10 troughs
                    window_start = max(0, trough_idx - 2)
                    window_end = min(len(prices), trough_idx + 3)
                    window_prices = prices[window_start:window_end]
                    local_trough_idx = trough_idx - window_start
                    
                    if 0 <= local_trough_idx < len(window_prices):
                        trough_price = window_prices[local_trough_idx]
                        # Trough should be lowest in its local window
                        assert trough_price <= np.min(window_prices), \
                            f"Trough at index {trough_idx} should be local minimum"
                            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Local extrema test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_support_resistance_levels(self, short_term_engineer, reference_time_series_data):
        """Test support and resistance level calculation."""
        df = reference_time_series_data['df']
        prices = reference_time_series_data['prices']
        
        try:
            # Calculate support/resistance levels
            sr_df = short_term_engineer.calculate_support_resistance(df, window=20, num_levels=3)
            
            # Check for support/resistance columns
            support_cols = [col for col in sr_df.columns if 'support' in col]
            resistance_cols = [col for col in sr_df.columns if 'resistance' in col]
            
            if len(support_cols) > 0:
                for support_col in support_cols:
                    support_values = sr_df[support_col].drop_nulls().to_numpy()
                    if len(support_values) > 0:
                        # Support levels should be reasonable relative to prices
                        price_range = np.ptp(prices)  # Peak-to-peak range
                        price_min = np.min(prices)
                        price_max = np.max(prices)
                        
                        # Support levels should be within reasonable range
                        assert np.all(support_values >= price_min - 0.1 * price_range), \
                            f"Support levels should be reasonable relative to price range"
                        assert np.all(support_values <= price_max + 0.1 * price_range), \
                            f"Support levels should be reasonable relative to price range"
            
            if len(resistance_cols) > 0:
                for resistance_col in resistance_cols:
                    resistance_values = sr_df[resistance_col].drop_nulls().to_numpy()
                    if len(resistance_values) > 0:
                        # Resistance levels should be reasonable relative to prices
                        price_range = np.ptp(prices)
                        price_min = np.min(prices)
                        price_max = np.max(prices)
                        
                        # Resistance levels should be within reasonable range
                        assert np.all(resistance_values >= price_min - 0.1 * price_range), \
                            f"Resistance levels should be reasonable relative to price range"
                        assert np.all(resistance_values <= price_max + 0.1 * price_range), \
                            f"Resistance levels should be reasonable relative to price range"
                            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Support/resistance test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.unit
@pytest.mark.mathematical
class TestShortTermPatternRecognition:
    """Test short-term pattern recognition features."""
    
    def test_price_pattern_detection(self, short_term_engineer, reference_time_series_data):
        """Test price pattern detection (reversals, continuations)."""
        df = reference_time_series_data['df']
        
        try:
            # Calculate pattern features
            pattern_df = short_term_engineer.detect_price_patterns(df, window=10)
            
            # Check for pattern columns
            pattern_cols = [col for col in pattern_df.columns if 'pattern' in col or 'reversal' in col]
            
            for pattern_col in pattern_cols:
                pattern_values = pattern_df[pattern_col].to_numpy()
                
                # Pattern indicators should be binary (0/1) or categorical
                unique_values = np.unique(pattern_values[~np.isnan(pattern_values)])
                
                # Should have reasonable number of unique values
                assert len(unique_values) <= 10, \
                    f"Pattern column {pattern_col} should have reasonable number of categories"
                
                # If binary, should be 0/1
                if len(unique_values) == 2:
                    assert set(unique_values) <= {0, 1}, \
                        f"Binary pattern column {pattern_col} should contain only 0/1"
                        
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Pattern detection test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_trend_strength_calculation(self, short_term_engineer, reference_time_series_data):
        """Test trend strength calculation for short-term analysis."""
        df = reference_time_series_data['df']
        
        try:
            # Calculate trend strength
            trend_df = short_term_engineer.calculate_trend_strength(df, windows=[5, 10, 20])
            
            for window in [5, 10, 20]:
                trend_col = next((col for col in trend_df.columns if f'trend_{window}' in col), None)
                if trend_col:
                    trend_values = trend_df[trend_col].drop_nulls().to_numpy()
                    
                    if len(trend_values) > 0:
                        # Trend strength should be between 0 and 1 (or -1 and 1 for directional)
                        assert np.all(trend_values >= -1.1), \
                            f"Trend strength {window} should be >= -1"
                        assert np.all(trend_values <= 1.1), \
                            f"Trend strength {window} should be <= 1"
                        
                        # Should be finite
                        assert np.all(np.isfinite(trend_values)), \
                            f"Trend strength {window} should be finite"
                            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Trend strength test skipped due to missing dependencies: {e}")
            else:
                raise e


@pytest.mark.integration
@pytest.mark.mathematical
class TestShortTermFeaturesPipeline:
    """Test complete short-term features pipeline."""
    
    def test_complete_short_term_pipeline(self, short_term_engineer, reference_time_series_data):
        """Test complete short-term features pipeline execution."""
        df = reference_time_series_data['df']
        
        try:
            # Run complete short-term analysis
            features_df = short_term_engineer.engineer_short_term_features(df)
            
            # Should return a DataFrame
            assert isinstance(features_df, pl.DataFrame), "Should return Polars DataFrame"
            
            # Should have same number of rows as input
            assert len(features_df) == len(df), "Should preserve input length"
            
            # Should have datetime column
            assert 'datetime' in features_df.columns, "Should contain datetime column"
            
            # Should have created short-term features
            feature_cols = [col for col in features_df.columns if col not in ['datetime', 'price']]
            assert len(feature_cols) > 0, "Should create short-term feature columns"
            
            # Check that features have reasonable value ranges
            for col in feature_cols:
                col_values = features_df[col].drop_nulls().to_numpy()
                if len(col_values) > 0:
                    # Should not have extreme outliers (beyond 10 standard deviations)
                    if np.std(col_values) > 0:
                        z_scores = np.abs((col_values - np.mean(col_values)) / np.std(col_values))
                        extreme_outliers = np.sum(z_scores > 10)
                        outlier_ratio = extreme_outliers / len(col_values)
                        assert outlier_ratio < 0.05, \
                            f"Feature {col} should not have excessive extreme outliers"
                            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Complete pipeline test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_short_term_features_temporal_safety(self, short_term_engineer, reference_time_series_data):
        """Test temporal safety of short-term features."""
        df = reference_time_series_data['df']
        
        try:
            # Calculate features
            features_df = short_term_engineer.engineer_short_term_features(df)
            
            # Check temporal alignment
            datetime_col = features_df['datetime']
            original_datetimes = df['datetime'].to_list()
            feature_datetimes = datetime_col.to_list()
            
            assert feature_datetimes == original_datetimes, \
                "Should preserve datetime ordering and values"
            
            # Check that NaN patterns make temporal sense
            for col in features_df.columns:
                if col not in ['datetime', 'price']:
                    col_values = features_df[col].to_numpy()
                    nan_indices = np.isnan(col_values)
                    
                    if np.any(nan_indices) and not np.all(nan_indices):
                        # Most NaN values should be at the beginning (for lag/rolling features)
                        first_valid = np.where(~nan_indices)[0]
                        if len(first_valid) > 0:
                            first_valid_idx = first_valid[0]
                            
                            # Initial NaN section should be reasonable
                            assert first_valid_idx < len(col_values) * 0.5, \
                                f"Feature {col} should not have excessive initial NaN values"
                                
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Temporal safety test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_short_term_features_with_gaps(self, short_term_engineer, edge_case_time_series):
        """Test short-term features with data gaps."""
        df_with_gaps = edge_case_time_series['with_gaps']
        
        try:
            # Should handle gaps gracefully
            features_df = short_term_engineer.engineer_short_term_features(df_with_gaps)
            
            # Should preserve input length
            assert len(features_df) == len(df_with_gaps), "Should preserve length with gaps"
            
            # Should not crash with gaps
            assert isinstance(features_df, pl.DataFrame), "Should return DataFrame with gaps"
            
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Gaps test skipped due to missing dependencies: {e}")
            else:
                raise e
    
    def test_short_term_features_consistency(self, short_term_engineer, reference_time_series_data):
        """Test consistency of short-term features across runs."""
        df = reference_time_series_data['df']
        
        try:
            # Run twice
            features_df_1 = short_term_engineer.engineer_short_term_features(df)
            features_df_2 = short_term_engineer.engineer_short_term_features(df)
            
            # Should be identical
            assert features_df_1.shape == features_df_2.shape, "Should produce consistent shapes"
            
            # Compare feature values
            for col in features_df_1.columns:
                if col in features_df_2.columns:
                    values_1 = features_df_1[col].to_numpy()
                    values_2 = features_df_2[col].to_numpy()
                    
                    # Should be exactly equal
                    np.testing.assert_array_equal(
                        values_1, values_2,
                        err_msg=f"Should produce consistent results for column {col}"
                    )
                    
        except Exception as e:
            if "No module named" in str(e) or "cannot import name" in str(e):
                pytest.skip(f"Consistency test skipped due to missing dependencies: {e}")
            else:
                raise e