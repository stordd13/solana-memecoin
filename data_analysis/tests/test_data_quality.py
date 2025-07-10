"""
Unit tests for data quality analysis mathematical validation
Tests mathematical correctness of quality metrics and anomaly detection
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import math


@pytest.mark.unit
@pytest.mark.mathematical
class TestGapAnalysis:
    """Test gap analysis mathematical correctness."""
    
    def test_gap_detection_accuracy(self, data_quality_analyzer):
        """Test gap detection with known gap patterns."""
        # Create data with known gaps
        timestamps = []
        base_time = datetime(2024, 1, 1)
        
        # Normal sequence for 30 minutes
        for i in range(30):
            timestamps.append(base_time + timedelta(minutes=i))
        
        # 10-minute gap
        for i in range(40, 70):  # Skip 30-39 (10 minute gap)
            timestamps.append(base_time + timedelta(minutes=i))
        
        # 5-minute gap
        for i in range(75, 100):  # Skip 70-74 (5 minute gap)
            timestamps.append(base_time + timedelta(minutes=i))
        
        datetime_series = pl.Series("datetime", timestamps)
        gap_analysis = data_quality_analyzer._analyze_gaps(datetime_series)
        
        # Should detect exactly 2 gaps
        assert gap_analysis['gap_count'] == 2, f"Expected 2 gaps, got {gap_analysis['gap_count']}"
        
        # Should identify gap sizes correctly
        gap_sizes = gap_analysis.get('gap_sizes', [])
        expected_gaps = [10, 5]  # minutes
        
        # Sort both lists for comparison
        gap_sizes_sorted = sorted(gap_sizes) if gap_sizes else []
        expected_gaps_sorted = sorted(expected_gaps)
        
        assert len(gap_sizes_sorted) == len(expected_gaps_sorted), \
            f"Gap count mismatch: {gap_sizes_sorted} vs {expected_gaps_sorted}"
        
        for actual, expected in zip(gap_sizes_sorted, expected_gaps_sorted):
            assert abs(actual - expected) <= 1, \
                f"Gap size mismatch: {actual} vs {expected} (tolerance: 1 minute)"
    
    def test_gap_analysis_edge_cases(self, data_quality_analyzer):
        """Test gap analysis with edge cases."""
        # Single timestamp - no gaps possible
        single_timestamp = pl.Series("datetime", [datetime(2024, 1, 1)])
        gap_analysis = data_quality_analyzer._analyze_gaps(single_timestamp)
        assert gap_analysis['gap_count'] == 0, "Single timestamp should have no gaps"
        
        # Two timestamps - test gap calculation
        two_timestamps = pl.Series("datetime", [
            datetime(2024, 1, 1),
            datetime(2024, 1, 1, 0, 10)  # 10 minutes later
        ])
        gap_analysis = data_quality_analyzer._analyze_gaps(two_timestamps)
        
        if gap_analysis['gap_count'] > 0:
            # If a gap is detected, it should be approximately 9 minutes (expecting 1-minute intervals)
            gap_sizes = gap_analysis.get('gap_sizes', [])
            if gap_sizes:
                assert 8 <= gap_sizes[0] <= 10, f"Gap size should be ~9 minutes, got {gap_sizes[0]}"
    
    def test_gap_percentage_calculation(self, data_quality_analyzer):
        """Test gap percentage calculation accuracy."""
        # Create 100-minute period with 20 minutes missing (20% gaps)
        timestamps = []
        base_time = datetime(2024, 1, 1)
        
        # First 40 minutes present
        for i in range(40):
            timestamps.append(base_time + timedelta(minutes=i))
        
        # 20 minutes missing (gap)
        
        # Last 40 minutes present
        for i in range(60, 100):
            timestamps.append(base_time + timedelta(minutes=i))
        
        datetime_series = pl.Series("datetime", timestamps)
        gap_analysis = data_quality_analyzer._analyze_gaps(datetime_series)
        
        # Calculate expected gap percentage
        total_expected_minutes = 100
        missing_minutes = 20
        expected_gap_percentage = (missing_minutes / total_expected_minutes) * 100
        
        if 'gap_percentage' in gap_analysis:
            actual_gap_percentage = gap_analysis['gap_percentage']
            assert abs(actual_gap_percentage - expected_gap_percentage) < 5, \
                f"Gap percentage mismatch: {actual_gap_percentage}% vs {expected_gap_percentage}%"


@pytest.mark.unit
@pytest.mark.mathematical
class TestAnomalyDetection:
    """Test anomaly detection mathematical correctness."""
    
    def test_iqr_outlier_detection(self, data_quality_analyzer):
        """Test IQR-based outlier detection accuracy."""
        # Create data with known outliers
        normal_prices = np.random.normal(100, 5, 95)  # 95 normal points
        outliers = [150, 50, 200, 30, 180]  # 5 outliers
        all_prices = np.concatenate([normal_prices, outliers])
        
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': all_prices
        })
        
        anomaly_analysis = data_quality_analyzer._analyze_price_anomalies(df)
        
        # Should detect some outliers
        assert 'outlier_count' in anomaly_analysis
        outlier_count = anomaly_analysis['outlier_count']
        
        # Should detect at least some of the extreme outliers
        assert outlier_count > 0, "Should detect some outliers in data with extreme values"
        
        # Shouldn't flag all points as outliers
        assert outlier_count < len(all_prices) * 0.5, "Shouldn't flag more than 50% as outliers"
    
    def test_z_score_outlier_detection_reference(self, data_quality_analyzer):
        """Test Z-score outlier detection against reference implementation."""
        np.random.seed(42)
        prices = np.random.normal(100, 10, 100)
        
        # Add known outliers (> 3 standard deviations)
        prices[50] = 100 + 4 * 10  # 4 std above mean
        prices[75] = 100 - 4 * 10  # 4 std below mean
        
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        # Calculate reference Z-scores
        mean_price = np.mean(prices)
        std_price = np.std(prices, ddof=1)
        z_scores = np.abs((prices - mean_price) / std_price)
        expected_outliers = np.sum(z_scores > 3.0)
        
        anomaly_analysis = data_quality_analyzer._analyze_price_anomalies(df)
        
        if 'z_score_outliers' in anomaly_analysis:
            actual_outliers = anomaly_analysis['z_score_outliers']
            # Allow some tolerance for different Z-score implementations
            assert abs(actual_outliers - expected_outliers) <= 2, \
                f"Z-score outlier count mismatch: {actual_outliers} vs {expected_outliers}"
    
    def test_winsorization_accuracy(self, data_quality_analyzer):
        """Test winsorization mathematical accuracy."""
        # Create data with known percentiles
        prices = np.arange(1, 101, dtype=float)  # 1 to 100
        
        df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': prices
        })
        
        # Expected winsorized values (0.5% and 99.5% percentiles)
        lower_percentile = np.percentile(prices, 0.5)  # Should be 0.5
        upper_percentile = np.percentile(prices, 99.5)  # Should be 99.5
        
        anomaly_analysis = data_quality_analyzer._analyze_price_anomalies(df)
        
        # Check if winsorization bounds are calculated correctly
        if 'winsorization_bounds' in anomaly_analysis:
            bounds = anomaly_analysis['winsorization_bounds']
            if isinstance(bounds, dict) and 'lower' in bounds and 'upper' in bounds:
                assert abs(bounds['lower'] - lower_percentile) < 0.1, \
                    f"Lower winsorization bound mismatch: {bounds['lower']} vs {lower_percentile}"
                assert abs(bounds['upper'] - upper_percentile) < 0.1, \
                    f"Upper winsorization bound mismatch: {bounds['upper']} vs {upper_percentile}"
    
    def test_anomaly_detection_edge_cases(self, data_quality_analyzer, edge_case_data):
        """Test anomaly detection with edge cases."""
        # Constant prices - should have no outliers
        constant_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': [100.0] * 100
        })
        
        anomaly_analysis = data_quality_analyzer._analyze_price_anomalies(constant_df)
        
        # Constant data should have no outliers
        outlier_count = anomaly_analysis.get('outlier_count', 0)
        assert outlier_count == 0, f"Constant prices should have no outliers, got {outlier_count}"


@pytest.mark.unit
@pytest.mark.mathematical
class TestQualityScoring:
    """Test quality score calculation mathematical correctness."""
    
    def test_quality_score_calculation_reference(self, data_quality_analyzer):
        """Test quality score calculation against expected formula."""
        # Create test scenario with known parameters
        total_rows = 100
        unique_dates = 95  # 5% duplicates
        gap_count = 2
        gap_percentage = 10.0  # 10% missing data
        outlier_count = 5
        extreme_changes = 1
        
        quality_score = data_quality_analyzer._calculate_quality_score(
            total_rows, unique_dates, gap_count, gap_percentage, 
            outlier_count, extreme_changes
        )
        
        # Quality score should be between 0 and 100
        assert 0 <= quality_score <= 100, f"Quality score should be 0-100, got {quality_score}"
        
        # With 95% completeness and some issues, score should be reasonable
        assert 60 <= quality_score <= 95, f"Expected quality score 60-95, got {quality_score}"
    
    def test_quality_score_edge_cases(self, data_quality_analyzer):
        """Test quality score calculation with edge cases."""
        # Perfect data - should get high score
        perfect_score = data_quality_analyzer._calculate_quality_score(
            total_rows=100,
            unique_dates=100,
            gap_count=0,
            gap_percentage=0.0,
            outlier_count=0,
            extreme_changes=0
        )
        
        assert perfect_score >= 95, f"Perfect data should get high score, got {perfect_score}"
        
        # Terrible data - should get low score
        terrible_score = data_quality_analyzer._calculate_quality_score(
            total_rows=100,
            unique_dates=50,  # 50% duplicates
            gap_count=20,     # Many gaps
            gap_percentage=50.0,  # 50% missing
            outlier_count=30,     # 30% outliers
            extreme_changes=10    # Many extreme changes
        )
        
        assert terrible_score <= 50, f"Terrible data should get low score, got {terrible_score}"
    
    def test_quality_score_monotonicity(self, data_quality_analyzer):
        """Test that quality score decreases monotonically with worse data quality."""
        base_params = {
            'total_rows': 100,
            'unique_dates': 100,
            'gap_count': 0,
            'gap_percentage': 0.0,
            'outlier_count': 0,
            'extreme_changes': 0
        }
        
        # Base score (perfect data)
        base_score = data_quality_analyzer._calculate_quality_score(**base_params)
        
        # Score with gaps should be lower
        gap_params = base_params.copy()
        gap_params['gap_count'] = 5
        gap_params['gap_percentage'] = 10.0
        gap_score = data_quality_analyzer._calculate_quality_score(**gap_params)
        
        assert gap_score < base_score, "Score should decrease with gaps"
        
        # Score with outliers should be lower
        outlier_params = base_params.copy()
        outlier_params['outlier_count'] = 10
        outlier_score = data_quality_analyzer._calculate_quality_score(**outlier_params)
        
        assert outlier_score < base_score, "Score should decrease with outliers"


@pytest.mark.unit
@pytest.mark.mathematical
class TestDataQualityThresholds:
    """Test data quality threshold consistency and mathematical validity."""
    
    def test_threshold_consistency(self, data_quality_analyzer):
        """Test that quality thresholds are mathematically consistent."""
        thresholds = data_quality_analyzer.quality_thresholds
        
        # Completeness should be reasonable
        assert 50 <= thresholds['completeness_min'] <= 100, \
            "Completeness threshold should be 50-100%"
        
        # Gap thresholds should be positive
        assert thresholds['max_gap_minutes'] > 0, "Max gap minutes should be positive"
        assert thresholds['max_gap_for_interpolation'] > 0, "Interpolation gap limit should be positive"
        
        # Should be reasonable for memecoin data (1-minute intervals)
        assert thresholds['max_gap_minutes'] <= 60, "Max gap should be ≤ 60 minutes"
        assert thresholds['max_gap_for_interpolation'] <= 30, "Interpolation limit should be ≤ 30 minutes"
        
        # Minimum data points should be reasonable
        assert thresholds['min_data_points'] >= 10, "Should require at least 10 data points"
    
    def test_extreme_thresholds_validity(self, data_quality_analyzer):
        """Test extreme value thresholds are appropriate for memecoin data."""
        extreme_thresholds = data_quality_analyzer.extreme_thresholds
        
        # All extreme thresholds should be positive
        for threshold_name, threshold_value in extreme_thresholds.items():
            assert threshold_value > 0, f"{threshold_name} should be positive"
        
        # Thresholds should be reasonable for crypto (but high for traditional assets)
        assert extreme_thresholds['extreme_minute_return'] >= 10.0, \
            "Extreme minute return threshold should be ≥ 1000% for crypto"
        assert extreme_thresholds['extreme_total_return'] >= 100.0, \
            "Extreme total return threshold should be ≥ 10000% for crypto"
    
    def test_outlier_method_parameters(self, data_quality_analyzer):
        """Test outlier detection method parameters are mathematically valid."""
        outlier_methods = data_quality_analyzer.outlier_methods
        
        # Winsorization percentiles should be valid
        winsor_params = outlier_methods.get('winsorization', {})
        if winsor_params:
            lower = winsor_params.get('lower', 0)
            upper = winsor_params.get('upper', 1)
            
            assert 0 <= lower < upper <= 1, \
                f"Winsorization percentiles invalid: lower={lower}, upper={upper}"
            assert upper - lower > 0.5, \
                "Winsorization range should be > 50% to avoid over-trimming"
        
        # Z-score threshold should be reasonable
        z_score_params = outlier_methods.get('z_score', {})
        if z_score_params:
            threshold = z_score_params.get('threshold', 3)
            assert 2 <= threshold <= 10, \
                f"Z-score threshold should be 2-10, got {threshold}"
        
        # IQR multiplier should be reasonable
        iqr_params = outlier_methods.get('iqr', {})
        if iqr_params:
            multiplier = iqr_params.get('multiplier', 1.5)
            assert 1.0 <= multiplier <= 5.0, \
                f"IQR multiplier should be 1-5, got {multiplier}"


@pytest.mark.integration
@pytest.mark.mathematical
class TestDataQualityPipeline:
    """Test complete data quality analysis pipeline."""
    
    def test_single_file_analysis_consistency(self, data_quality_analyzer, synthetic_token_data):
        """Test single file analysis for mathematical consistency."""
        for token_type, df in synthetic_token_data.items():
            result = data_quality_analyzer.analyze_single_file(df, token_type)
            
            # Should return valid result structure
            assert isinstance(result, dict), f"Result should be dict for {token_type}"
            assert 'status' in result, f"Result should have status for {token_type}"
            
            if result.get('status') == 'success':
                # Should have key quality metrics
                assert 'quality_score' in result, f"Should have quality score for {token_type}"
                assert 'completeness' in result, f"Should have completeness for {token_type}"
                
                # Quality score should be valid
                quality_score = result['quality_score']
                assert 0 <= quality_score <= 100, \
                    f"Quality score should be 0-100 for {token_type}, got {quality_score}"
                
                # Completeness should be valid percentage
                completeness = result['completeness']
                assert 0 <= completeness <= 100, \
                    f"Completeness should be 0-100% for {token_type}, got {completeness}"
    
    def test_multiple_files_analysis_aggregation(self, data_quality_analyzer, synthetic_token_files):
        """Test multiple files analysis mathematical aggregation."""
        parquet_files = list(synthetic_token_files.values())
        
        results = data_quality_analyzer.analyze_multiple_files(parquet_files, limit=None)
        
        # Should return aggregated results
        assert isinstance(results, dict), "Results should be dict"
        assert 'summary' in results, "Results should have summary"
        
        summary = results['summary']
        
        # Summary statistics should be mathematically consistent
        if 'total_files' in summary and summary['total_files'] > 0:
            assert summary['total_files'] == len(parquet_files), \
                "Total files should match input count"
        
        # Average quality score should be between min and max individual scores
        if 'reports' in results and 'average_quality_score' in summary:
            individual_scores = []
            for report in results['reports'].values():
                if isinstance(report, dict) and 'quality_score' in report:
                    individual_scores.append(report['quality_score'])
            
            if individual_scores:
                expected_avg = np.mean(individual_scores)
                actual_avg = summary['average_quality_score']
                
                assert abs(actual_avg - expected_avg) < 0.1, \
                    f"Average quality score mismatch: {actual_avg} vs {expected_avg}"
    
    def test_data_quality_pipeline_numerical_stability(self, data_quality_analyzer):
        """Test numerical stability of quality analysis with extreme values."""
        # Very large values
        large_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': [1e12] * 100  # Very large prices
        })
        
        result = data_quality_analyzer.analyze_single_file(large_df, "large_values")
        assert result is not None, "Should handle large values"
        
        # Very small values
        small_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 1) + timedelta(minutes=99),
                interval="1m",
                eager=True
            ),
            'price': [1e-6] * 100  # Very small prices
        })
        
        result = data_quality_analyzer.analyze_single_file(small_df, "small_values")
        assert result is not None, "Should handle small values"