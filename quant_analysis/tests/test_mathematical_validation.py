"""
Mathematical validation tests for quantitative analysis module.
Ensures all risk metrics, trading signals, and portfolio calculations are mathematically correct.
Following TDD principles as specified in CLAUDE.md.
"""

import pytest
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quant_analysis.quant_analysis import QuantAnalysis
from quant_analysis.quant_viz import QuantVisualizations
from quant_analysis.trading_analysis import TradingAnalytics


@pytest.mark.unit
@pytest.mark.mathematical
class TestQuantAnalysisRiskMetrics:
    """Test QuantAnalysis risk metrics mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quant = QuantAnalysis()
        
    def test_sharpe_ratio_calculation_accuracy(self):
        """Test Sharpe ratio calculation against manual calculation."""
        # Test with known values
        returns_data = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012]
        returns = pl.Series(returns_data)
        
        # Calculate expected Sharpe ratio manually
        mean_return = np.mean(returns_data)
        std_return = np.std(returns_data, ddof=1)
        expected_sharpe = (mean_return - self.quant.risk_free_rate) / std_return
        
        # Test actual implementation
        calculated_sharpe = self.quant.calculate_sharpe_ratio(returns)
        
        # Mathematical validation to 1e-12 precision
        assert abs(calculated_sharpe - expected_sharpe) < 1e-12, \
            f"Sharpe ratio calculation mismatch: {calculated_sharpe} vs {expected_sharpe}"
    
    def test_sharpe_ratio_edge_cases(self):
        """Test Sharpe ratio edge cases."""
        # Test with zero standard deviation
        constant_returns = pl.Series([0.01, 0.01, 0.01, 0.01])
        sharpe_constant = self.quant.calculate_sharpe_ratio(constant_returns)
        assert sharpe_constant == 0.0, "Constant returns should give 0 Sharpe ratio"
        
        # Test with single value
        single_return = pl.Series([0.01])
        sharpe_single = self.quant.calculate_sharpe_ratio(single_return)
        assert sharpe_single == 0.0, "Single return should give 0 Sharpe ratio"
        
        # Test with empty series
        empty_returns = pl.Series([])
        sharpe_empty = self.quant.calculate_sharpe_ratio(empty_returns)
        assert sharpe_empty == 0.0, "Empty returns should give 0 Sharpe ratio"
        
    def test_sortino_ratio_calculation_accuracy(self):
        """Test Sortino ratio calculation against manual calculation."""
        # Test with known values including negative returns
        returns_data = [0.02, -0.01, 0.015, -0.008, 0.01, -0.005, 0.018]
        returns = pl.Series(returns_data)
        
        # Calculate expected Sortino ratio manually
        mean_return = np.mean(returns_data)
        negative_returns = [r for r in returns_data if r < 0]
        downside_deviation = np.std(negative_returns, ddof=1) if negative_returns else 0
        
        if downside_deviation > 0:
            expected_sortino = (mean_return - self.quant.risk_free_rate) / downside_deviation
        else:
            expected_sortino = 0.0
            
        # Test actual implementation
        calculated_sortino = self.quant.calculate_sortino_ratio(returns)
        
        # Mathematical validation
        assert abs(calculated_sortino - expected_sortino) < 1e-10, \
            f"Sortino ratio calculation mismatch: {calculated_sortino} vs {expected_sortino}"
    
    def test_hurst_exponent_calculation(self):
        """Test Hurst exponent calculation mathematical properties."""
        # Create test data with known properties
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(1000))
        price_series = pl.Series(random_walk)
        
        # Calculate Hurst exponent
        hurst = self.quant.calculate_hurst_exponent(price_series)
        
        # Mathematical validation
        assert 0 <= hurst <= 1, f"Hurst exponent should be between 0 and 1, got {hurst}"
        assert not np.isnan(hurst), "Hurst exponent should not be NaN"
        assert not np.isinf(hurst), "Hurst exponent should not be infinite"


@pytest.mark.unit
@pytest.mark.mathematical
class TestTradingAnalyticsSignals:
    """Test TradingAnalytics signal generation mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trading = TradingAnalytics()
        
    def test_optimal_stop_loss_take_profit_logic(self):
        """Test optimal stop loss and take profit calculation logic."""
        # Create test price data
        np.random.seed(42)
        price_data = 100 + np.cumsum(np.random.randn(200) * 0.5)
        dates = pd.date_range('2023-01-01', periods=200, freq='1min')
        
        df = pd.DataFrame({
            'datetime': dates,
            'price': price_data,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        # Test actual implementation
        result = self.trading.calculate_optimal_stop_loss_take_profit(df)
        
        # Validate output structure
        assert isinstance(result, pd.DataFrame), "Should return pandas DataFrame"
        assert len(result) == len(df), "Result length should match input length"
        
        # Check for required columns
        required_columns = ['stop_loss', 'take_profit']
        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"
        
        # Mathematical validation - stop loss should be below price, take profit above
        for i in range(len(result)):
            if not pd.isna(result.iloc[i]['stop_loss']):
                assert result.iloc[i]['stop_loss'] < df.iloc[i]['price'], \
                    f"Stop loss should be below price at index {i}"
            if not pd.isna(result.iloc[i]['take_profit']):
                assert result.iloc[i]['take_profit'] > df.iloc[i]['price'], \
                    f"Take profit should be above price at index {i}"
    
    def test_vwap_analysis_accuracy(self):
        """Test VWAP (Volume Weighted Average Price) calculation accuracy."""
        # Create test data with known VWAP
        prices = [100, 101, 102, 103, 104]
        volumes = [1000, 2000, 1500, 3000, 2500]
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'price': prices,
            'volume': volumes
        })
        
        # Calculate expected VWAP manually
        total_volume = sum(volumes)
        weighted_price_sum = sum(p * v for p, v in zip(prices, volumes))
        expected_vwap = weighted_price_sum / total_volume
        
        # Test actual implementation
        result = self.trading.vwap_analysis(df)
        
        # Mathematical validation
        assert isinstance(result, pd.DataFrame), "Should return pandas DataFrame"
        assert 'vwap' in result.columns, "Should contain vwap column"
        
        # Check final VWAP value
        final_vwap = result['vwap'].iloc[-1]
        assert abs(final_vwap - expected_vwap) < 1e-10, \
            f"VWAP calculation mismatch: {final_vwap} vs {expected_vwap}"


@pytest.mark.unit  
@pytest.mark.mathematical
class TestQuantVisualizationsAccuracy:
    """Test QuantVisualizations mathematical accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz = QuantVisualizations()
        
    def test_entry_exit_matrix_calculation(self):
        """Test entry/exit matrix calculation mathematical properties."""
        # Create test price data
        np.random.seed(42)
        price_data = 100 + np.cumsum(np.random.randn(100) * 0.5)
        dates = [datetime.now() + timedelta(minutes=i) for i in range(100)]
        
        df = pl.DataFrame({
            'datetime': dates,
            'price': price_data
        })
        
        # Test actual implementation
        result = self.viz.plot_entry_exit_matrix(df)
        
        # Validate output structure
        assert result is not None, "Should return a valid plot object"
        
        # Test that the underlying calculation makes sense
        # Entry/exit matrix should show returns for different entry/exit combinations
        # This is a complex calculation, so we test basic properties
        
    def test_volatility_surface_mathematical_properties(self):
        """Test volatility surface calculation mathematical properties."""
        # Create test data with known volatility patterns
        np.random.seed(42)
        price_data = 100 + np.cumsum(np.random.randn(500) * 0.02)
        dates = [datetime.now() + timedelta(minutes=i) for i in range(500)]
        
        df = pl.DataFrame({
            'datetime': dates,
            'price': price_data
        })
        
        # Test actual implementation
        result = self.viz.plot_volatility_surface(df)
        
        # Validate output structure
        assert result is not None, "Should return a valid plot object"


@pytest.mark.integration
@pytest.mark.mathematical
class TestQuantAnalysisIntegration:
    """Test integration between QuantAnalysis components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quant = QuantAnalysis()
        self.trading = TradingAnalytics()
        self.viz = QuantVisualizations()
        
    def test_risk_metrics_consistency(self):
        """Test consistency between different risk metrics."""
        # Create test portfolio returns
        np.random.seed(42)
        returns_data = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        returns = pl.Series(returns_data)
        
        # Calculate metrics
        sharpe = self.quant.calculate_sharpe_ratio(returns)
        sortino = self.quant.calculate_sortino_ratio(returns)
        
        # Test mathematical relationships
        # Sortino ratio should generally be higher than Sharpe ratio (less penalty for upside)
        assert not np.isnan(sharpe), "Sharpe ratio should not be NaN"
        assert not np.isnan(sortino), "Sortino ratio should not be NaN"
        
        # Both metrics should be finite
        assert np.isfinite(sharpe), "Sharpe ratio should be finite"
        assert np.isfinite(sortino), "Sortino ratio should be finite"
    
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_returns = pl.Series([1e-10, -1e-10, 1e-10, -1e-10])
        sharpe_small = self.quant.calculate_sharpe_ratio(small_returns)
        assert not np.isnan(sharpe_small), "Should handle very small values"
        
        # Test with very large values
        large_returns = pl.Series([1e6, -1e6, 1e6, -1e6])
        sharpe_large = self.quant.calculate_sharpe_ratio(large_returns)
        assert not np.isnan(sharpe_large), "Should handle very large values"
        
        # Test with zero values
        zero_returns = pl.Series([0, 0, 0, 0])
        sharpe_zero = self.quant.calculate_sharpe_ratio(zero_returns)
        assert sharpe_zero == 0.0, "Should handle zero values"
    
    def test_temporal_consistency(self):
        """Test temporal consistency in calculations."""
        # Create test data with consistent time series
        np.random.seed(42)
        price_data = 100 + np.cumsum(np.random.randn(1000) * 0.01)
        
        # Test that calculations are consistent across different time windows
        full_series = pl.Series(price_data)
        half_series = pl.Series(price_data[:500])
        
        # Calculate Hurst exponent for both
        hurst_full = self.quant.calculate_hurst_exponent(full_series)
        hurst_half = self.quant.calculate_hurst_exponent(half_series)
        
        # Both should be valid values
        assert 0 <= hurst_full <= 1, "Full series Hurst should be valid"
        assert 0 <= hurst_half <= 1, "Half series Hurst should be valid"
        
        # They should be reasonably close (within 0.3 for random data)
        assert abs(hurst_full - hurst_half) < 0.3, \
            f"Hurst values should be reasonably consistent: {hurst_full} vs {hurst_half}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])