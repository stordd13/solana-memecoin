"""
Test configuration and fixtures for feature_engineering module
Provides comprehensive test infrastructure for mathematical validation of feature calculations
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from advanced_feature_engineering import AdvancedFeatureEngineer
from create_directional_targets import create_directional_targets
from short_term_features import ShortTermFeatureEngineer
from correlation_analysis import TokenCorrelationAnalyzer


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def feature_engineer():
    """Create AdvancedFeatureEngineer instance for testing."""
    return AdvancedFeatureEngineer()


@pytest.fixture
def short_term_engineer():
    """Create ShortTermFeatureEngineer instance for testing."""
    return ShortTermFeatureEngineer()


@pytest.fixture
def correlation_analyzer():
    """Create TokenCorrelationAnalyzer instance for testing."""
    return TokenCorrelationAnalyzer()


@pytest.fixture
def reference_time_series_data():
    """Generate time series with known mathematical properties for validation."""
    np.random.seed(42)  # Reproducible results
    
    # Create 1000-point time series (simulating ~16 hours of minute data)
    n_points = 1000
    
    # Base price series with controlled properties
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_points)  # 2% volatility
    log_returns = np.cumsum(returns)
    prices = base_price * np.exp(log_returns)
    
    # Create timestamps
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_points)]
    
    # Create polars DataFrame
    df = pl.DataFrame({
        'datetime': timestamps,
        'price': prices
    })
    
    # Calculate reference technical indicators for validation
    reference_indicators = _calculate_reference_indicators(prices, returns)
    
    return {
        'df': df,
        'prices': prices,
        'returns': returns,
        'log_returns': log_returns,
        'timestamps': timestamps,
        'n_points': n_points,
        'reference_indicators': reference_indicators
    }


@pytest.fixture
def ml_safe_feature_examples():
    """Generate examples of ML-safe and unsafe features for testing."""
    n_points = 200
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_points))
    
    df = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_points)],
        'price': prices
    })
    
    # ML-Safe features (should pass validation)
    safe_features = {
        'rolling_mean_10': df['price'].rolling_mean(window_size=10),
        'rolling_std_20': df['price'].rolling_std(window_size=20),
        'price_lag_1': df['price'].shift(1),
        'price_lag_5': df['price'].shift(5),
        'rsi_14': _calculate_simple_rsi(df['price'].to_numpy(), 14),
        'returns': df['price'].pct_change()
    }
    
    # ML-Unsafe features (should fail validation)
    unsafe_features = {
        'future_return_1': df['price'].shift(-1) / df['price'] - 1,  # Uses future data
        'total_return': df['price'] / df['price'][0] - 1,  # Uses first value (global stat)
        'max_price': [df['price'].max()] * len(df),  # Constant feature (global stat)
        'final_price': [df['price'][-1]] * len(df),  # Uses final value
        'whole_series_mean': [df['price'].mean()] * len(df)  # Global statistic
    }
    
    return {
        'df': df,
        'safe_features': safe_features,
        'unsafe_features': unsafe_features,
        'prices': prices
    }


@pytest.fixture
def target_creation_test_data():
    """Generate test data for directional target creation."""
    n_points = 100
    # Create predictable price pattern for testing
    base_prices = []
    for i in range(n_points):
        if i < 30:
            base_prices.append(100 + i * 0.5)  # Uptrend
        elif i < 60:
            base_prices.append(115 - (i - 30) * 0.3)  # Downtrend  
        else:
            base_prices.append(106 + np.sin((i - 60) * 0.2) * 2)  # Oscillation
    
    df = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_points)],
        'price': base_prices
    })
    
    # Calculate expected targets manually
    horizons = [5, 15, 30, 60]
    expected_targets = {}
    
    for h in horizons:
        # Directional targets (1 if price goes up, 0 if down)
        directional = []
        # Return targets (percentage change)
        returns = []
        
        for i in range(len(base_prices)):
            if i + h < len(base_prices):
                future_price = base_prices[i + h]
                current_price = base_prices[i]
                
                # Directional: 1 if up, 0 if down
                directional.append(1 if future_price > current_price else 0)
                
                # Return: percentage change
                returns.append((future_price - current_price) / current_price)
            else:
                directional.append(None)  # NaN for end-of-series
                returns.append(None)
        
        expected_targets[h] = {
            'directional': directional,
            'returns': returns
        }
    
    return {
        'df': df,
        'prices': base_prices,
        'horizons': horizons,
        'expected_targets': expected_targets
    }


@pytest.fixture
def correlation_test_data():
    """Generate multi-token data for correlation analysis testing."""
    np.random.seed(42)
    n_points = 200
    n_tokens = 5
    
    # Create correlated time series
    base_returns = np.random.normal(0, 0.02, n_points)
    
    tokens_data = {}
    for i in range(n_tokens):
        # Add some correlation structure
        correlation_factor = 0.3 + i * 0.1
        token_returns = (
            correlation_factor * base_returns + 
            (1 - correlation_factor) * np.random.normal(0, 0.02, n_points)
        )
        
        prices = 100 * np.exp(np.cumsum(token_returns))
        
        tokens_data[f'token_{i}'] = pl.DataFrame({
            'datetime': [datetime(2024, 1, 1) + timedelta(minutes=j) for j in range(n_points)],
            'price': prices
        })
    
    # Calculate expected correlation matrix manually
    price_matrix = np.column_stack([
        tokens_data[f'token_{i}']['price'].to_numpy() for i in range(n_tokens)
    ])
    
    expected_correlation_matrix = np.corrcoef(price_matrix.T)
    
    return {
        'tokens_data': tokens_data,
        'n_tokens': n_tokens,
        'n_points': n_points,
        'expected_correlation_matrix': expected_correlation_matrix,
        'price_matrix': price_matrix
    }


@pytest.fixture
def edge_case_time_series():
    """Generate edge case time series for robustness testing."""
    datasets = {}
    
    # Constant prices (no variation)
    datasets['constant'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)],
        'price': [100.0] * 100
    })
    
    # Single point
    datasets['single_point'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1)],
        'price': [100.0]
    })
    
    # Two points
    datasets['two_points'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
        'price': [100.0, 105.0]
    })
    
    # Extreme volatility
    extreme_returns = np.random.normal(0, 0.5, 100)  # 50% volatility
    extreme_prices = 100 * np.exp(np.cumsum(extreme_returns))
    datasets['extreme_volatility'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)],
        'price': extreme_prices
    })
    
    # Missing data (gaps)
    gap_timestamps = []
    gap_prices = []
    for i in range(50):
        gap_timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
        gap_prices.append(100 + i * 0.1)
    # Skip 20 minutes (gap)
    for i in range(70, 100):
        gap_timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
        gap_prices.append(100 + i * 0.1)
    
    datasets['with_gaps'] = pl.DataFrame({
        'datetime': gap_timestamps,
        'price': gap_prices
    })
    
    # Very large numbers
    datasets['large_values'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)],
        'price': [1e12 + i * 1e9 for i in range(100)]
    })
    
    # Very small numbers
    datasets['small_values'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)],
        'price': [1e-6 + i * 1e-9 for i in range(100)]
    })
    
    return datasets


@pytest.fixture
def technical_indicator_references():
    """Calculate reference technical indicators using standard formulas."""
    # Test data - need at least 26 points for MACD calculation
    np.random.seed(42)  # For reproducible test data
    base_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110])
    # Extend with additional points for MACD
    additional_prices = 110 + np.cumsum(np.random.normal(0, 1, 20))
    prices = np.concatenate([base_prices, additional_prices])
    
    references = {}
    
    # Simple Moving Average
    references['sma_5'] = _calculate_reference_sma(prices, 5)
    
    # RSI
    references['rsi_14'] = _calculate_reference_rsi(prices, min(14, len(prices) - 1))
    
    # MACD
    references['macd'] = _calculate_reference_macd(prices)
    
    # Bollinger Bands
    references['bollinger'] = _calculate_reference_bollinger(prices, 5, 2)
    
    # Volatility
    references['volatility'] = _calculate_reference_volatility(prices, 5)
    
    return {
        'prices': prices,
        'references': references
    }


def _calculate_reference_indicators(prices, returns):
    """Calculate reference technical indicators for validation."""
    indicators = {}
    
    # Simple statistics
    indicators['mean'] = np.mean(prices)
    indicators['std'] = np.std(prices, ddof=1)
    indicators['volatility'] = np.std(returns, ddof=1)
    
    # Technical indicators
    if len(prices) >= 14:
        indicators['rsi_14'] = _calculate_simple_rsi(prices, 14)
    
    if len(prices) >= 10:
        indicators['sma_10'] = _calculate_reference_sma(prices, 10)
        indicators['bollinger'] = _calculate_reference_bollinger(prices, 10, 2)
    
    return indicators


def _calculate_simple_rsi(prices, period):
    """Calculate simple RSI for testing (basic implementation)."""
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.full(len(prices), np.nan)
    avg_losses = np.full(len(prices), np.nan)
    
    # Simple moving averages for RSI
    for i in range(period, len(prices)):
        avg_gains[i] = np.mean(gains[i-period:i])
        avg_losses[i] = np.mean(losses[i-period:i])
    
    rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_reference_sma(prices, window):
    """Calculate simple moving average reference."""
    sma = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        sma[i] = np.mean(prices[i - window + 1:i + 1])
    return sma


def _calculate_reference_rsi(prices, period):
    """Calculate RSI reference implementation."""
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.full(len(prices), np.nan)
    
    for i in range(period, len(prices)):
        avg_gain = np.mean(gains[i-period:i])
        avg_loss = np.mean(losses[i-period:i])
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_reference_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD reference implementation."""
    if len(prices) < slow:
        return {'macd': np.full(len(prices), np.nan), 
                'signal': np.full(len(prices), np.nan)}
    
    # Exponential moving averages (simplified)
    ema_fast = _calculate_ema(prices, fast)
    ema_slow = _calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line[~np.isnan(macd_line)], signal)
    
    # Pad signal line to match length
    signal_padded = np.full(len(prices), np.nan)
    signal_padded[-len(signal_line):] = signal_line
    
    return {'macd': macd_line, 'signal': signal_padded}


def _calculate_ema(prices, period):
    """Calculate exponential moving average."""
    alpha = 2 / (period + 1)
    ema = np.full(len(prices), np.nan)
    
    if len(prices) < period:
        return ema  # Return all NaN if insufficient data
    
    # Start with simple average
    ema[period - 1] = np.mean(prices[:period])
    
    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema


def _calculate_reference_bollinger(prices, window, num_std):
    """Calculate Bollinger Bands reference implementation."""
    sma = _calculate_reference_sma(prices, window)
    
    rolling_std = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        rolling_std[i] = np.std(prices[i - window + 1:i + 1], ddof=1)
    
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }


def _calculate_reference_volatility(prices, window):
    """Calculate rolling volatility reference implementation."""
    returns = np.diff(prices) / prices[:-1]
    volatility = np.full(len(prices), np.nan)
    
    for i in range(window, len(prices)):
        vol_returns = returns[i - window:i]
        volatility[i] = np.std(vol_returns, ddof=1)
    
    return volatility


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for each test."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Setup any global test configuration
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    yield
    
    # Cleanup after test if needed
    pass