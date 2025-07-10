"""
Test configuration and fixtures for data_cleaning module
Provides test infrastructure for mathematical validation of cleaning operations
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from clean_tokens import CategoryAwareTokenCleaner
from analyze_exclusions import analyze_exclusion_reasons
from generate_graduated_datasets import generate_graduated_datasets


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def token_cleaner():
    """Create CategoryAwareTokenCleaner instance for testing."""
    return CategoryAwareTokenCleaner()


@pytest.fixture
def reference_mathematical_data():
    """Generate reference data with known mathematical properties for validation."""
    np.random.seed(42)  # Reproducible results
    
    # Base price series (100 points, 1-minute intervals)
    base_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    base_prices = np.maximum(base_prices, 0.01)  # Ensure positive prices
    
    # Calculate expected mathematical properties
    returns = np.diff(base_prices) / base_prices[:-1]
    price_mean = np.mean(base_prices)
    price_std = np.std(base_prices, ddof=1)
    price_cv = price_std / price_mean if price_mean > 0 else 0
    
    # Log returns and log CV
    log_prices = np.log(base_prices + 1e-10)
    log_returns = np.diff(log_prices)
    log_price_cv = np.std(log_prices) / np.mean(log_prices) if np.mean(log_prices) != 0 else 0
    
    # Entropy calculation
    unique_prices = len(np.unique(base_prices))
    hist, _ = np.histogram(base_prices, bins=min(10, unique_prices))
    probs = hist / np.sum(hist) if np.sum(hist) > 0 else np.ones(len(hist)) / len(hist)
    probs = probs[probs > 0]  # Remove zero probabilities
    entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    normalized_entropy = entropy / np.log2(len(probs)) if len(probs) > 1 else 0
    
    return {
        'prices': base_prices,
        'returns': returns,
        'log_returns': log_returns,
        'price_mean': price_mean,
        'price_std': price_std,
        'price_cv': price_cv,
        'log_price_cv': log_price_cv,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'unique_prices': unique_prices,
        'price_ratio': np.max(base_prices) / np.min(base_prices),
        'datetime_series': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(len(base_prices))]
    }


@pytest.fixture
def test_token_datasets():
    """Generate various token datasets for comprehensive testing."""
    datasets = {}
    
    # Clean normal token (no issues)
    datasets['clean_normal'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': 100 + np.cumsum(np.random.normal(0, 0.5, 100))  # Gentle random walk
    })
    
    # Token with staircase pattern (constant prices)
    staircase_prices = []
    for i in range(20):
        staircase_prices.extend([100.0 + i] * 5)  # 5 minutes at each price level
    datasets['staircase_pattern'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': staircase_prices
    })
    
    # Token with gaps (missing time periods)
    gap_timestamps = []
    gap_prices = []
    base_time = datetime(2024, 1, 1)
    for i in range(50):
        gap_timestamps.append(base_time + timedelta(minutes=i))
        gap_prices.append(100.0 + i * 0.1)
    # 30-minute gap
    for i in range(80, 130):  # Skip 50-79 (30 minute gap)
        gap_timestamps.append(base_time + timedelta(minutes=i))
        gap_prices.append(100.0 + i * 0.1)
    
    datasets['with_gaps'] = pl.DataFrame({
        'datetime': gap_timestamps,
        'price': gap_prices
    })
    
    # Token with extreme price ratio (potential corruption)
    extreme_prices = [0.001] * 30 + [1000.0] * 30 + [0.001] * 40  # 1M:1 ratio
    datasets['extreme_ratio'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': extreme_prices
    })
    
    # Token with duplicated timestamps
    duplicate_timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i//2) for i in range(100)]
    datasets['duplicate_timestamps'] = pl.DataFrame({
        'datetime': duplicate_timestamps,
        'price': np.random.uniform(99, 101, 100)
    })
    
    # Token with zero prices
    zero_prices = [100.0] * 30 + [0.0] * 20 + [100.0] * 50
    datasets['with_zeros'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': zero_prices
    })
    
    # Token with negative prices
    negative_prices = [100.0] * 30 + [-10.0] * 20 + [100.0] * 50
    datasets['with_negatives'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': negative_prices
    })
    
    # High volatility token (frequent large changes)
    high_vol_prices = [100.0]
    for i in range(99):
        # Alternate between +50% and -33% changes
        if i % 2 == 0:
            high_vol_prices.append(high_vol_prices[-1] * 1.5)
        else:
            high_vol_prices.append(high_vol_prices[-1] * 0.67)
    
    datasets['high_volatility'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': high_vol_prices
    })
    
    # Low variability token (very stable)
    stable_prices = [100 + 0.001 * i for i in range(100)]  # Tiny linear trend
    datasets['low_variability'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': stable_prices
    })
    
    return datasets


@pytest.fixture
def edge_case_datasets():
    """Generate edge case datasets for robustness testing."""
    datasets = {}
    
    # Single data point
    datasets['single_point'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1)],
        'price': [100.0]
    })
    
    # Two data points
    datasets['two_points'] = pl.DataFrame({
        'datetime': [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
        'price': [100.0, 150.0]
    })
    
    # All same prices (constant)
    datasets['all_constant'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': [42.0] * 100
    })
    
    # Very large prices (numerical stability test)
    datasets['very_large_prices'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': [1e15 + i for i in range(100)]
    })
    
    # Very small prices (numerical stability test)
    datasets['very_small_prices'] = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=99),
            interval="1m",
            eager=True
        ),
        'price': [1e-10 + i * 1e-12 for i in range(100)]
    })
    
    # Empty dataframe
    datasets['empty'] = pl.DataFrame({
        'datetime': [],
        'price': []
    }, schema=[('datetime', pl.Datetime), ('price', pl.Float64)])
    
    return datasets


@pytest.fixture
def synthetic_token_files(test_token_datasets, temp_data_dir):
    """Create synthetic token files for file-based testing."""
    file_paths = {}
    
    for dataset_name, df in test_token_datasets.items():
        file_path = temp_data_dir / f"{dataset_name}_token.parquet"
        df.write_parquet(file_path)
        file_paths[dataset_name] = file_path
    
    return file_paths


@pytest.fixture
def graduated_thresholds():
    """Provide reference graduated thresholds for testing."""
    return {
        'short_term': {
            'staircase_unique_ratio': 0.3,
            'staircase_variation_ratio': 0.5,
            'staircase_relative_std': 0.02,
            'variability_cv_threshold': 0.01,
            'variability_log_cv_threshold': 0.005,
            'variability_entropy_threshold': 0.3,
            'max_gap_minutes': 15,
            'max_ratio_threshold': 1000
        },
        'medium_term': {
            'staircase_unique_ratio': 0.2,
            'staircase_variation_ratio': 0.4,
            'staircase_relative_std': 0.01,
            'variability_cv_threshold': 0.005,
            'variability_log_cv_threshold': 0.003,
            'variability_entropy_threshold': 0.2,
            'max_gap_minutes': 30,
            'max_ratio_threshold': 10000
        },
        'long_term': {
            'staircase_unique_ratio': 0.1,
            'staircase_variation_ratio': 0.3,
            'staircase_relative_std': 0.005,
            'variability_cv_threshold': 0.003,
            'variability_log_cv_threshold': 0.002,
            'variability_entropy_threshold': 0.1,
            'max_gap_minutes': 60,
            'max_ratio_threshold': 100000
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for each test."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Setup any global test configuration
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    yield
    
    # Cleanup after test if needed
    pass