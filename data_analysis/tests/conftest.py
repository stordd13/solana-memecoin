"""
Pytest configuration and fixtures for data_analysis module tests
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple
import sys
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_analysis.data_loader import DataLoader
from data_analysis.data_quality import DataQualityAnalyzer
from data_analysis.price_analysis import PriceAnalyzer
from data_analysis.analyze_tokens import TokenAnalyzer


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data that gets cleaned up after tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def data_loader():
    """Create a DataLoader instance for testing."""
    return DataLoader()


@pytest.fixture
def data_quality_analyzer():
    """Create a DataQualityAnalyzer instance for testing."""
    return DataQualityAnalyzer()


@pytest.fixture
def price_analyzer():
    """Create a PriceAnalyzer instance for testing."""
    return PriceAnalyzer()


@pytest.fixture
def token_analyzer():
    """Create a TokenAnalyzer instance for testing."""
    return TokenAnalyzer()


@pytest.fixture
def synthetic_token_data():
    """Generate synthetic token data with known statistical properties for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 100 minutes of data (standard memecoin lifespan)
    n_points = 100
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(minutes=n_points-1),
        interval="1m",
        eager=True
    )
    
    # Test case 1: Normal behavior token (gentle trends, moderate volatility)
    normal_prices = 100 + np.cumsum(np.random.normal(0, 0.02, n_points))  # 2% std per minute
    normal_df = pl.DataFrame({
        'datetime': timestamps,
        'price': normal_prices
    })
    
    # Test case 2: Extreme pump token (1000%+ increase)
    pump_base = 100
    pump_multiplier = np.concatenate([
        np.ones(30),  # Stable for 30 minutes
        np.linspace(1, 11, 40),  # 1000% pump over 40 minutes
        np.ones(30) * 11  # Stable at peak
    ])
    pump_noise = np.random.normal(1, 0.01, n_points)  # 1% noise
    pump_prices = pump_base * pump_multiplier * pump_noise
    pump_df = pl.DataFrame({
        'datetime': timestamps,
        'price': pump_prices
    })
    
    # Test case 3: Extreme dump token (99% decrease)
    dump_base = 1000
    dump_multiplier = np.concatenate([
        np.ones(20),  # Stable for 20 minutes
        np.linspace(1, 0.01, 50),  # 99% dump over 50 minutes
        np.ones(30) * 0.01  # Dead at bottom
    ])
    dump_noise = np.random.normal(1, 0.005, n_points)  # 0.5% noise
    dump_prices = dump_base * dump_multiplier * dump_noise
    dump_df = pl.DataFrame({
        'datetime': timestamps,
        'price': dump_prices
    })
    
    # Test case 4: High volatility token (no clear trend)
    volatile_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.1, n_points)))  # 10% std per minute
    volatile_df = pl.DataFrame({
        'datetime': timestamps,
        'price': volatile_prices
    })
    
    # Test case 5: Constant prices (edge case)
    constant_df = pl.DataFrame({
        'datetime': timestamps,
        'price': [100.0] * n_points
    })
    
    # Test case 6: Data with gaps (missing values)
    gap_timestamps = timestamps[::2]  # Every other minute
    gap_prices = normal_prices[::2]
    gap_df = pl.DataFrame({
        'datetime': gap_timestamps,
        'price': gap_prices
    })
    
    return {
        'normal_behavior': normal_df,
        'extreme_pump': pump_df,
        'extreme_dump': dump_df,
        'high_volatility': volatile_df,
        'constant_price': constant_df,
        'with_gaps': gap_df
    }


@pytest.fixture
def expected_statistical_properties():
    """Expected statistical properties for synthetic data validation."""
    return {
        'normal_behavior': {
            'initial_price': 100,
            'price_range': (95, 105),  # Approximate range
            'volatility_range': (0.01, 0.05),  # Expected volatility range
            'trend_type': 'random_walk'
        },
        'extreme_pump': {
            'initial_price': 100,
            'final_price_min': 1000,  # At least 1000% increase
            'max_return': 9.0,  # At least 900% total return
            'trend_type': 'strong_upward'
        },
        'extreme_dump': {
            'initial_price': 1000,
            'final_price_max': 20,  # At most 2% of initial
            'max_drawdown': -0.98,  # At least 98% drawdown
            'trend_type': 'strong_downward'
        },
        'high_volatility': {
            'initial_price': 100,
            'volatility_min': 0.08,  # At least 8% volatility
            'trend_type': 'volatile'
        },
        'constant_price': {
            'price_value': 100.0,
            'volatility_max': 1e-10,  # Essentially zero
            'trend_type': 'flat'
        }
    }


@pytest.fixture
def synthetic_token_files(temp_data_dir, synthetic_token_data):
    """Create synthetic token parquet files for testing data loading."""
    file_paths = {}
    
    for token_name, df in synthetic_token_data.items():
        file_path = temp_data_dir / f"{token_name}.parquet"
        df.write_parquet(file_path)
        file_paths[token_name] = file_path
    
    return file_paths


@pytest.fixture
def edge_case_data():
    """Generate edge case data for robust testing."""
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(minutes=49),
        interval="1m",
        eager=True
    )
    
    return {
        'single_point': pl.DataFrame({
            'datetime': [datetime(2024, 1, 1)],
            'price': [100.0]
        }),
        'two_points': pl.DataFrame({
            'datetime': timestamps[:2],
            'price': [100.0, 150.0]
        }),
        'with_nan': pl.DataFrame({
            'datetime': timestamps,
            'price': [100.0 if i % 10 != 5 else None for i in range(50)]
        }),
        'with_inf': pl.DataFrame({
            'datetime': timestamps,
            'price': [100.0 if i != 25 else float('inf') for i in range(50)]
        }),
        'negative_prices': pl.DataFrame({
            'datetime': timestamps,
            'price': [-100.0] * 50
        }),
        'zero_prices': pl.DataFrame({
            'datetime': timestamps,
            'price': [0.0] * 50
        })
    }


@pytest.fixture
def reference_calculations():
    """Reference calculations for validation against numpy/scipy implementations."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    return {
        'prices': prices,
        'returns': np.diff(prices) / prices[:-1],
        'log_returns': np.diff(np.log(prices)),
        'mean': np.mean(prices),
        'std': np.std(prices, ddof=1),
        'min': np.min(prices),
        'max': np.max(prices),
        'percentiles': {
            'p5': np.percentile(prices, 5),
            'p25': np.percentile(prices, 25),
            'p50': np.percentile(prices, 50),
            'p75': np.percentile(prices, 75),
            'p95': np.percentile(prices, 95)
        },
        'volatility': np.std(np.diff(prices) / prices[:-1], ddof=1)
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "mathematical: marks tests that validate mathematical correctness"
    )
    config.addinivalue_line(
        "markers", "streamlit: marks tests that validate Streamlit display accuracy"
    )