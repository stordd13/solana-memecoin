"""
Pytest configuration and fixtures for time series autocorrelation tests
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple
import sys

# Add the parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


@pytest.fixture
def analyzer():
    """Create an AutocorrelationClusteringAnalyzer instance for testing."""
    return AutocorrelationClusteringAnalyzer()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data that gets cleaned up after tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def synthetic_price_series():
    """Generate synthetic price series with known properties for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Different types of synthetic series
    series_data = {}
    
    # 1. Random walk (should have low autocorrelation)
    n_points = 200
    random_walk = np.cumsum(np.random.normal(0, 0.01, n_points)) + 100
    series_data['random_walk'] = random_walk
    
    # 2. AR(1) process with known autocorrelation
    phi = 0.7  # AR coefficient
    ar1_series = [100]  # Starting value
    for _ in range(n_points - 1):
        next_val = phi * ar1_series[-1] + np.random.normal(0, 0.5)
        ar1_series.append(next_val)
    series_data['ar1_phi07'] = np.array(ar1_series)
    
    # 3. Trend series (should have high autocorrelation)
    trend_series = 100 + 0.1 * np.arange(n_points) + np.random.normal(0, 0.2, n_points)
    series_data['trend'] = trend_series
    
    # 4. Cyclical series
    t = np.arange(n_points)
    cyclical_series = 100 + 10 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.5, n_points)
    series_data['cyclical'] = cyclical_series
    
    # 5. White noise (should have near-zero autocorrelation except at lag 0)
    white_noise = 100 + np.random.normal(0, 1, n_points)
    series_data['white_noise'] = white_noise
    
    return series_data


@pytest.fixture
def synthetic_token_data(temp_data_dir, synthetic_price_series):
    """Create synthetic token parquet files with known properties."""
    token_files = {}
    
    for series_name, prices in synthetic_price_series.items():
        # Create different lifespan categories
        for category, length_mult in [('sprint', 0.5), ('standard', 1.0), ('marathon', 2.0)]:
            # Adjust series length for category
            series_length = int(len(prices) * length_mult)
            if series_length < 50:  # Minimum length
                series_length = 50
            
            # Resample or extend the series
            if series_length <= len(prices):
                category_prices = prices[:series_length]
            else:
                # Extend by repeating the pattern
                repeats = (series_length // len(prices)) + 1
                extended = np.tile(prices, repeats)
                category_prices = extended[:series_length]
            
            # Create datetime index
            timestamps = pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=series_length-1),
                interval="1m",
                eager=True
            )
            
            # Create DataFrame
            df = pl.DataFrame({
                'datetime': timestamps,
                'price': category_prices
            })
            
            # Add log_price column
            df = df.with_columns([
                pl.col('price').log().alias('log_price')
            ])
            
            # Save to parquet
            token_name = f"{series_name}_{category}"
            file_path = temp_data_dir / f"{token_name}.parquet"
            df.write_parquet(file_path)
            
            token_files[token_name] = {
                'path': file_path,
                'expected_length': series_length,
                'expected_category': category,
                'series_type': series_name
            }
    
    return token_files


@pytest.fixture
def known_acf_values():
    """Provide known ACF values for validation."""
    return {
        'ar1_phi07': {
            'theoretical_acf': lambda k, phi=0.7: phi**k,  # AR(1) theoretical ACF
            'expected_lag1': 0.7,
            'expected_lag5': 0.7**5
        },
        'white_noise': {
            'expected_lag0': 1.0,
            'expected_lag1': 0.0,  # Should be near zero
            'tolerance': 0.1  # Allow for sampling variation
        }
    }


@pytest.fixture
def clustering_test_data():
    """Generate test data for clustering validation."""
    np.random.seed(42)
    
    # Create distinct clusters with known properties
    cluster_data = []
    cluster_labels = []
    
    # Cluster 1: Low volatility, slight upward trend
    for i in range(20):
        series = 100 + 0.05 * np.arange(100) + np.random.normal(0, 0.5, 100)
        cluster_data.append(series)
        cluster_labels.append(0)
    
    # Cluster 2: High volatility, no trend
    for i in range(20):
        series = 100 + np.random.normal(0, 2, 100)
        cluster_data.append(series)
        cluster_labels.append(1)
    
    # Cluster 3: Strong downward trend
    for i in range(20):
        series = 100 - 0.1 * np.arange(100) + np.random.normal(0, 0.8, 100)
        cluster_data.append(series)
        cluster_labels.append(2)
    
    return {
        'data': cluster_data,
        'true_labels': cluster_labels,
        'n_clusters': 3
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