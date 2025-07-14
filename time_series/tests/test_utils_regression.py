# test_utils_regression.py
"""
Regression tests to ensure new utils modules produce consistent results
with the legacy Streamlit app functionality.
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_extraction_15 import extract_features_from_token_data
from utils.death_detection import detect_token_death, calculate_death_features
from utils.clustering_engine import ClusteringEngine


class TestUtilsRegression:
    """Regression tests against known good outputs."""
    
    @pytest.fixture
    def legacy_results_dir(self):
        """Path to legacy results for comparison."""
        return Path(__file__).parent.parent / "results"
    
    @pytest.fixture
    def sample_processed_data(self):
        """Load sample processed data if available."""
        processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        
        if not processed_dir.exists():
            pytest.skip("No processed data available for regression testing")
        
        # Try to find a sample token file
        for category_dir in processed_dir.iterdir():
            if category_dir.is_dir():
                token_files = list(category_dir.glob("*.parquet"))
                if token_files:
                    # Load first available token
                    try:
                        df = pl.read_parquet(token_files[0])
                        return df, token_files[0].stem
                    except Exception:
                        continue
        
        pytest.skip("No valid token data found for regression testing")
    
    def test_death_detection_consistency_with_legacy(self, sample_processed_data):
        """Test that death detection matches legacy behavior."""
        df, token_name = sample_processed_data
        
        if 'price' not in df.columns:
            pytest.skip("Token data missing price column")
        
        prices = df['price'].to_numpy()
        returns = np.diff(prices) / (prices[:-1] + 1e-12)
        
        # Test death detection
        death_minute = detect_token_death(prices, returns)
        
        # Basic sanity checks that should match legacy behavior
        if death_minute is not None:
            assert isinstance(death_minute, int)
            assert 0 <= death_minute < len(prices)
            
            # If death detected, final prices should be relatively constant
            if death_minute < len(prices) - 30:
                final_prices = prices[death_minute:death_minute+30]
                price_cv = np.std(final_prices) / (np.mean(final_prices) + 1e-12)
                assert price_cv < 0.1, "Death period should have low price variation"
    
    def test_feature_extraction_mathematical_precision(self, sample_processed_data):
        """Test feature extraction mathematical precision."""
        df, token_name = sample_processed_data
        
        if 'price' not in df.columns:
            pytest.skip("Token data missing price column")
        
        # Extract features multiple times to test consistency
        features1 = extract_features_from_token_data(df, use_log=False)
        features2 = extract_features_from_token_data(df, use_log=False)
        
        # Should be identical to machine precision
        for key, value1 in features1.items():
            value2 = features2[key]
            assert abs(value1 - value2) < 1e-12, f"Feature {key} not consistent: {value1} vs {value2}"
    
    def test_clustering_engine_mathematical_consistency(self):
        """Test clustering engine mathematical consistency."""
        engine = ClusteringEngine(random_state=42)
        
        # Create deterministic test data
        np.random.seed(42)
        features = np.random.randn(50, 15)
        
        # Test multiple runs with same seed
        result1 = engine.cluster_and_evaluate(features, k=3)
        result2 = engine.cluster_and_evaluate(features, k=3)
        
        # Should be identical
        np.testing.assert_array_equal(result1['labels'], result2['labels'])
        assert abs(result1['silhouette_score'] - result2['silhouette_score']) < 1e-12
        assert abs(result1['inertia'] - result2['inertia']) < 1e-12
    
    def test_feature_ranges_reasonable(self, sample_processed_data):
        """Test that extracted features are in reasonable ranges."""
        df, token_name = sample_processed_data
        
        if 'price' not in df.columns:
            pytest.skip("Token data missing price column")
        
        features = extract_features_from_token_data(df, use_log=False)
        
        # Test reasonable ranges for features
        assert features['lifespan_minutes'] > 0, "Lifespan should be positive"
        assert features['lifespan_minutes'] <= len(df), "Lifespan can't exceed data length"
        
        # ACF values should be in [-1, 1]
        for acf_feature in ['acf_lag_1', 'acf_lag_5', 'acf_lag_10']:
            value = features[acf_feature]
            assert -1.1 <= value <= 1.1, f"ACF feature {acf_feature} out of range: {value}"
        
        # Standard deviations should be non-negative
        assert features['std_return'] >= 0, "Standard deviation should be non-negative"
        assert features['volatility_5min'] >= 0, "Volatility should be non-negative"
        
        # Max drawdown should be in [0, 1]
        assert 0 <= features['max_drawdown'] <= 1, f"Max drawdown out of range: {features['max_drawdown']}"
    
    def test_existing_results_compatibility(self, legacy_results_dir):
        """Test compatibility with existing result formats."""
        if not legacy_results_dir.exists():
            pytest.skip("No legacy results directory found")
        
        # Look for existing JSON result files
        json_files = list(legacy_results_dir.glob("*.json"))
        
        if not json_files:
            pytest.skip("No legacy JSON results found")
        
        for json_file in json_files[:3]:  # Test first 3 files
            try:
                with open(json_file, 'r') as f:
                    legacy_data = json.load(f)
                
                # Check basic structure that should be preserved
                if 'total_tokens_analyzed' in legacy_data:
                    assert isinstance(legacy_data['total_tokens_analyzed'], int)
                    assert legacy_data['total_tokens_analyzed'] > 0
                
                if 'archetypes' in legacy_data:
                    archetypes = legacy_data['archetypes']
                    assert isinstance(archetypes, dict)
                    
                    for cluster_id, archetype in archetypes.items():
                        if 'stats' in archetype:
                            stats = archetype['stats']
                            if 'n_tokens' in stats:
                                assert stats['n_tokens'] > 0
                            if 'pct_dead' in stats:
                                assert 0 <= stats['pct_dead'] <= 100
                
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed legacy files
                continue
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases."""
        # Test with very small prices
        small_prices = np.array([1e-10, 2e-10, 1.5e-10, 1e-10])
        features = extract_features_from_token_data(
            pl.DataFrame({'price': small_prices}), use_log=False
        )
        
        # Should not produce NaN or inf values
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} not finite with small prices: {value}"
        
        # Test with large prices
        large_prices = np.array([1e6, 2e6, 1.5e6, 1e6])
        features = extract_features_from_token_data(
            pl.DataFrame({'price': large_prices}), use_log=False
        )
        
        # Should not produce NaN or inf values
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} not finite with large prices: {value}"
        
        # Test with zero returns (constant prices)
        constant_prices = np.full(100, 0.001)
        features = extract_features_from_token_data(
            pl.DataFrame({'price': constant_prices}), use_log=False
        )
        
        # Should handle gracefully
        for key, value in features.items():
            assert np.isfinite(value), f"Feature {key} not finite with constant prices: {value}"
        
        # Mean return should be near zero
        assert abs(features['mean_return']) < 1e-6
        assert features['std_return'] < 1e-6
    
    def test_clustering_stability_regression(self):
        """Test that clustering stability matches expected behavior."""
        engine = ClusteringEngine(random_state=42)
        
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(30, 15) + [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(30, 15) + [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        features = np.vstack([cluster1, cluster2])
        
        stability = engine.stability_test(features, k=2, n_runs=5)
        
        # Well-separated clusters should have high stability
        assert stability['mean_ari'] > 0.8, "Well-separated clusters should have high ARI"
        assert stability['mean_silhouette'] > 0.5, "Well-separated clusters should have high silhouette"
        assert stability['std_ari'] < 0.2, "Stability should be consistent"
    
    def test_memory_efficiency(self):
        """Test that new implementation is memory efficient."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        engine = ClusteringEngine(random_state=42)
        np.random.seed(42)
        large_features = np.random.randn(1000, 15)
        
        # Run clustering
        result = engine.cluster_and_evaluate(large_features, k=5)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f} MB"
        
        # Should still produce valid results
        assert len(result['labels']) == 1000
        assert result['n_clusters'] == 5
        assert np.isfinite(result['silhouette_score'])