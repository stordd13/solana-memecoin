# test_phase1_day1_2_baseline.py
"""
Tests for Phase 1 Day 1-2 baseline assessment script.
Focuses on A/B test reproducibility and mathematical validation.
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_extraction_15 import extract_features_from_returns, ESSENTIAL_FEATURES
from utils.clustering_engine import ClusteringEngine
from utils.death_detection import detect_token_death


class TestPhase1Day12Baseline:
    """Test suite for baseline assessment A/B testing."""
    
    @pytest.fixture
    def sample_token_data(self):
        """Create sample token data for testing."""
        # Generate synthetic price data
        np.random.seed(42)
        n_points = 200
        
        # Create realistic memecoin price pattern
        base_price = 0.000001
        prices = [base_price]
        
        for i in range(n_points - 1):
            # Random walk with occasional pumps
            if np.random.random() < 0.02:  # 2% chance of pump
                multiplier = np.random.uniform(2, 10)
            else:
                multiplier = np.random.uniform(0.98, 1.02)
            
            new_price = prices[-1] * multiplier
            prices.append(max(new_price, 0.0))  # Ensure non-negative
        
        return np.array(prices)
    
    @pytest.fixture
    def clustering_engine(self):
        """Create clustering engine for testing."""
        return ClusteringEngine(random_state=42)
    
    def test_feature_extraction_consistency(self, sample_token_data):
        """Test that feature extraction is deterministic."""
        prices = sample_token_data
        returns = np.diff(prices) / (prices[:-1] + 1e-12)
        
        # Extract features multiple times
        features1 = extract_features_from_returns(returns, prices, use_log=False)
        features2 = extract_features_from_returns(returns, prices, use_log=False)
        
        # Should be identical
        for key in ESSENTIAL_FEATURES.keys():
            assert key in features1
            assert key in features2
            assert abs(features1[key] - features2[key]) < 1e-12, f"Feature {key} not consistent"
    
    def test_raw_vs_log_feature_differences(self, sample_token_data):
        """Test that raw vs log returns produce different features."""
        prices = sample_token_data
        returns = np.diff(prices) / (prices[:-1] + 1e-12)
        
        raw_features = extract_features_from_returns(returns, prices, use_log=False)
        log_features = extract_features_from_returns(returns, prices, use_log=True)
        
        # Some features should be different (especially return-based ones)
        different_features = ['mean_return', 'std_return', 'acf_lag_1']
        
        for feature in different_features:
            assert feature in raw_features
            assert feature in log_features
            # Allow for some numerical tolerance but expect differences
            if abs(raw_features[feature]) > 1e-6 or abs(log_features[feature]) > 1e-6:
                assert abs(raw_features[feature] - log_features[feature]) > 1e-10
    
    def test_death_detection_consistency(self, sample_token_data):
        """Test death detection reproducibility."""
        prices = sample_token_data
        returns = np.diff(prices) / (prices[:-1] + 1e-12)
        
        # Add death pattern to end of token
        death_prices = np.concatenate([prices, np.full(50, prices[-1])])
        death_returns = np.diff(death_prices) / (death_prices[:-1] + 1e-12)
        
        death_minute1 = detect_token_death(death_prices, death_returns)
        death_minute2 = detect_token_death(death_prices, death_returns)
        
        assert death_minute1 == death_minute2, "Death detection not consistent"
        assert death_minute1 is not None, "Should detect death in synthetic dead token"
    
    def test_clustering_reproducibility(self, clustering_engine):
        """Test clustering reproducibility with same random seed."""
        # Create synthetic feature matrix
        np.random.seed(42)
        features = np.random.randn(100, 15)  # 100 tokens, 15 features
        
        result1 = clustering_engine.cluster_and_evaluate(features, k=5)
        result2 = clustering_engine.cluster_and_evaluate(features, k=5)
        
        # Labels should be identical with same random seed
        np.testing.assert_array_equal(result1['labels'], result2['labels'])
        assert abs(result1['silhouette_score'] - result2['silhouette_score']) < 1e-12
    
    def test_stability_metrics_calculation(self, clustering_engine):
        """Test stability metrics calculation accuracy."""
        # Create simple 2-cluster data for predictable results
        np.random.seed(42)
        cluster1 = np.random.randn(50, 15) + [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(50, 15) + [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        features = np.vstack([cluster1, cluster2])
        
        stability = clustering_engine.stability_test(features, k=2, n_runs=3)
        
        # Check required stability metrics
        assert 'mean_ari' in stability
        assert 'std_ari' in stability
        assert 'mean_silhouette' in stability
        assert stability['stability_runs'] == 3
        assert len(stability['ari_scores']) == 3
        
        # For well-separated clusters, should have high stability
        assert stability['mean_ari'] > 0.5, "Should have reasonable ARI for well-separated clusters"
        assert stability['mean_silhouette'] > 0.0, "Should have positive silhouette score"
    
    def test_optimal_k_selection(self, clustering_engine):
        """Test optimal K selection methods."""
        # Create data with clear 3-cluster structure
        np.random.seed(42)
        cluster_size = 30
        cluster1 = np.random.randn(cluster_size, 15) + [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(cluster_size, 15) + [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster3 = np.random.randn(cluster_size, 15) + [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        features = np.vstack([cluster1, cluster2, cluster3])
        
        k_analysis = clustering_engine.find_optimal_k(features, k_range=range(2, 6))
        
        # Check structure
        assert 'k_range' in k_analysis
        assert 'inertias' in k_analysis
        assert 'silhouette_scores' in k_analysis
        assert 'optimal_k_elbow' in k_analysis
        assert 'optimal_k_silhouette' in k_analysis
        
        # Check reasonable results
        assert k_analysis['optimal_k_elbow'] in [2, 3, 4, 5]
        assert k_analysis['optimal_k_silhouette'] in [2, 3, 4, 5]
        assert len(k_analysis['inertias']) == 4  # 2,3,4,5
        assert len(k_analysis['silhouette_scores']) == 4
        
        # Inertias should generally decrease
        inertias = k_analysis['inertias']
        assert inertias[0] > inertias[-1], "Inertia should generally decrease with more clusters"
    
    def test_scale_invariance_preparation(self, sample_token_data):
        """Test preparation for scale invariance testing."""
        # Create high and low base price versions
        low_prices = sample_token_data * 0.000001  # Very low base price
        high_prices = sample_token_data * 0.01     # Higher base price
        
        # Extract features from both
        low_returns = np.diff(low_prices) / (low_prices[:-1] + 1e-12)
        high_returns = np.diff(high_prices) / (high_prices[:-1] + 1e-12)
        
        low_features = extract_features_from_returns(low_returns, low_prices, use_log=False)
        high_features = extract_features_from_returns(high_returns, high_prices, use_log=False)
        
        # Some features should be scale-invariant, others shouldn't
        scale_invariant_features = ['acf_lag_1', 'acf_lag_5', 'acf_lag_10']
        
        for feature in scale_invariant_features:
            # ACF should be similar regardless of scale
            diff = abs(low_features[feature] - high_features[feature])
            assert diff < 0.1, f"Feature {feature} should be approximately scale-invariant"
    
    def test_essential_features_completeness(self, sample_token_data):
        """Test that all 15 essential features are extracted."""
        prices = sample_token_data
        returns = np.diff(prices) / (prices[:-1] + 1e-12)
        
        features = extract_features_from_returns(returns, prices, use_log=False)
        
        # Check all 15 features are present
        assert len(features) == 15, f"Expected 15 features, got {len(features)}"
        
        for feature_name in ESSENTIAL_FEATURES.keys():
            assert feature_name in features, f"Missing essential feature: {feature_name}"
            
            # Check feature is finite
            value = features[feature_name]
            assert np.isfinite(value), f"Feature {feature_name} is not finite: {value}"
            
            # Check feature type matches expectation
            expected_type = ESSENTIAL_FEATURES[feature_name]
            if expected_type == bool:
                assert isinstance(value, (bool, int)), f"Feature {feature_name} should be boolean-like"
            elif expected_type in [int, float]:
                assert isinstance(value, (int, float)), f"Feature {feature_name} should be numeric"
    
    def test_edge_case_handling(self):
        """Test handling of edge cases in feature extraction."""
        # Test very short time series
        short_prices = np.array([0.001, 0.002])
        short_returns = np.array([1.0])
        
        features = extract_features_from_returns(short_returns, short_prices, use_log=False)
        
        # Should not crash and should return all 15 features
        assert len(features) == 15
        for value in features.values():
            assert np.isfinite(value), "All features should be finite even for short series"
        
        # Test constant prices (dead token)
        constant_prices = np.full(100, 0.001)
        constant_returns = np.zeros(99)
        
        features = extract_features_from_returns(constant_returns, constant_prices, use_log=False)
        
        # Should detect as dead
        assert features['is_dead'] == True
        assert features['death_minute'] != -1
        
        # Volatility should be zero or near zero
        assert features['std_return'] < 1e-6
        assert features['volatility_5min'] < 1e-6