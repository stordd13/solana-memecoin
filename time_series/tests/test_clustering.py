"""
Unit tests for clustering functionality
Tests clustering algorithms, stability, and validation
"""

import pytest
import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import make_blobs
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


@pytest.mark.unit
class TestClusteringAlgorithms:
    """Test different clustering algorithms."""
    
    def test_kmeans_clustering(self, analyzer, clustering_test_data):
        """Test K-means clustering with known data."""
        # Use known clustered data
        data = np.array(clustering_test_data['data'])
        true_labels = clustering_test_data['true_labels']
        n_clusters = clustering_test_data['n_clusters']
        
        # Flatten data for clustering (treat each series as features)
        feature_matrix = data.reshape(data.shape[0], -1)
        
        # Perform clustering
        clustering_results = analyzer.perform_clustering(
            feature_matrix, 
            method='kmeans', 
            n_clusters=n_clusters,
            find_optimal_k=False
        )
        
        assert 'labels' in clustering_results, "Results should contain labels"
        assert len(clustering_results['labels']) == len(true_labels), "Labels length should match data"
        
        # Check clustering quality with Adjusted Rand Index
        ari = adjusted_rand_score(true_labels, clustering_results['labels'])
        assert ari > 0.3, f"Clustering quality (ARI={ari}) should be reasonable"
    
    def test_hierarchical_clustering(self, analyzer):
        """Test hierarchical clustering."""
        # Create simple clustered data
        X, y = make_blobs(n_samples=60, centers=3, n_features=10, random_state=42)
        
        clustering_results = analyzer.perform_clustering(
            X, method='hierarchical', n_clusters=3, find_optimal_k=False
        )
        
        assert 'labels' in clustering_results, "Results should contain labels"
        assert len(np.unique(clustering_results['labels'])) <= 3, "Should not exceed requested clusters"
        
        # Check clustering quality
        silhouette = silhouette_score(X, clustering_results['labels'])
        assert silhouette > 0.1, f"Silhouette score ({silhouette}) should be reasonable"
    
    def test_dbscan_clustering(self, analyzer):
        """Test DBSCAN clustering."""
        # Create data with clear clusters
        X, y = make_blobs(n_samples=100, centers=3, n_features=5, 
                         cluster_std=0.5, random_state=42)
        
        clustering_results = analyzer.perform_clustering(
            X, method='dbscan', find_optimal_k=False
        )
        
        assert 'labels' in clustering_results, "Results should contain labels"
        
        # DBSCAN may find different number of clusters or noise points (-1)
        unique_labels = np.unique(clustering_results['labels'])
        assert len(unique_labels) >= 1, "Should find at least one cluster"
        
        # Most points should be assigned to clusters (not noise)
        noise_ratio = np.sum(clustering_results['labels'] == -1) / len(clustering_results['labels'])
        assert noise_ratio < 0.5, f"Too many noise points: {noise_ratio:.2%}"
    
    def test_find_optimal_k(self, analyzer):
        """Test optimal K finding with elbow method."""
        # Create data with clear number of clusters
        X, y = make_blobs(n_samples=120, centers=4, n_features=8, random_state=42)
        
        clustering_results = analyzer.perform_clustering(
            X, method='kmeans', find_optimal_k=True
        )
        
        assert 'elbow_analysis' in clustering_results, "Should include elbow analysis"
        assert 'optimal_k_elbow' in clustering_results['elbow_analysis'], "Should find optimal K"
        
        optimal_k = clustering_results['elbow_analysis']['optimal_k_elbow']
        assert 2 <= optimal_k <= 10, f"Optimal K ({optimal_k}) should be reasonable"
    
    def test_clustering_stability(self, analyzer):
        """Test clustering stability across multiple runs."""
        np.random.seed(42)  # For reproducible test
        
        # Create stable data
        X, y = make_blobs(n_samples=90, centers=3, n_features=6, 
                         cluster_std=0.3, random_state=42)
        
        # Run clustering multiple times
        results = []
        for _ in range(3):
            clustering_result = analyzer.perform_clustering(
                X, method='kmeans', n_clusters=3, find_optimal_k=False
            )
            results.append(clustering_result['labels'])
        
        # Results should be consistent (high ARI between runs)
        ari_01 = adjusted_rand_score(results[0], results[1])
        ari_02 = adjusted_rand_score(results[0], results[2])
        
        assert ari_01 > 0.8, f"Clustering should be stable across runs (ARI={ari_01})"
        assert ari_02 > 0.8, f"Clustering should be stable across runs (ARI={ari_02})"


@pytest.mark.unit
class TestClusteringValidation:
    """Test clustering validation and quality metrics."""
    
    def test_silhouette_analysis(self, analyzer):
        """Test silhouette analysis in clustering results."""
        # Create well-separated clusters
        X, y = make_blobs(n_samples=60, centers=3, n_features=5, 
                         cluster_std=0.8, random_state=42)
        
        clustering_results = analyzer.perform_clustering(
            X, method='kmeans', n_clusters=3, find_optimal_k=False
        )
        
        # Calculate silhouette score for validation
        if len(np.unique(clustering_results['labels'])) > 1:
            silhouette = silhouette_score(X, clustering_results['labels'])
            assert silhouette > 0, "Silhouette score should be positive for good clustering"
            assert silhouette < 1, "Silhouette score should be less than 1"
    
    def test_cluster_characteristics_analysis(self, analyzer, synthetic_token_data):
        """Test cluster characteristics analysis."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=6)
        
        if len(token_data) < 3:
            pytest.skip("Need at least 3 tokens for clustering test")
        
        # Prepare data and get some results
        feature_matrix, token_names = analyzer.prepare_price_only_data(
            token_data, method='returns'
        )
        
        if len(feature_matrix) < 3:
            pytest.skip("Need at least 3 feature vectors for clustering test")
        
        # Perform clustering
        clustering_results = analyzer.perform_clustering(
            feature_matrix, method='kmeans', n_clusters=2, find_optimal_k=False
        )
        
        # Analyze cluster characteristics
        cluster_stats = analyzer.analyze_cluster_characteristics(
            token_data, clustering_results['labels'], token_names
        )
        
        assert isinstance(cluster_stats, dict), "Cluster stats should be dict"
        
        # Check structure of cluster stats
        for cluster_id, stats in cluster_stats.items():
            assert 'n_tokens' in stats, "Should include token count"
            assert stats['n_tokens'] > 0, "Cluster should have at least one token"
            
            # Should include meaningful statistics
            if 'avg_return' in stats:
                assert isinstance(stats['avg_return'], (int, float)), "Return should be numeric"
            if 'avg_volatility' in stats:
                assert isinstance(stats['avg_volatility'], (int, float)), "Volatility should be numeric"
                assert stats['avg_volatility'] >= 0, "Volatility should be non-negative"
    
    def test_empty_cluster_handling(self, analyzer):
        """Test handling of empty clusters."""
        # Create data that might result in empty clusters
        X = np.array([[1, 1], [2, 2], [100, 100]])  # One outlier
        
        # Try clustering with more clusters than reasonable
        clustering_results = analyzer.perform_clustering(
            X, method='kmeans', n_clusters=5, find_optimal_k=False
        )
        
        # Should handle gracefully without crashing
        assert 'labels' in clustering_results, "Should return labels even with potential empty clusters"
        
        # Number of unique labels should be reasonable
        n_unique = len(np.unique(clustering_results['labels']))
        assert n_unique <= len(X), "Cannot have more clusters than data points"


@pytest.mark.unit
class TestDTWClustering:
    """Test DTW (Dynamic Time Warping) clustering for variable-length sequences."""
    
    def test_dtw_clustering_basic(self, analyzer, synthetic_token_data):
        """Test basic DTW clustering functionality."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=5)
        
        if len(token_data) < 3:
            pytest.skip("Need at least 3 tokens for DTW clustering test")
        
        # Perform DTW clustering
        dtw_results = analyzer.dtw_clustering_variable_length(
            token_data, use_log_price=True, n_clusters=2, max_tokens=5
        )
        
        assert 'labels' in dtw_results, "DTW results should contain labels"
        assert 'distance_matrix' in dtw_results, "DTW results should contain distance matrix"
        assert 'cluster_stats' in dtw_results, "DTW results should contain cluster stats"
        
        # Check distance matrix properties
        distance_matrix = dtw_results['distance_matrix']
        assert distance_matrix.shape[0] == distance_matrix.shape[1], "Distance matrix should be square"
        assert np.all(np.diag(distance_matrix) == 0), "Diagonal should be zero"
        assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix should be symmetric"
    
    def test_dtw_cluster_stats(self, analyzer, synthetic_token_data):
        """Test DTW cluster statistics."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=4)
        
        if len(token_data) < 2:
            pytest.skip("Need at least 2 tokens for DTW test")
        
        dtw_results = analyzer.dtw_clustering_variable_length(
            token_data, use_log_price=True, n_clusters=2, max_tokens=4
        )
        
        # Check cluster statistics
        cluster_stats = dtw_results['cluster_stats']
        
        for cluster_id, stats in cluster_stats.items():
            assert 'n_tokens' in stats, "Should include token count"
            assert 'avg_length' in stats, "Should include average length"
            assert 'min_length' in stats, "Should include minimum length"
            assert 'max_length' in stats, "Should include maximum length"
            
            # Validate length statistics
            assert stats['min_length'] <= stats['avg_length'] <= stats['max_length'], \
                "Length statistics should be consistent"
            assert stats['min_length'] > 0, "Minimum length should be positive"
    
    @pytest.mark.slow
    def test_dtw_clustering_performance(self, analyzer, synthetic_token_data):
        """Test DTW clustering doesn't take too long (marked as slow test)."""
        import time
        
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        token_data = analyzer.load_raw_prices(temp_dir, limit=3)  # Small limit for speed
        
        if len(token_data) < 2:
            pytest.skip("Need at least 2 tokens for DTW performance test")
        
        start_time = time.time()
        
        dtw_results = analyzer.dtw_clustering_variable_length(
            token_data, use_log_price=True, n_clusters=2, max_tokens=3
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 30, f"DTW clustering took too long: {elapsed_time:.2f} seconds"
        assert dtw_results is not None, "DTW clustering should produce results"


@pytest.mark.unit
class TestClusteringErrorHandling:
    """Test error handling in clustering functions."""
    
    def test_clustering_with_insufficient_data(self, analyzer):
        """Test clustering with too few data points."""
        # Single data point
        X = np.array([[1, 2, 3]])
        
        # Should handle gracefully
        try:
            clustering_results = analyzer.perform_clustering(
                X, method='kmeans', n_clusters=2, find_optimal_k=False
            )
            # If it succeeds, should have reasonable results
            assert len(clustering_results['labels']) == 1, "Should return one label"
        except ValueError:
            # It's also acceptable to raise a clear error
            pass
    
    def test_clustering_with_nan_data(self, analyzer):
        """Test clustering with NaN values in data."""
        # Data with NaN values
        X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        
        # Should handle NaN values
        clustering_results = analyzer.perform_clustering(
            X, method='kmeans', n_clusters=2, find_optimal_k=False
        )
        
        # Should produce valid results (NaN handling is done in the method)
        assert 'labels' in clustering_results, "Should handle NaN values gracefully"
        assert not np.any(np.isnan(clustering_results['labels'])), "Labels should not be NaN"
    
    def test_clustering_with_constant_features(self, analyzer):
        """Test clustering with constant features."""
        # Data where all values in one feature are the same
        X = np.array([[1, 5, 3], [2, 5, 4], [3, 5, 5]])
        
        clustering_results = analyzer.perform_clustering(
            X, method='kmeans', n_clusters=2, find_optimal_k=False
        )
        
        # Should handle constant features gracefully
        assert 'labels' in clustering_results, "Should handle constant features"
        assert len(clustering_results['labels']) == 3, "Should return correct number of labels"
    
    def test_dtw_clustering_edge_cases(self, analyzer, temp_data_dir):
        """Test DTW clustering with edge cases."""
        # Create minimal token data
        minimal_df = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=49),
                interval="1m",
                eager=True
            ),
            'price': [100.0] * 50  # Constant prices
        })
        minimal_df = minimal_df.with_columns([pl.col('price').log().alias('log_price')])
        
        file_path = temp_data_dir / "minimal_token.parquet"
        minimal_df.write_parquet(file_path)
        
        token_data = analyzer.load_raw_prices(temp_data_dir)
        
        # Should handle edge case gracefully
        try:
            dtw_results = analyzer.dtw_clustering_variable_length(
                token_data, use_log_price=True, n_clusters=2, max_tokens=1
            )
            # If successful, should have reasonable structure
            assert 'labels' in dtw_results, "Should return labels"
        except (ValueError, RuntimeError):
            # It's acceptable to fail gracefully with clear errors
            pass


@pytest.mark.unit  
class TestClusteringMetrics:
    """Test clustering evaluation metrics."""
    
    def test_elbow_method_analysis(self, analyzer):
        """Test elbow method for optimal K selection."""
        # Create data with clear optimal K
        X, y = make_blobs(n_samples=150, centers=4, n_features=6, random_state=42)
        
        # Find optimal K
        elbow_analysis = analyzer.find_optimal_clusters(X, max_k=8)
        
        assert 'inertias' in elbow_analysis, "Should include inertias"
        assert 'silhouette_scores' in elbow_analysis, "Should include silhouette scores"
        assert 'optimal_k_elbow' in elbow_analysis, "Should find optimal K by elbow"
        assert 'optimal_k_silhouette' in elbow_analysis, "Should find optimal K by silhouette"
        
        # Optimal K should be reasonable
        optimal_k_elbow = elbow_analysis['optimal_k_elbow']
        optimal_k_silhouette = elbow_analysis['optimal_k_silhouette']
        
        assert 2 <= optimal_k_elbow <= 8, f"Elbow optimal K ({optimal_k_elbow}) should be reasonable"
        assert 2 <= optimal_k_silhouette <= 8, f"Silhouette optimal K ({optimal_k_silhouette}) should be reasonable"
        
        # Should be close to true number of clusters (4)
        assert abs(optimal_k_elbow - 4) <= 2, "Elbow method should find approximately correct K"
    
    def test_clustering_quality_metrics(self, analyzer):
        """Test various clustering quality metrics."""
        # Create well-separated data
        X, true_labels = make_blobs(n_samples=90, centers=3, n_features=4, 
                                   cluster_std=0.5, random_state=42)
        
        clustering_results = analyzer.perform_clustering(
            X, method='kmeans', n_clusters=3, find_optimal_k=False
        )
        
        predicted_labels = clustering_results['labels']
        
        # Calculate quality metrics
        ari = adjusted_rand_score(true_labels, predicted_labels)
        silhouette = silhouette_score(X, predicted_labels)
        
        # Quality should be good for well-separated data
        assert ari > 0.5, f"ARI ({ari}) should be good for well-separated data"
        assert silhouette > 0.3, f"Silhouette ({silhouette}) should be good for well-separated data"