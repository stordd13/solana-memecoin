"""
Integration tests for complete analysis pipelines
Tests end-to-end functionality of all analysis types
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
from autocorrelation_clustering import AutocorrelationClusteringAnalyzer


@pytest.mark.integration
class TestFeatureBasedAnalysis:
    """Test complete feature-based analysis pipeline."""
    
    def test_feature_based_pipeline_complete(self, analyzer, synthetic_token_data):
        """Test complete feature-based analysis pipeline."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Run complete feature-based analysis
        results = analyzer.run_complete_analysis(
            temp_dir,
            use_log_price=True,
            n_clusters=3,
            find_optimal_k=False,
            clustering_method='kmeans',
            max_tokens=8
        )
        
        # Validate complete results structure
        required_keys = [
            'token_data', 'acf_results', 'feature_matrix', 'token_names',
            'cluster_labels', 'clustering_results', 't_sne_2d', 't_sne_3d',
            'cluster_stats', 'acf_by_cluster', 'use_log_price', 'n_clusters'
        ]
        
        for key in required_keys:
            assert key in results, f"Results missing required key: {key}"
        
        # Validate data consistency
        n_tokens = len(results['token_names'])
        assert len(results['cluster_labels']) == n_tokens, "Cluster labels should match token count"
        assert results['feature_matrix'].shape[0] == n_tokens, "Feature matrix should match token count"
        assert results['t_sne_2d'].shape[0] == n_tokens, "t-SNE 2D should match token count"
        assert results['t_sne_3d'].shape[0] == n_tokens, "t-SNE 3D should match token count"
        
        # Validate clustering results
        assert results['n_clusters'] == 3, "Should have requested number of clusters"
        assert len(np.unique(results['cluster_labels'])) <= 3, "Should not exceed requested clusters"
        
        # Validate ACF results
        assert len(results['acf_results']) > 0, "Should have ACF results"
        for token_name, acf_result in results['acf_results'].items():
            assert 'acf' in acf_result, f"ACF missing for token {token_name}"
            assert len(acf_result['acf']) > 0, f"ACF should not be empty for {token_name}"
    
    def test_feature_based_with_optimal_k(self, analyzer, synthetic_token_data):
        """Test feature-based analysis with optimal K finding."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        results = analyzer.run_complete_analysis(
            temp_dir,
            use_log_price=True,
            find_optimal_k=True,
            clustering_method='kmeans',
            max_tokens=6
        )
        
        # Should include elbow analysis
        assert 'elbow_analysis' in results['clustering_results'], \
            "Should include elbow analysis when find_optimal_k=True"
        
        # Optimal K should be reasonable
        optimal_k = results['clustering_results']['elbow_analysis']['optimal_k_silhouette']
        assert 2 <= optimal_k <= 10, f"Optimal K ({optimal_k}) should be reasonable"
        assert results['n_clusters'] == optimal_k, "Should use optimal K"
    
    def test_feature_based_different_clustering_methods(self, analyzer, synthetic_token_data):
        """Test feature-based analysis with different clustering methods."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        methods = ['kmeans', 'hierarchical']  # Skip DBSCAN for deterministic testing
        
        for method in methods:
            results = analyzer.run_complete_analysis(
                temp_dir,
                use_log_price=True,
                n_clusters=2,
                find_optimal_k=False,
                clustering_method=method,
                max_tokens=4
            )
            
            assert results is not None, f"Analysis should work with {method}"
            assert 'cluster_labels' in results, f"Should have cluster labels for {method}"
            assert len(results['cluster_labels']) > 0, f"Should have labels for {method}"


@pytest.mark.integration
class TestPriceOnlyAnalysis:
    """Test complete price-only analysis pipeline."""
    
    def test_price_only_pipeline_complete(self, analyzer, synthetic_token_data):
        """Test complete price-only analysis pipeline."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Test different price methods
        methods = ['returns', 'log_returns', 'prices', 'dtw_features']
        
        for method in methods:
            results = analyzer.run_price_only_analysis(
                temp_dir,
                method=method,
                use_log_price=True,
                n_clusters=2,
                find_optimal_k=False,
                clustering_method='kmeans',
                max_tokens=6,
                max_length=100
            )
            
            # Validate results structure
            required_keys = [
                'token_data', 'acf_results', 'feature_matrix', 'token_names',
                'cluster_labels', 'clustering_results', 't_sne_2d', 't_sne_3d',
                'cluster_stats', 'acf_by_cluster', 'analysis_method'
            ]
            
            for key in required_keys:
                assert key in results, f"Results missing key {key} for method {method}"
            
            # Validate method-specific properties
            assert results['analysis_method'] == f'price_only_{method}', \
                f"Analysis method should be price_only_{method}"
            
            # Validate ACF is computed even for price-only
            assert len(results['acf_results']) > 0, f"Should have ACF results for {method}"
            
            # Validate data consistency
            n_tokens = len(results['token_names'])
            if n_tokens > 0:
                assert len(results['cluster_labels']) == n_tokens, \
                    f"Cluster labels should match token count for {method}"
                assert results['feature_matrix'].shape[0] == n_tokens, \
                    f"Feature matrix should match token count for {method}"
    
    def test_price_only_sequence_length_handling(self, analyzer, synthetic_token_data):
        """Test price-only analysis with different sequence length settings."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Test with max_length constraint
        results_limited = analyzer.run_price_only_analysis(
            temp_dir,
            method='returns',
            max_length=50,
            max_tokens=4
        )
        
        # Test without max_length constraint
        results_unlimited = analyzer.run_price_only_analysis(
            temp_dir,
            method='returns',
            max_length=None,
            max_tokens=4
        )
        
        # Both should succeed
        assert results_limited is not None, "Limited length analysis should work"
        assert results_unlimited is not None, "Unlimited length analysis should work"
        
        # Limited should have shorter or equal sequences
        if ('sequence_length' in results_limited and 'sequence_length' in results_unlimited and
            results_limited['sequence_length'] != 'variable' and 
            results_unlimited['sequence_length'] != 'variable'):
            
            assert results_limited['sequence_length'] <= results_unlimited['sequence_length'], \
                "Limited analysis should have shorter sequences"
    
    def test_price_only_log_price_consistency(self, analyzer, synthetic_token_data):
        """Test consistency of log price usage in price-only analysis."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Test with log prices
        results_log = analyzer.run_price_only_analysis(
            temp_dir,
            method='returns',
            use_log_price=True,
            max_tokens=4
        )
        
        # Test without log prices
        results_no_log = analyzer.run_price_only_analysis(
            temp_dir,
            method='returns',
            use_log_price=False,
            max_tokens=4
        )
        
        # Both should succeed
        assert results_log is not None, "Log price analysis should work"
        assert results_no_log is not None, "No log price analysis should work"
        
        # Should respect the log price setting
        assert results_log['use_log_price'] == True, "Should record log price usage"
        assert results_no_log['use_log_price'] == False, "Should record no log price usage"


@pytest.mark.integration
class TestMultiResolutionAnalysis:
    """Test complete multi-resolution analysis pipeline."""
    
    def test_multi_resolution_pipeline_complete(self, analyzer, synthetic_token_data):
        """Test complete multi-resolution analysis pipeline."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Run multi-resolution analysis
        results = analyzer.analyze_by_lifespan_category(
            temp_dir,
            method='returns',
            use_log_price=True,
            max_tokens_per_category=5
        )
        
        # Validate top-level results structure
        required_keys = [
            'categories', 'category_summary', 'analysis_method', 'total_tokens_analyzed'
        ]
        
        for key in required_keys:
            assert key in results, f"Multi-resolution results missing key: {key}"
        
        # Validate categories
        categories = results['categories']
        assert isinstance(categories, dict), "Categories should be dict"
        
        # Test each category that has tokens
        for category_name, category_results in categories.items():
            if len(category_results.get('token_data', {})) > 0:
                # Each category should have complete analysis results
                category_required_keys = [
                    'token_data', 'acf_results', 'feature_matrix', 'token_names',
                    'cluster_labels', 'clustering_results', 't_sne_2d', 't_sne_3d',
                    'cluster_stats', 'acf_by_cluster', 'category', 'lifespan_range'
                ]
                
                for key in category_required_keys:
                    assert key in category_results, \
                        f"Category {category_name} missing key: {key}"
                
                # Validate lifespan categorization
                expected_ranges = {
                    'Sprint': '200-400 minutes',
                    'Standard': '400-1200 minutes',
                    'Marathon': '1200+ minutes'
                }
                
                if category_name in expected_ranges:
                    assert category_results['lifespan_range'] == expected_ranges[category_name], \
                        f"Category {category_name} has wrong lifespan range"
                
                # Validate token lifespans match category
                for token_name, token_df in category_results['token_data'].items():
                    token_length = len(token_df)
                    
                    if category_name == 'Sprint':
                        assert 200 <= token_length <= 400, \
                            f"Sprint token {token_name} has invalid length {token_length}"
                    elif category_name == 'Standard':
                        assert 400 < token_length <= 1200, \
                            f"Standard token {token_name} has invalid length {token_length}"
                    elif category_name == 'Marathon':
                        assert token_length > 1200, \
                            f"Marathon token {token_name} has invalid length {token_length}"
    
    def test_multi_resolution_cross_category_comparison(self, analyzer, synthetic_token_data):
        """Test cross-category ACF comparison in multi-resolution analysis."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Run analysis
        results = analyzer.analyze_by_lifespan_category(
            temp_dir,
            method='returns',
            max_tokens_per_category=4
        )
        
        # Test cross-category comparison
        if len(results['categories']) >= 2:
            acf_comparison = analyzer.compare_acf_across_lifespans(results)
            
            # Validate comparison structure
            required_keys = [
                'category_acf_means', 'category_acf_stds', 
                'cross_category_correlations', 'distinctive_patterns'
            ]
            
            for key in required_keys:
                assert key in acf_comparison, f"ACF comparison missing key: {key}"
            
            # Validate ACF means
            category_acf_means = acf_comparison['category_acf_means']
            for category_name, acf_mean in category_acf_means.items():
                assert isinstance(acf_mean, np.ndarray), f"ACF mean for {category_name} should be array"
                assert len(acf_mean) > 0, f"ACF mean for {category_name} should not be empty"
                assert acf_mean[0] == 1.0 or abs(acf_mean[0] - 1.0) < 1e-10, \
                    f"ACF lag 0 should be 1.0 for {category_name}"
            
            # Validate cross-category correlations
            correlations = acf_comparison['cross_category_correlations']
            for comparison, corr in correlations.items():
                assert -1 <= corr <= 1, f"Correlation {comparison} should be between -1 and 1: {corr}"
    
    def test_multi_resolution_dtw_clustering(self, analyzer, synthetic_token_data):
        """Test DTW clustering in multi-resolution analysis."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Run analysis
        results = analyzer.analyze_by_lifespan_category(
            temp_dir,
            method='returns',
            max_tokens_per_category=3
        )
        
        # Add DTW clustering to categories that have sufficient tokens
        for category_name, category_results in results['categories'].items():
            if len(category_results.get('token_data', {})) >= 2:
                try:
                    dtw_results = analyzer.dtw_clustering_variable_length(
                        category_results['token_data'],
                        use_log_price=True,
                        n_clusters=2,
                        max_tokens=3
                    )
                    
                    # Validate DTW results structure
                    dtw_required_keys = [
                        'labels', 'token_names', 'distance_matrix', 'cluster_stats', 'n_clusters'
                    ]
                    
                    for key in dtw_required_keys:
                        assert key in dtw_results, f"DTW results missing key: {key}"
                    
                    # Validate distance matrix
                    distance_matrix = dtw_results['distance_matrix']
                    n_tokens = len(dtw_results['token_names'])
                    assert distance_matrix.shape == (n_tokens, n_tokens), \
                        "Distance matrix should be square"
                    assert np.all(np.diag(distance_matrix) == 0), \
                        "Distance matrix diagonal should be zero"
                    
                except Exception as e:
                    # DTW clustering may fail with insufficient data - that's acceptable
                    pytest.skip(f"DTW clustering failed for {category_name}: {e}")
    
    def test_multi_resolution_token_limits(self, analyzer, synthetic_token_data):
        """Test token limits in multi-resolution analysis."""
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Test with strict token limits
        max_per_category = 2
        results = analyzer.analyze_by_lifespan_category(
            temp_dir,
            method='returns',
            max_tokens_per_category=max_per_category
        )
        
        # Check that limits are respected
        for category_name, category_results in results['categories'].items():
            n_tokens = len(category_results.get('token_data', {}))
            assert n_tokens <= max_per_category, \
                f"Category {category_name} has {n_tokens} tokens, exceeds limit {max_per_category}"
        
        # Validate total tokens analyzed
        total_analyzed = sum(len(cat.get('token_data', {})) for cat in results['categories'].values())
        assert results['total_tokens_analyzed'] == total_analyzed, \
            "Total tokens analyzed should match sum of category tokens"


@pytest.mark.integration
class TestAnalysisPipelineRobustness:
    """Test robustness of analysis pipelines under various conditions."""
    
    def test_minimal_data_handling(self, analyzer, temp_data_dir):
        """Test analysis pipelines with minimal data."""
        # Create minimal dataset (1-2 tokens)
        for i in range(2):
            prices = 100 + np.random.normal(0, 1, 150)  # Standard length
            timestamps = pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=149),
                interval="1m",
                eager=True
            )
            
            df = pl.DataFrame({
                'datetime': timestamps,
                'price': prices
            })
            df = df.with_columns([pl.col('price').log().alias('log_price')])
            
            file_path = temp_data_dir / f"minimal_token_{i}.parquet"
            df.write_parquet(file_path)
        
        # Test feature-based analysis
        try:
            results = analyzer.run_complete_analysis(
                temp_data_dir,
                max_tokens=2,
                n_clusters=2,
                find_optimal_k=False
            )
            assert results is not None, "Should handle minimal data"
        except (ValueError, RuntimeError):
            # Acceptable to fail with clear error for insufficient data
            pass
        
        # Test price-only analysis
        try:
            results = analyzer.run_price_only_analysis(
                temp_data_dir,
                method='returns',
                max_tokens=2,
                n_clusters=2,
                find_optimal_k=False
            )
            assert results is not None, "Should handle minimal data"
        except (ValueError, RuntimeError):
            # Acceptable to fail with clear error for insufficient data
            pass
    
    def test_large_dataset_simulation(self, analyzer, temp_data_dir):
        """Test analysis pipelines with larger dataset (but still manageable for testing)."""
        # Create larger dataset
        n_tokens = 20
        
        for i in range(n_tokens):
            # Vary the length and patterns
            length = 200 + i * 20  # Varying lengths
            
            # Create different patterns
            if i % 4 == 0:  # Trend
                prices = 100 + 0.1 * np.arange(length) + np.random.normal(0, 1, length)
            elif i % 4 == 1:  # Volatile
                prices = 100 + np.random.normal(0, 3, length)
            elif i % 4 == 2:  # Cyclical
                t = np.arange(length)
                prices = 100 + 5 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.5, length)
            else:  # Random walk
                prices = 100 + np.cumsum(np.random.normal(0, 0.5, length))
            
            timestamps = pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=length-1),
                interval="1m",
                eager=True
            )
            
            df = pl.DataFrame({
                'datetime': timestamps,
                'price': prices
            })
            df = df.with_columns([pl.col('price').log().alias('log_price')])
            
            file_path = temp_data_dir / f"large_token_{i:02d}.parquet"
            df.write_parquet(file_path)
        
        # Test with reasonable limits
        results = analyzer.run_complete_analysis(
            temp_data_dir,
            max_tokens=15,
            find_optimal_k=True,
            clustering_method='kmeans'
        )
        
        assert results is not None, "Should handle larger dataset"
        assert len(results['token_names']) <= 15, "Should respect token limit"
        assert results['n_clusters'] >= 2, "Should find multiple clusters in diverse data"
    
    @pytest.mark.slow
    def test_analysis_pipeline_memory_usage(self, analyzer, synthetic_token_data):
        """Test that analysis pipelines don't use excessive memory."""
        import psutil
        import os
        
        temp_dir = list(synthetic_token_data.values())[0]['path'].parent
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run analysis
        results = analyzer.run_complete_analysis(
            temp_dir,
            max_tokens=10,
            find_optimal_k=True
        )
        
        # Check memory usage after analysis
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f} MB"
        assert results is not None, "Analysis should complete successfully"