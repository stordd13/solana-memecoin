# unified_analysis.py
"""
Unified analysis module that eliminates redundant clustering approaches
Provides a single, streamlined path for all analysis types
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from time_series.archetype_utils import (
    extract_essential_features, categorize_by_lifespan, 
    detect_token_death, prepare_token_data
)
from time_series.behavioral_archetype_analysis import BehavioralArchetypeAnalyzer


class UnifiedAnalysisEngine:
    """
    Unified analysis engine that handles all analysis types with a single clustering approach.
    Eliminates redundancy between multi-resolution and behavioral archetype analysis.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.behavioral_analyzer = BehavioralArchetypeAnalyzer()
        
    def run_unified_analysis(self, 
                           data_path: Path,
                           analysis_type: str,
                           time_series_method: str = "log_returns",
                           max_tokens: Optional[int] = None,
                           max_tokens_per_category: Optional[int] = None,
                           sample_ratio: Optional[float] = None,
                           clustering_method: str = "kmeans",
                           find_optimal_k: bool = True,
                           n_clusters: Optional[int] = None) -> Dict:
        """
        Unified analysis function that handles all analysis types with consistent approach.
        
        Args:
            data_path: Path to data directory
            analysis_type: Type of analysis ("lifespan", "behavioral", "price_only")
            time_series_method: Method for time series ("returns", "log_returns", "prices", "log_prices")
            max_tokens: Maximum tokens for behavioral/price analysis
            max_tokens_per_category: Maximum tokens per category for lifespan analysis  
            sample_ratio: Sampling ratio for faster processing
            clustering_method: Clustering algorithm
            find_optimal_k: Whether to find optimal K
            n_clusters: Fixed number of clusters
            
        Returns:
            Dictionary with unified analysis results
        """
        print(f"DEBUG: Starting unified analysis - type: {analysis_type}, method: {time_series_method}")
        
        # Step 1: Load and prepare data based on analysis type
        if analysis_type == "lifespan":
            token_data = self._load_lifespan_data(data_path, max_tokens_per_category, sample_ratio)
        elif analysis_type == "behavioral":
            token_data = self._load_behavioral_data(data_path, max_tokens, sample_ratio)
        else:  # price_only
            token_data = self._load_price_only_data(data_path, max_tokens)
            
        if not token_data:
            raise ValueError("No valid token data loaded")
            
        print(f"DEBUG: Loaded {len(token_data)} tokens for {analysis_type} analysis")
        
        # Step 2: Extract unified features (always use 15-feature approach)
        features_df = self._extract_unified_features(token_data, time_series_method)
        
        # Step 3: Perform clustering with quality analysis
        clustering_results = self._perform_unified_clustering(
            features_df, clustering_method, find_optimal_k, n_clusters
        )
        
        # Step 4: Generate analysis-specific results
        if analysis_type == "lifespan":
            return self._generate_lifespan_results(token_data, features_df, clustering_results)
        elif analysis_type == "behavioral":
            return self._generate_behavioral_results(token_data, features_df, clustering_results)
        else:  # price_only
            return self._generate_price_only_results(token_data, features_df, clustering_results)
    
    def _load_lifespan_data(self, data_path: Path, max_tokens_per_category: Optional[int], 
                           sample_ratio: Optional[float]) -> Dict[str, pl.DataFrame]:
        """Load and categorize data by lifespan for lifespan analysis."""
        print("DEBUG: Loading data for lifespan analysis...")
        
        # Load from processed categories
        all_token_data = self.behavioral_analyzer.load_categorized_tokens(
            data_path, limit=None, sample_ratio=None
        )
        
        # Categorize by lifespan
        categorized_tokens = categorize_by_lifespan(all_token_data, {})
        
        # Apply category limits and sampling
        if max_tokens_per_category or sample_ratio:
            categorized_tokens = self._apply_category_sampling(
                categorized_tokens, max_tokens_per_category, sample_ratio
            )
        
        # Flatten for unified processing (we'll track categories in metadata)
        flattened_tokens = {}
        for category, tokens in categorized_tokens.items():
            for token_name, token_df in tokens.items():
                # Add category metadata
                token_df = token_df.with_columns(pl.lit(category).alias('lifespan_category'))
                flattened_tokens[token_name] = token_df
                
        return flattened_tokens
    
    def _load_behavioral_data(self, data_path: Path, max_tokens: Optional[int], 
                             sample_ratio: Optional[float]) -> Dict[str, pl.DataFrame]:
        """Load data for behavioral archetype analysis."""
        print("DEBUG: Loading data for behavioral analysis...")
        
        return self.behavioral_analyzer.load_categorized_tokens(
            data_path, limit=max_tokens, sample_ratio=sample_ratio
        )
    
    def _load_price_only_data(self, data_path: Path, max_tokens: Optional[int]) -> Dict[str, pl.DataFrame]:
        """Load data for price-only analysis."""
        print("DEBUG: Loading data for price-only analysis...")
        
        # For price-only, load from raw data directory
        from time_series.autocorrelation_clustering import AutocorrelationClusteringAnalyzer
        analyzer = AutocorrelationClusteringAnalyzer()
        return analyzer.load_raw_prices(data_path, limit=max_tokens)
    
    def _extract_unified_features(self, token_data: Dict[str, pl.DataFrame], 
                                 time_series_method: str) -> pl.DataFrame:
        """Extract 15 essential features using unified approach."""
        print(f"DEBUG: Extracting unified features using method: {time_series_method}")
        
        # Validate inputs
        if not token_data:
            raise ValueError("No token data provided for feature extraction")
            
        if time_series_method not in ["returns", "log_returns", "prices", "log_prices"]:
            raise ValueError(f"Unknown time series method: {time_series_method}")
        
        # Map time series method to use_log_returns parameter
        use_log_returns = time_series_method in ["log_returns", "log_prices"]
        
        try:
            # Use the proven 15-feature extraction
            features_df = extract_essential_features(token_data, use_log_returns=use_log_returns)
            
            if features_df is None or features_df.height == 0:
                raise ValueError("Feature extraction returned empty result")
                
            print(f"DEBUG: Extracted features for {features_df.height} tokens")
            print(f"DEBUG: Feature columns: {[col for col in features_df.columns if col not in ['token', 'category']]}")
            
            return features_df
            
        except Exception as e:
            print(f"ERROR: Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction failed: {e}") from e
    
    def _perform_unified_clustering(self, features_df: pl.DataFrame, 
                                   clustering_method: str, find_optimal_k: bool, 
                                   n_clusters: Optional[int]) -> Dict:
        """Perform clustering with quality analysis and imbalance detection."""
        print(f"DEBUG: Performing unified clustering with method: {clustering_method}")
        
        # Prepare features for clustering
        feature_cols = [col for col in features_df.columns 
                       if col not in ['token', 'category', 'lifespan_category']]
        X = features_df.select(feature_cols).to_numpy()
        
        # Handle NaN values
        if np.any(np.isnan(X)):
            print("DEBUG: Applying median imputation for NaN values")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Check feature variance and remove zero-variance features
        feature_variances = np.var(X, axis=0)
        valid_feature_mask = feature_variances > 1e-10
        
        if np.sum(valid_feature_mask) < 2:
            print("WARNING: Insufficient feature variance for meaningful clustering!")
            return self._create_minimal_clustering_result(len(features_df))
        
        if np.sum(~valid_feature_mask) > 0:
            print(f"DEBUG: Removing {np.sum(~valid_feature_mask)} zero-variance features")
            X = X[:, valid_feature_mask]
            feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_feature_mask[i]]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal K if requested
        if find_optimal_k and clustering_method == 'kmeans':
            optimal_k = self._find_optimal_k(X_scaled)
            if n_clusters is None:
                n_clusters = optimal_k
        
        if n_clusters is None:
            n_clusters = 5  # Default
        
        # Perform clustering
        labels, clusterer = self._cluster_data(X_scaled, clustering_method, n_clusters)
        
        # Analyze cluster quality
        quality_metrics = self._analyze_cluster_quality(X_scaled, labels, features_df)
        
        return {
            'labels': labels,
            'clusterer': clusterer,
            'n_clusters': len(np.unique(labels)),
            'feature_cols': feature_cols,
            'X_scaled': X_scaled,
            'quality_metrics': quality_metrics,
            'scaler': self.scaler
        }
    
    def _find_optimal_k(self, X_scaled: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        n_samples = X_scaled.shape[0]
        max_k = min(max_k, n_samples // 2)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            
            if k <= n_samples - 1:  # Silhouette requires at least 2 clusters with different labels
                try:
                    sil_score = silhouette_score(X_scaled, labels)
                    silhouette_scores.append(sil_score)
                except:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
        
        # Use elbow method (simplified)
        if len(inertias) >= 2:
            diffs = np.diff(inertias)
            optimal_k = k_range[np.argmin(diffs)] if len(diffs) > 0 else k_range[0]
        else:
            optimal_k = k_range[0]
        
        print(f"DEBUG: Optimal K found: {optimal_k}")
        return optimal_k
    
    def _cluster_data(self, X_scaled: np.ndarray, method: str, n_clusters: int) -> Tuple[np.ndarray, object]:
        """Perform clustering with specified method."""
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X_scaled)
            
        elif method == 'dbscan':
            # Use adaptive DBSCAN parameters
            n_features = X_scaled.shape[1]
            min_samples = max(n_features, 5)
            
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            eps = np.percentile(distances[:, -1], 90)
            
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(X_scaled)
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(X_scaled)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels, clusterer
    
    def _analyze_cluster_quality(self, X_scaled: np.ndarray, labels: np.ndarray, 
                                features_df: pl.DataFrame) -> Dict:
        """Analyze cluster quality and detect imbalances."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_samples = len(labels)
        
        # Calculate cluster sizes
        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
        cluster_percentages = {label: size / n_samples * 100 for label, size in cluster_sizes.items()}
        
        # Detect severe imbalance (one cluster > 80% of data)
        max_cluster_pct = max(cluster_percentages.values())
        is_severely_imbalanced = max_cluster_pct > 80
        
        # Calculate quality metrics
        try:
            silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0
        except:
            silhouette = 0
            
        try:
            davies_bouldin = davies_bouldin_score(X_scaled, labels) if n_clusters > 1 else float('inf')
        except:
            davies_bouldin = float('inf')
        
        # Analyze what's causing imbalance
        imbalance_analysis = self._analyze_imbalance_causes(features_df, labels)
        
        return {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'cluster_percentages': cluster_percentages,
            'max_cluster_percentage': max_cluster_pct,
            'is_severely_imbalanced': is_severely_imbalanced,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'imbalance_analysis': imbalance_analysis
        }
    
    def _analyze_imbalance_causes(self, features_df: pl.DataFrame, labels: np.ndarray) -> Dict:
        """Analyze what's causing cluster imbalance."""
        # Add cluster labels to features
        features_with_clusters = features_df.with_columns(pl.Series('cluster', labels))
        
        # Analyze death rates by cluster
        if 'is_dead' in features_df.columns:
            death_by_cluster = features_with_clusters.group_by('cluster').agg([
                pl.col('is_dead').mean().alias('death_rate'),
                pl.count().alias('size')
            ])
            
            death_analysis = {
                row['cluster']: {
                    'death_rate': row['death_rate'],
                    'size': row['size']
                }
                for row in death_by_cluster.iter_rows(named=True)
            }
        else:
            death_analysis = {}
        
        # Find most discriminative features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['token', 'category', 'lifespan_category']]
        
        feature_discrimination = {}
        for feature in feature_cols[:5]:  # Top 5 features
            try:
                cluster_means = features_with_clusters.group_by('cluster').agg(
                    pl.col(feature).mean().alias('mean')
                )
                cluster_means_dict = {row['cluster']: row['mean'] for row in cluster_means.iter_rows(named=True)}
                feature_discrimination[feature] = cluster_means_dict
            except:
                continue
        
        return {
            'death_analysis': death_analysis,
            'feature_discrimination': feature_discrimination
        }
    
    def _create_minimal_clustering_result(self, n_tokens: int) -> Dict:
        """Create minimal clustering result for edge cases."""
        return {
            'labels': np.zeros(n_tokens, dtype=int),
            'clusterer': None,
            'n_clusters': 1,
            'feature_cols': [],
            'X_scaled': np.array([]),
            'quality_metrics': {
                'n_clusters': 1,
                'cluster_sizes': {0: n_tokens},
                'cluster_percentages': {0: 100.0},
                'max_cluster_percentage': 100.0,
                'is_severely_imbalanced': True,
                'silhouette_score': 0,
                'davies_bouldin_score': float('inf'),
                'imbalance_analysis': {'warning': 'Insufficient data variance for clustering'}
            }
        }
    
    def _apply_category_sampling(self, categorized_tokens: Dict[str, Dict], 
                                max_tokens_per_category: Optional[int],
                                sample_ratio: Optional[float]) -> Dict[str, Dict]:
        """Apply sampling to categorized tokens."""
        import random
        
        sampled_categories = {}
        
        for category_name, category_tokens in categorized_tokens.items():
            if len(category_tokens) == 0:
                sampled_categories[category_name] = {}
                continue
            
            # Apply category limit
            if max_tokens_per_category and len(category_tokens) > max_tokens_per_category:
                token_names = list(category_tokens.keys())
                sampled_names = random.sample(token_names, max_tokens_per_category)
                category_tokens = {name: category_tokens[name] for name in sampled_names}
            
            # Apply sampling ratio
            if sample_ratio and 0 < sample_ratio < 1:
                sample_size = max(1, int(len(category_tokens) * sample_ratio))
                sample_size = min(sample_size, len(category_tokens))
                token_names = list(category_tokens.keys())
                sampled_names = random.sample(token_names, sample_size)
                category_tokens = {name: category_tokens[name] for name in sampled_names}
            
            sampled_categories[category_name] = category_tokens
            print(f"DEBUG: {category_name}: {len(category_tokens)} tokens after sampling")
        
        return sampled_categories
    
    def _generate_lifespan_results(self, token_data: Dict, features_df: pl.DataFrame, 
                                  clustering_results: Dict) -> Dict:
        """Generate results for lifespan analysis."""
        # Group results by lifespan category
        features_with_clusters = features_df.with_columns(pl.Series('cluster', clustering_results['labels']))
        
        # Validate that lifespan_category column exists
        if 'lifespan_category' not in features_with_clusters.columns:
            print("WARNING: 'lifespan_category' column missing from features. This indicates an issue in data loading.")
            print(f"Available columns: {features_with_clusters.columns}")
            # Return minimal results structure
            return {
                'analysis_type': 'lifespan',
                'categories': {},
                'clustering_results': clustering_results,
                'total_tokens_analyzed': features_df.height,
                'quality_metrics': clustering_results['quality_metrics'],
                'error': 'Missing lifespan_category column'
            }
        
        category_results = {}
        for category in ['Sprint', 'Standard', 'Marathon']:
            try:
                category_tokens = features_with_clusters.filter(
                    pl.col('lifespan_category') == category
                )
                
                if category_tokens.height > 0:
                    category_results[category] = {
                        'n_tokens': category_tokens.height,
                        'cluster_distribution': category_tokens.group_by('cluster').count().to_dicts(),
                        'features': category_tokens
                    }
                else:
                    print(f"DEBUG: No tokens found for category: {category}")
            except Exception as e:
                print(f"ERROR: Failed to process category {category}: {e}")
                continue
        
        # Extract backward compatibility data for lifespan analysis
        token_names = features_df['token'].to_list() if 'token' in features_df.columns else []
        cluster_labels = clustering_results['labels']
        
        return {
            # New unified structure
            'analysis_type': 'lifespan',
            'categories': category_results,
            'clustering_results': clustering_results,
            'total_tokens_analyzed': features_df.height,
            'quality_metrics': clustering_results['quality_metrics'],
            
            # Legacy compatibility keys for display functions
            'token_names': token_names,
            'cluster_labels': cluster_labels,
            'n_clusters': clustering_results['n_clusters'],
            'token_data': token_data,  # Original token DataFrames
            'analysis_method': 'lifespan_analysis'
        }
    
    def _generate_behavioral_results(self, token_data: Dict, features_df: pl.DataFrame, 
                                    clustering_results: Dict) -> Dict:
        """Generate results for behavioral archetype analysis."""
        # Use the behavioral analyzer's archetype identification
        features_with_clusters = features_df.with_columns(pl.Series('cluster', clustering_results['labels']))
        
        # Create archetype analysis
        archetypes = self._identify_behavioral_archetypes(features_with_clusters)
        
        # Extract backward compatibility data
        token_names = features_df['token'].to_list() if 'token' in features_df.columns else []
        cluster_labels = clustering_results['labels']
        
        return {
            # New unified structure
            'analysis_type': 'behavioral',
            'features_df': features_with_clusters,
            'clustering_results': clustering_results,
            'archetypes': archetypes,
            'total_tokens_analyzed': features_df.height,
            'quality_metrics': clustering_results['quality_metrics'],
            
            # Legacy compatibility keys for display functions
            'token_names': token_names,
            'cluster_labels': cluster_labels,
            'n_clusters': clustering_results['n_clusters'],
            'token_data': token_data,  # Original token DataFrames
            'analysis_method': 'behavioral_analysis'
        }
    
    def _generate_price_only_results(self, token_data: Dict, features_df: pl.DataFrame, 
                                    clustering_results: Dict) -> Dict:
        """Generate results for price-only analysis."""
        features_with_clusters = features_df.with_columns(pl.Series('cluster', clustering_results['labels']))
        
        # Extract backward compatibility data for price-only analysis
        token_names = features_df['token'].to_list() if 'token' in features_df.columns else []
        cluster_labels = clustering_results['labels']
        
        return {
            # New unified structure
            'analysis_type': 'price_only',
            'features_df': features_with_clusters,
            'clustering_results': clustering_results,
            'total_tokens_analyzed': features_df.height,
            'quality_metrics': clustering_results['quality_metrics'],
            
            # Legacy compatibility keys for display functions
            'token_names': token_names,
            'cluster_labels': cluster_labels,
            'n_clusters': clustering_results['n_clusters'],
            'token_data': token_data,  # Original token DataFrames
            'analysis_method': 'price_only_analysis'
        }
    
    def _identify_behavioral_archetypes(self, features_with_clusters: pl.DataFrame) -> Dict:
        """Identify behavioral archetypes from clustered features."""
        archetypes = {}
        
        for cluster_id in features_with_clusters['cluster'].unique():
            cluster_data = features_with_clusters.filter(pl.col('cluster') == cluster_id)
            
            # Calculate cluster statistics
            stats = {
                'n_tokens': cluster_data.height,
                'pct_of_total': cluster_data.height / features_with_clusters.height * 100,
            }
            
            # Add death rate if available
            if 'is_dead' in cluster_data.columns:
                stats['pct_dead'] = cluster_data.select(pl.col('is_dead').mean() * 100).item()
            
            # Simple archetype naming based on cluster characteristics
            archetype_name = self._determine_archetype_name(cluster_data, stats)
            
            archetypes[cluster_id] = {
                'name': archetype_name,
                'stats': stats
            }
        
        return archetypes
    
    def _determine_archetype_name(self, cluster_data: pl.DataFrame, stats: Dict) -> str:
        """Determine archetype name based on cluster characteristics."""
        pct_dead = stats.get('pct_dead', 0)
        n_tokens = stats['n_tokens']
        
        if pct_dead > 90:
            return f"Death Cluster ({n_tokens} tokens)"
        elif pct_dead > 50:
            return f"Mixed Behavior ({n_tokens} tokens)"
        else:
            return f"Survivor Cluster ({n_tokens} tokens)"