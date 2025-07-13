# behavioural_archetype_analysis.py
"""
Behavioral Archetype Analysis for Memecoin Tokens
Identifies 5-8 distinct behavioral patterns including death patterns
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Import utilities from archetype_utils
from .archetype_utils import (
    detect_token_death, calculate_death_features, extract_lifecycle_features,
    extract_early_features, prepare_token_data, extract_essential_features,
    generate_archetype_docs, categorize_by_lifespan
)


class BehavioralArchetypeAnalyzer:
    """
    Analyzes memecoin tokens to identify behavioral archetypes including death patterns.
    Focuses on early detection using only first 5 minutes of data.
    """
    
    def __init__(self):
        self.token_features = {}
        self.death_info = {}
        self.clusters = None
        self.archetype_names = None
        self.early_detection_model = None
        self.scaler = RobustScaler()  # Use RobustScaler for extreme outliers
        
    def load_categorized_tokens(self, processed_dir: Path, limit: Optional[int] = None, sample_ratio: Optional[float] = None) -> Dict[str, pl.DataFrame]:
        """
        Load tokens from all categories in processed directory.
        
        Args:
            processed_dir: Path to processed data directory
            limit: Optional limit on tokens per category
            sample_ratio: Optional sampling ratio (0.1 = 10% sample) for faster processing
            
        Returns:
            Dictionary mapping token names to DataFrames
        """
        token_data = {}
        categories = ['normal_behavior_tokens', 'tokens_with_extremes', 'dead_tokens', 'tokens_with_gaps']
        
        for category in categories:
            category_path = processed_dir / category
            if not category_path.exists():
                print(f"DEBUG: Category directory not found: {category_path}")
                continue
                
            parquet_files = list(category_path.glob("*.parquet"))
            if limit:
                parquet_files = parquet_files[:limit]
            
            print(f"DEBUG: Loading {len(parquet_files)} tokens from {category}...")
            
            for file_path in tqdm(parquet_files, desc=f"Loading {category}"):
                try:
                    df = pl.read_parquet(file_path)
                    token_name = file_path.stem
                    
                    # Ensure we have required columns
                    if 'datetime' in df.columns and 'price' in df.columns:
                        # Add category info
                        df = df.with_columns([
                            pl.lit(category).alias('category')
                        ])
                        token_data[token_name] = df
                        
                except Exception as e:
                    print(f"DEBUG: Error loading {file_path}: {e}")
                    continue
        
        print(f"DEBUG: Total tokens loaded: {len(token_data)}")
        
        # Apply stratified sampling if requested (for faster processing)
        if sample_ratio is not None and 0 < sample_ratio < 1:
            import random
            from collections import defaultdict
            
            total_tokens = len(token_data)
            target_sample_size = int(total_tokens * sample_ratio)
            
            print(f"DEBUG: Applying stratified sampling for {target_sample_size} tokens ({sample_ratio*100:.1f}%)...")
            
            # First, categorize all tokens by lifespan (quick death detection)
            lifespan_categories = defaultdict(list)
            
            print("DEBUG: Performing quick lifespan categorization for stratified sampling...")
            for token_name, df in token_data.items():
                prices = df['price'].to_numpy()
                returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
                
                death_minute = detect_token_death(prices, returns, min_death_duration=30)
                
                # Calculate active lifespan (before death or full length if alive)
                active_lifespan = death_minute if death_minute is not None else len(df)
                
                # Categorize by active lifespan
                if 0 <= active_lifespan <= 400:
                    lifespan_categories['Sprint'].append(token_name)
                elif 400 < active_lifespan <= 1200:
                    lifespan_categories['Standard'].append(token_name)
                elif active_lifespan > 1200:
                    lifespan_categories['Marathon'].append(token_name)
            
            # Show natural distribution
            print("DEBUG: Natural lifespan distribution:")
            total_categorized = sum(len(tokens) for tokens in lifespan_categories.values())
            for category, tokens in lifespan_categories.items():
                pct = len(tokens) / total_categorized * 100 if total_categorized > 0 else 0
                print(f"  {category}: {len(tokens)} tokens ({pct:.1f}%)")
            
            # Apply stratified sampling to maintain proportions
            sampled_token_data = {}
            
            for category, token_list in lifespan_categories.items():
                if len(token_list) == 0:
                    continue
                
                # Calculate sample size for this category (proportional)
                category_proportion = len(token_list) / total_categorized
                category_sample_size = max(1, int(target_sample_size * category_proportion))
                
                # Don't sample more than available
                category_sample_size = min(category_sample_size, len(token_list))
                
                # Random sample within category
                sampled_tokens = random.sample(token_list, category_sample_size)
                
                print(f"DEBUG:  Sampling {len(sampled_tokens)} tokens from {category}")
                
                # Add to sampled dataset
                for token_name in sampled_tokens:
                    sampled_token_data[token_name] = token_data[token_name]
            
            print(f"DEBUG: Stratified sampling complete: {len(sampled_token_data)} tokens")
            return sampled_token_data
        
        return token_data
    
    def extract_all_features(self, token_data: Dict[str, pl.DataFrame], use_log_returns: bool = False) -> pl.DataFrame:
        """
        Extract the 15 essential features for each token (CEO requirement).
        
        Args:
            token_data: Dictionary mapping token names to DataFrames (pre-loaded and possibly sampled)
            use_log_returns: Whether to use modified log returns (for A/B testing)
            
        Returns:
            DataFrame with 15 essential features for each token
        """
        print(f"DEBUG: Extracting 15 essential features for {len(token_data)} tokens...")
        
        # Use centralized feature extraction from utils with pre-loaded tokens
        features_df = extract_essential_features(token_data, use_log_returns=use_log_returns)
        
        # Print summary statistics
        self._print_data_summary(features_df)
        
        # Verify we have exactly 15 features (excluding metadata)
        feature_cols = [col for col in features_df.columns if col not in ['token', 'category']]
        print(f"DEBUG: Using exactly {len(feature_cols)} features (target: 15)")
        print(f"DEBUG: Features: {feature_cols}")
        
        return features_df
    
    def _print_data_summary(self, features_df: pl.DataFrame):
        """Print summary statistics about the data."""
        total_tokens = features_df.height
        dead_tokens = features_df['is_dead'].sum()
        alive_tokens = total_tokens - dead_tokens
        
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Total tokens: {total_tokens:,}")
        print(f"Dead tokens: {dead_tokens:,} ({dead_tokens/total_tokens*100:.1f}%)")
        print(f"Alive tokens: {alive_tokens:,} ({alive_tokens/total_tokens*100:.1f}%)")
        
        # Death type distribution (not in 15 essential features)
        if dead_tokens > 0:
            print("\nDead Token Analysis:")
            avg_lifespan = features_df.filter(pl.col('is_dead')).select(pl.col('lifespan_minutes').mean()).item()
            print(f"  Average lifespan: {avg_lifespan:.1f} minutes")
            
            # Show lifespan distribution
            short_death = features_df.filter(pl.col('is_dead') & (pl.col('lifespan_minutes') <= 400)).height
            medium_death = features_df.filter(pl.col('is_dead') & (pl.col('lifespan_minutes') > 400) & (pl.col('lifespan_minutes') <= 1200)).height
            long_death = features_df.filter(pl.col('is_dead') & (pl.col('lifespan_minutes') > 1200)).height
            
            print(f"  Short lifespan (≤400 min): {short_death} ({short_death/dead_tokens*100:.1f}%)")
            print(f"  Medium lifespan (400-1200 min): {medium_death} ({medium_death/dead_tokens*100:.1f}%)")
            print(f"  Long lifespan (>1200 min): {long_death} ({long_death/dead_tokens*100:.1f}%)")
        
        # Category distribution
        print("\nCategory Distribution:")
        category_counts = features_df.group_by('category').agg(pl.count().alias('count'))
        for row in category_counts.iter_rows(named=True):
            cat = row['category']
            count = row['count']
            print(f"  {cat}: {count} ({count/total_tokens*100:.1f}%)")
    
    def perform_clustering(self, features_df: pl.DataFrame, n_clusters_range=[3, 4, 5, 6, 7, 8, 9, 10]) -> Dict:
        """
        Perform clustering using elbow method for K selection (NO PCA).
        
        Args:
            features_df: DataFrame with extracted features
            n_clusters_range: List of cluster numbers to try
            
        Returns:
            Dictionary with clustering results
        """
        # Use centralized clustering from autocorrelation_clustering
        from .autocorrelation_clustering import AutocorrelationClusteringAnalyzer
        analyzer = AutocorrelationClusteringAnalyzer()
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['token', 'category', 'lifespan_category']]
        X = features_df.select(feature_cols).to_numpy()
        
        print(f"DEBUG: Clustering with {len(feature_cols)} features: {feature_cols}")
        
        # Handle NaN values with median imputation
        if np.any(np.isnan(X)):
            print("DEBUG: NaN values detected in features, applying median imputation...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("DEBUG: NaN values have been imputed using median strategy")
        
        # Check feature variance BEFORE scaling
        feature_variances = np.var(X, axis=0)
        zero_variance_mask = feature_variances < 1e-10
        valid_feature_mask = ~zero_variance_mask
        
        print(f"DEBUG: Feature variance analysis:")
        print(f"  Zero variance features: {np.sum(zero_variance_mask)}/{len(feature_cols)}")
        print(f"  Valid features for clustering: {np.sum(valid_feature_mask)}")
        
        if np.sum(valid_feature_mask) < 2:
            print("DEBUG: WARNING - Insufficient features with variance for meaningful clustering!")
            print("DEBUG: This indicates tokens are too similar (likely all dead with same characteristics)")
            # Return minimal clustering result
            n_tokens = len(features_df)
            return {
                'X_scaled': X,
                'feature_cols': feature_cols,
                'elbow_results': {'optimal_k_elbow': 2, 'optimal_k_silhouette': 2},
                'kmeans': {2: {
                    'model': None,
                    'labels': np.zeros(n_tokens, dtype=int),  # All tokens in one cluster
                    'silhouette_score': 0.0,
                    'davies_bouldin_score': 0.0
                }},
                'dbscan': None,
                'hierarchical': {},
                'best_k': 2,
                'warning': 'Insufficient feature variance for meaningful clustering'
            }
        
        # Filter features to only those with variance for clustering
        if np.sum(zero_variance_mask) > 0:
            print(f"DEBUG: Removing {np.sum(zero_variance_mask)} zero-variance features for clustering")
            X_filtered = X[:, valid_feature_mask]
            filtered_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_feature_mask[i]]
            print(f"DEBUG: Using {len(filtered_feature_cols)} features: {filtered_feature_cols}")
        else:
            X_filtered = X
            filtered_feature_cols = feature_cols
        
        # Standardize features (NO PCA - CEO requirement)
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        # Find optimal K
        print("\nDEBUG: Finding optimal K using elbow method...")
        elbow_results = analyzer.find_optimal_clusters(X_scaled, max_clusters=n_clusters_range[-1])
        
        # Use elbow method result (CEO requirement)
        best_k = elbow_results['optimal_k_elbow']
        
        # Store results
        clustering_results = {
            'X_scaled': X_scaled,
            'feature_cols': filtered_feature_cols,  # Use filtered features
            'original_feature_cols': feature_cols,   # Keep original for reference
            'elbow_results': elbow_results,
            'kmeans': {},
            'dbscan': None,
            'hierarchical': {},
            'best_k': best_k
        }
        
        # K-means clustering with best K
        print(f"\nDEBUG: Performing K-means clustering with K={best_k}...")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        
        clustering_results['kmeans'][best_k] = {
            'model': kmeans,
            'labels': labels,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score
        }
        
        print(f"DEBUG:  K={best_k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
        
        # DBSCAN for outlier detection with adaptive parameters
        print("\nDEBUG: Performing DBSCAN clustering with adaptive parameters...")
        
        # Calculate appropriate eps using k-distance graph method
        from sklearn.neighbors import NearestNeighbors
        n_features = X_scaled.shape[1]
        min_samples = max(n_features, 5)  # At least number of dimensions
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        # Sort distances to k-th nearest neighbor
        k_distances = distances[:, -1]
        k_distances_sorted = np.sort(k_distances)
        
        # Find elbow point (use 90th percentile for robustness)
        eps = np.percentile(k_distances_sorted, 90)
        
        print(f"DEBUG: DBSCAN parameters - eps={eps:.3f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = (dbscan_labels == -1).sum()
        
        clustering_results['dbscan'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            'n_clusters': n_clusters_dbscan,
            'n_outliers': n_outliers
        }
        print(f"DEBUG:  DBSCAN found {n_clusters_dbscan} clusters and {n_outliers} outliers")
        
        # t-SNE for visualization (use scaled data, not PCA)
        print("\nDEBUG: Performing t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        clustering_results['X_tsne'] = X_tsne
        
        return clustering_results
    
    def test_clustering_stability(self, features_df: pl.DataFrame, optimal_k: int, n_runs: int = 10) -> Dict:
        """
        Test clustering stability using the AutocorrelationClusteringAnalyzer method.
        
        Args:
            features_df: DataFrame with features
            optimal_k: Number of clusters to test
            n_runs: Number of stability runs
            
        Returns:
            Dictionary with stability results
        """
        from .autocorrelation_clustering import AutocorrelationClusteringAnalyzer
        analyzer = AutocorrelationClusteringAnalyzer()
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['token', 'category', 'lifespan_category']]
        X = features_df.select(feature_cols).to_numpy()
        
        # Handle NaN values with median imputation
        if np.any(np.isnan(X)):
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Delegate to the AutocorrelationClusteringAnalyzer
        return analyzer.test_clustering_stability(X_scaled, optimal_k, n_runs)
    
    def identify_archetypes(self, features_df: pl.DataFrame, clustering_results) -> Dict:
        """
        Identify and name behavioral archetypes based on clustering results.
        
        Args:
            features_df: DataFrame with features
            clustering_results: Results from clustering analysis
            
        Returns:
            Dictionary with archetype information
        """
        # Use best K-means result
        best_k = clustering_results['best_k']
        labels = clustering_results['kmeans'][best_k]['labels']
        
        # Add cluster labels to features
        features_df = features_df.with_columns(pl.Series('cluster', labels))
        
        # Analyze each cluster
        archetypes = {}
        
        print("\n" + "="*60)
        print("BEHAVIORAL ARCHETYPE ANALYSIS")
        print("="*60)
        
        for cluster_id in range(best_k):
            cluster_data = features_df.filter(pl.col('cluster') == cluster_id)
            n_tokens = cluster_data.height
            
            # Calculate cluster statistics using the 15 features
            stats = {
                'n_tokens': n_tokens,
                'pct_of_total': n_tokens / features_df.height * 100,
                'pct_dead': cluster_data.select(pl.col('is_dead').mean() * 100).item(),
                'avg_lifespan': cluster_data.select(pl.col('lifespan_minutes').mean()).item(),
                'avg_return_magnitude': cluster_data.select(pl.col('return_5min').mean()).item(),
                'avg_volatility_5min': cluster_data.select(pl.col('volatility_5min').mean()).item(),
                'avg_mean_return': cluster_data.select(pl.col('mean_return').mean()).item(),
                'avg_std_return': cluster_data.select(pl.col('std_return').mean()).item(),
                'avg_max_drawdown': cluster_data.select(pl.col('max_drawdown').mean()).item(),
                'avg_price_change_5min': cluster_data.select(pl.col('price_change_ratio_5min').mean()).item()
            }
            
            # Determine archetype name based on characteristics
            archetype_name = self._determine_archetype_name(cluster_data, stats)
            
            # Get representative examples
            examples = cluster_data.sort('lifespan_minutes', descending=True).head(10)['token'].to_list()
            
            # Extract ACF signature (average ACF values)
            acf_cols = [col for col in cluster_data.columns if col.startswith('acf_lag_')]
            if acf_cols:
                acf_signature = cluster_data.select(pl.mean(acf_cols)).row(0, named=True)
            else:
                acf_signature = {}
            
            archetypes[cluster_id] = {
                'name': archetype_name,
                'stats': stats,
                'examples': examples,
                'acf_signature': acf_signature
            }
            
            # Print summary
            print(f"\nCluster {cluster_id}: {archetype_name}")
            print(f"  Tokens: {n_tokens} ({stats['pct_of_total']:.1f}%)")
            print(f"  Dead: {stats['pct_dead']:.1f}%")
            print(f"  Avg lifespan: {stats['avg_lifespan']:.1f} minutes")
            print(f"  Avg max return (5min): {stats['avg_return_magnitude']*100:.1f}%")
            print(f"  Example tokens: {', '.join(examples[:3])}...")
        
        self.archetype_names = {i: arch['name'] for i, arch in archetypes.items()}
        return archetypes
    
    def _determine_archetype_name(self, cluster_data: pl.DataFrame, stats) -> str:
        """Determine archetype name based on cluster characteristics."""
        pct_dead = stats['pct_dead']
        avg_lifespan = stats['avg_lifespan']
        avg_max_return = stats['avg_return_magnitude']
        avg_volatility = stats['avg_volatility_5min']
        
        # Death-based patterns
        if pct_dead > 90:
            if avg_lifespan < 60:
                if avg_max_return > 0.5:
                    return "Quick Pump & Death"
                else:
                    return "Dead on Arrival"
            elif avg_lifespan < 360:
                return "Slow Bleed"
            else:
                return "Extended Decline"
        
        # Mixed patterns
        elif pct_dead > 50:
            if avg_volatility > 0.1:
                return "Phoenix Attempt"
            else:
                return "Zombie Walker"
        
        # Survivor patterns
        else:
            if avg_max_return > 0.3 and avg_volatility > 0.1:
                return "Survivor Pump"
            elif avg_volatility < 0.05:
                return "Stable Survivor"
            else:
                return "Survivor Organic"
    
    def create_early_detection_rules(self, features_df: pl.DataFrame, archetypes) -> DecisionTreeClassifier:
        """
        Create classification rules using only first 5 minutes of data.
        
        Args:
            features_df: DataFrame with features
            archetypes: Archetype information
            
        Returns:
            Trained decision tree classifier
        """
        # Select only early features (5-minute)
        early_feature_cols = [col for col in features_df.columns if '5min' in col]
        
        if not early_feature_cols:
            print("DEBUG: Warning: No early features found for classification")
            return None
        
        X = features_df.select(early_feature_cols).fill_null(0).to_numpy()
        y = features_df['cluster'].to_numpy()
        
        # Train decision tree for interpretable rules
        print("\nDEBUG: Training early detection classifier...")
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
        clf.fit(X, y)
        
        # Print feature importance
        feature_importance = sorted(zip(early_feature_cols, clf.feature_importances_), 
                                  key=lambda x: x[1], reverse=True)
        
        print("\nDEBUG: Top 5 most important early detection features:")
        for feat, imp in feature_importance[:5]:
            print(f"  {feat}: {imp:.3f}")
        
        # Calculate accuracy (using same data for simplicity - in production use CV)
        accuracy = clf.score(X, y)
        print(f"\nDEBUG: Early detection accuracy: {accuracy:.1%}")
        
        self.early_detection_model = clf
        return clf
    
    def save_results(self, features_df: pl.DataFrame, clustering_results, archetypes, output_dir: Path):
        """Save analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if cluster column exists, if not add it
        if 'cluster' not in features_df.columns:
            print("DEBUG: Warning: 'cluster' column missing, adding from clustering results...")
            best_k = clustering_results['best_k']
            labels = clustering_results['kmeans'][best_k]['labels']
            features_df = features_df.with_columns(pl.Series('cluster', labels))
        
        # Save archetype assignments
        required_columns = ['token', 'category', 'cluster', 'is_dead', 'lifespan_minutes']
        available_columns = [col for col in required_columns if col in features_df.columns]
        
        if len(available_columns) < len(required_columns):
            missing_columns = set(required_columns) - set(available_columns)
            print(f"DEBUG: Warning: Missing columns in features_df: {missing_columns}")
            # Add default values for missing columns
            for col in missing_columns:
                if col == 'cluster':
                    best_k = clustering_results['best_k']
                    labels = clustering_results['kmeans'][best_k]['labels']
                    features_df = features_df.with_columns(pl.Series('cluster', labels))
                elif col == 'category':
                    features_df = features_df.with_columns(pl.lit('unknown').alias('category'))
                elif col == 'is_dead':
                    features_df = features_df.with_columns(pl.lit(False).alias('is_dead'))
                elif col == 'lifespan_minutes':
                    features_df = features_df.with_columns(pl.lit(1440).alias('lifespan_minutes'))
            available_columns = required_columns
        
        assignments = features_df.select(available_columns)
        assignments = assignments.with_columns(
            pl.col('cluster').map_elements(lambda x: self.archetype_names.get(x, str(x)), return_dtype=pl.Utf8).alias('archetype')
        )
        assignments.write_csv(output_dir / f"archetype_assignments_{timestamp}.csv")
        
        # Save archetype statistics
        archetype_stats = []
        for cluster_id, arch in archetypes.items():
            stats = arch['stats'].copy()
            stats['cluster_id'] = cluster_id
            stats['archetype_name'] = arch['name']
            archetype_stats.append(stats)
        
        pl.DataFrame(archetype_stats).write_csv(
            output_dir / f"archetype_statistics_{timestamp}.csv"
        )
        
        # Save comprehensive report
        report = {
            'timestamp': timestamp,
            'n_tokens': features_df.height,
            'n_archetypes': len(archetypes),
            'pct_dead': features_df.select(pl.col('is_dead').mean() * 100).item(),
            'archetypes': archetypes,
            'clustering_metrics': {
                'best_k': clustering_results['best_k'],
                'silhouette_score': clustering_results['kmeans'][clustering_results['best_k']]['silhouette_score'],
                'dbscan_outliers': clustering_results['dbscan']['n_outliers']
            }
        }
        
        with open(output_dir / f"archetype_analysis_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDEBUG: Results saved to {output_dir}")
        
        return timestamp