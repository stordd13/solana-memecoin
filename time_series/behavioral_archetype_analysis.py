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
from sklearn.impute import SimpleImputer  # Added for NaN handling
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from .archetype_utils import (
    detect_token_death, calculate_death_features, extract_lifecycle_features,
    extract_early_features, prepare_token_data
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
                print(f"Warning: Category directory not found: {category_path}")
                continue
                
            parquet_files = list(category_path.glob("*.parquet"))
            if limit:
                parquet_files = parquet_files[:limit]
            
            print(f"\nLoading {len(parquet_files)} tokens from {category}...")
            
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
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        print(f"\nTotal tokens loaded: {len(token_data)}")
        
        # Apply stratified sampling if requested (for faster processing)
        if sample_ratio is not None and 0 < sample_ratio < 1:
            import random
            from collections import defaultdict
            
            total_tokens = len(token_data)
            target_sample_size = int(total_tokens * sample_ratio)
            
            print(f"Applying stratified sampling for {target_sample_size} tokens ({sample_ratio*100:.1f}%)...")
            
            # First, categorize all tokens by lifespan (quick death detection)
            lifespan_categories = defaultdict(list)
            
            print("Performing quick lifespan categorization for stratified sampling...")
            for token_name, df in token_data.items():
                prices = df['price'].to_numpy()
                returns = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
                
                # Import death detection from archetype_utils
                from time_series.archetype_utils import detect_token_death
                death_minute = detect_token_death(prices, returns, window=30)
                
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
            print("Natural lifespan distribution:")
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
                
                print(f"  Sampling {len(sampled_tokens)} tokens from {category}")
                
                # Add to sampled dataset
                for token_name in sampled_tokens:
                    sampled_token_data[token_name] = token_data[token_name]
            
            print(f"Stratified sampling complete: {len(sampled_token_data)} tokens")
            return sampled_token_data
        
        return token_data
    
    def extract_all_features(self, token_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Extract ONLY the 15 essential features for each token (CEO requirement).
        
        Args:
            token_data: Dictionary mapping token names to DataFrames
            
        Returns:
            DataFrame with only 15 essential features for each token
        """
        all_features = []
        
        print("\nExtracting 15 essential features for all tokens...")
        for token_name, df in tqdm(token_data.items(), desc="Processing tokens"):
            try:
                # Prepare data
                prices, returns, death_minute = prepare_token_data(df)
                
                # Store death info
                self.death_info[token_name] = death_minute
                
                # Extract all features first
                features = {'token': token_name, 'category': df['category'][0]}
                
                # Death features
                death_features = calculate_death_features(prices, returns, death_minute)
                features.update(death_features)
                
                # Lifecycle features (pre-death if applicable)
                lifecycle_features = extract_lifecycle_features(prices, returns, death_minute)
                features.update(lifecycle_features)
                
                # Early detection features (first 5 minutes)
                early_features = extract_early_features(prices, returns, window_minutes=5)
                features.update(early_features)
                
                # SELECT ONLY THE 15 ESSENTIAL FEATURES (CEO requirement)
                essential_features = {
                    'token': token_name,
                    'category': df['category'][0],
                    
                    # Death (3)
                    'is_dead': features.get('is_dead', False),
                    'death_minute': features.get('death_minute', None),
                    'lifespan_minutes': features.get('lifespan_minutes', len(prices)),
                    
                    # Core stats (4)
                    'mean_return': features.get('mean_return', 0),
                    'std_return': features.get('std_return', 0),
                    'volatility_5min': features.get('volatility_5min', 0),
                    'max_drawdown': features.get('max_drawdown', 0),
                    
                    # ACF (3)
                    'acf_lag_1': features.get('acf_lag_1', 0),
                    'acf_lag_5': features.get('acf_lag_5', 0),
                    'acf_lag_10': features.get('acf_lag_10', 0),
                    
                    # Early detection (5)
                    'return_magnitude_5min': features.get('return_magnitude_5min', 0),
                    'trend_direction_5min': features.get('trend_direction_5min', 0),
                    'price_change_ratio_5min': features.get('price_change_ratio_5min', 0),
                    'autocorrelation_5min': features.get('autocorrelation_5min', 0),
                    # Note: volatility_5min already included in core stats
                }
                
                # Store raw data reference
                self.token_features[token_name] = {
                    'prices': prices,
                    'returns': returns,
                    'death_minute': death_minute,
                    'features': essential_features
                }
                
                all_features.append(essential_features)
                
            except Exception as e:
                print(f"Error processing {token_name}: {e}")
                continue
        
        # Convert to Polars DataFrame
        features_df = pl.DataFrame(all_features)
        
        # Print summary statistics
        self._print_data_summary(features_df)
        
        # Verify we have exactly 15 features (excluding metadata)
        feature_cols = [col for col in features_df.columns if col not in ['token', 'category']]
        print(f"\n✅ Using exactly {len(feature_cols)} features (target: 15)")
        print(f"Features: {feature_cols}")
        
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
    
    def _find_optimal_k_elbow(self, X_scaled: np.ndarray, k_range: range) -> Dict:
        """
        Find optimal K using elbow method (from autocorrelation_clustering.py).
        
        Args:
            X_scaled: Scaled feature matrix
            k_range: Range of K values to test
            
        Returns:
            Dictionary with elbow analysis results
        """
        inertias = []
        silhouette_scores = []
        
        print(f"Testing K values: {list(k_range)}")
        
        # Check for NaN values before K-means
        if np.any(np.isnan(X_scaled)):
            print("Warning: NaN values detected in scaled features, applying median imputation...")
            imputer = SimpleImputer(strategy='median')
            X_scaled = imputer.fit_transform(X_scaled)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
            
                inertias.append(kmeans.inertia_)
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
                
                print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
            except Exception as e:
                print(f"  K={k}: Failed - {e}")
                # Use fallback values if clustering fails
                inertias.append(np.inf)
                silhouette_scores.append(0.0)
        
        # Find elbow using distance method (from autocorrelation_clustering.py)
        optimal_k_elbow = 3  # Default
        
        if len(inertias) >= 3 and len(k_range) > 1:
            # Normalize the data to [0,1] range
            k_norm = np.array(list(k_range)) - min(k_range)
            # Safety check for empty arrays
            if len(k_norm) > 0 and max(k_norm) > 0:
                k_norm = k_norm / max(k_norm)
            
            inertias_norm = np.array(inertias) - min(inertias)
            # Safety check for empty arrays and avoid division by zero
            if len(inertias_norm) > 0 and max(inertias_norm) > 0:
                inertias_norm = inertias_norm / max(inertias_norm)
            
            # Calculate distance from each point to line connecting first and last points
            distances = []
            for i in range(len(k_norm)):
                x1, y1 = k_norm[0], inertias_norm[0]
                x2, y2 = k_norm[-1], inertias_norm[-1]
                x0, y0 = k_norm[i], inertias_norm[i]
                
                # Distance formula: |ax0 + by0 + c| / sqrt(a^2 + b^2)
                if x2 != x1:
                    a = y2 - y1
                    b = -(x2 - x1)
                    c = (x2 - x1) * y1 - (y2 - y1) * x1
                    distance = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
                else:
                    distance = abs(x0 - x1)
                distances.append(distance)
            
            # Find the point with maximum distance (the elbow)
            if len(distances) > 0:
                elbow_idx = np.argmax(distances)
                k_range_list = list(k_range)
                if elbow_idx < len(k_range_list):
                    optimal_k_elbow = k_range_list[elbow_idx]
        
        # Find optimal K using silhouette score
        if len(silhouette_scores) > 0:
            k_range_list = list(k_range)
            sil_idx = np.argmax(silhouette_scores)
            if sil_idx < len(k_range_list):
                optimal_k_silhouette = k_range_list[sil_idx]
        else:
            optimal_k_silhouette = optimal_k_elbow
        
        print(f"Elbow method suggests K={optimal_k_elbow}")
        print(f"Silhouette method suggests K={optimal_k_silhouette}")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette
        }

    def perform_clustering(self, features_df: pl.DataFrame, n_clusters_range=[3, 4, 5, 6, 7, 8, 9, 10]) -> Dict:
        """
        Perform clustering using elbow method for K selection (NO PCA).
        
        Args:
            features_df: DataFrame with extracted features
            n_clusters_range: List of cluster numbers to try
            
        Returns:
            Dictionary with clustering results
        """
        # Select features for clustering (exclude metadata)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['token', 'category', 'lifespan_category']]
        X = features_df.select(feature_cols).to_numpy()
        
        print(f"Clustering with {len(feature_cols)} features: {feature_cols}")
        
        # Handle NaN values with median imputation (same pattern as run_kmeans_clustering)
        if np.any(np.isnan(X)):
            print("Warning: NaN values detected in features, applying median imputation...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            print("NaN values have been imputed using median strategy")
        
        # Standardize features (NO PCA - CEO requirement)
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal K using elbow method
        print("\nFinding optimal K using elbow method...")
        elbow_results = self._find_optimal_k_elbow(X_scaled, range(n_clusters_range[0], n_clusters_range[-1]+1))
        
        # Use elbow method result (CEO requirement)
        best_k = elbow_results['optimal_k_elbow']
        
        # Store results
        clustering_results = {
            'X_scaled': X_scaled,
            'feature_cols': feature_cols,
            'elbow_results': elbow_results,
            'kmeans': {},
            'dbscan': None,
            'hierarchical': {},
            'best_k': best_k
        }
        
        # K-means clustering with best K
        print(f"\nPerforming K-means clustering with K={best_k}...")
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
        
        print(f"  K={best_k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
        
        # DBSCAN for outlier detection
        print("\nPerforming DBSCAN clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = (dbscan_labels == -1).sum()
        
        clustering_results['dbscan'] = {
            'model': dbscan,
            'labels': dbscan_labels,
            'n_clusters': n_clusters_dbscan,
            'n_outliers': n_outliers
        }
        print(f"  DBSCAN found {n_clusters_dbscan} clusters and {n_outliers} outliers")
        
        # t-SNE for visualization (use scaled data, not PCA)
        print("\nPerforming t-SNE for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        clustering_results['X_tsne'] = X_tsne
        
        return clustering_results
    
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
            
            # Calculate cluster statistics using only the 15 features we have
            stats = {
                'n_tokens': n_tokens,
                'pct_of_total': n_tokens / features_df.height * 100,
                'pct_dead': cluster_data.select(pl.col('is_dead').mean() * 100).item(),
                'avg_lifespan': cluster_data.select(pl.col('lifespan_minutes').mean()).item(),
                'avg_return_magnitude': cluster_data.select(pl.col('return_magnitude_5min').mean()).item(),
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
            print("Warning: No early features found for classification")
            return None
        
        X = features_df.select(early_feature_cols).fill_null(0).to_numpy()
        y = features_df['cluster'].to_numpy()
        
        # Train decision tree for interpretable rules
        print("\nTraining early detection classifier...")
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
        clf.fit(X, y)
        
        # Print feature importance
        feature_importance = sorted(zip(early_feature_cols, clf.feature_importances_), 
                                  key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 most important early detection features:")
        for feat, imp in feature_importance[:5]:
            print(f"  {feat}: {imp:.3f}")
        
        # Calculate accuracy (using same data for simplicity - in production use CV)
        accuracy = clf.score(X, y)
        print(f"\nEarly detection accuracy: {accuracy:.1%}")
        
        self.early_detection_model = clf
        return clf
    
    def save_results(self, features_df: pl.DataFrame, clustering_results, archetypes, output_dir: Path):
        """Save analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if cluster column exists, if not add it
        if 'cluster' not in features_df.columns:
            print("Warning: 'cluster' column missing, adding from clustering results...")
            best_k = clustering_results['best_k']
            labels = clustering_results['kmeans'][best_k]['labels']
            features_df = features_df.with_columns(pl.Series('cluster', labels))
        
        # Save archetype assignments
        required_columns = ['token', 'category', 'cluster', 'is_dead', 'lifespan_minutes']
        available_columns = [col for col in required_columns if col in features_df.columns]
        
        if len(available_columns) < len(required_columns):
            missing_columns = set(required_columns) - set(available_columns)
            print(f"Warning: Missing columns in features_df: {missing_columns}")
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
            pl.col('cluster').replace({k: v for k, v in enumerate(self.archetype_names.values())}).alias('archetype')
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
        
        print(f"\nResults saved to {output_dir}")
        
        return timestamp
    
    def extract_14_features(self, token_data):
        """
        Extract 14 essential features for baseline clustering (CEO requirements).
        """
        features_list = []
        
        for token_name, token_df in token_data.items():
            # Handle both pandas and polars DataFrames
            if hasattr(token_df, 'is_empty'):  # Polars DataFrame
                if token_df.is_empty():
                    continue
                df = token_df
            else:  # Pandas DataFrame
                if len(token_df) == 0:
                    continue
                df = pl.from_pandas(token_df)
            
            # Basic preparation
            prices = df['price'].to_numpy()
            
            # Safe returns calculation with epsilon to avoid division by zero
            epsilon = 1e-10
            if 'return' in df.columns:
                returns = df['return'].to_numpy()
            else:
                # Avoid division by zero by using maximum of price or epsilon
                safe_prices = np.maximum(prices[:-1], epsilon)
                returns = np.diff(prices) / safe_prices
            
            # Skip if insufficient data
            if len(returns) < 10:
                continue
            
            # Death detection
            from .archetype_utils import detect_token_death
            death_minute = detect_token_death(prices, returns)
            is_dead = death_minute is not None
            lifespan_minutes = death_minute if is_dead else len(prices)
            
            # Core statistics (4 features) with winsorization for extreme values
            # Winsorize full returns series to handle extreme values (clip top/bottom 1%)
            def winsorize_data(data, limits=(0.01, 0.01)):
                """Custom winsorization function to clip extreme values"""
                lower_percentile = limits[0] * 100
                upper_percentile = (1 - limits[1]) * 100
                lower_bound = np.percentile(data, lower_percentile)
                upper_bound = np.percentile(data, upper_percentile)
                return np.clip(data, lower_bound, upper_bound)
            
            returns_winsorized = winsorize_data(returns, limits=(0.01, 0.01))
            
            mean_return = np.mean(returns_winsorized)
            std_return = np.std(returns_winsorized)
            
            # Volatility calculation with winsorized returns
            early_returns_vol = returns_winsorized[:5] if len(returns_winsorized) >= 5 else returns_winsorized
            volatility_5min = np.std(early_returns_vol)
            
            # Safe max drawdown calculation
            max_price = np.max(prices)
            if max_price > epsilon:
                max_drawdown = np.max(np.maximum.accumulate(prices) - prices) / max_price
            else:
                max_drawdown = 0.0
            
            # ACF features (3 features) using winsorized returns for stability
            from statsmodels.tsa.stattools import acf
            try:
                # Use winsorized returns for ACF calculation to avoid numerical issues
                acf_values = acf(returns_winsorized, nlags=min(10, len(returns_winsorized)//2), fft=True, missing='drop')
                acf_lag_1 = acf_values[1] if len(acf_values) > 1 else 0
                acf_lag_5 = acf_values[5] if len(acf_values) > 5 else 0
                acf_lag_10 = acf_values[10] if len(acf_values) > 10 else 0
            except:
                acf_lag_1 = acf_lag_5 = acf_lag_10 = 0
            
            # Early detection features (4 features) with winsorization for extreme values
            early_returns = returns[:5] if len(returns) >= 5 else returns
            
            # Winsorize extreme returns to handle billion % gains (clip top/bottom 1%)
            early_returns_winsorized = winsorize_data(early_returns, limits=(0.01, 0.01))
            
            return_magnitude_5min = np.mean(np.abs(early_returns_winsorized))
            trend_direction_5min = np.mean(early_returns_winsorized)
            
            # Safe price change ratio calculation
            first_price = max(prices[0], epsilon)
            price_change_ratio_5min = (prices[min(4, len(prices)-1)] - prices[0]) / first_price
            
            # Early autocorrelation using winsorized returns
            try:
                early_acf = acf(early_returns_winsorized, nlags=min(3, len(early_returns_winsorized)//2), fft=True, missing='drop')
                autocorrelation_5min = early_acf[1] if len(early_acf) > 1 else 0
            except:
                autocorrelation_5min = 0
            
            # Compile features
            features = {
                'token': token_name,
                # Death features (3)
                'is_dead': int(is_dead),
                'death_minute': death_minute if is_dead else -1,
                'lifespan_minutes': lifespan_minutes,
                # Core stats (4)
                'mean_return': mean_return,
                'std_return': std_return,
                'volatility_5min': volatility_5min,
                'max_drawdown': max_drawdown,
                # ACF (3)
                'acf_lag_1': acf_lag_1,
                'acf_lag_5': acf_lag_5,
                'acf_lag_10': acf_lag_10,
                # Early detection (4)
                'return_magnitude_5min': return_magnitude_5min,
                'trend_direction_5min': trend_direction_5min,
                'price_change_ratio_5min': price_change_ratio_5min,
                'autocorrelation_5min': autocorrelation_5min
            }
            
            features_list.append(features)
        
        return pl.DataFrame(features_list)
    
    def find_optimal_k_elbow(self, features_df: pl.DataFrame, k_range):
        """
        Find optimal K using elbow method.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.impute import SimpleImputer
        
        # Validate input
        if features_df.height == 0:
            print("Warning: Empty features DataFrame")
            return k_range[0]
        
        # Prepare features (exclude non-numeric)
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        if not feature_cols:
            print("Warning: No feature columns found")
            return k_range[0]
            
        X = features_df.select(feature_cols).to_numpy()
        
        # Debug: Check original features (only if issues detected)
        has_nan = np.any(np.isnan(X))
        has_inf = np.any(np.isinf(X))
        if has_nan or has_inf:
            print(f"DEBUG find_optimal_k_elbow: X shape: {X.shape}, NaN: {has_nan}, Inf: {has_inf}")
        
        # Check if we have enough samples
        if X.shape[0] < k_range[0]:
            print(f"Warning: Not enough samples ({X.shape[0]}) for minimum K ({k_range[0]})")
            return min(2, X.shape[0]) if X.shape[0] > 1 else 1
        
        # Handle NaN/Inf values BEFORE scaling
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("DEBUG find_optimal_k_elbow: Applying NaN/Inf cleanup...")
            # First replace inf with large finite values
            X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)
            # Then apply median imputation for remaining NaN values
            if np.any(np.isnan(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            print(f"DEBUG find_optimal_k_elbow: After cleanup - NaN: {np.any(np.isnan(X))}, Inf: {np.any(np.isinf(X))}")
        
        # Scale features using RobustScaler for extreme outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Final safety check after scaling
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("DEBUG find_optimal_k_elbow: NaN/Inf detected after scaling, applying final cleanup...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
            X_scaled = np.clip(X_scaled, -5.0, 5.0)
            print(f"DEBUG find_optimal_k_elbow: After cleanup - NaN: {np.any(np.isnan(X_scaled))}, Inf: {np.any(np.isinf(X_scaled))}")
        
        # Calculate inertia for different K values
        inertias = []
        # Ensure k_range doesn't exceed number of samples
        max_k = min(k_range[1], X.shape[0] - 1)
        min_k = min(k_range[0], max_k)
        
        if max_k < min_k:
            print(f"Warning: Cannot cluster {X.shape[0]} samples with requested K range {k_range}")
            return min_k
        
        k_values = list(range(min_k, max_k + 1))
        
        for k in k_values:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            except Exception as e:
                print(f"Warning: K-means failed for k={k}: {e}")
                # Use previous inertia or a default value
                inertias.append(inertias[-1] if inertias else 0.0)
        
        # Find elbow using simple method
        # Calculate rate of change
        if len(inertias) < 3:
            return k_values[0]
        
        # Find point with maximum decrease in rate of change
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        
        # Find elbow as point where second derivative is maximum
        elbow_idx = np.argmax(diff_diffs) + 1
        optimal_k = k_values[elbow_idx]
        
        return optimal_k
    
    def run_kmeans_clustering(self, features_df: pl.DataFrame, k):
        """
        Run K-means clustering with given K.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import silhouette_score
        from sklearn.impute import SimpleImputer
        
        # Validate input
        if features_df.height == 0:
            print("Warning: Empty features DataFrame for clustering")
            return {'labels': [], 'silhouette_score': 0.0, 'k': k}
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        if not feature_cols:
            print("Warning: No feature columns found for clustering")
            return {'labels': [], 'silhouette_score': 0.0, 'k': k}
            
        X = features_df.select(feature_cols).to_numpy()
        
        # Debug: Check original features (only if issues detected)
        has_nan = np.any(np.isnan(X))
        has_inf = np.any(np.isinf(X))
        if has_nan or has_inf:
            print(f"DEBUG run_kmeans_clustering: X shape: {X.shape}, NaN: {has_nan}, Inf: {has_inf}")
        
        # Handle NaN/Inf values BEFORE scaling
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("DEBUG run_kmeans_clustering: Applying NaN/Inf cleanup...")
            # First replace inf with large finite values
            X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)
            # Then apply median imputation for remaining NaN values
            if np.any(np.isnan(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            print(f"DEBUG run_kmeans_clustering: After cleanup - NaN: {np.any(np.isnan(X))}, Inf: {np.any(np.isinf(X))}")
        else:
            print("DEBUG run_kmeans_clustering: No NaN/Inf values detected")
        
        # Validate cluster count
        if k > len(X):
            print(f"Warning: K ({k}) cannot be larger than number of samples ({len(X)})")
            k = min(k, len(X))
        
        if k < 1:
            k = 1
        
        # Scale features using RobustScaler for extreme outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Final safety check after scaling
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("DEBUG run_kmeans_clustering: NaN/Inf detected after scaling, applying final cleanup...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
            X_scaled = np.clip(X_scaled, -5.0, 5.0)
            print(f"DEBUG run_kmeans_clustering: After cleanup - NaN: {np.any(np.isnan(X_scaled))}, Inf: {np.any(np.isinf(X_scaled))}")
        
        # Run clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics (handle edge cases)
        try:
            # Silhouette score requires at least 2 clusters and at least 2 samples per cluster
            unique_labels = np.unique(labels)
            if len(unique_labels) >= 2 and len(X_scaled) >= 2:
                silhouette_avg = silhouette_score(X_scaled, labels)
            else:
                silhouette_avg = 0.0
                print(f"Warning: Cannot calculate silhouette score with {len(unique_labels)} clusters and {len(X_scaled)} samples")
        except Exception as e:
            print(f"Warning: Silhouette score calculation failed: {e}")
            silhouette_avg = 0.0
        
        return {
            'labels': labels,
            'kmeans_model': kmeans,
            'scaler': scaler,
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_,
            'k': k
        }
    
    def test_clustering_stability(self, features_df: pl.DataFrame, k, n_runs):
        """
        Test clustering stability using multiple runs.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import adjusted_rand_score
        from sklearn.impute import SimpleImputer
        
        # Validate input
        if features_df.height == 0:
            print("Warning: Empty features DataFrame for stability testing")
            return {'ari_scores': [], 'average_ari': 0.0, 'min_ari': 0.0, 'max_ari': 0.0, 'n_runs': 0}
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        if not feature_cols:
            print("Warning: No feature columns found for stability testing")
            return {'ari_scores': [], 'average_ari': 0.0, 'min_ari': 0.0, 'max_ari': 0.0, 'n_runs': 0}
            
        X = features_df.select(feature_cols).to_numpy()
        
        # Debug: Check original features (only if issues detected)
        has_nan = np.any(np.isnan(X))
        has_inf = np.any(np.isinf(X))
        if has_nan or has_inf:
            print(f"DEBUG test_clustering_stability: X shape: {X.shape}, NaN: {has_nan}, Inf: {has_inf}")
        
        # Handle NaN/Inf values BEFORE scaling
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("DEBUG test_clustering_stability: Applying NaN/Inf cleanup...")
            # First replace inf with large finite values
            X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)
            # Then apply median imputation for remaining NaN values
            if np.any(np.isnan(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            print(f"DEBUG test_clustering_stability: After cleanup - NaN: {np.any(np.isnan(X))}, Inf: {np.any(np.isinf(X))}")
        else:
            print("DEBUG test_clustering_stability: No NaN/Inf values detected")
        
        # Validate cluster count
        if k > len(X):
            print(f"Warning: K ({k}) cannot be larger than number of samples ({len(X)}) for stability testing")
            k = min(k, len(X))
        
        if k < 1:
            k = 1
        
        # Scale features using RobustScaler for extreme outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Final safety check after scaling
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("DEBUG test_clustering_stability: NaN/Inf detected after scaling, applying final cleanup...")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
            X_scaled = np.clip(X_scaled, -5.0, 5.0)
            print(f"DEBUG test_clustering_stability: After cleanup - NaN: {np.any(np.isnan(X_scaled))}, Inf: {np.any(np.isinf(X_scaled))}")
        
        # Run multiple times
        all_labels = []
        for i in range(n_runs):
            try:
                kmeans = KMeans(n_clusters=k, random_state=i, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                all_labels.append(labels)
            except Exception as e:
                print(f"DEBUG test_clustering_stability: K-means run {i} failed: {e}")
                # Add a dummy labels array to maintain consistent structure
                all_labels.append(np.zeros(X_scaled.shape[0], dtype=int))
        
        # Calculate ARI between all pairs
        ari_scores = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                try:
                    ari = adjusted_rand_score(all_labels[i], all_labels[j])
                    ari_scores.append(ari)
                except Exception as e:
                    print(f"Warning: ARI calculation failed for run {i} vs {j}: {e}")
        
        # Handle empty ari_scores
        if not ari_scores:
            return {
                'ari_scores': [],
                'average_ari': 0.0,
                'min_ari': 0.0,
                'max_ari': 0.0,
                'n_runs': n_runs,
                'all_labels': all_labels
            }
        
        return {
            'ari_scores': ari_scores,
            'average_ari': np.mean(ari_scores),
            'min_ari': np.min(ari_scores),
            'max_ari': np.max(ari_scores),
            'n_runs': n_runs,
            'all_labels': all_labels
        }