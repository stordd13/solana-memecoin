"""
Behavioral Archetype Analysis for Memecoin Tokens
Identifies 5-8 distinct behavioral patterns including death patterns
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
        self.scaler = StandardScaler()
        
    def load_categorized_tokens(self, processed_dir: Path, limit: Optional[int] = None) -> Dict[str, pl.DataFrame]:
        """
        Load tokens from all categories in processed directory.
        
        Args:
            processed_dir: Path to processed data directory
            limit: Optional limit on tokens per category
            
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
        return token_data
    
    def extract_all_features(self, token_data: Dict[str, pl.DataFrame]) -> pd.DataFrame:
        """
        Extract ONLY the 15 essential features for each token (CEO requirement).
        
        Args:
            token_data: Dictionary mapping token names to DataFrames
            
        Returns:
            DataFrame with only 15 essential features for each token
        """
        import pandas as pd
        
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
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Print summary statistics
        self._print_data_summary(features_df)
        
        # Verify we have exactly 15 features (excluding metadata)
        feature_cols = [col for col in features_df.columns if col not in ['token', 'category']]
        print(f"\n✅ Using exactly {len(feature_cols)} features (target: 15)")
        print(f"Features: {feature_cols}")
        
        return features_df
    
    def _print_data_summary(self, features_df):
        """Print summary statistics about the data."""
        total_tokens = len(features_df)
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
            avg_lifespan = features_df[features_df['is_dead']]['lifespan_minutes'].mean()
            print(f"  Average lifespan: {avg_lifespan:.1f} minutes")
            
            # Show lifespan distribution
            short_death = (features_df[features_df['is_dead']]['lifespan_minutes'] <= 400).sum()
            medium_death = ((features_df[features_df['is_dead']]['lifespan_minutes'] > 400) & 
                           (features_df[features_df['is_dead']]['lifespan_minutes'] <= 1200)).sum()
            long_death = (features_df[features_df['is_dead']]['lifespan_minutes'] > 1200).sum()
            
            print(f"  Short lifespan (≤400 min): {short_death} ({short_death/dead_tokens*100:.1f}%)")
            print(f"  Medium lifespan (400-1200 min): {medium_death} ({medium_death/dead_tokens*100:.1f}%)")
            print(f"  Long lifespan (>1200 min): {long_death} ({long_death/dead_tokens*100:.1f}%)")
        
        # Category distribution
        print("\nCategory Distribution:")
        category_counts = features_df['category'].value_counts()
        for cat, count in category_counts.items():
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
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(sil_score)
            
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Find elbow using distance method (from autocorrelation_clustering.py)
        optimal_k_elbow = 3  # Default
        
        if len(inertias) >= 3:
            # Normalize the data to [0,1] range
            k_norm = np.array(list(k_range)) - min(k_range)
            k_norm = k_norm / max(k_norm) if max(k_norm) > 0 else k_norm
            
            inertias_norm = np.array(inertias) - min(inertias)
            inertias_norm = inertias_norm / max(inertias_norm) if max(inertias_norm) > 0 else inertias_norm
            
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
            elbow_idx = np.argmax(distances)
            optimal_k_elbow = list(k_range)[elbow_idx]
        
        # Find optimal K using silhouette score
        optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
        
        print(f"Elbow method suggests K={optimal_k_elbow}")
        print(f"Silhouette method suggests K={optimal_k_silhouette}")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette
        }

    def perform_clustering(self, features_df, n_clusters_range=[3, 4, 5, 6, 7, 8, 9, 10]) -> Dict:
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
        X = features_df[feature_cols].fillna(0).values
        
        print(f"Clustering with {len(feature_cols)} features: {feature_cols}")
        
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
    
    def identify_archetypes(self, features_df, clustering_results) -> Dict:
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
        features_df['cluster'] = labels
        
        # Analyze each cluster
        archetypes = {}
        
        print("\n" + "="*60)
        print("BEHAVIORAL ARCHETYPE ANALYSIS")
        print("="*60)
        
        for cluster_id in range(best_k):
            cluster_data = features_df[features_df['cluster'] == cluster_id]
            n_tokens = len(cluster_data)
            
            # Calculate cluster statistics using only the 15 features we have
            stats = {
                'n_tokens': n_tokens,
                'pct_of_total': n_tokens / len(features_df) * 100,
                'pct_dead': cluster_data['is_dead'].mean() * 100,
                'avg_lifespan': cluster_data['lifespan_minutes'].mean(),
                'avg_return_magnitude': cluster_data['return_magnitude_5min'].mean(),
                'avg_volatility_5min': cluster_data['volatility_5min'].mean(),
                'avg_mean_return': cluster_data['mean_return'].mean(),
                'avg_std_return': cluster_data['std_return'].mean(),
                'avg_max_drawdown': cluster_data['max_drawdown'].mean(),
                'avg_price_change_5min': cluster_data['price_change_ratio_5min'].mean()
            }
            
            # Determine archetype name based on characteristics
            archetype_name = self._determine_archetype_name(cluster_data, stats)
            
            # Get representative examples
            examples = cluster_data.nlargest(10, 'lifespan_minutes')['token'].tolist()
            
            # Extract ACF signature (average ACF values)
            acf_cols = [col for col in cluster_data.columns if col.startswith('acf_lag_')]
            if acf_cols:
                acf_signature = cluster_data[acf_cols].mean().to_dict()
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
            print(f"  Avg max return (5min): {stats['avg_max_return']*100:.1f}%")
            print(f"  Example tokens: {', '.join(examples[:3])}...")
        
        self.archetype_names = {i: arch['name'] for i, arch in archetypes.items()}
        return archetypes
    
    def _determine_archetype_name(self, cluster_data, stats) -> str:
        """Determine archetype name based on cluster characteristics."""
        pct_dead = stats['pct_dead']
        avg_lifespan = stats['avg_lifespan']
        avg_max_return = stats['avg_max_return']
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
    
    def create_early_detection_rules(self, features_df, archetypes) -> DecisionTreeClassifier:
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
        
        X = features_df[early_feature_cols].fillna(0).values
        y = features_df['cluster'].values
        
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
    
    def save_results(self, features_df, clustering_results, archetypes, output_dir: Path):
        """Save analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save archetype assignments
        assignments = features_df[['token', 'category', 'cluster', 'is_dead', 'lifespan_minutes']]
        assignments['archetype'] = assignments['cluster'].map(self.archetype_names)
        assignments.to_csv(output_dir / f"archetype_assignments_{timestamp}.csv", index=False)
        
        # Save archetype statistics
        archetype_stats = []
        for cluster_id, arch in archetypes.items():
            stats = arch['stats'].copy()
            stats['cluster_id'] = cluster_id
            stats['archetype_name'] = arch['name']
            archetype_stats.append(stats)
        
        import pandas as pd
        pd.DataFrame(archetype_stats).to_csv(
            output_dir / f"archetype_statistics_{timestamp}.csv", index=False
        )
        
        # Save comprehensive report
        report = {
            'timestamp': timestamp,
            'n_tokens': len(features_df),
            'n_archetypes': len(archetypes),
            'pct_dead': features_df['is_dead'].mean() * 100,
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
        import pandas as pd
        
        features_list = []
        
        for token_name, token_df in token_data.items():
            # Handle both pandas and polars DataFrames
            if hasattr(token_df, 'is_empty'):  # Polars DataFrame
                if token_df.is_empty():
                    continue
                df = token_df.to_pandas()
            else:  # Pandas DataFrame
                if len(token_df) == 0:
                    continue
                df = token_df
            
            # Basic preparation
            prices = df['price'].values
            returns = df['return'].values if 'return' in df.columns else np.diff(prices) / prices[:-1]
            
            # Skip if insufficient data
            if len(returns) < 10:
                continue
            
            # Death detection
            from .archetype_utils import detect_token_death
            death_minute = detect_token_death(prices, returns)
            is_dead = death_minute is not None
            lifespan_minutes = death_minute if is_dead else len(prices)
            
            # Core statistics (4 features)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            volatility_5min = np.std(returns[:5]) if len(returns) >= 5 else np.std(returns)
            max_drawdown = np.max(np.maximum.accumulate(prices) - prices) / np.max(prices)
            
            # ACF features (3 features)
            from statsmodels.tsa.stattools import acf
            try:
                acf_values = acf(returns, nlags=min(10, len(returns)//2), fft=True, missing='drop')
                acf_lag_1 = acf_values[1] if len(acf_values) > 1 else 0
                acf_lag_5 = acf_values[5] if len(acf_values) > 5 else 0
                acf_lag_10 = acf_values[10] if len(acf_values) > 10 else 0
            except:
                acf_lag_1 = acf_lag_5 = acf_lag_10 = 0
            
            # Early detection features (4 features)
            early_returns = returns[:5] if len(returns) >= 5 else returns
            return_magnitude_5min = np.mean(np.abs(early_returns))
            trend_direction_5min = np.mean(early_returns)
            price_change_ratio_5min = (prices[min(4, len(prices)-1)] - prices[0]) / prices[0]
            
            # Early autocorrelation
            try:
                early_acf = acf(early_returns, nlags=min(3, len(early_returns)//2), fft=True, missing='drop')
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
        
        return pd.DataFrame(features_list)
    
    def find_optimal_k_elbow(self, features_df, k_range):
        """
        Find optimal K using elbow method.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Validate input
        if features_df is None or len(features_df) == 0:
            print("Warning: Empty features DataFrame")
            return k_range[0]
        
        # Prepare features (exclude non-numeric)
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        if not feature_cols:
            print("Warning: No feature columns found")
            return k_range[0]
            
        X = features_df[feature_cols].values
        
        # Check if we have enough samples
        if X.shape[0] < k_range[0]:
            print(f"Warning: Not enough samples ({X.shape[0]}) for minimum K ({k_range[0]})")
            return min(2, X.shape[0]) if X.shape[0] > 1 else 1
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
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
    
    def run_kmeans_clustering(self, features_df, k):
        """
        Run K-means clustering with given K.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        # Validate input
        if features_df is None or len(features_df) == 0:
            print("Warning: Empty features DataFrame")
            return {'labels': [], 'silhouette_score': 0.0, 'k': k}
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        if not feature_cols:
            print("Warning: No feature columns found")
            return {'labels': [], 'silhouette_score': 0.0, 'k': k}
            
        X = features_df[feature_cols].values
        
        # Validate cluster count
        if k > len(X):
            print(f"Warning: K ({k}) cannot be larger than number of samples ({len(X)})")
            k = min(k, len(X))
        
        if k < 1:
            k = 1
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
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
    
    def test_clustering_stability(self, features_df, k, n_runs):
        """
        Test clustering stability using multiple runs.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import adjusted_rand_score
        
        # Validate input
        if features_df is None or len(features_df) == 0:
            print("Warning: Empty features DataFrame for stability testing")
            return {'ari_scores': [], 'average_ari': 0.0, 'min_ari': 0.0, 'max_ari': 0.0, 'n_runs': 0}
        
        # Prepare features
        feature_cols = [col for col in features_df.columns if col not in ['token']]
        if not feature_cols:
            print("Warning: No feature columns found for stability testing")
            return {'ari_scores': [], 'average_ari': 0.0, 'min_ari': 0.0, 'max_ari': 0.0, 'n_runs': 0}
            
        X = features_df[feature_cols].values
        
        # Validate cluster count
        if k > len(X):
            print(f"Warning: K ({k}) cannot be larger than number of samples ({len(X)}) for stability testing")
            k = min(k, len(X))
        
        if k < 1:
            k = 1
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run multiple times
        all_labels = []
        for i in range(n_runs):
            kmeans = KMeans(n_clusters=k, random_state=i, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            all_labels.append(labels)
        
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