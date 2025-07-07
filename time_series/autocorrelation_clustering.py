"""
Time Series Autocorrelation and Clustering Analysis
Focuses on raw price/log price analysis without cleaning or feature engineering
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from dtaidistance import dtw
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm


class AutocorrelationClusteringAnalyzer:
    """
    Analyze time series using autocorrelation, clustering, and dimensionality reduction
    Works directly with raw prices and log prices
    """
    
    def __init__(self):
        self.max_lag = 100  # Maximum lag for autocorrelation
        self.n_clusters = 5  # Default number of clusters
        
    def load_raw_prices(self, data_dir: Path, limit: Optional[int] = None) -> Dict[str, pl.DataFrame]:
        """
        Load raw price data from parquet files
        
        Args:
            data_dir: Directory containing token parquet files
            limit: Optional limit on number of tokens to load
            
        Returns:
            Dictionary mapping token names to DataFrames with price data
        """
        token_data = {}
        
        # Find all parquet files
        parquet_files = list(data_dir.rglob("*.parquet"))
        if limit:
            parquet_files = parquet_files[:limit]
            
        print(f"Loading {len(parquet_files)} token files...")
        
        for file_path in tqdm(parquet_files, desc="Loading tokens"):
            try:
                df = pl.read_parquet(file_path)
                token_name = file_path.stem
                
                # Ensure we have datetime and price columns
                if 'datetime' in df.columns and 'price' in df.columns:
                    # Sort by datetime
                    df = df.sort('datetime')
                    
                    # Add log price
                    df = df.with_columns([
                        pl.col('price').log().alias('log_price')
                    ])
                    
                    token_data[token_name] = df
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        print(f"Successfully loaded {len(token_data)} tokens")
        return token_data
    
    def compute_autocorrelation(self, series: np.ndarray, max_lag: Optional[int] = None) -> Dict:
        """
        Compute autocorrelation and partial autocorrelation for a series
        
        Args:
            series: Price or log-price series
            max_lag: Maximum lag to compute
            
        Returns:
            Dictionary with ACF, PACF values and statistics
        """
        if max_lag is None:
            max_lag = min(self.max_lag, len(series) // 4)
            
        # Remove NaN values
        series_clean = series[~np.isnan(series)]
        
        if len(series_clean) < 3:  # Need minimum 3 points for meaningful ACF
            return {
                'acf': np.array([]),
                'pacf': np.array([]),
                'significant_lags': [],
                'decay_rate': np.nan
            }
        
        # Adjust max_lag if series is too short
        max_lag = min(max_lag, len(series_clean) // 2 - 1, len(series_clean) - 2)
        max_lag = max(max_lag, 1)  # Ensure at least lag 1
        
        # Compute ACF and PACF
        acf_values = acf(series_clean, nlags=max_lag, fft=True)
        pacf_values = pacf(series_clean, nlags=max_lag)
        
        # Find significant lags (outside 95% confidence interval)
        n = len(series_clean)
        confidence_interval = 1.96 / np.sqrt(n)
        significant_lags = np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
        
        # Estimate decay rate (for AR processes)
        # Find first zero crossing or minimum
        zero_crossings = np.where(np.diff(np.sign(acf_values)))[0]
        if len(zero_crossings) > 0:
            decay_rate = -np.log(np.abs(acf_values[1])) if acf_values[1] != 0 else np.inf
        else:
            decay_rate = np.nan
            
        return {
            'acf': acf_values,
            'pacf': pacf_values,
            'significant_lags': significant_lags.tolist(),
            'decay_rate': decay_rate,
            'first_zero_crossing': int(zero_crossings[0]) if len(zero_crossings) > 0 else max_lag
        }
    
    def compute_all_autocorrelations(self, token_data: Dict[str, pl.DataFrame], 
                                   use_log_price: bool = True) -> Dict[str, Dict]:
        """
        Compute autocorrelation for all tokens
        
        Args:
            token_data: Dictionary of token DataFrames
            use_log_price: Whether to use log prices instead of raw prices
            
        Returns:
            Dictionary mapping token names to autocorrelation results
        """
        results = {}
        price_col = 'log_price' if use_log_price else 'price'
        
        print(f"Computing autocorrelations for {len(token_data)} tokens...")
        
        for token_name, df in tqdm(token_data.items(), desc="Computing ACF"):
            if price_col in df.columns:
                prices = df[price_col].to_numpy()
                results[token_name] = self.compute_autocorrelation(prices)
            else:
                print(f"Warning: {token_name} missing {price_col} column")
                
        return results
    
    def extract_time_series_features(self, series: np.ndarray, acf_result: Dict) -> np.ndarray:
        """
        Extract features from time series for clustering
        
        Features include:
        - ACF values at specific lags
        - Statistical moments
        - Trend characteristics
        - Volatility measures
        
        Returns exactly 16 features for consistent clustering
        """
        # Remove NaN values
        series_clean = series[~np.isnan(series)]
        
        if len(series_clean) < 10:
            return np.zeros(16)  # Return zero features if too short
        
        features = []
        
        # Basic statistics (5 features)
        features.extend([
            np.mean(series_clean),
            np.std(series_clean),
            np.median(series_clean),
            np.percentile(series_clean, 25),
            np.percentile(series_clean, 75)
        ])
        
        # Returns statistics (4 features)
        returns = np.diff(series_clean) / series_clean[:-1]
        returns_clean = returns[~np.isnan(returns)]
        
        if len(returns_clean) > 0:
            features.extend([
                np.mean(returns_clean),
                np.std(returns_clean),
                np.min(returns_clean),
                np.max(returns_clean)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # ACF features (6 features)
        if 'acf' in acf_result and len(acf_result['acf']) > 10:
            # ACF at lags 1, 5, 10
            features.extend([
                acf_result['acf'][1],
                acf_result['acf'][5],
                acf_result['acf'][10]
            ])
            # Number of significant lags
            features.append(len(acf_result['significant_lags']))
            # Decay rate
            features.append(acf_result['decay_rate'] if not np.isnan(acf_result['decay_rate']) else 0)
            # First zero crossing
            features.append(acf_result['first_zero_crossing'])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Trend measure (1 feature)
        if len(series_clean) > 1:
            x = np.arange(len(series_clean))
            slope = np.polyfit(x, series_clean, 1)[0]
            features.append(slope)
        else:
            features.append(0)
        
        # Convert to numpy array and handle any remaining NaN/inf values
        features = np.array(features, dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure exactly 16 features (5 + 4 + 6 + 1 = 16)
        if len(features) != 16:
            print(f"Warning: Expected 16 features, got {len(features)}. Padding/truncating...")
            if len(features) < 16:
                features = np.pad(features, (0, 16 - len(features)), 'constant')
            else:
                features = features[:16]
        
        return features
    
    def prepare_clustering_data(self, token_data: Dict[str, pl.DataFrame], 
                              acf_results: Dict[str, Dict],
                              use_log_price: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for clustering
        
        Returns:
            Feature matrix and list of token names
        """
        price_col = 'log_price' if use_log_price else 'price'
        
        feature_matrix = []
        token_names = []
        
        for token_name, df in token_data.items():
            if token_name in acf_results and price_col in df.columns:
                prices = df[price_col].to_numpy()
                features = self.extract_time_series_features(prices, acf_results[token_name])
                
                # Check for NaN values in features
                if not np.any(np.isnan(features)):
                    feature_matrix.append(features)
                    token_names.append(token_name)
                else:
                    print(f"Warning: Skipping {token_name} due to NaN features")
                    
        feature_matrix = np.array(feature_matrix)
        
        # Final check for any remaining NaN values
        if np.any(np.isnan(feature_matrix)):
            print("Warning: NaN values detected in feature matrix, applying imputation...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            feature_matrix = imputer.fit_transform(feature_matrix)
                
        return feature_matrix, token_names
    
    def find_optimal_clusters(self, feature_matrix: np.ndarray, 
                            max_clusters: int = None,
                            max_k: int = None) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            feature_matrix: Feature matrix for clustering
            max_clusters: Maximum number of clusters to test (default: 15)
            max_k: Alternative parameter name for max_clusters
            
        Returns:
            Dictionary with elbow analysis results and optimal K suggestions
        """
        # Handle both parameter names for backwards compatibility
        if max_k is not None:
            max_clusters = max_k
        elif max_clusters is None:
            max_clusters = 15
            
        from sklearn.metrics import silhouette_score
        
        # Add feature validation
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Feature variance: {np.var(feature_matrix, axis=0)}")
        print(f"Features with zero variance: {np.sum(np.var(feature_matrix, axis=0) == 0)}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Check for any remaining issues after scaling
        print(f"Scaled features - any NaN: {np.any(np.isnan(features_scaled))}")
        print(f"Scaled features - any Inf: {np.any(np.isinf(features_scaled))}")
        
        # Test different numbers of clusters
        k_range = range(2, min(max_clusters + 1, len(features_scaled)))
        inertias = []
        silhouette_scores = []
        
        print(f"Testing K from 2 to {max(k_range)}...")
        
        for k in tqdm(k_range, desc="Finding optimal K"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(features_scaled):  # Need at least 2 clusters but not all points in separate clusters
                try:
                    sil_score = silhouette_score(features_scaled, labels)
                    silhouette_scores.append(sil_score)
                    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}, Clusters found={len(unique_labels)}")
                except Exception as e:
                    print(f"K={k}: Error calculating silhouette score: {e}")
                    silhouette_scores.append(0)
            else:
                print(f"K={k}: Invalid clustering (found {len(unique_labels)} unique labels), silhouette=0")
                silhouette_scores.append(0)
        
        # Find elbow using improved "knee" detection
        if len(inertias) >= 3:
            # Method 1: Use the "kneedle" algorithm approach
            # Normalize the data to [0,1] range
            k_norm = np.array(list(k_range)) - min(k_range)
            k_norm = k_norm / max(k_norm) if max(k_norm) > 0 else k_norm
            
            inertias_norm = np.array(inertias) - min(inertias)
            inertias_norm = inertias_norm / max(inertias_norm) if max(inertias_norm) > 0 else inertias_norm
            
            # Calculate distance from each point to line connecting first and last points
            distances = []
            for i in range(len(k_norm)):
                # Distance from point to line y = mx + b where line connects (k_norm[0], inertias_norm[0]) to (k_norm[-1], inertias_norm[-1])
                x1, y1 = k_norm[0], inertias_norm[0]
                x2, y2 = k_norm[-1], inertias_norm[-1]
                x0, y0 = k_norm[i], inertias_norm[i]
                
                # Distance formula: |ax0 + by0 + c| / sqrt(a^2 + b^2)
                # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
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
            
            # Method 2: Also try second derivative approach for comparison
            first_deriv = np.diff(inertias)
            second_deriv = np.diff(first_deriv)
            
            # For elbow, we want the point where rate of decrease slows down most
            # This corresponds to the most negative second derivative
            second_deriv_elbow_idx = np.argmin(second_deriv) + 2  # +2 because of double diff
            second_deriv_optimal_k = list(k_range)[second_deriv_elbow_idx] if second_deriv_elbow_idx < len(k_range) else list(k_range)[-1]
            
            # Use the distance method as primary, but store both
            print(f"Elbow detection: Distance method suggests K={optimal_k_elbow}, Second derivative suggests K={second_deriv_optimal_k}")
            
        else:
            optimal_k_elbow = 3  # Default fallback
        
        # Find optimal K using silhouette score
        optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
        
        # Debug information
        print(f"K range tested: {list(k_range)}")
        print(f"Inertias: {[f'{x:.2f}' for x in inertias]}")
        print(f"Silhouette scores: {[f'{x:.3f}' for x in silhouette_scores]}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f} at K={optimal_k_silhouette}")
        print(f"Selected elbow K: {optimal_k_elbow}")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'scaler': scaler
        }

    def perform_clustering(self, feature_matrix: np.ndarray, 
                         method: str = 'kmeans',
                         n_clusters: Optional[int] = None,
                         find_optimal_k: bool = False) -> Dict:
        """
        Perform time series clustering
        
        Args:
            feature_matrix: Feature matrix for clustering
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (for methods that require it)
            find_optimal_k: Whether to find optimal number of clusters first
            
        Returns:
            Dictionary with cluster labels and clustering object
        """
        # Handle NaN values if any remain
        if np.any(np.isnan(feature_matrix)):
            print("Warning: NaN values in feature matrix, applying median imputation...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            feature_matrix = imputer.fit_transform(feature_matrix)
        
        # Find optimal number of clusters if requested
        elbow_analysis = None
        if find_optimal_k and method == 'kmeans':
            elbow_analysis = self.find_optimal_clusters(feature_matrix)
            if n_clusters is None:
                # Use silhouette score as primary criterion, elbow as secondary
                n_clusters = elbow_analysis['optimal_k_silhouette']
                print(f"Optimal K found: {n_clusters} (silhouette), {elbow_analysis['optimal_k_elbow']} (elbow)")
        
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        # Validate n_clusters against data size
        n_samples = feature_matrix.shape[0]
        if method in ['kmeans', 'hierarchical'] and n_clusters > n_samples:
            print(f"Warning: n_clusters ({n_clusters}) > n_samples ({n_samples}), adjusting to {n_samples}")
            n_clusters = n_samples
        
        # For very small datasets, ensure minimum viable clustering
        if method in ['kmeans', 'hierarchical'] and n_clusters > n_samples // 2 and n_samples > 2:
            n_clusters = max(2, n_samples // 2)
            print(f"Adjusting n_clusters to {n_clusters} for small dataset")
            
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(features_scaled)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        result = {
            'labels': labels,
            'clusterer': clusterer,
            'scaler': scaler,
            'n_clusters': len(np.unique(labels[labels >= 0])),  # Exclude noise points for DBSCAN
            'elbow_analysis': elbow_analysis
        }
        
        return result
    
    def compute_tsne(self, feature_matrix: np.ndarray, 
                    n_components: int = 2,
                    perplexity: float = 30.0) -> np.ndarray:
        """
        Compute t-SNE embedding for visualization
        
        Args:
            feature_matrix: Feature matrix
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            
        Returns:
            t-SNE embedded coordinates
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply PCA first if high dimensional or if we have more features than samples
        n_samples, n_features = features_scaled.shape
        
        if n_features > 50 and n_samples > 50:
            # For large datasets, reduce to 50 components
            pca = PCA(n_components=50)
            features_scaled = pca.fit_transform(features_scaled)
        elif n_features > n_samples:
            # If more features than samples, reduce dimensions
            n_components = min(n_samples - 1, n_features // 2)
            if n_components > 1 and n_components < n_features:
                pca = PCA(n_components=n_components)
                features_scaled = pca.fit_transform(features_scaled)
            
        # Adjust perplexity for small datasets
        adjusted_perplexity = min(perplexity, (features_scaled.shape[0] - 1) / 3.0)
        adjusted_perplexity = max(adjusted_perplexity, 1.0)
        
        # Use exact method for small datasets or high dimensions
        method = 'exact' if features_scaled.shape[0] < 1000 or n_components >= 4 else 'barnes_hut'
        
        # Compute t-SNE
        tsne = TSNE(n_components=n_components, 
                   perplexity=adjusted_perplexity,
                   random_state=42,
                   max_iter=1000,
                   method=method)
        
        embedding = tsne.fit_transform(features_scaled)
        
        return embedding
    
    def create_visualization_plots(self, results: Dict) -> Dict[str, go.Figure]:
        """
        Create comprehensive visualization plots
        
        Args:
            results: Dictionary containing all analysis results
            
        Returns:
            Dictionary of plotly figures
        """
        figures = {}
        
        # 1. t-SNE scatter plot with clusters
        if 't_sne_2d' in results:
            fig = go.Figure()
            
            embedding = results['t_sne_2d']
            labels = results['cluster_labels']
            token_names = results['token_names']
            
            # Create scatter plot for each cluster
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                fig.add_trace(go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers+text',
                    name=f'Cluster {cluster_id}',
                    text=[token_names[i] for i in np.where(mask)[0]],
                    textposition='top center',
                    marker=dict(size=10),
                    hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ))
                
            fig.update_layout(
                title='t-SNE Visualization of Token Clusters',
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                height=600,
                showlegend=True
            )
            
            figures['tsne_2d'] = fig
            
        # 2. 3D t-SNE plot if available
        if 't_sne_3d' in results:
            fig = go.Figure()
            
            embedding = results['t_sne_3d']
            labels = results['cluster_labels']
            
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                fig.add_trace(go.Scatter3d(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    z=embedding[mask, 2],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    text=[token_names[i] for i in np.where(mask)[0]],
                    marker=dict(size=5),
                    hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
                ))
                
            fig.update_layout(
                title='3D t-SNE Visualization',
                scene=dict(
                    xaxis_title='t-SNE 1',
                    yaxis_title='t-SNE 2',
                    zaxis_title='t-SNE 3'
                ),
                height=700
            )
            
            figures['tsne_3d'] = fig
            
        # 3. ACF heatmap for clusters
        if 'acf_by_cluster' in results:
            fig = make_subplots(
                rows=1, cols=len(results['acf_by_cluster']),
                subplot_titles=[f'Cluster {i}' for i in sorted(results['acf_by_cluster'].keys())]
            )
            
            for idx, (cluster_id, acf_data) in enumerate(sorted(results['acf_by_cluster'].items())):
                # Average ACF for cluster
                avg_acf = np.mean(acf_data, axis=0)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(avg_acf))),
                        y=avg_acf,
                        mode='lines',
                        name=f'Cluster {cluster_id}',
                        showlegend=False
                    ),
                    row=1, col=idx+1
                )
                
            fig.update_layout(
                title='Average ACF by Cluster',
                height=400,
                showlegend=False
            )
            
            figures['acf_clusters'] = fig
        
        # 4. Elbow curve if available
        if 'clustering_results' in results and results['clustering_results']['elbow_analysis']:
            elbow_data = results['clustering_results']['elbow_analysis']
            
            # Create subplot with elbow curve and silhouette scores
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Elbow Method (Inertia)', 'Silhouette Score'),
                x_title='Number of Clusters (K)'
            )
            
            # Elbow curve
            fig.add_trace(
                go.Scatter(
                    x=elbow_data['k_range'],
                    y=elbow_data['inertias'],
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Mark optimal K from elbow
            optimal_k_elbow = elbow_data['optimal_k_elbow']
            if optimal_k_elbow in elbow_data['k_range']:
                idx = elbow_data['k_range'].index(optimal_k_elbow)
                fig.add_trace(
                    go.Scatter(
                        x=[optimal_k_elbow],
                        y=[elbow_data['inertias'][idx]],
                        mode='markers',
                        name=f'Elbow K={optimal_k_elbow}',
                        marker=dict(color='red', size=12, symbol='star')
                    ),
                    row=1, col=1
                )
            
            # Silhouette scores
            fig.add_trace(
                go.Scatter(
                    x=elbow_data['k_range'],
                    y=elbow_data['silhouette_scores'],
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='green', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )
            
            # Mark optimal K from silhouette
            optimal_k_sil = elbow_data['optimal_k_silhouette']
            if optimal_k_sil in elbow_data['k_range']:
                idx = elbow_data['k_range'].index(optimal_k_sil)
                fig.add_trace(
                    go.Scatter(
                        x=[optimal_k_sil],
                        y=[elbow_data['silhouette_scores'][idx]],
                        mode='markers',
                        name=f'Best Silhouette K={optimal_k_sil}',
                        marker=dict(color='orange', size=12, symbol='star')
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title='Optimal Number of Clusters Analysis',
                height=400,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
            fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
            fig.update_yaxes(title_text="Inertia", row=1, col=1)
            fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            
            figures['elbow_analysis'] = fig
            
        return figures
    
    def analyze_cluster_characteristics(self, token_data: Dict[str, pl.DataFrame],
                                      cluster_labels: np.ndarray,
                                      token_names: List[str]) -> Dict:
        """
        Analyze characteristics of each cluster
        
        Returns:
            Dictionary with cluster statistics and characteristics
        """
        cluster_stats = {}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_tokens = [token_names[i] for i in np.where(mask)[0]]
            
            # Collect price statistics for cluster
            price_stats = []
            returns_stats = []
            lengths = []
            
            for token in cluster_tokens:
                if token in token_data:
                    df = token_data[token]
                    prices = df['price'].to_numpy()
                    returns = np.diff(prices) / prices[:-1]
                    
                    price_stats.append({
                        'mean': np.mean(prices),
                        'std': np.std(prices),
                        'min': np.min(prices),
                        'max': np.max(prices)
                    })
                    
                    returns_stats.append({
                        'mean': np.mean(returns),
                        'std': np.std(returns),
                        'skew': np.mean((returns - np.mean(returns))**3) / np.std(returns)**3,
                        'kurtosis': np.mean((returns - np.mean(returns))**4) / np.std(returns)**4
                    })
                    
                    lengths.append(len(prices))
                    
            cluster_stats[cluster_id] = {
                'n_tokens': len(cluster_tokens),
                'tokens': cluster_tokens[:10],  # First 10 tokens
                'avg_length': np.mean(lengths),
                'price_characteristics': {
                    'avg_volatility': np.mean([s['std'] for s in price_stats]),
                    'avg_return': np.mean([s['mean'] for s in returns_stats]),
                    'avg_return_volatility': np.mean([s['std'] for s in returns_stats])
                }
            }
            
        return cluster_stats

    def run_complete_analysis(self, data_dir: Path, 
                            use_log_price: bool = True,
                            n_clusters: Optional[int] = None,
                            find_optimal_k: bool = True,
                            clustering_method: str = 'kmeans',
                            max_tokens: Optional[int] = None) -> Dict:
        """
        Run complete autocorrelation and clustering analysis
        
        Args:
            data_dir: Directory containing token data
            use_log_price: Whether to use log prices
            n_clusters: Number of clusters (if None, will find optimal)
            find_optimal_k: Whether to find optimal number of clusters
            clustering_method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            max_tokens: Maximum number of tokens to analyze (None = no limit)
            
        Returns:
            Dictionary with all results
        """
        # Load data (no limit by default)
        token_data = self.load_raw_prices(data_dir, limit=max_tokens)
        
        if len(token_data) == 0:
            raise ValueError("No token data loaded")
            
        print(f"Loaded {len(token_data)} tokens for analysis")
            
        # Compute autocorrelations
        acf_results = self.compute_all_autocorrelations(token_data, use_log_price)
        
        # Prepare clustering data
        feature_matrix, token_names = self.prepare_clustering_data(token_data, acf_results, use_log_price)
        
        if len(feature_matrix) == 0:
            raise ValueError("No valid features extracted from tokens")
            
        print(f"Prepared feature matrix: {feature_matrix.shape}")
        
        # Perform clustering with optimal K finding
        clustering_results = self.perform_clustering(
            feature_matrix, 
            method=clustering_method, 
            n_clusters=n_clusters,
            find_optimal_k=find_optimal_k
        )
        
        # Compute t-SNE (applied directly on tokens, not clusters!)
        print("Computing t-SNE embeddings...")
        tsne_2d = self.compute_tsne(feature_matrix, n_components=2)
        tsne_3d = self.compute_tsne(feature_matrix, n_components=3)
        
        # Analyze clusters
        cluster_stats = self.analyze_cluster_characteristics(
            token_data, 
            clustering_results['labels'], 
            token_names
        )
        
        # Organize ACF by cluster
        acf_by_cluster = {}
        for cluster_id in np.unique(clustering_results['labels']):
            mask = clustering_results['labels'] == cluster_id
            cluster_tokens = [token_names[i] for i in np.where(mask)[0]]
            cluster_acfs = [acf_results[token]['acf'] for token in cluster_tokens if token in acf_results]
            
            # Pad to same length
            max_len = max(len(acf) for acf in cluster_acfs)
            padded_acfs = []
            for acf in cluster_acfs:
                padded = np.pad(acf, (0, max_len - len(acf)), mode='constant', constant_values=0)
                padded_acfs.append(padded)
                
            acf_by_cluster[cluster_id] = np.array(padded_acfs)
            
        # Compile results
        results = {
            'token_data': token_data,
            'acf_results': acf_results,
            'feature_matrix': feature_matrix,
            'token_names': token_names,
            'cluster_labels': clustering_results['labels'],
            'clustering_results': clustering_results,
            't_sne_2d': tsne_2d,
            't_sne_3d': tsne_3d,
            'cluster_stats': cluster_stats,
            'acf_by_cluster': acf_by_cluster,
            'use_log_price': use_log_price,
            'n_clusters': clustering_results['n_clusters']  # Use actual number of clusters found
        }
        
        return results
    
    def prepare_price_only_data(self, token_data: Dict[str, pl.DataFrame], 
                               method: str = 'returns', 
                               use_log_price: bool = True,
                               max_length: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare price data for clustering without feature engineering
        
        Args:
            token_data: Dictionary of token DataFrames
            method: 'returns', 'log_returns', 'prices', 'log_prices', or 'dtw_features'
            use_log_price: Whether to use log prices
            max_length: Maximum sequence length (None = use minimum length across all tokens)
            
        Returns:
            Price matrix and list of token names
        """
        from scipy.stats import skew
        
        price_col = 'log_price' if use_log_price else 'price'
        
        valid_tokens = []
        price_series = []
        
        # First pass: collect all valid price series
        for token_name, df in token_data.items():
            if price_col in df.columns:
                prices = df[price_col].to_numpy()
                if len(prices) > 10:  # Minimum length requirement
                    valid_tokens.append(token_name)
                    price_series.append(prices)
        
        if not price_series:
            raise ValueError("No valid price series found")
        
        # Determine sequence length based on method
        lengths = [len(series) for series in price_series]
        
        if method in ['returns', 'log_returns']:
            # For returns, we need N+1 prices to get N returns
            max_returns_length = min(lengths) - 1
            if max_length is None:
                target_length = max_returns_length
            else:
                target_length = min(max_length, max_returns_length)
        else:
            # For prices and other methods, use the series length directly
            if max_length is None:
                target_length = min(lengths)
            else:
                target_length = min(max_length, min(lengths))
        
        print(f"Using sequence length: {target_length} (method: {method}, range: {min(lengths)}-{max(lengths)})")
        
        # Second pass: prepare data based on method
        feature_matrix = []
        final_tokens = []
        
        for i, prices in enumerate(price_series):
            if method == 'returns':
                # Use returns (more stationary)
                # We already calculated target_length to account for this
                if len(prices) >= target_length + 1:
                    returns = np.diff(prices[-(target_length+1):])  # Get target_length returns
                    if len(returns) == target_length:
                        feature_matrix.append(returns)
                        final_tokens.append(valid_tokens[i])
                    else:
                        print(f"Unexpected returns length for {valid_tokens[i]}: {len(returns)} vs {target_length}")
                else:
                    print(f"Insufficient data for {valid_tokens[i]}: {len(prices)} points, need {target_length+1} for {target_length} returns")
                    
            elif method == 'log_returns':
                # Use log returns
                if len(prices) >= target_length + 1:
                    log_returns = np.diff(np.log(prices[-(target_length+1):]))
                    if len(log_returns) == target_length:
                        # Remove any inf/nan values
                        if not np.any(np.isnan(log_returns)) and not np.any(np.isinf(log_returns)):
                            feature_matrix.append(log_returns)
                            final_tokens.append(valid_tokens[i])
                        else:
                            print(f"Skipping {valid_tokens[i]} due to invalid log returns")
                    else:
                        print(f"Unexpected log returns length for {valid_tokens[i]}: {len(log_returns)} vs {target_length}")
                else:
                    print(f"Insufficient data for {valid_tokens[i]}: {len(prices)} points, need {target_length+1} for {target_length} log returns")
                        
            elif method == 'prices':
                # Use raw prices directly (regardless of use_log_price setting)
                raw_prices_col = 'price'  # Always use raw prices for this method
                if raw_prices_col in token_data[valid_tokens[i]].columns:
                    raw_prices = token_data[valid_tokens[i]][raw_prices_col].to_numpy()
                    segment = raw_prices[-target_length:]
                    if len(segment) >= target_length:
                        feature_matrix.append(segment)
                        final_tokens.append(valid_tokens[i])
                        
            elif method == 'log_prices':
                # Use log-transformed prices directly
                segment = prices[-target_length:]
                if len(segment) >= target_length:
                    # For log prices, we already have log values if use_log_price=True
                    # Otherwise, take log of the segment
                    if not use_log_price:
                        segment = np.log(segment)
                    feature_matrix.append(segment)
                    final_tokens.append(valid_tokens[i])
                    
            elif method == 'dtw_features':
                # Extract statistical features from the raw series
                segment = prices[-target_length:]
                if len(segment) >= target_length:
                    # Extract features: quantiles, autocorrelations, etc.
                    features = []
                    # Quantiles
                    quantiles = np.percentile(segment, [10, 25, 50, 75, 90])
                    features.extend(quantiles)
                    # Returns stats
                    returns = np.diff(segment)
                    if len(returns) > 0:
                        mean_ret = np.mean(returns)
                        std_ret = np.std(returns)
                        skew_ret = skew(returns) if len(returns) > 2 else 0
                        features.extend([mean_ret, std_ret, skew_ret])
                    else:
                        features.extend([0, 0, 0])
                    # Trend
                    trend_slope = np.polyfit(np.arange(len(segment)), segment, 1)[0]
                    features.append(trend_slope)
                    # Volatility in different windows
                    for window in [5, 10, 20]:
                        if len(returns) >= window:
                            rolling_vol = [np.std(returns[j:j+window]) for j in range(len(returns)-window+1)]
                            vol_mean = np.mean(rolling_vol)
                            features.append(vol_mean)
                        else:
                            features.append(0)
                    
                    # Convert to numpy array and clean NaN/inf values
                    features = np.array(features, dtype=float)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Only add if no NaN/inf values remain
                    if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        feature_matrix.append(features)
                        final_tokens.append(valid_tokens[i])
                    else:
                        print(f"Skipping {valid_tokens[i]} due to invalid DTW features")
            else:
                raise ValueError(f"Unknown method: {method}")
        
        feature_matrix = np.array(feature_matrix)
        
        # Final check for NaN/inf values and clean if necessary
        if len(feature_matrix) > 0:
            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                print("Warning: Found NaN/inf values in feature matrix, cleaning...")
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Price-only feature matrix shape: {feature_matrix.shape}")
        print(f"Method: {method}, Valid tokens: {len(final_tokens)}")
        
        return feature_matrix, final_tokens
    
    def run_price_only_analysis(self, data_dir: Path,
                               method: str = 'returns',
                               use_log_price: bool = True,
                               n_clusters: Optional[int] = None,
                               find_optimal_k: bool = True,
                               clustering_method: str = 'kmeans',
                               max_tokens: Optional[int] = None,
                               max_length: Optional[int] = None) -> Dict:
        """
        Run clustering analysis using only price data with ACF computation
        
        Args:
            data_dir: Directory containing token data
            method: 'returns', 'log_returns', 'prices', 'log_prices', or 'dtw_features'
            use_log_price: Whether to use log prices
            n_clusters: Number of clusters (if None, will find optimal)
            find_optimal_k: Whether to find optimal number of clusters
            clustering_method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            max_tokens: Maximum number of tokens to analyze
            max_length: Maximum sequence length for price series
            
        Returns:
            Dictionary with all results including ACF
        """
        # Load data
        token_data = self.load_raw_prices(data_dir, limit=max_tokens)
        
        if len(token_data) == 0:
            raise ValueError("No token data loaded")
            
        print(f"Loaded {len(token_data)} tokens for price-only analysis")
        
        # Compute autocorrelations for all tokens (this is what ACF is for!)
        print("Computing autocorrelations for all tokens...")
        acf_results = self.compute_all_autocorrelations(token_data, use_log_price)
        
        # Prepare price-only data for clustering
        feature_matrix, token_names = self.prepare_price_only_data(
            token_data, method=method, use_log_price=use_log_price, max_length=max_length
        )
        
        if len(feature_matrix) == 0:
            raise ValueError("No valid price data extracted from tokens")
            
        print(f"Prepared price feature matrix: {feature_matrix.shape}")
        
        # Perform clustering
        clustering_results = self.perform_clustering(
            feature_matrix, 
            method=clustering_method, 
            n_clusters=n_clusters,
            find_optimal_k=find_optimal_k
        )
        
        # Compute t-SNE
        print("Computing t-SNE embeddings for price data...")
        tsne_2d = self.compute_tsne(feature_matrix, n_components=2)
        tsne_3d = self.compute_tsne(feature_matrix, n_components=3)
        
        # Analyze clusters (only for tokens we have data for)
        filtered_token_data = {token: token_data[token] for token in token_names if token in token_data}
        cluster_stats = self.analyze_cluster_characteristics(
            filtered_token_data, 
            clustering_results['labels'], 
            token_names
        )
        
        # Organize ACF by cluster (same as feature-based analysis)
        acf_by_cluster = {}
        for cluster_id in np.unique(clustering_results['labels']):
            mask = clustering_results['labels'] == cluster_id
            cluster_tokens = [token_names[i] for i in np.where(mask)[0]]
            cluster_acfs = [acf_results[token]['acf'] for token in cluster_tokens if token in acf_results]
            
            if cluster_acfs:
                # Pad to same length
                max_len = max(len(acf) for acf in cluster_acfs)
                padded_acfs = []
                for acf in cluster_acfs:
                    padded = np.pad(acf, (0, max_len - len(acf)), mode='constant', constant_values=0)
                    padded_acfs.append(padded)
                    
                acf_by_cluster[cluster_id] = np.array(padded_acfs)
        
        # Compile results
        results = {
            'token_data': filtered_token_data,
            'acf_results': acf_results,  # Include ACF results!
            'feature_matrix': feature_matrix,
            'token_names': token_names,
            'cluster_labels': clustering_results['labels'],
            'clustering_results': clustering_results,
            't_sne_2d': tsne_2d,
            't_sne_3d': tsne_3d,
            'cluster_stats': cluster_stats,
            'acf_by_cluster': acf_by_cluster,  # Include ACF by cluster
            'use_log_price': use_log_price,
            'n_clusters': clustering_results['n_clusters'],
            'analysis_method': f'price_only_{method}',
            'sequence_length': feature_matrix.shape[1] if method != 'dtw_features' else 'variable'
        }
        
        return results

    # ================================
    # PHASE 1A: MULTI-RESOLUTION ACF ANALYSIS METHODS
    # ================================
    
    def analyze_by_lifespan_category(self, data_dir: Path, 
                                   method: str = 'returns',
                                   use_log_price: bool = True,
                                   max_tokens_per_category: Optional[int] = None) -> Dict:
        """
        Analyze tokens by lifespan categories: Sprint, Standard, Marathon
        
        Args:
            data_dir: Directory containing token data
            method: Price transformation method
            use_log_price: Whether to use log prices
            max_tokens_per_category: Maximum tokens per category
            
        Returns:
            Dictionary with results for each lifespan category
        """
        # Load all token data
        all_token_data = self.load_raw_prices(data_dir)
        
        # Categorize tokens by lifespan
        categories = {
            'Sprint': {},      # 200-400 minutes
            'Standard': {},    # 400-1200 minutes  
            'Marathon': {}     # 1200+ minutes
        }
        
        for token_name, df in all_token_data.items():
            lifespan = len(df)
            
            if 200 <= lifespan <= 400:
                categories['Sprint'][token_name] = df
            elif 400 < lifespan <= 1200:
                categories['Standard'][token_name] = df
            elif lifespan > 1200:
                categories['Marathon'][token_name] = df
        
        print(f"Token distribution by lifespan:")
        for category, tokens in categories.items():
            print(f"  {category}: {len(tokens)} tokens")
            
        # Analyze each category separately
        results_by_category = {}
        
        for category_name, token_data in categories.items():
            if len(token_data) == 0:
                print(f"Skipping {category_name} - no tokens in this category")
                continue
                
            print(f"\nAnalyzing {category_name} category ({len(token_data)} tokens)...")
            
            # Limit tokens per category if specified
            if max_tokens_per_category and len(token_data) > max_tokens_per_category:
                # Sample randomly to get diverse representation
                import random
                token_names = list(token_data.keys())
                sampled_names = random.sample(token_names, max_tokens_per_category)
                token_data = {name: token_data[name] for name in sampled_names}
                print(f"  Sampled {len(token_data)} tokens for analysis")
            
            try:
                # For category-specific analysis, we need to work with the subset
                # Create a temporary directory-like structure
                temp_results = {}
                
                # Compute ACF for category tokens
                acf_results = self.compute_all_autocorrelations(token_data, use_log_price)
                
                # Prepare price data for clustering
                feature_matrix, token_names = self.prepare_price_only_data(
                    token_data, method=method, use_log_price=use_log_price
                )
                
                if len(feature_matrix) == 0:
                    print(f"No valid feature matrix for {category_name}")
                    continue
                
                # Perform clustering
                clustering_results = self.perform_clustering(
                    feature_matrix, 
                    method='kmeans',
                    find_optimal_k=True
                )
                
                # Compute t-SNE
                tsne_2d = self.compute_tsne(feature_matrix, n_components=2)
                tsne_3d = self.compute_tsne(feature_matrix, n_components=3)
                
                # Analyze clusters
                cluster_stats = self.analyze_cluster_characteristics(
                    token_data, 
                    clustering_results['labels'], 
                    token_names
                )
                
                # Organize ACF by cluster
                acf_by_cluster = {}
                for cluster_id in np.unique(clustering_results['labels']):
                    mask = clustering_results['labels'] == cluster_id
                    cluster_tokens = [token_names[i] for i in np.where(mask)[0]]
                    cluster_acfs = [acf_results[token]['acf'] for token in cluster_tokens if token in acf_results]
                    
                    if cluster_acfs:
                        max_len = max(len(acf) for acf in cluster_acfs)
                        padded_acfs = []
                        for acf in cluster_acfs:
                            padded = np.pad(acf, (0, max_len - len(acf)), mode='constant', constant_values=0)
                            padded_acfs.append(padded)
                            
                        acf_by_cluster[cluster_id] = np.array(padded_acfs)
                
                # Compile category results
                category_results = {
                    'token_data': token_data,
                    'acf_results': acf_results,
                    'feature_matrix': feature_matrix,
                    'token_names': token_names,
                    'cluster_labels': clustering_results['labels'],
                    'clustering_results': clustering_results,
                    't_sne_2d': tsne_2d,
                    't_sne_3d': tsne_3d,
                    'cluster_stats': cluster_stats,
                    'acf_by_cluster': acf_by_cluster,
                    'use_log_price': use_log_price,
                    'n_clusters': clustering_results['n_clusters'],
                    'category': category_name,
                    'lifespan_range': self._get_lifespan_range(category_name),
                    'analysis_method': f'multi_resolution_{method}'
                }
                
                results_by_category[category_name] = category_results
                
            except Exception as e:
                print(f"Error analyzing {category_name}: {e}")
                continue
        
        # Compile multi-resolution results
        multi_resolution_results = {
            'categories': results_by_category,
            'category_summary': {
                category: {
                    'n_tokens': len(results['token_data']),
                    'n_clusters': results['n_clusters'],
                    'lifespan_range': results['lifespan_range']
                }
                for category, results in results_by_category.items()
            },
            'analysis_method': f'multi_resolution_{method}',
            'total_tokens_analyzed': sum(len(results['token_data']) for results in results_by_category.values())
        }
        
        return multi_resolution_results
    
    def _get_lifespan_range(self, category_name: str) -> str:
        """Get lifespan range description for category"""
        ranges = {
            'Sprint': '200-400 minutes',
            'Standard': '400-1200 minutes', 
            'Marathon': '1200+ minutes'
        }
        return ranges.get(category_name, 'Unknown')
    
    def compute_multi_resolution_acf(self, token_data: Dict[str, pl.DataFrame],
                                   time_horizons: List[int] = None) -> Dict:
        """
        Compute ACF at multiple time horizons for cross-resolution analysis
        
        Args:
            token_data: Dictionary of token DataFrames
            time_horizons: List of maximum lags to compute (default: [20, 50, 100])
            
        Returns:
            Dictionary with ACF results at different time horizons
        """
        if time_horizons is None:
            time_horizons = [20, 50, 100]  # Short, medium, long horizons
            
        multi_resolution_acf = {}
        
        for horizon in time_horizons:
            print(f"Computing ACF with max lag {horizon}...")
            
            # Temporarily set max_lag for this horizon
            original_max_lag = self.max_lag
            self.max_lag = horizon
            
            try:
                # Compute ACF for all tokens at this horizon
                acf_results = self.compute_all_autocorrelations(token_data, use_log_price=True)
                multi_resolution_acf[f'horizon_{horizon}'] = acf_results
                
            except Exception as e:
                print(f"Error computing ACF for horizon {horizon}: {e}")
                multi_resolution_acf[f'horizon_{horizon}'] = {}
            finally:
                # Restore original max_lag
                self.max_lag = original_max_lag
        
        return multi_resolution_acf
    
    def dtw_clustering_variable_length(self, token_data: Dict[str, pl.DataFrame],
                                     use_log_price: bool = True,
                                     n_clusters: int = 5,
                                     max_tokens: Optional[int] = None) -> Dict:
        """
        Perform DTW-based clustering for variable-length sequences
        
        Args:
            token_data: Dictionary of token DataFrames
            use_log_price: Whether to use log prices
            n_clusters: Number of clusters
            max_tokens: Maximum number of tokens to analyze
            
        Returns:
            Dictionary with DTW clustering results
        """
        from dtaidistance import dtw
        from sklearn.cluster import AgglomerativeClustering
        
        # Prepare data
        price_col = 'log_price' if use_log_price else 'price'
        
        valid_tokens = []
        price_series = []
        
        # Collect price series (keeping original lengths)
        for token_name, df in token_data.items():
            if price_col in df.columns:
                prices = df[price_col].to_numpy()
                if len(prices) >= 50:  # Minimum length for meaningful DTW
                    valid_tokens.append(token_name)
                    price_series.append(prices)
                    
        if len(price_series) == 0:
            raise ValueError("No valid price series for DTW clustering")
            
        # Limit tokens if specified
        if max_tokens and len(price_series) > max_tokens:
            import random
            indices = random.sample(range(len(price_series)), max_tokens)
            price_series = [price_series[i] for i in indices]
            valid_tokens = [valid_tokens[i] for i in indices]
        
        print(f"Computing DTW distances for {len(price_series)} variable-length sequences...")
        
        # Compute DTW distance matrix
        n_series = len(price_series)
        distance_matrix = np.zeros((n_series, n_series))
        
        for i in tqdm(range(n_series), desc="Computing DTW distances"):
            for j in range(i+1, n_series):
                try:
                    # Use DTW distance with window constraint for efficiency
                    window_size = max(10, int(0.1 * max(len(price_series[i]), len(price_series[j]))))
                    dist = dtw.distance(price_series[i], price_series[j], window=window_size)
                    
                    # Handle invalid distance values
                    if np.isnan(dist) or np.isinf(dist) or dist < 0:
                        # Use Euclidean distance of summary statistics as fallback
                        mean_diff = abs(np.mean(price_series[i]) - np.mean(price_series[j]))
                        std_diff = abs(np.std(price_series[i]) - np.std(price_series[j]))
                        dist = mean_diff + std_diff
                    
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
                    
                except Exception as e:
                    print(f"Error computing DTW between series {i} and {j}: {e}")
                    # Use Euclidean distance of summary statistics as fallback
                    try:
                        mean_diff = abs(np.mean(price_series[i]) - np.mean(price_series[j]))
                        std_diff = abs(np.std(price_series[i]) - np.std(price_series[j]))
                        fallback_dist = mean_diff + std_diff
                        distance_matrix[i, j] = fallback_dist
                        distance_matrix[j, i] = fallback_dist
                    except:
                        # Last resort: use a large but finite distance
                        distance_matrix[i, j] = 1000.0
                        distance_matrix[j, i] = 1000.0
        
        # Validate distance matrix before clustering
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            print("Warning: Distance matrix contains NaN or inf values, cleaning...")
            distance_matrix = np.nan_to_num(distance_matrix, nan=1000.0, posinf=1000.0, neginf=0.0)
        
        # Ensure matrix is symmetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Perform hierarchical clustering on DTW distances
        try:
            # Use precomputed distance matrix
            clusterer = AgglomerativeClustering(
                n_clusters=min(n_clusters, len(valid_tokens)), 
                metric='precomputed',
                linkage='average'
            )
            labels = clusterer.fit_predict(distance_matrix)
            
        except Exception as e:
            print(f"DTW clustering failed: {e}")
            # Fallback to random assignment
            labels = np.random.randint(0, min(n_clusters, len(valid_tokens)), size=len(valid_tokens))
        
        # Analyze cluster characteristics
        cluster_stats = {}
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_tokens = [valid_tokens[i] for i in np.where(mask)[0]]
            cluster_lengths = [len(price_series[i]) for i in np.where(mask)[0]]
            
            cluster_stats[cluster_id] = {
                'n_tokens': int(np.sum(mask)),
                'token_names': cluster_tokens,
                'avg_length': float(np.mean(cluster_lengths)),
                'std_length': float(np.std(cluster_lengths)),
                'min_length': int(np.min(cluster_lengths)),
                'max_length': int(np.max(cluster_lengths))
            }
        
        return {
            'labels': labels,
            'token_names': valid_tokens,
            'distance_matrix': distance_matrix,
            'cluster_stats': cluster_stats,
            'n_clusters': len(np.unique(labels)),
            'method': 'dtw_variable_length'
        }
    
    def compare_acf_across_lifespans(self, multi_resolution_results: Dict) -> Dict:
        """
        Compare ACF patterns across different lifespan categories
        
        Args:
            multi_resolution_results: Results from analyze_by_lifespan_category()
            
        Returns:
            Dictionary with cross-lifespan ACF comparison
        """
        if 'categories' not in multi_resolution_results:
            raise ValueError("Invalid multi-resolution results format")
            
        acf_comparison = {
            'category_acf_means': {},
            'category_acf_stds': {},
            'cross_category_correlations': {},
            'distinctive_patterns': {}
        }
        
        categories = multi_resolution_results['categories']
        
        # Compute mean ACF patterns for each category
        for category_name, results in categories.items():
            if 'acf_by_cluster' in results:
                all_acfs = []
                
                # Collect all ACF patterns in this category
                for cluster_id, cluster_acfs in results['acf_by_cluster'].items():
                    for acf in cluster_acfs:
                        all_acfs.append(acf)
                
                if all_acfs:
                    # Pad to same length
                    max_len = max(len(acf) for acf in all_acfs)
                    padded_acfs = []
                    for acf in all_acfs:
                        padded = np.pad(acf, (0, max_len - len(acf)), mode='constant', constant_values=0)
                        padded_acfs.append(padded)
                    
                    acf_matrix = np.array(padded_acfs)
                    
                    # Compute statistics
                    acf_comparison['category_acf_means'][category_name] = np.mean(acf_matrix, axis=0)
                    acf_comparison['category_acf_stds'][category_name] = np.std(acf_matrix, axis=0)
        
        # Compute cross-category correlations
        category_names = list(acf_comparison['category_acf_means'].keys())
        for i, cat1 in enumerate(category_names):
            for j, cat2 in enumerate(category_names):
                if i <= j:  # Upper triangular matrix
                    mean1 = acf_comparison['category_acf_means'][cat1]
                    mean2 = acf_comparison['category_acf_means'][cat2]
                    
                    # Trim to same length
                    min_len = min(len(mean1), len(mean2))
                    corr = np.corrcoef(mean1[:min_len], mean2[:min_len])[0, 1]
                    
                    acf_comparison['cross_category_correlations'][f'{cat1}_vs_{cat2}'] = corr
        
        # Identify distinctive patterns (largest differences)
        if len(category_names) >= 2:
            for cat in category_names:
                mean_acf = acf_comparison['category_acf_means'][cat]
                
                # Compare with other categories
                other_cats = [c for c in category_names if c != cat]
                if other_cats:
                    other_means = [acf_comparison['category_acf_means'][c] for c in other_cats]
                    
                    # Find lags where this category differs most
                    max_differences = []
                    for lag in range(min(len(mean_acf), min(len(m) for m in other_means))):
                        cat_value = mean_acf[lag]
                        other_values = [m[lag] for m in other_means]
                        avg_other = np.mean(other_values)
                        difference = abs(cat_value - avg_other)
                        max_differences.append(difference)
                    
                    # Find top distinctive lags
                    distinctive_lags = np.argsort(max_differences)[-5:][::-1]  # Top 5
                    
                    acf_comparison['distinctive_patterns'][cat] = {
                        'distinctive_lags': distinctive_lags.tolist(),
                        'differences': [max_differences[lag] for lag in distinctive_lags]
                    }
        
        return acf_comparison 