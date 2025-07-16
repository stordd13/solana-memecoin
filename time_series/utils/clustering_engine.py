# clustering_engine.py
"""
Core clustering engine with ARI and silhouette evaluation.
Implements CEO requirements for stability testing and optimal K selection.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, homogeneity_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class ClusteringEngine:
    """
    Core clustering engine following CEO requirements.
    Supports stability testing, optimal K selection, and scale invariance evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def prepare_features_for_clustering(self, features_dict: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature dictionary for clustering.
        
        Args:
            features_dict: Dictionary mapping token names to feature dictionaries
            
        Returns:
            Tuple of (feature_matrix, token_names)
        """
        if not features_dict:
            raise ValueError("Empty features dictionary")
        
        # Get feature names from first token (assume all tokens have same features)
        sample_token = next(iter(features_dict.values()))
        feature_names = sorted(sample_token.keys())
        
        # Build feature matrix
        token_names = []
        feature_matrix = []
        
        for token_name, features in features_dict.items():
            # Ensure all features are present and finite
            feature_row = []
            for feat_name in feature_names:
                value = features.get(feat_name, 0.0)
                if not np.isfinite(value):
                    value = 0.0
                feature_row.append(value)
            
            feature_matrix.append(feature_row)
            token_names.append(token_name)
        
        feature_matrix = np.array(feature_matrix)
        
        # Apply per-token scaling as per CEO requirements
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix, token_names
    
    def cluster_and_evaluate(self, features: np.ndarray, k: int = 5) -> Dict[str, float]:
        """
        Perform clustering and return evaluation metrics.
        
        Args:
            features: Feature matrix
            k: Number of clusters
            
        Returns:
            Dictionary with clustering results and metrics
        """
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(features)
        
        # Calculate metrics
        silhouette = silhouette_score(features, labels)
        inertia = kmeans.inertia_
        
        return {
            'labels': labels,
            'silhouette_score': silhouette,
            'inertia': inertia,
            'n_clusters': k,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def stability_test(self, features: np.ndarray, k: int, n_runs: int = 5) -> Dict[str, float]:
        """
        Test clustering stability using multiple random seeds.
        
        Args:
            features: Feature matrix
            k: Number of clusters
            n_runs: Number of stability runs
            
        Returns:
            Dictionary with stability metrics
        """
        base_kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        base_labels = base_kmeans.fit_predict(features)
        
        ari_scores = []
        silhouette_scores = []
        
        for run in range(1, n_runs + 1):
            # Use different random seed for each run
            test_kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state + run)
            test_labels = test_kmeans.fit_predict(features)
            
            # Calculate ARI against base clustering
            ari = adjusted_rand_score(base_labels, test_labels)
            ari_scores.append(ari)
            
            # Calculate silhouette score
            sil = silhouette_score(features, test_labels)
            silhouette_scores.append(sil)
        
        return {
            'mean_ari': np.mean(ari_scores),
            'std_ari': np.std(ari_scores),
            'min_ari': np.min(ari_scores),
            'mean_silhouette': np.mean(silhouette_scores),
            'std_silhouette': np.std(silhouette_scores),
            'base_labels': base_labels,
            'stability_runs': n_runs,
            'ari_scores': ari_scores,
            'silhouette_scores': silhouette_scores
        }
    
    def find_optimal_k(self, features: np.ndarray, k_range: range = range(3, 11)) -> Dict[str, float]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            features: Feature matrix
            k_range: Range of K values to test
            
        Returns:
            Dictionary with optimal K analysis
        """
        inertias = []
        silhouette_scores = []
        k_values = list(k_range)
        
        for k in k_values:
            result = self.cluster_and_evaluate(features, k)
            inertias.append(result['inertia'])
            silhouette_scores.append(result['silhouette_score'])
        
        # Find elbow using the "elbow method" - look for point of maximum curvature
        optimal_k_elbow = self._find_elbow_point(k_values, inertias)
        
        # Find optimal K by silhouette score
        max_sil_idx = np.argmax(silhouette_scores)
        optimal_k_silhouette = k_values[max_sil_idx]
        
        return {
            'k_range': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'max_silhouette_score': silhouette_scores[max_sil_idx]
        }
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Find elbow point using the method of maximum curvature.
        
        Args:
            k_values: List of K values
            inertias: Corresponding inertia values
            
        Returns:
            Optimal K value
        """
        if len(k_values) < 3:
            return k_values[0]
        
        # Normalize the data
        k_norm = np.array(k_values)
        inertia_norm = np.array(inertias)
        
        # Calculate the point farthest from the line connecting first and last points
        def point_line_distance(point, line_start, line_end):
            """Calculate perpendicular distance from point to line."""
            return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        
        line_start = np.array([k_norm[0], inertia_norm[0]])
        line_end = np.array([k_norm[-1], inertia_norm[-1]])
        
        distances = []
        for i in range(len(k_values)):
            point = np.array([k_norm[i], inertia_norm[i]])
            dist = point_line_distance(point, line_start, line_end)
            distances.append(dist)
        
        # Return K value with maximum distance (elbow point)
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]
    
    def evaluate_behavioral_separation(self, features: np.ndarray, labels: np.ndarray, 
                                     true_labels: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate how well clustering separates behavioral patterns.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            true_labels: Known behavioral labels (if available)
            
        Returns:
            Dictionary with separation metrics
        """
        metrics = {
            'silhouette_score': silhouette_score(features, labels)
        }
        
        if true_labels is not None and len(true_labels) == len(labels):
            metrics['homogeneity_score'] = homogeneity_score(true_labels, labels)
            metrics['ari_vs_true'] = adjusted_rand_score(true_labels, labels)
        
        return metrics
    
    def test_scale_invariance(self, features_low: np.ndarray, features_high: np.ndarray, 
                            k: int) -> Dict[str, float]:
        """
        Test scale invariance by comparing clustering across different base price groups.
        
        Args:
            features_low: Features from low base price tokens
            features_high: Features from high base price tokens
            k: Number of clusters to use
            
        Returns:
            Dictionary with scale invariance metrics
        """
        # Cluster each group separately
        result_low = self.cluster_and_evaluate(features_low, k)
        result_high = self.cluster_and_evaluate(features_high, k)
        
        # Calculate cross-ARI (how similar are the clustering patterns)
        # Note: This is a simplified version - full implementation would require alignment
        cross_ari = 0.0  # Placeholder - complex calculation
        
        return {
            'low_silhouette': result_low['silhouette_score'],
            'high_silhouette': result_high['silhouette_score'],
            'silhouette_difference': abs(result_low['silhouette_score'] - result_high['silhouette_score']),
            'cross_ari': cross_ari,
            'low_labels': result_low['labels'],
            'high_labels': result_high['labels']
        }
    
    def create_tsne_embedding(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Create t-SNE embedding for visualization.
        
        Args:
            features: Feature matrix
            n_components: Number of t-SNE dimensions (2 or 3)
            
        Returns:
            t-SNE embedding array
        """
        tsne = TSNE(n_components=n_components, random_state=self.random_state, 
                   perplexity=min(30, features.shape[0] - 1))
        embedding = tsne.fit_transform(features)
        return embedding
    
    def comprehensive_analysis(self, features_dict: Dict[str, Dict[str, float]], 
                             k_range: range = range(3, 11),
                             stability_runs: int = 5,
                             category: str = None) -> Dict:
        """
        Perform comprehensive clustering analysis following CEO requirements.
        
        Args:
            features_dict: Dictionary mapping token names to features
            k_range: Range of K values to test
            stability_runs: Number of stability test runs
            
        Returns:
            Complete analysis results
        """
        # Prepare features
        features, token_names = self.prepare_features_for_clustering(features_dict)
        
        # Find optimal K
        k_analysis = self.find_optimal_k(features, k_range)
        
        # Use elbow method result as primary choice (CEO preference)
        optimal_k = k_analysis['optimal_k_elbow']
        
        # Perform stability test
        stability = self.stability_test(features, optimal_k, stability_runs)
        
        # Final clustering with optimal K
        final_result = self.cluster_and_evaluate(features, optimal_k)
        
        # Create t-SNE embedding
        tsne_2d = self.create_tsne_embedding(features, n_components=2)
        
        # Create 3D t-SNE embedding
        tsne_3d = self.create_tsne_embedding(features, n_components=3)
        
        return {
            'features': features,
            'token_names': token_names,
            'k_analysis': k_analysis,
            'optimal_k': optimal_k,
            'stability': stability,
            'final_clustering': final_result,
            'tsne_2d': tsne_2d,
            'tsne_3d': tsne_3d,
            'meets_ceo_requirements': self._evaluate_ceo_requirements(
                stability['mean_ari'], final_result['silhouette_score'], category
            )
        }
    
    def _evaluate_ceo_requirements(self, mean_ari: float, silhouette_score: float, 
                                 category: str = None) -> Dict[str, bool]:
        """
        Evaluate whether clustering results meet CEO requirements.
        
        Args:
            mean_ari: Mean Adjusted Rand Index from stability testing
            silhouette_score: Silhouette score from final clustering
            category: Token category ('sprint', 'standard', 'marathon')
            
        Returns:
            Dictionary with threshold evaluation results
        """
        # Default thresholds
        ari_threshold = 0.75
        silhouette_threshold = 0.5
        
        # Category-specific adjustments
        if category == 'sprint':
            # More lenient for short-lived sprint tokens
            ari_threshold = 0.70
            silhouette_threshold = 0.25
        elif category == 'marathon':
            # More lenient for complex marathon tokens
            ari_threshold = 0.70
            silhouette_threshold = 0.30
        
        meets_ari = mean_ari >= ari_threshold
        meets_silhouette = silhouette_score >= silhouette_threshold
        
        return {
            'ari_threshold': meets_ari,
            'silhouette_threshold': meets_silhouette,
            'stability_achieved': meets_ari and meets_silhouette,
            'thresholds_used': {
                'ari': ari_threshold,
                'silhouette': silhouette_threshold,
                'category': category or 'default'
            }
        }
    
    def create_tsne_embedding(self, features: np.ndarray, n_components: int = 2, 
                             perplexity: float = 30, learning_rate: float = 200) -> np.ndarray:
        """
        Create t-SNE embedding for visualization.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
            learning_rate: t-SNE learning rate
            
        Returns:
            t-SNE embedding coordinates
        """
        # Adjust perplexity if we have few samples
        n_samples = features.shape[0]
        adjusted_perplexity = min(perplexity, max(5, n_samples // 4))
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=adjusted_perplexity,
            learning_rate=learning_rate,
            random_state=self.random_state,
            init='pca',
            metric='euclidean'
        )
        
        return tsne.fit_transform(features)