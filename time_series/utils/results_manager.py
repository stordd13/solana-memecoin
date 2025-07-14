# results_manager.py
"""
Results management system for timestamped saving and loading of analysis results.
Supports JSON, CSV, and PNG exports with versioning.
"""

import json
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsManager:
    """
    Manages saving and loading of analysis results with timestamps and versioning.
    """
    
    def __init__(self, base_results_dir: Path):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_timestamp(self) -> str:
        """Generate timestamp string for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_analysis_results(self, results: Dict[str, Any], analysis_name: str, 
                            phase_dir: str, include_plots: bool = True) -> str:
        """
        Save complete analysis results with timestamp.
        
        Args:
            results: Analysis results dictionary
            analysis_name: Name of the analysis (e.g., 'baseline_assessment')
            phase_dir: Phase directory name (e.g., 'phase1_day1_2_baseline')
            include_plots: Whether to generate and save plots
            
        Returns:
            Timestamp string used for file naming
        """
        timestamp = self.generate_timestamp()
        results_dir = self.base_results_dir / phase_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        json_file = results_dir / f"{analysis_name}_{timestamp}.json"
        self._save_json_results(results, json_file)
        
        # Save feature data as CSV if available
        if 'features_dict' in results:
            csv_file = results_dir / f"{analysis_name}_features_{timestamp}.csv"
            self._save_features_csv(results['features_dict'], csv_file)
        
        # Save clustering assignments if available
        if 'final_clustering' in results and 'token_names' in results:
            assignments_file = results_dir / f"{analysis_name}_assignments_{timestamp}.csv"
            self._save_clustering_assignments(results, assignments_file)
        
        # Generate and save plots
        if include_plots:
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            self._generate_analysis_plots(results, plots_dir, analysis_name, timestamp)
        
        print(f"Results saved to {results_dir} with timestamp {timestamp}")
        return timestamp
    
    def _save_json_results(self, results: Dict[str, Any], file_path: Path):
        """Save results as JSON, handling numpy arrays."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_features_csv(self, features_dict: Dict[str, Dict[str, float]], file_path: Path):
        """Save features dictionary as CSV."""
        if not features_dict:
            return
        
        # Convert to DataFrame
        data = []
        for token_name, features in features_dict.items():
            row = {'token': token_name}
            row.update(features)
            data.append(row)
        
        df = pl.DataFrame(data)
        df.write_csv(file_path)
    
    def _save_clustering_assignments(self, results: Dict[str, Any], file_path: Path):
        """Save clustering assignments as CSV."""
        if 'final_clustering' not in results or 'token_names' not in results:
            return
        
        token_names = results['token_names']
        labels = results['final_clustering']['labels']
        
        df = pl.DataFrame({
            'token': token_names,
            'cluster': labels.tolist() if isinstance(labels, np.ndarray) else labels
        })
        df.write_csv(file_path)
    
    def _generate_analysis_plots(self, results: Dict[str, Any], plots_dir: Path, 
                               analysis_name: str, timestamp: str):
        """Generate and save analysis plots."""
        try:
            # K-selection plots
            if 'k_analysis' in results:
                self._plot_k_selection(results['k_analysis'], plots_dir, analysis_name, timestamp)
            
            # Stability plots
            if 'stability' in results:
                self._plot_stability(results['stability'], plots_dir, analysis_name, timestamp)
            
            # t-SNE plots
            if 'tsne_2d' in results and 'final_clustering' in results:
                self._plot_tsne(results, plots_dir, analysis_name, timestamp)
            
            # Feature correlation plot
            if 'features_dict' in results:
                self._plot_feature_correlation(results['features_dict'], plots_dir, analysis_name, timestamp)
                
        except Exception as e:
            print(f"Warning: Failed to generate some plots: {e}")
    
    def _plot_k_selection(self, k_analysis: Dict, plots_dir: Path, analysis_name: str, timestamp: str):
        """Generate K-selection plots (elbow and silhouette)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Elbow plot
        ax1.plot(k_analysis['k_range'], k_analysis['inertias'], 'bo-')
        ax1.axvline(k_analysis['optimal_k_elbow'], color='red', linestyle='--', 
                   label=f"Elbow K={k_analysis['optimal_k_elbow']}")
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(k_analysis['k_range'], k_analysis['silhouette_scores'], 'go-')
        ax2.axvline(k_analysis['optimal_k_silhouette'], color='red', linestyle='--',
                   label=f"Best Silhouette K={k_analysis['optimal_k_silhouette']}")
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{analysis_name}_k_selection_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability(self, stability: Dict, plots_dir: Path, analysis_name: str, timestamp: str):
        """Generate stability analysis plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ARI scores across runs
        runs = list(range(1, len(stability['ari_scores']) + 1))
        ax1.plot(runs, stability['ari_scores'], 'bo-')
        ax1.axhline(0.75, color='red', linestyle='--', label='CEO Threshold (0.75)')
        ax1.axhline(stability['mean_ari'], color='green', linestyle='-', 
                   label=f"Mean ARI ({stability['mean_ari']:.3f})")
        ax1.set_xlabel('Stability Run')
        ax1.set_ylabel('ARI Score')
        ax1.set_title('Stability Test - ARI Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores across runs
        ax2.plot(runs, stability['silhouette_scores'], 'go-')
        ax2.axhline(0.5, color='red', linestyle='--', label='CEO Threshold (0.5)')
        ax2.axhline(stability['mean_silhouette'], color='green', linestyle='-',
                   label=f"Mean Silhouette ({stability['mean_silhouette']:.3f})")
        ax2.set_xlabel('Stability Run')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Stability Test - Silhouette Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{analysis_name}_stability_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tsne(self, results: Dict, plots_dir: Path, analysis_name: str, timestamp: str):
        """Generate t-SNE visualization."""
        tsne_coords = results['tsne_2d']
        labels = results['final_clustering']['labels']
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Token Clusters')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(plots_dir / f"{analysis_name}_tsne_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlation(self, features_dict: Dict, plots_dir: Path, 
                                analysis_name: str, timestamp: str):
        """Generate feature correlation heatmap."""
        if not features_dict:
            return
        
        # Convert to DataFrame for correlation calculation
        data = []
        for features in features_dict.values():
            data.append(list(features.values()))
        
        if not data:
            return
        
        feature_names = list(next(iter(features_dict.values())).keys())
        df = pl.DataFrame(data, schema=feature_names)
        
        # Calculate correlation matrix
        corr_matrix = df.to_pandas().corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        plt.savefig(plots_dir / f"{analysis_name}_correlation_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_analysis_results(self, analysis_name: str, phase_dir: str, 
                            timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from file.
        
        Args:
            analysis_name: Name of the analysis
            phase_dir: Phase directory name
            timestamp: Specific timestamp to load (if None, loads most recent)
            
        Returns:
            Analysis results dictionary or None if not found
        """
        results_dir = self.base_results_dir / phase_dir
        
        if not results_dir.exists():
            return None
        
        # Find matching files
        if timestamp:
            json_file = results_dir / f"{analysis_name}_{timestamp}.json"
        else:
            # Find most recent file
            pattern = f"{analysis_name}_*.json"
            json_files = list(results_dir.glob(pattern))
            if not json_files:
                return None
            json_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        if not json_file.exists():
            return None
        
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    
    def list_available_results(self, phase_dir: str) -> List[Dict[str, str]]:
        """
        List all available analysis results in a phase directory.
        
        Args:
            phase_dir: Phase directory name
            
        Returns:
            List of dictionaries with analysis info
        """
        results_dir = self.base_results_dir / phase_dir
        
        if not results_dir.exists():
            return []
        
        json_files = list(results_dir.glob("*.json"))
        
        results = []
        for json_file in json_files:
            parts = json_file.stem.split('_')
            if len(parts) >= 3:
                analysis_name = '_'.join(parts[:-2])
                timestamp = '_'.join(parts[-2:])
                
                results.append({
                    'analysis_name': analysis_name,
                    'timestamp': timestamp,
                    'file_path': str(json_file),
                    'modified_time': datetime.fromtimestamp(json_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return results